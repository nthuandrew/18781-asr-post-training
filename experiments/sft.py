"""Config-driven supervised fine-tuning for OWSM-style ASR models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from experiments.asr_common import (
    ASRDataset,
    CommonConfig,
    JsonlLogger,
    OWSMModel,
    evaluate,
    load_yaml,
    parse_common_config,
    print_and_save_comparison,
    resolve_training_steps,
    set_seed,
    write_resolved_config,
)


@dataclass
class SFTTrainingConfig:
    steps: int | None = 10
    epochs: int | None = None
    learning_rate: float = 3.0e-5
    max_grad_norm: float = 1.0
    batch_size: int = 1


@dataclass
class SFTConfig:
    common: CommonConfig
    training: SFTTrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return {**self.common.to_dict(), "training": asdict(self.training)}


def load_config(path: str | Path) -> SFTConfig:
    values = load_yaml(path)
    return SFTConfig(
        common=parse_common_config(values),
        training=SFTTrainingConfig(**values.get("training", {})),
    )


def train(config: SFTConfig, train_set: ASRDataset) -> Path:
    runner = OWSMModel(config.common.model, config.common.runtime)
    optimizer = torch.optim.Adam(
        runner.model.parameters(), lr=config.training.learning_rate
    )
    history: list[dict[str, Any]] = []
    total_steps = resolve_training_steps(
        steps=config.training.steps,
        epochs=config.training.epochs,
        dataset_size=len(train_set),
    )
    if config.training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    progress = tqdm(
        total=total_steps,
        desc="SFT train",
        unit="sample",
        dynamic_ncols=True,
    )
    with JsonlLogger(config.common.runtime.output_dir, "training_log.jsonl") as logger:
        for optimizer_step, batch_start in enumerate(
            range(0, total_steps, config.training.batch_size)
        ):
            batch_end = min(batch_start + config.training.batch_size, total_steps)
            samples = [
                train_set[step % len(train_set)] for step in range(batch_start, batch_end)
            ]
            if runner.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runner.device)
            optimizer.zero_grad(set_to_none=True)
            loss, component_metrics = runner.sft_loss_batch(samples)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                runner.model.parameters(), config.training.max_grad_norm
            )
            optimizer.step()
            speech_lengths = [len(sample["speech"]) for sample in samples]
            record: dict[str, Any] = {
                "optimizer_step": optimizer_step,
                "sample_step_start": batch_start,
                "sample_step_end": batch_end - 1,
                "epoch": batch_start // len(train_set) + 1,
                "batch_size": len(samples),
                "loss": float(loss.detach().cpu()),
                "component_metrics": component_metrics,
                "sample_ids": [sample["id"] for sample in samples],
                "speech_samples_min": min(speech_lengths),
                "speech_samples_max": max(speech_lengths),
                "speech_seconds_total": float(
                    sum(speech_lengths) / config.common.dataset.sampling_rate
                ),
                "reference_words_total": sum(
                    len(sample["reference"].split()) for sample in samples
                ),
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if runner.device.type == "cuda":
                record["gpu_memory_allocated_gb"] = float(
                    torch.cuda.memory_allocated(runner.device) / 1024**3
                )
                record["gpu_memory_reserved_gb"] = float(
                    torch.cuda.memory_reserved(runner.device) / 1024**3
                )
                record["gpu_peak_memory_allocated_gb"] = float(
                    torch.cuda.max_memory_allocated(runner.device) / 1024**3
                )
                record["gpu_peak_memory_reserved_gb"] = float(
                    torch.cuda.max_memory_reserved(runner.device) / 1024**3
                )
            history.append(record)
            logger.write(record)
            progress.update(len(samples))
            progress.set_postfix(
                loss=f"{record['loss']:.4f}", grad_norm=f"{record['grad_norm']:.4f}"
            )
    progress.close()

    path = Path(config.common.runtime.output_dir) / "sft.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": runner.model.state_dict(),
            "history": history,
            "losses": [float(row["loss"]) for row in history],
            "config": config.to_dict(),
        },
        path,
    )
    del runner
    torch.cuda.empty_cache()
    return path


def run(config: SFTConfig, stage: str = "all") -> dict[str, dict[str, float]]:
    common = config.common
    if common.runtime.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seed(common.runtime.seed)
    write_resolved_config(common.runtime.output_dir, config.to_dict())
    train_set = ASRDataset(common.dataset, train=True, seed=common.runtime.seed)
    eval_set = ASRDataset(common.dataset, train=False, seed=common.runtime.seed)
    checkpoint = Path(common.runtime.output_dir) / "sft.pt"
    if stage in {"all", "train"}:
        checkpoint = train(config, train_set)

    summaries: dict[str, dict[str, float]] = {}
    if stage in {"all", "eval"}:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing SFT checkpoint: {checkpoint}")
        summaries["base"] = evaluate(common, eval_set, None, "base")
        summaries["sft"] = evaluate(common, eval_set, checkpoint, "sft")
        print_and_save_comparison(common.runtime.output_dir, summaries)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--stage", choices=("all", "train", "eval"), default="all")
    args = parser.parse_args(argv)
    run(load_config(args.config), args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
