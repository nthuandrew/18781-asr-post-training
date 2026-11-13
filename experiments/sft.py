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
    OWSMModel,
    evaluate,
    load_yaml,
    parse_common_config,
    print_and_save_comparison,
    set_seed,
    write_resolved_config,
)


@dataclass
class SFTTrainingConfig:
    steps: int = 10
    learning_rate: float = 3.0e-5
    max_grad_norm: float = 1.0


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
    losses: list[float] = []
    progress = tqdm(
        range(config.training.steps),
        desc="SFT train",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
        sample = train_set[step % len(train_set)]
        optimizer.zero_grad(set_to_none=True)
        loss = runner.sft_loss(sample)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runner.model.parameters(), config.training.max_grad_norm
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(
            loss=f"{losses[-1]:.4f}", grad_norm=f"{float(grad_norm):.4f}"
        )

    path = Path(config.common.runtime.output_dir) / "sft.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": runner.model.state_dict(), "losses": losses, "config": config.to_dict()},
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
