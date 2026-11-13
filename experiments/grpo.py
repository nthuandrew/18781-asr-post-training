"""Config-driven group-relative policy optimization for ASR."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from evaluation.metrics import compute_format_reward
from experiments.asr_common import (
    ASRDataset,
    CommonConfig,
    JsonlLogger,
    OWSMModel,
    RewardConfig,
    evaluate,
    load_yaml,
    parse_common_config,
    print_and_save_comparison,
    resolve_training_steps,
    set_seed,
    write_resolved_config,
)


@dataclass
class GRPOTrainingConfig:
    steps: int | None = 5
    epochs: int | None = None
    learning_rate: float = 1.0e-6
    num_hypotheses: int = 4
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    max_grad_norm: float = 1.0
    log_candidates: bool = False
    optimization_epochs: int = 1
    batch_size: int = 1


@dataclass
class GRPOConfig:
    common: CommonConfig
    training: GRPOTrainingConfig
    reward: RewardConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.common.to_dict(),
            "training": asdict(self.training),
            "reward": asdict(self.reward),
        }


def load_config(path: str | Path) -> GRPOConfig:
    values = load_yaml(path)
    return GRPOConfig(
        common=parse_common_config(values),
        training=GRPOTrainingConfig(**values.get("training", {})),
        reward=RewardConfig(**values.get("reward", {})),
    )


def group_advantages(rewards: Sequence[float], eps: float = 1.0e-6) -> list[float]:
    if not rewards:
        return []
    values = np.asarray(rewards, dtype=np.float32)
    if len(values) == 1 or float(values.std()) < eps:
        return [0.0 for _ in rewards]
    normalized = (values - values.mean()) / (values.std() + eps)
    return [float(value) for value in normalized]


def train(config: GRPOConfig, train_set: ASRDataset) -> Path:
    common = config.common
    training = config.training
    runner = OWSMModel(
        common.model, common.runtime, nbest=training.num_hypotheses
    )
    reference = copy.deepcopy(runner.model).eval()
    reference.requires_grad_(False)
    optimizer = torch.optim.Adam(
        runner.model.parameters(), lr=training.learning_rate
    )
    history: list[dict[str, Any]] = []
    total_steps = resolve_training_steps(
        steps=training.steps,
        epochs=training.epochs,
        dataset_size=len(train_set),
    )
    if training.optimization_epochs <= 0:
        raise ValueError("training.optimization_epochs must be positive")
    if training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")

    progress = tqdm(
        range(total_steps),
        desc="GRPO train",
        unit="step",
        dynamic_ncols=True,
    )
    pending: list[dict[str, Any]] = []
    optimizer_step = 0
    skipped = 0

    def update_batch() -> None:
        nonlocal optimizer_step
        if not pending:
            return
        samples = [group["sample"] for group in pending]
        hypothesis_groups = [group["hypotheses"] for group in pending]
        runner.model.eval()
        if runner.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runner.device)
        with torch.no_grad():
            old_logps = runner.batch_sequence_log_probs(
                samples, hypothesis_groups, length_normalize=True
            )
            policy_model = runner.model
            runner.model = reference
            reference_logps = runner.batch_sequence_log_probs(
                samples, hypothesis_groups, length_normalize=True
            )
            runner.model = policy_model
        for group, old_logp, reference_logp in zip(
            pending, old_logps, reference_logps
        ):
            group["old_logp"] = old_logp
            group["reference_logp"] = reference_logp
            group["sample_record"]["old_policy_logps"] = [
                float(value) for value in old_logp.cpu()
            ]
            group["sample_record"]["reference_logps"] = [
                float(value) for value in reference_logp.cpu()
            ]
            sample_logger.write(group["sample_record"])
        for optimization_epoch in range(training.optimization_epochs):
            optimizer.zero_grad(set_to_none=True)
            group_metrics: list[dict[str, Any]] = []
            group_losses: list[torch.Tensor] = []
            policy_logps = runner.batch_sequence_log_probs(
                samples, hypothesis_groups, length_normalize=True
            )
            for group, policy_logp in zip(pending, policy_logps):
                ratio = torch.exp(policy_logp - group["old_logp"])
                clipped = ratio.clamp(
                    1.0 - training.clip_epsilon, 1.0 + training.clip_epsilon
                )
                policy_loss = -torch.minimum(
                    ratio * group["advantages"], clipped * group["advantages"]
                ).mean()
                log_ratio = group["reference_logp"] - policy_logp
                kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean().clamp_min(0.0)
                loss = policy_loss + training.kl_beta * kl
                group_losses.append(loss)
                detached_ratio = ratio.detach()
                group_metrics.append(
                    {
                        "sample_id": group["sample"]["id"],
                        "loss": float(loss.detach().cpu()),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "kl": float(kl.detach().cpu()),
                        "mean_ratio": float(detached_ratio.mean().cpu()),
                        "ratio_min": float(detached_ratio.min().cpu()),
                        "ratio_max": float(detached_ratio.max().cpu()),
                        "clip_fraction": float(
                            (
                                (detached_ratio < 1.0 - training.clip_epsilon)
                                | (detached_ratio > 1.0 + training.clip_epsilon)
                            ).float().mean().cpu()
                        ),
                    }
                )
            loss = torch.stack(group_losses).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                runner.model.parameters(), training.max_grad_norm
            )
            optimizer.step()
            record: dict[str, Any] = {
                "optimizer_step": optimizer_step,
                "optimization_epoch": optimization_epoch,
                "epoch": pending[0]["epoch"],
                "batch_size": len(pending),
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(np.mean([row["policy_loss"] for row in group_metrics])),
                "group_losses": [row["loss"] for row in group_metrics],
                "mean_reward": float(np.mean([group["mean_reward"] for group in pending])),
                "reward_std": float(np.mean([group["reward_std"] for group in pending])),
                "kl": float(np.mean([row["kl"] for row in group_metrics])),
                "mean_ratio": float(np.mean([row["mean_ratio"] for row in group_metrics])),
                "ratio_min": float(min(row["ratio_min"] for row in group_metrics)),
                "ratio_max": float(max(row["ratio_max"] for row in group_metrics)),
                "clip_fraction": float(np.mean([row["clip_fraction"] for row in group_metrics])),
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "sample_ids": [group["sample"]["id"] for group in pending],
                "group_metrics": group_metrics,
                "skipped_samples_so_far": skipped,
            }
            if runner.device.type == "cuda":
                record["gpu_peak_memory_allocated_gb"] = float(
                    torch.cuda.max_memory_allocated(runner.device) / 1024**3
                )
                record["gpu_peak_memory_reserved_gb"] = float(
                    torch.cuda.max_memory_reserved(runner.device) / 1024**3
                )
            history.append(record)
            training_logger.write(record)
            optimizer_step += 1
        pending.clear()
        progress.set_postfix(
            loss=f"{record['loss']:.4f}",
            reward=f"{record['mean_reward']:.4f}",
            kl=f"{record['kl']:.4f}",
            clip=f"{record['clip_fraction']:.2f}",
            grad_norm=f"{record['grad_norm']:.4f}",
        )

    with (
        JsonlLogger(common.runtime.output_dir, "training_log.jsonl") as training_logger,
        JsonlLogger(common.runtime.output_dir, "training_samples.jsonl") as sample_logger,
    ):
        for step in progress:
            sample = train_set[step % len(train_set)]
            hypotheses = runner.generate(sample["speech"])
            rewards = [
                compute_format_reward(
                    sample["reference"],
                    hypothesis,
                    format_weight=config.reward.format_weight,
                    wer_weight=config.reward.wer_weight,
                    wer_tolerance=config.reward.wer_tolerance,
                )
                for hypothesis in hypotheses
            ]
            if training.log_candidates:
                for rank, (hypothesis, reward) in enumerate(zip(hypotheses, rewards), start=1):
                    tqdm.write(f"  candidate {rank}: reward={reward:.4f} text={hypothesis!r}")
            advantage_values = group_advantages(rewards)
            advantages = torch.tensor(advantage_values, device=runner.device)
            sample_record: dict[str, Any] = {
                "sample_step": step,
                "epoch": step // len(train_set) + 1,
                "sample_id": sample["id"],
                "reference": sample["reference"],
                "hypotheses": hypotheses,
                "rewards": [float(value) for value in rewards],
                "advantages": advantage_values,
                "accepted": len(hypotheses) >= 2 and torch.count_nonzero(advantages).item() > 0,
            }
            if not sample_record["accepted"]:
                skipped += 1
                sample_record["skip_reason"] = "fewer_than_two_hypotheses_or_no_reward_variance"
                sample_logger.write(sample_record)
                progress.set_postfix(status="skipped: no reward variance")
            else:
                pending.append(
                    {
                        "sample": sample,
                        "sample_record": sample_record,
                        "hypotheses": hypotheses,
                        "advantages": advantages,
                        "mean_reward": float(np.mean(rewards)),
                        "reward_std": float(np.std(rewards)),
                        "epoch": sample_record["epoch"],
                    }
                )
            if len(pending) >= training.batch_size or (step + 1) % len(train_set) == 0:
                update_batch()
        update_batch()

    path = Path(common.runtime.output_dir) / "grpo.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": runner.model.state_dict(), "history": history, "config": config.to_dict()},
        path,
    )
    del runner, reference
    torch.cuda.empty_cache()
    return path


def run(config: GRPOConfig, stage: str = "all") -> dict[str, dict[str, float]]:
    common = config.common
    if common.runtime.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seed(common.runtime.seed)
    write_resolved_config(common.runtime.output_dir, config.to_dict())
    train_set = ASRDataset(common.dataset, train=True, seed=common.runtime.seed)
    eval_set = ASRDataset(common.dataset, train=False, seed=common.runtime.seed)
    checkpoint = Path(common.runtime.output_dir) / "grpo.pt"
    if stage in {"all", "train"}:
        checkpoint = train(config, train_set)

    summaries: dict[str, dict[str, float]] = {}
    if stage in {"all", "eval"}:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing GRPO checkpoint: {checkpoint}")
        summaries["base"] = evaluate(common, eval_set, None, "base")
        summaries["grpo"] = evaluate(common, eval_set, checkpoint, "grpo")
        print_and_save_comparison(common.runtime.output_dir, summaries)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--stage", choices=("all", "train", "eval"), default="all")
    args = parser.parse_args(argv)
    run(load_config(args.config), args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
