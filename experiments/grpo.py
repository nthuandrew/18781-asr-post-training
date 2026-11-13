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
    OWSMModel,
    RewardConfig,
    evaluate,
    load_yaml,
    parse_common_config,
    print_and_save_comparison,
    set_seed,
    write_resolved_config,
)


@dataclass
class GRPOTrainingConfig:
    steps: int = 5
    learning_rate: float = 1.0e-6
    num_hypotheses: int = 4
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    max_grad_norm: float = 1.0
    log_candidates: bool = False


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
    history: list[dict[str, float]] = []

    progress = tqdm(
        range(training.steps),
        desc="GRPO train",
        unit="step",
        dynamic_ncols=True,
    )
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
        advantages = torch.tensor(group_advantages(rewards), device=runner.device)
        if len(hypotheses) < 2 or torch.count_nonzero(advantages).item() == 0:
            progress.set_postfix(status="skipped: no reward variance")
            continue

        optimizer.zero_grad(set_to_none=True)
        runner.model.eval()
        policy_logp = runner.sequence_log_probs(
            sample, hypotheses, length_normalize=True
        )
        old_logp = policy_logp.detach()
        with torch.no_grad():
            policy_model = runner.model
            runner.model = reference
            reference_logp = runner.sequence_log_probs(
                sample, hypotheses, length_normalize=True
            )
            runner.model = policy_model

        ratio = torch.exp(policy_logp - old_logp)
        clipped = ratio.clamp(1.0 - training.clip_epsilon, 1.0 + training.clip_epsilon)
        policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        log_ratio = reference_logp - policy_logp
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        loss = policy_loss + training.kl_beta * kl
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runner.model.parameters(), training.max_grad_norm
        )
        optimizer.step()
        record = {
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(np.mean(rewards)),
            "kl": float(kl.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        history.append(record)
        progress.set_postfix(
            loss=f"{record['loss']:.4f}",
            reward=f"{record['mean_reward']:.4f}",
            kl=f"{record['kl']:.4f}",
            grad_norm=f"{record['grad_norm']:.4f}",
        )

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
