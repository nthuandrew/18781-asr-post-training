"""Config-driven direct preference optimization for ASR."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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
class DPOTrainingConfig:
    steps: int = 5
    learning_rate: float = 5.0e-6
    num_hypotheses: int = 4
    beta: float = 0.1
    max_grad_norm: float = 1.0
    log_candidates: bool = False


@dataclass
class DPOConfig:
    common: CommonConfig
    training: DPOTrainingConfig
    reward: RewardConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.common.to_dict(),
            "training": asdict(self.training),
            "reward": asdict(self.reward),
        }


def load_config(path: str | Path) -> DPOConfig:
    values = load_yaml(path)
    return DPOConfig(
        common=parse_common_config(values),
        training=DPOTrainingConfig(**values.get("training", {})),
        reward=RewardConfig(**values.get("reward", {})),
    )


def dpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy_logps.shape != reference_logps.shape or policy_logps.numel() != 2:
        raise ValueError("DPO expects matching [chosen, rejected] log-probabilities")
    policy_margin = policy_logps[0] - policy_logps[1]
    reference_margin = reference_logps[0] - reference_logps[1]
    logits = float(beta) * (policy_margin - reference_margin)
    return -F.logsigmoid(logits), logits


def select_preference_pair(
    hypotheses: Sequence[str], rewards: Sequence[float]
) -> tuple[str, str] | None:
    if len(hypotheses) != len(rewards) or len(hypotheses) < 2:
        return None
    chosen_index = int(np.argmax(rewards))
    rejected_index = int(np.argmin(rewards))
    if chosen_index == rejected_index or rewards[chosen_index] <= rewards[rejected_index]:
        return None
    return hypotheses[chosen_index], hypotheses[rejected_index]


def train(config: DPOConfig, train_set: ASRDataset) -> Path:
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
    history: list[dict[str, float | str]] = []

    progress = tqdm(
        range(training.steps),
        desc="DPO train",
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
        pair = select_preference_pair(hypotheses, rewards)
        if pair is None:
            progress.set_postfix(status="skipped: no preference")
            continue
        chosen, rejected = pair

        optimizer.zero_grad(set_to_none=True)
        runner.model.eval()
        policy_logps = runner.sequence_log_probs(
            sample, [chosen, rejected], length_normalize=False
        )
        with torch.no_grad():
            policy_model = runner.model
            runner.model = reference
            reference_logps = runner.sequence_log_probs(
                sample, [chosen, rejected], length_normalize=False
            )
            runner.model = policy_model

        loss, logits = dpo_loss(policy_logps, reference_logps, training.beta)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runner.model.parameters(), training.max_grad_norm
        )
        optimizer.step()
        record: dict[str, float | str] = {
            "loss": float(loss.detach().cpu()),
            "preference_logit": float(logits.detach().cpu()),
            "chosen_reward": float(max(rewards)),
            "rejected_reward": float(min(rewards)),
            "grad_norm": float(grad_norm.detach().cpu()),
            "chosen": chosen,
            "rejected": rejected,
        }
        history.append(record)
        progress.set_postfix(
            loss=f"{record['loss']:.4f}",
            reward_margin=f"{record['chosen_reward'] - record['rejected_reward']:.4f}",
            grad_norm=f"{record['grad_norm']:.4f}",
        )

    path = Path(common.runtime.output_dir) / "dpo.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": runner.model.state_dict(), "history": history, "config": config.to_dict()},
        path,
    )
    del runner, reference
    torch.cuda.empty_cache()
    return path


def run(config: DPOConfig, stage: str = "all") -> dict[str, dict[str, float]]:
    common = config.common
    if common.runtime.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seed(common.runtime.seed)
    write_resolved_config(common.runtime.output_dir, config.to_dict())
    train_set = ASRDataset(common.dataset, train=True, seed=common.runtime.seed)
    eval_set = ASRDataset(common.dataset, train=False, seed=common.runtime.seed)
    checkpoint = Path(common.runtime.output_dir) / "dpo.pt"
    if stage in {"all", "train"}:
        checkpoint = train(config, train_set)

    summaries: dict[str, dict[str, float]] = {}
    if stage in {"all", "eval"}:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing DPO checkpoint: {checkpoint}")
        summaries["base"] = evaluate(common, eval_set, None, "base")
        summaries["dpo"] = evaluate(common, eval_set, checkpoint, "dpo")
        print_and_save_comparison(common.runtime.output_dir, summaries)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    parser.add_argument("--stage", choices=("all", "train", "eval"), default="all")
    args = parser.parse_args(argv)
    run(load_config(args.config), args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
