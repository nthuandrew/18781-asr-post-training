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

from evaluation.metrics import compute_format_reward, wer_score
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
class DPOTrainingConfig:
    steps: int | None = 5
    epochs: int | None = None
    learning_rate: float = 5.0e-6
    num_hypotheses: int = 4
    beta: float = 0.1
    max_grad_norm: float = 1.0
    log_candidates: bool = False
    min_reward_margin: float = 0.0
    max_chosen_wer: float | None = None
    batch_size: int = 1


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
    hypotheses: Sequence[str],
    rewards: Sequence[float],
    *,
    wers: Sequence[float] | None = None,
    min_reward_margin: float = 0.0,
    max_chosen_wer: float | None = None,
) -> tuple[str, str] | None:
    if len(hypotheses) != len(rewards) or len(hypotheses) < 2:
        return None
    chosen_index = int(np.argmax(rewards))
    rejected_index = int(np.argmin(rewards))
    reward_margin = rewards[chosen_index] - rewards[rejected_index]
    if chosen_index == rejected_index or reward_margin < min_reward_margin:
        return None
    if max_chosen_wer is not None:
        if wers is None or len(wers) != len(hypotheses):
            raise ValueError("Candidate WERs are required when max_chosen_wer is set")
        if wers[chosen_index] > max_chosen_wer:
            return None
    if reward_margin <= 0.0:
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
    history: list[dict[str, Any]] = []
    total_steps = resolve_training_steps(
        steps=training.steps,
        epochs=training.epochs,
        dataset_size=len(train_set),
    )

    if training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    progress = tqdm(
        range(total_steps),
        desc="DPO train",
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
        samples = [item["sample"] for item in pending]
        pairs = [[item["chosen"], item["rejected"]] for item in pending]
        runner.model.eval()
        if runner.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runner.device)
        with torch.no_grad():
            policy_model = runner.model
            runner.model = reference
            reference_logps = runner.batch_sequence_log_probs(
                samples, pairs, length_normalize=False
            )
            runner.model = policy_model
        policy_logps = runner.batch_sequence_log_probs(
            samples, pairs, length_normalize=False
        )
        loss_outputs = [
            dpo_loss(policy_group, reference_group, training.beta)
            for policy_group, reference_group in zip(policy_logps, reference_logps)
        ]
        losses = [output[0] for output in loss_outputs]
        logit_tensors = [output[1] for output in loss_outputs]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runner.model.parameters(), training.max_grad_norm
        )
        optimizer.step()
        logits = [float(value.detach().cpu()) for value in logit_tensors]
        loss_values = [float(value.detach().cpu()) for value in losses]
        policy_margins = [
            float((values[0] - values[1]).detach().cpu()) for values in policy_logps
        ]
        reference_margins = [
            float((values[0] - values[1]).cpu()) for values in reference_logps
        ]
        for index, item in enumerate(pending):
            item["sample_record"].update(
                {
                    "policy_logps": [
                        float(value) for value in policy_logps[index].detach().cpu()
                    ],
                    "reference_logps": [
                        float(value) for value in reference_logps[index].cpu()
                    ],
                    "loss": loss_values[index],
                    "preference_logit": logits[index],
                }
            )
            sample_logger.write(item["sample_record"])
        record: dict[str, Any] = {
            "optimizer_step": optimizer_step,
            "epoch": pending[0]["epoch"],
            "batch_size": len(pending),
            "loss": float(batch_loss.detach().cpu()),
            "sample_losses": loss_values,
            "preference_logit": float(np.mean(logits)),
            "preference_logits": logits,
            "reward_margin": float(np.mean([item["reward_margin"] for item in pending])),
            "chosen_reward": float(np.mean([item["chosen_reward"] for item in pending])),
            "rejected_reward": float(np.mean([item["rejected_reward"] for item in pending])),
            "chosen_wer": float(np.mean([item["chosen_wer"] for item in pending])),
            "rejected_wer": float(np.mean([item["rejected_wer"] for item in pending])),
            "policy_margin": float(np.mean(policy_margins)),
            "reference_margin": float(np.mean(reference_margins)),
            "grad_norm": float(grad_norm.detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "sample_ids": [item["sample_id"] for item in pending],
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
            reward_margin=f"{record['reward_margin']:.4f}",
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
            wers = [wer_score(sample["reference"], hypothesis) for hypothesis in hypotheses]
            if training.log_candidates:
                for rank, (hypothesis, reward) in enumerate(zip(hypotheses, rewards), start=1):
                    tqdm.write(f"  candidate {rank}: reward={reward:.4f} text={hypothesis!r}")
            pair = select_preference_pair(
                hypotheses,
                rewards,
                wers=wers,
                min_reward_margin=training.min_reward_margin,
                max_chosen_wer=training.max_chosen_wer,
            )
            sample_record: dict[str, Any] = {
                "sample_step": step,
                "epoch": step // len(train_set) + 1,
                "sample_id": sample["id"],
                "reference": sample["reference"],
                "hypotheses": hypotheses,
                "rewards": [float(value) for value in rewards],
                "wers": [float(value) for value in wers],
                "accepted": pair is not None,
            }
            if pair is None:
                skipped += 1
                sample_record["skip_reason"] = "no_pair_after_reward_margin_or_wer_gates"
                sample_logger.write(sample_record)
                progress.set_postfix(status="skipped: no preference")
            else:
                chosen, rejected = pair
                chosen_index = hypotheses.index(chosen)
                rejected_index = hypotheses.index(rejected)
                sample_record.update(
                    {
                        "chosen_index": chosen_index,
                        "rejected_index": rejected_index,
                        "chosen": chosen,
                        "rejected": rejected,
                        "chosen_reward": float(rewards[chosen_index]),
                        "rejected_reward": float(rewards[rejected_index]),
                        "chosen_wer": float(wers[chosen_index]),
                        "rejected_wer": float(wers[rejected_index]),
                    }
                )
                pending.append(
                    {
                        "sample": sample,
                        "sample_record": sample_record,
                        "chosen": chosen,
                        "rejected": rejected,
                        "epoch": sample_record["epoch"],
                        "sample_id": sample["id"],
                        "reward_margin": rewards[chosen_index] - rewards[rejected_index],
                        "chosen_reward": rewards[chosen_index],
                        "rejected_reward": rewards[rejected_index],
                        "chosen_wer": wers[chosen_index],
                        "rejected_wer": wers[rejected_index],
                    }
                )
            if len(pending) >= training.batch_size or (step + 1) % len(train_set) == 0:
                update_batch()
        update_batch()

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
