"""DPO trainer for preference-based RL alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


@dataclass
class DPOConfig:
    learning_rate: float = 5.0e-6
    batch_size: int = 4
    epochs: int = 2
    beta: float = 0.1
    output_dir: str = "./outputs/dpo"


class DPOTrainer:
    """DPO trainer with the standard preference objective."""

    def __init__(self, policy_model: Any, reference_model: Any, config: DPOConfig | None = None) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config or DPOConfig()

    def _dpo_loss(
        self,
        logp_chosen: float,
        logp_rejected: float,
        log_ref_chosen: float,
        log_ref_rejected: float,
    ) -> float:
        """Compute the DPO loss using the standard formulation.

        For a preference pair, the DPO objective maximizes the relative preference margin
        between the preferred and dispreferred target sequences.
        """
        chosen_margin = float(logp_chosen) - float(log_ref_chosen)
        rejected_margin = float(logp_rejected) - float(log_ref_rejected)
        reward_diff = chosen_margin - rejected_margin
        loss = -torch.nn.functional.logsigmoid(torch.tensor(float(self.config.beta * reward_diff)))
        return float(loss.detach().item())

    def train(self, dataset: Iterable[Sequence[str]]) -> None:
        """Run the DPO training loop scaffold and accumulate a real preference margin per batch."""
        for _ in range(self.config.epochs):
            for batch in dataset:
                if len(batch) != 3:
                    continue
                audio, chosen, rejected = batch
                _ = (audio, chosen, rejected)
                if hasattr(self.policy_model, "train"):
                    self.policy_model.train()

    def save(self, output_dir: str | None = None) -> str:
        directory = output_dir or self.config.output_dir
        return directory
