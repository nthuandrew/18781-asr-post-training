"""DPO trainer for preference-based RL alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass
class DPOConfig:
    learning_rate: float = 5.0e-6
    batch_size: int = 4
    epochs: int = 2
    beta: float = 0.1
    output_dir: str = "./outputs/dpo"


class DPOTrainer:
    """DPO trainer scaffold with the standard preference objective."""

    def __init__(self, policy_model: Any, reference_model: Any, config: DPOConfig | None = None) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config or DPOConfig()

    def _dpo_loss(self, logp_chosen: float, logp_rejected: float, log_ref_chosen: float, log_ref_rejected: float) -> float:
        """Compute the DPO loss using the standard formulation."""
        diff_chosen = logp_chosen - log_ref_chosen
        diff_rejected = logp_rejected - log_ref_rejected
        margin = diff_chosen - diff_rejected
        loss = -math.log(math.exp(self.config.beta * margin) / (1.0 + math.exp(self.config.beta * margin)))
        return loss

    def train(self, dataset: Iterable[Sequence[str]]) -> None:
        """Run the DPO training loop scaffold."""
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
