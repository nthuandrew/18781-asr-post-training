"""GRPO trainer for group-relative RL optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from evaluation.metrics import compute_format_reward


@dataclass
class GRPOConfig:
    learning_rate: float = 1.0e-6
    batch_size: int = 2
    epochs: int = 2
    num_hypotheses: int = 4
    kl_coeff: float = 0.05
    output_dir: str = "./outputs/grpo"


class GRPOTrainer:
    """A scaffold for GRPO / policy-gradient training."""

    def __init__(self, model: Any, reference_model: Any, config: GRPOConfig | None = None) -> None:
        self.model = model
        self.reference_model = reference_model
        self.config = config or GRPOConfig()

    def compute_rewards(self, hypotheses: Sequence[str], reference: str) -> List[float]:
        """Compute a reward for each hypothesis in a group using the ASR format-alignment objective."""
        rewards: List[float] = []
        for hypothesis in hypotheses:
            reward = compute_format_reward(reference, hypothesis)
            if not hypothesis.strip():
                reward -= 1.0
            rewards.append(float(reward))
        return rewards

    def train(self, dataset: Iterable[Any]) -> None:
        """Run the GRPO training loop scaffold."""
        for _ in range(self.config.epochs):
            for batch in dataset:
                _ = batch
                if hasattr(self.model, "train"):
                    self.model.train()

    def save(self, output_dir: str | None = None) -> str:
        directory = output_dir or self.config.output_dir
        return directory
