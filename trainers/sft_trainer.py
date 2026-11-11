"""Baseline supervised fine-tuning trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


@dataclass
class SFTConfig:
    learning_rate: float = 1.0e-5
    batch_size: int = 8
    epochs: int = 3
    output_dir: str = "./outputs/sft"


class SFTTrainer:
    """Minimal SFT baseline for ASR post-training."""

    def __init__(self, model: Any, config: SFTConfig | None = None) -> None:
        self.model = model
        self.config = config or SFTConfig()

    def train(self, dataset: Iterable[Any]) -> None:
        """Run a placeholder fine-tuning loop."""
        for _ in range(self.config.epochs):
            for batch in dataset:
                _ = batch
                if hasattr(self.model, "train"):
                    self.model.train()

    def save(self, output_dir: str | None = None) -> str:
        directory = output_dir or self.config.output_dir
        return directory
