"""Base interface for ASR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ASRModelInterface(ABC):
    """Common interface for ASR model wrappers used in RL post-training."""

    @abstractmethod
    def forward_log_probs(self, audio: Any, text: str) -> Any:
        """Return log-probabilities for the target text sequence."""

    @abstractmethod
    def generate(self, audio: Any, **kwargs: Any) -> str:
        """Decode audio to text using the model's generate method."""

    def predict(self, audio: Any, **kwargs: Any) -> str:
        """Convenience wrapper for model inference."""
        return self.generate(audio, **kwargs)
