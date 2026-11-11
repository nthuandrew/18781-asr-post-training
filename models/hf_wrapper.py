"""Hugging Face wrapper for Transformer-based ASR models."""

from __future__ import annotations

from typing import Any

from .base import ASRModelInterface


class HFASRWrapper(ASRModelInterface):
    """Wrapper for Hugging Face Whisper or similar Seq2Seq ASR models."""

    def __init__(self, model_name_or_path: str, processor: Any | None = None, model: Any | None = None, **kwargs: Any) -> None:
        self.model_name_or_path = model_name_or_path
        self.processor = processor
        self.model = model
        self.kwargs = kwargs

    def forward_log_probs(self, audio: Any, text: str) -> Any:
        """Compute token-level log-probabilities for a target text sequence."""
        if self.model is None:
            raise ValueError("A Hugging Face model instance is required for forward_log_probs().")
        raise NotImplementedError("HF log-prob extraction is not implemented in this scaffold.")

    def generate(self, audio: Any, **kwargs: Any) -> str:
        """Generate a transcript using the Hugging Face model."""
        if self.model is None or self.processor is None:
            raise ValueError("A Hugging Face processor and model are required for generation.")
        raise NotImplementedError("HF generation is not implemented in this scaffold.")
