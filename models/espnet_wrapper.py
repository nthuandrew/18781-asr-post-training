"""ESPnet-based wrapper for OWSM-style ASR models."""

from __future__ import annotations

from typing import Any

from .base import ASRModelInterface


class EspnetASRWrapper(ASRModelInterface):
    """Wrapper for ESPnet models, intended for OWSM v4 or similar architectures."""

    def __init__(self, model_path: str, tokenizer: Any | None = None, **kwargs: Any) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def forward_log_probs(self, audio: Any, text: str) -> Any:
        """Compute log-probabilities over the target sequence.

        Real implementation should:
        1. run the encoder over the audio,
        2. tokenize the target text,
        3. pass memory and target tokens into the decoder,
        4. collect token log-probabilities for the target sequence.
        """
        raise NotImplementedError("ESPnet log-prob extraction is not implemented in this scaffold.")

    def generate(self, audio: Any, **kwargs: Any) -> str:
        """Generate a transcript from audio using the ESPnet model."""
        raise NotImplementedError("ESPnet generation is not implemented in this scaffold.")
