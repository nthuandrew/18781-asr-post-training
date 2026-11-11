"""Hugging Face wrapper for Transformer-based ASR models."""

from __future__ import annotations

from typing import Any

import torch

from .base import ASRModelInterface

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None


class HFASRWrapper(ASRModelInterface):
    """Wrapper for Hugging Face Whisper or similar Seq2Seq ASR models."""

    def __init__(
        self,
        model_name_or_path: str,
        processor: Any | None = None,
        model: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.processor = processor
        self.model = model
        self.kwargs = kwargs

        if self.model is None and AutoModelForSpeechSeq2Seq is not None and model_name_or_path:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, **kwargs)
        if self.processor is None and AutoProcessor is not None and model_name_or_path:
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    def forward_log_probs(self, audio: Any, text: str) -> Any:
        """Compute token-level log-probabilities for a target text sequence."""
        if self.model is None or self.processor is None:
            raise ValueError("A Hugging Face model and processor are required for forward_log_probs().")

        inputs = self.processor(audio, text, return_tensors="pt", padding=True, truncation=True)
        labels = inputs["labels"]
        outputs = self.model(**inputs)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs

    def generate(self, audio: Any, **kwargs: Any) -> str:
        """Generate a transcript using the Hugging Face model."""
        if self.model is None or self.processor is None:
            raise ValueError("A Hugging Face processor and model are required for generation.")

        model_inputs = self.processor(audio, return_tensors="pt")
        generated_ids = self.model.generate(**model_inputs, **kwargs)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
