"""ESPnet-based wrapper for OWSM-style ASR models."""

from __future__ import annotations

from typing import Any

from .base import ASRModelInterface

try:
    from espnet2.bin.s2t_inference import Speech2Text
except ImportError:  # pragma: no cover - optional dependency
    Speech2Text = None


class EspnetASRWrapper(ASRModelInterface):
    """Wrapper for ESPnet models, intended for OWSM v4 or similar architectures."""

    def __init__(self, model_path: str | None = None, tokenizer: Any | None = None, model: Any | None = None, **kwargs: Any) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = model or kwargs.get("model")
        self.kwargs = dict(kwargs)

        if self.model is None and model_path is not None:
            if Speech2Text is None:
                raise ImportError("ESPnet is required for the OWSM wrapper. Install the ESPnet train/s2t dependencies first.")
            self.model = Speech2Text.from_pretrained(model_path, **self.kwargs)

    @staticmethod
    def _normalize_audio(audio: Any) -> Any:
        """Convert dataset audio decoders into the waveform shape expected by `Speech2Text`."""
        if isinstance(audio, dict):
            return audio.get("array")
        if hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            if hasattr(samples, "data"):
                tensor = samples.data
                if hasattr(tensor, "detach"):
                    tensor = tensor.detach()
                if getattr(tensor, "ndim", 0) > 1 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if hasattr(tensor, "numpy"):
                    return tensor.numpy()
                return tensor
        return audio

    @staticmethod
    def _extract_text_from_result(result: Any) -> str:
        """Normalize the notebook-style `Speech2Text` output into a plain transcript string."""
        if isinstance(result, (list, tuple)):
            if not result:
                return ""
            first = result[0]
            if isinstance(first, (list, tuple)) and first:
                nested = first[0]
                if isinstance(nested, tuple) and nested:
                    return str(nested[0])
                if isinstance(nested, dict):
                    return str(nested.get("text", ""))
            if isinstance(first, tuple) and first:
                return str(first[0])
            if isinstance(first, dict):
                return str(first.get("text", ""))
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(result)

    def forward_log_probs(self, audio: Any, text: str) -> Any:
        """Compute log-probabilities over the target sequence.

        ESPnet's `Speech2Text` exposes decoding rather than a direct log-probability API,
        so this wrapper raises an explicit error unless the backend provides a dedicated
        log-probability helper.
        """
        if self.model is None:
            raise ValueError("An ESPnet model instance is required for forward_log_probs().")
        if hasattr(self.model, "ctc_log_probs"):
            return self.model.ctc_log_probs(audio, text)
        if hasattr(self.model, "forward_log_probs"):
            return self.model.forward_log_probs(audio, text)
        raise NotImplementedError("ESPnet log-prob extraction is not available for the loaded backend.")

    def generate(self, audio: Any, **kwargs: Any) -> str:
        """Generate a transcript from audio using the ESPnet model."""
        if self.model is None:
            raise ValueError("An ESPnet model instance is required for generation.")

        decode_kwargs = dict(self.kwargs)
        decode_kwargs.update(kwargs)
        normalized_audio = self._normalize_audio(audio)
        result = self.model(normalized_audio, **decode_kwargs)
        return self._extract_text_from_result(result)
