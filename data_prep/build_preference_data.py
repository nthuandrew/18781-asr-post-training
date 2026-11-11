"""Build preference pairs for DPO-style RL training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class PreferenceExample:
    audio_path: str
    chosen: str
    rejected: str
    score: float = 0.0


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _format_score(text: str) -> float:
    score = 0.0
    if re.search(r"\b(?:um|uh|erm|mm)\b", text.lower()):
        return 0.0
    score += 1.0
    if re.search(r"\b\d+(?:\.\d+)?\b", text):
        score += 1.0
    return score


def _wer_constraint(candidate: str, reference: str, threshold: float = 0.15) -> bool:
    """A placeholder WER check.

    Real implementations should use a robust WER calculator such as jiwer.
    """
    if not reference:
        return True
    candidate_norm = _normalize_whitespace(candidate)
    reference_norm = _normalize_whitespace(reference)
    if candidate_norm == reference_norm:
        return True
    if len(reference_norm.split()) == 0:
        return True
    return abs(len(candidate_norm.split()) - len(reference_norm.split())) / max(len(reference_norm.split()), 1) <= threshold


def build_preference_pairs(
    samples: Iterable[Dict[str, str]],
    model: Any,
    num_hypotheses: int = 4,
    wer_threshold: float = 0.15,
) -> List[PreferenceExample]:
    """Generate chosen/rejected pairs from model hypotheses.

    This function is a skeleton: it expects the model to provide a `generate()` method
    and ranks candidate outputs according to formatting quality while keeping WER under a
    threshold. The real pipeline should call the model on audio and compare to the literal transcript.
    """
    pairs: List[PreferenceExample] = []

    for sample in samples:
        audio_path = sample.get("audio_path", "")
        literal = sample.get("transcript", "")
        if not audio_path or not literal:
            continue

        hypotheses: List[str] = []
        for _ in range(num_hypotheses):
            prediction = model.generate(audio_path)
            hypotheses.append(prediction)

        scored: List[tuple[str, float]] = []
        for hyp in hypotheses:
            format_score = _format_score(hyp)
            if not _wer_constraint(hyp, literal, threshold=wer_threshold):
                continue
            scored.append((hyp, format_score))

        if not scored:
            scored = [(literal, 2.0)]

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        chosen_text, chosen_score = ranked[0]
        rejected_text, rejected_score = ranked[-1]

        pairs.append(
            PreferenceExample(
                audio_path=audio_path,
                chosen=chosen_text,
                rejected=rejected_text,
                score=float(chosen_score - rejected_score),
            )
        )

    return pairs
