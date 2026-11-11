"""Evaluation metrics for ASR alignment quality."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List


def wer_score(reference: str, prediction: str) -> float:
    """A lightweight placeholder WER estimate.

    Replace with jiwer.wer for production use.
    """
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens:
        return 0.0
    errors = sum(1 for token in ref_tokens if token not in pred_tokens)
    return errors / len(ref_tokens)


def format_compliance(text: str) -> float:
    """A simple formatting heuristic for number and disfluency compliance."""
    score = 0.0
    if re.search(r"\b(?:um|uh|erm|mm)\b", text.lower()):
        return 0.0
    score += 0.5
    if re.search(r"\b\d+(?:\.\d+)?\b", text):
        score += 0.5
    return score


def harmful_edit_rate(reference: str, prediction: str) -> float:
    """Estimate semantic drift risk for transcript editing."""
    ref_tokens = set(reference.lower().split())
    pred_tokens = set(prediction.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = len(ref_tokens & pred_tokens)
    return 1.0 - (overlap / len(ref_tokens))


def reward_hacking_tracker(format_score: float, wer: float, threshold: float = 0.15) -> bool:
    """Return True when format quality improves while WER is unacceptably worse."""
    return format_score > 0.8 and wer > threshold


def compute_metrics(reference: str, prediction: str) -> Dict[str, float]:
    """Compute a minimal set of evaluation metrics."""
    score = wer_score(reference, prediction)
    fmt_score = format_compliance(prediction)
    harmful = harmful_edit_rate(reference, prediction)
    return {
        "wer": score,
        "format_compliance": fmt_score,
        "harmful_edit_rate": harmful,
    }
