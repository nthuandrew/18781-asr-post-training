"""Evaluation metrics for ASR alignment quality."""

from __future__ import annotations

import re
from typing import Dict

try:
    from jiwer import wer as jiwer_wer
except ImportError:  # pragma: no cover - jiwer is optional for lightweight environments.
    jiwer_wer = None


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _levenshtein_distance(reference: str, hypothesis: str) -> int:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if ref_tokens == hyp_tokens:
        return 0

    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dp = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def wer_score(reference: str, prediction: str) -> float:
    """Compute WER using jiwer when available, otherwise fall back to token-level Levenshtein."""
    reference_norm = _normalize_text(reference)
    prediction_norm = _normalize_text(prediction)
    if not reference_norm:
        return 0.0
    if jiwer_wer is not None:
        return float(jiwer_wer(reference_norm, prediction_norm))

    distance = _levenshtein_distance(reference_norm, prediction_norm)
    ref_length = max(len(reference_norm.split()), 1)
    return distance / ref_length


def format_compliance(text: str) -> float:
    """Score basic transcript formatting without judging lexical accuracy.

    The score is intentionally simple and auditable.  Lexical correctness is
    measured separately with WER, so this function only checks disfluencies,
    casing, punctuation, and written-number usage.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0

    has_disfluency = bool(re.search(r"\b(?:um|uh|erm|mm|you\s+know)\b", normalized))
    score = 0.25 if not has_disfluency else 0.0
    stripped = (text or "").strip()
    if stripped[:1].isupper():
        score += 0.25
    if re.search(r"[.!?][\"')\]]?$", stripped):
        score += 0.25
    if re.search(r"\b\d+(?:[.,]\d+)?\b", stripped):
        score += 0.25
    return min(score, 1.0)


def compute_format_reward(
    reference: str,
    hypothesis: str,
    format_weight: float = 1.0,
    wer_weight: float = 1.5,
    wer_tolerance: float = 0.0,
) -> float:
    """Compute a reward that prefers clean text while penalizing WER regression."""
    format_score = format_compliance(hypothesis)
    wer = wer_score(reference, hypothesis)
    penalty = max(0.0, wer - wer_tolerance)
    return float(format_weight * format_score - wer_weight * penalty)


def harmful_edit_rate(reference: str, prediction: str) -> float:
    """Estimate how much the generated caption diverges semantically from the original transcript."""
    ref_tokens = set(_normalize_text(reference).lower().split())
    pred_tokens = set(_normalize_text(prediction).lower().split())
    if not ref_tokens:
        return 0.0
    overlap = len(ref_tokens & pred_tokens)
    return 1.0 - (overlap / len(ref_tokens))


def reward_hacking_tracker(format_score: float, wer: float, threshold: float = 0.15) -> bool:
    """Return True when format quality improves while WER degrades beyond a tolerable threshold."""
    return format_score > 0.8 and wer > threshold


def compute_metrics(reference: str, prediction: str) -> Dict[str, float]:
    """Compute a minimal set of evaluation metrics."""
    return {
        "wer": wer_score(reference, prediction),
        "format_compliance": format_compliance(prediction),
        "harmful_edit_rate": harmful_edit_rate(reference, prediction),
    }
