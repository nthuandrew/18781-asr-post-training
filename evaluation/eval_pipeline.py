"""End-to-end validation pipeline for ASR post-training."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .metrics import compute_metrics


def run_evaluation(model: Any, eval_sets: Iterable[Dict[str, str]]) -> List[Dict[str, float]]:
    """Run inference over benchmark sets and compute metrics for each example."""
    results: List[Dict[str, float]] = []
    for sample in eval_sets:
        reference = sample.get("reference", "")
        audio = sample.get("audio_path", "")
        prediction = model.generate(audio)
        results.append(compute_metrics(reference, prediction))
    return results
