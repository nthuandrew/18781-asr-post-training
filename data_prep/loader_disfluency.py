"""Utilities for loading disfluency-style ASR datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_disfluency_dataset(root: str | Path) -> List[Dict[str, Any]]:
    """Load a local disfluency dataset.

    The real implementation should read local transcripts or manifests and return a list
    of samples with audio_path, transcript, and optionally a fluent_reference field.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Disfluency dataset root does not exist: {root_path}")

    items: List[Dict[str, Any]] = []
    for transcript_file in sorted(root_path.rglob("*.txt")):
        text = transcript_file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            items.append(
                {
                    "audio_path": str(transcript_file.with_suffix(".wav")),
                    "transcript": text,
                    "fluent_reference": text,
                }
            )
    return items
