"""Utilities for loading NVIDIA Numb3rs-style ASR datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_numb3rs_dataset(root: str | Path) -> List[Dict[str, Any]]:
    """Load Numb3rs data from a local root directory.

    Expected format is a JSONL or JSON manifest, with each example containing at least
    an audio path and transcript text.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Numb3rs dataset root does not exist: {root_path}")

    records: List[Dict[str, Any]] = []
    manifest_path = root_path / "manifest.json"

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for item in data:
            records.append({
                "audio_path": str(root_path / item.get("audio_path", "")),
                "transcript": item.get("transcript", ""),
            })
        return records

    for manifest_file in sorted(root_path.rglob("*.jsonl")):
        with manifest_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                records.append({
                    "audio_path": str(root_path / item.get("audio_path", "")),
                    "transcript": item.get("transcript", ""),
                })

    return records
