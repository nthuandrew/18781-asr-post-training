"""Utilities for downloading and preparing open ASR datasets for format-alignment experiments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional for lightweight local runs.
    load_dataset = None


DISFLUENCY_TOKENS = ("uh", "um", "erm", "mm", "like", "you know")
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
NUMBER_PHRASE_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:\s+(?:and\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))*\b",
    flags=re.IGNORECASE,
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_number_words(text: str) -> int | None:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    filtered = [token for token in tokens if token not in {"and"}]
    if not filtered:
        return None

    total = 0
    current = 0
    for token in filtered:
        if token in NUMBER_WORDS and NUMBER_WORDS[token] < 100:
            current += NUMBER_WORDS[token]
        elif token == "hundred":
            current *= 100
        else:
            if token in NUMBER_WORDS:
                total += current + NUMBER_WORDS[token]
                current = 0
            else:
                return None
    return total + current


def _number_to_words(value: int) -> str:
    words = [
        "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = ["", "", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if value < 20:
        return words[value]
    if value < 100:
        tens_word = tens[value // 10]
        unit_word = words[value % 10] if value % 10 else ""
        return f"{tens_word} {unit_word}".strip()
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        if remainder == 0:
            return f"{words[hundreds]} hundred"
        return f"{words[hundreds]} hundred and { _number_to_words(remainder) }".strip()
    return str(value)


def format_transcript_for_fleurs(text: str) -> str:
    """Normalize a reference transcript into a cleaner ASR-style format for FLEURS experiments."""
    cleaned = _normalize_whitespace(text)
    cleaned = re.sub(r"\b(?:uh|um|erm|mm|like|you know)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    matches = list(NUMBER_PHRASE_RE.finditer(cleaned.lower()))
    if matches:
        last_match = matches[-1]
        phrase = last_match.group(0)
        numeric_value = _parse_number_words(phrase)
        if numeric_value is not None:
            cleaned = cleaned.replace(phrase, f"{phrase} ({numeric_value})", 1)

    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
    if not cleaned.endswith((".", "!", "?")):
        cleaned = cleaned + "."
    return _normalize_whitespace(cleaned)


def score_format_compliance(text: str) -> float:
    """Return a heuristic score for formatting compliance on FLEURS-style transcripts."""
    cleaned = _normalize_whitespace(text).lower()
    if not cleaned:
        return 0.0

    score = 0.0
    if not any(token in cleaned.split() for token in DISFLUENCY_TOKENS):
        score += 0.5

    if re.search(r"\b\d+(?:\.\d+)?\b", cleaned):
        score += 0.25

    if re.search(r"[.,!?]$", cleaned):
        score += 0.25

    return min(score, 1.0)


def load_fleurs_dataset(
    language: str = "en_us",
    split: str = "train",
    cache_dir: str | Path | None = None,
    max_examples: int | None = None,
) -> List[Dict[str, Any]]:
    """Load the FLEURS dataset and return a lightweight list of rows for ASR pipeline work."""
    if load_dataset is None:
        raise ImportError("datasets is required to load FLEURS. Install requirements first.")

    ds = load_dataset("google/fleurs", language, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    rows: List[Dict[str, Any]] = []
    for index, example in enumerate(ds):
        if max_examples is not None and index >= max_examples:
            break
        rows.append(
            {
                "id": example.get("id", str(index)),
                "audio": example.get("audio"),
                "transcript": example.get("transcription", example.get("text", "")),
                "language": language,
            }
        )
    return rows


def build_fleurs_format_rows(
    dataset: Iterable[Dict[str, Any]],
    max_examples: int | None = None,
) -> List[Dict[str, str]]:
    """Build a dataset of `(audio_path, transcript, formatted_transcript)` rows for FLEURS experiments."""
    rows: List[Dict[str, str]] = []
    for index, row in enumerate(dataset):
        if max_examples is not None and index >= max_examples:
            break
        transcript = str(row.get("transcript", "")).strip()
        formatted = format_transcript_for_fleurs(transcript)
        if not transcript:
            continue
        rows.append(
            {
                "audio_path": str(row.get("audio_path", row.get("id", ""))),
                "transcript": transcript,
                "formatted_transcript": formatted,
                "format_score": str(score_format_compliance(formatted)),
            }
        )
    return rows


def build_fleurs_preference_pairs(
    rows: Sequence[Dict[str, str]],
    max_pairs: int | None = None,
) -> List[Dict[str, str]]:
    """Create a simple preference-pair structure for DPO/FLEURS experiments.

    In the current stage we intentionally keep this as a deterministic, rule-based scaffolding
    that will later be replaced by model-generated hypotheses and WER-aware ranking.
    """
    pairs: List[Dict[str, str]] = []
    for index, row in enumerate(rows):
        if max_pairs is not None and index >= max_pairs:
            break
        transcript = str(row.get("transcript", "")).strip()
        formatted = str(row.get("formatted_transcript", transcript)).strip()
        chosen = formatted if score_format_compliance(formatted) >= score_format_compliance(transcript) else transcript
        rejected = transcript if chosen == formatted else formatted
        pairs.append({
            "audio_path": str(row.get("audio_path", "")),
            "chosen": chosen,
            "rejected": rejected,
            "reference": transcript,
        })
    return pairs


def build_fleurs_training_batch(
    rows: Sequence[Dict[str, str]],
    max_examples: int | None = None,
    *,
    reward_weight_format: float = 1.0,
    reward_weight_wer: float = 1.5,
) -> List[Dict[str, Any]]:
    """Build a batch contract for ASR RL training with chosen/rejected targets and reward metadata."""
    batch: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if max_examples is not None and index >= max_examples:
            break

        reference = str(row.get("transcript", "")).strip()
        if not reference:
            continue

        formatted = str(row.get("formatted_transcript", format_transcript_for_fleurs(reference))).strip()
        pair = build_fleurs_preference_pairs([row], max_pairs=1)[0]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        reward = 0.0
        if chosen:
            from evaluation.metrics import compute_format_reward
            reward = compute_format_reward(
                reference,
                chosen,
                format_weight=reward_weight_format,
                wer_weight=reward_weight_wer,
            )

        batch.append(
            {
                "audio_path": str(row.get("audio_path", row.get("id", ""))),
                "reference": reference,
                "chosen": chosen,
                "rejected": rejected,
                "formatted": formatted,
                "reward": float(reward),
            }
        )
    return batch
