"""Deterministic training-target transforms for ASR formatting experiments."""

from __future__ import annotations

import re
from typing import Callable


_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
    "hundredth": 100,
    "thousandth": 1_000,
}
_DECADES = {
    "twenties": 20,
    "thirties": 30,
    "forties": 40,
    "fifties": 50,
    "sixties": 60,
    "seventies": 70,
    "eighties": 80,
    "nineties": 90,
}
_NUMBER_TOKENS = set(_SMALL) | set(_TENS) | set(_SCALES) | set(_ORDINALS)


def _ordinal_suffix(value: int) -> str:
    if 10 <= value % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _parse_cardinal(tokens: list[str]) -> int | None:
    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        if token == "and":
            continue
        if token in _SMALL:
            current += _SMALL[token]
            saw_number = True
        elif token in _TENS:
            current += _TENS[token]
            saw_number = True
        elif token == "hundred":
            current = max(current, 1) * 100
            saw_number = True
        elif token in {"thousand", "million"}:
            total += max(current, 1) * _SCALES[token]
            current = 0
            saw_number = True
        else:
            return None
    return total + current if saw_number else None


def _render_number_span(tokens: list[str]) -> str | None:
    # Common spoken-year and decade forms need different composition from
    # ordinary cardinals: "nineteen forty" is 1940, not 59.
    if len(tokens) == 2 and tokens[0] in {"eighteen", "nineteen"}:
        if tokens[1] in _DECADES:
            return f"{_SMALL[tokens[0]] * 100 + _DECADES[tokens[1]]}s"
        second = _SMALL.get(tokens[1], _TENS.get(tokens[1]))
        if second is not None:
            return str(_SMALL[tokens[0]] * 100 + second)
    if tokens == ["twenty", "twenty"]:
        return "2020"

    ordinal_index = next(
        (index for index, token in enumerate(tokens) if token in _ORDINALS), None
    )
    if ordinal_index is not None:
        if ordinal_index != len(tokens) - 1:
            return None
        prefix = _parse_cardinal(tokens[:ordinal_index]) or 0
        ordinal = _ORDINALS[tokens[-1]]
        if tokens[-1] in {"hundredth", "thousandth"}:
            value = max(prefix, 1) * ordinal
        else:
            value = prefix + ordinal
        return f"{value}{_ordinal_suffix(value)}"

    value = _parse_cardinal(tokens)
    return None if value is None else str(value)


def inverse_normalize_number_words(text: str) -> str:
    """Replace unambiguous English number-word spans while preserving other words."""
    words = text.split()
    output: list[str] = []
    index = 0
    while index < len(words):
        token = words[index].lower()
        if token not in _NUMBER_TOKENS:
            output.append(words[index])
            index += 1
            continue

        end = index + 1
        while end < len(words):
            candidate = words[end].lower()
            if candidate in _NUMBER_TOKENS:
                end += 1
                continue
            if (
                candidate == "and"
                and end + 1 < len(words)
                and words[end + 1].lower() in _NUMBER_TOKENS
            ):
                end += 1
                continue
            break
        span = [word.lower() for word in words[index:end]]
        rendered = _render_number_span(span)
        if rendered is None:
            output.extend(words[index:end])
        else:
            output.append(rendered)
        index = end
    return " ".join(output)


def format_sft_target_v1(text: str) -> str:
    """Create a conservative display-form SFT target from a lexical transcript.

    This transform preserves all non-number lexical tokens, applies deterministic
    English inverse text normalization, and adds sentence-final punctuation. It
    intentionally does not infer commas or capitalization.
    """
    formatted = re.sub(r"\s+", " ", (text or "")).strip()
    if not formatted:
        return ""
    formatted = inverse_normalize_number_words(formatted)
    if not re.search(r"[.!?][\"')\]]?$", formatted):
        formatted += "."
    return formatted


_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "none": lambda text: text,
    "format_sft_v1": format_sft_target_v1,
}


def transform_training_text(text: str, transform: str) -> str:
    try:
        function = _TRANSFORMS[transform]
    except KeyError as error:
        choices = ", ".join(sorted(_TRANSFORMS))
        raise ValueError(f"Unknown train_text_transform={transform!r}; choose {choices}") from error
    return function(text)
