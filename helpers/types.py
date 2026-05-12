"""Shared data types and parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NewsSample:
    """Single labeled news datapoint used for training and evaluation."""

    text: str
    label: int


def normalize_label(raw_label: str) -> int:
    """Map raw label values from common formats into binary direction labels."""
    value = raw_label.strip().lower()
    positive = {"1", "up", "positive", "bullish", "rise", "gain"}
    negative = {"0", "-1", "down", "negative", "bearish", "fall", "loss"}
    if value in positive:
        return 1
    if value in negative:
        return 0
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {raw_label}") from exc
    return 1 if numeric > 0 else 0
