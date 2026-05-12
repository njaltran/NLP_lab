"""CSV loading utilities for financial-news datasets."""

from __future__ import annotations

import csv
from typing import List

from .types import NewsSample, normalize_label


def load_financial_news_csv(path: str) -> List[NewsSample]:
    """Load financial news samples from CSV with flexible text/label column names."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return []

    text_candidates = ("text", "headline", "title", "news", "content")
    label_candidates = ("label", "target", "movement", "sentiment", "direction")
    text_key = next((k for k in text_candidates if k in rows[0]), None)
    label_key = next((k for k in label_candidates if k in rows[0]), None)
    if not text_key or not label_key:
        raise ValueError(
            "CSV must include one text column "
            f"{text_candidates} and one label column {label_candidates}"
        )

    samples = []
    for row in rows:
        text = (row.get(text_key) or "").strip()
        raw_label = (row.get(label_key) or "").strip()
        if not text or not raw_label:
            continue
        samples.append(NewsSample(text=text, label=normalize_label(raw_label)))
    return samples
