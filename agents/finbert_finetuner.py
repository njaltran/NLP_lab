"""Reusable per-head FinBERT training engine.

Nadi (`agents/nadi_classifier.py`) owns fine-tuning directly (ADR 0001): it
calls `train_finbert()` here once per directional head it needs to retrain.
Each head is a genuine 2-class classifier — its own movement class versus
`neutral` — trained on a filtered subset of the data, not one-vs-rest
(ADR 0002): the up-head never sees `down` rows, and vice versa.
"""

import pandas as pd

HEADS = ("up", "down")


def rows_for_head(frame: pd.DataFrame, head: str) -> pd.DataFrame:
    """Filter to the rows a directional head is allowed to train on: its own
    movement class plus `neutral`. The opposite movement class is dropped
    entirely — the head never sees it during training (ADR 0002)."""
    if head not in HEADS:
        raise ValueError(f"head must be 'up' or 'down', got {head!r}")
    return frame[frame["label"].isin({head, "neutral"})]
