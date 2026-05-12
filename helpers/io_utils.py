"""IO helpers for manual evaluation artifacts."""

from __future__ import annotations

import csv
from typing import Dict, Iterable


def write_manual_eval_csv(rows: Iterable[Dict[str, str]], output_path: str) -> None:
    """Write manual evaluation rows to CSV if at least one row is present."""
    rows_list = list(rows)
    if not rows_list:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)
