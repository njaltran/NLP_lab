"""Shared constants for stock prediction helpers."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")
MIN_SAMPLES_REQUIRED = 10
REQUIRED_SEARCH_KEYS = {"include_bigrams", "min_token_length", "smoothing"}
DEFAULT_SEARCH_SPACE = [
    {"include_bigrams": False, "min_token_length": 2, "smoothing": 1.0},
    {"include_bigrams": True, "min_token_length": 2, "smoothing": 1.0},
    {"include_bigrams": False, "min_token_length": 3, "smoothing": 0.7},
    {"include_bigrams": True, "min_token_length": 3, "smoothing": 0.7},
]
