"""Text preprocessing agent for feature extraction."""

from __future__ import annotations

from typing import List

from .constants import TOKEN_PATTERN


class ProcessingAgent:
    """Preprocess financial-news text into token features for modeling."""

    def __init__(self, include_bigrams: bool = False, min_token_length: int = 2):
        """Initialize preprocessing options."""
        if min_token_length < 1:
            raise ValueError("min_token_length must be at least 1.")
        self.include_bigrams = include_bigrams
        self.min_token_length = min_token_length

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text and optionally append adjacent bigram tokens."""
        tokens = [
            tok.lower()
            for tok in TOKEN_PATTERN.findall(text)
            if len(tok) >= self.min_token_length
        ]
        if self.include_bigrams:
            tokens += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens
