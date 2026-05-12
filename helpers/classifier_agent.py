"""Naive Bayes classification and explanation agent."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence


class ClassifierAgent:
    """Multinomial Naive Bayes classifier with token-level explanations."""

    def __init__(self, smoothing: float = 1.0):
        """Initialize classifier hyperparameters and model state."""
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative.")
        self.smoothing = smoothing
        self.class_counts = Counter()
        self.token_counts = {0: Counter(), 1: Counter()}
        self.total_tokens = {0: 0, 1: 0}
        self.vocabulary = set()
        self.fitted = False

    def fit(self, tokenized_texts: Sequence[Sequence[str]], labels: Sequence[int]) -> None:
        """Fit model statistics from tokenized texts and binary labels."""
        self.class_counts.clear()
        self.token_counts = {0: Counter(), 1: Counter()}
        self.total_tokens = {0: 0, 1: 0}
        self.vocabulary = set()
        for tokens, label in zip(tokenized_texts, labels):
            self.class_counts[label] += 1
            for token in tokens:
                self.vocabulary.add(token)
                self.token_counts[label][token] += 1
                self.total_tokens[label] += 1
        if not self.vocabulary:
            raise ValueError("Training data produced an empty vocabulary; check input text content.")
        self.fitted = True

    def _class_log_prior(self, label: int) -> float:
        """Return log prior probability for a class label."""
        total = sum(self.class_counts.values())
        if total == 0:
            raise ValueError("Model has not been fitted.")
        return math.log(self.class_counts[label] / total)

    def _token_log_prob(self, token: str, label: int) -> float:
        """Return smoothed log likelihood of token under a class."""
        vocab_size = max(len(self.vocabulary), 1)
        numerator = self.token_counts[label][token] + self.smoothing
        denominator = self.total_tokens[label] + self.smoothing * vocab_size
        return math.log(numerator / denominator)

    def predict_one(self, tokens: Sequence[str]) -> int:
        """Predict binary direction label for one tokenized sample."""
        if not self.fitted:
            raise ValueError("Model has not been fitted.")
        scores = {}
        for label in (0, 1):
            score = self._class_log_prior(label)
            score += sum(self._token_log_prob(token, label) for token in tokens)
            scores[label] = score
        return 1 if scores[1] >= scores[0] else 0

    def predict(self, tokenized_texts: Sequence[Sequence[str]]) -> List[int]:
        """Predict binary direction labels for multiple tokenized samples."""
        return [self.predict_one(tokens) for tokens in tokenized_texts]

    @staticmethod
    def _direction_from_impact(impact: float) -> str:
        """Convert token impact sign into a readable market-direction hint."""
        if impact > 0:
            return "bullish"
        if impact < 0:
            return "bearish"
        return "neutral"

    def _token_impact_scores(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Aggregate each token's contribution toward bullish vs bearish score."""
        token_impacts = defaultdict(float)
        for token in tokens:
            token_impacts[token] += self._token_log_prob(token, 1) - self._token_log_prob(
                token, 0
            )
        return dict(token_impacts)

    def explain_prediction(self, tokens: Sequence[str], top_k: int = 5) -> str:
        """Return top token-level evidence used to justify a prediction."""
        if not self.fitted:
            raise ValueError("Model has not been fitted.")
        token_impacts = self._token_impact_scores(tokens)
        ranked = sorted(token_impacts.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        if not ranked:
            return "No strong token-level evidence was found."
        fragments = []
        for token, impact in ranked:
            direction = self._direction_from_impact(impact)
            fragments.append(f"{token}({direction},{impact:.2f})")
        return "Top evidence tokens: " + ", ".join(fragments)
