from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")
MIN_SAMPLES_REQUIRED = 10
REQUIRED_SEARCH_KEYS = {"include_bigrams", "min_token_length", "smoothing"}
DEFAULT_SEARCH_SPACE = [
    {"include_bigrams": False, "min_token_length": 2, "smoothing": 1.0},
    {"include_bigrams": True, "min_token_length": 2, "smoothing": 1.0},
    {"include_bigrams": False, "min_token_length": 3, "smoothing": 0.7},
    {"include_bigrams": True, "min_token_length": 3, "smoothing": 0.7},
]


@dataclass(frozen=True)
class NewsSample:
    text: str
    label: int


def _normalize_label(raw_label: str) -> int:
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


def load_financial_news_csv(path: str) -> List[NewsSample]:
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
        samples.append(NewsSample(text=text, label=_normalize_label(raw_label)))
    return samples


class ProcessingAgent:
    def __init__(self, include_bigrams: bool = False, min_token_length: int = 2):
        if min_token_length < 1:
            raise ValueError("min_token_length must be at least 1.")
        self.include_bigrams = include_bigrams
        self.min_token_length = min_token_length

    def tokenize(self, text: str) -> List[str]:
        tokens = [
            tok.lower()
            for tok in TOKEN_PATTERN.findall(text)
            if len(tok) >= self.min_token_length
        ]
        if self.include_bigrams:
            tokens += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens


class ClassifierAgent:
    def __init__(self, smoothing: float = 1.0):
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative.")
        self.smoothing = smoothing
        self.class_counts = Counter()
        self.token_counts = {0: Counter(), 1: Counter()}
        self.total_tokens = {0: 0, 1: 0}
        self.vocabulary = set()
        self.fitted = False

    def fit(self, tokenized_texts: Sequence[Sequence[str]], labels: Sequence[int]) -> None:
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
        total = sum(self.class_counts.values())
        if total == 0:
            raise ValueError("Model has not been fitted.")
        return math.log(self.class_counts[label] / total)

    def _token_log_prob(self, token: str, label: int) -> float:
        vocab_size = max(len(self.vocabulary), 1)
        numerator = self.token_counts[label][token] + self.smoothing
        denominator = self.total_tokens[label] + self.smoothing * vocab_size
        return math.log(numerator / denominator)

    def predict_one(self, tokens: Sequence[str]) -> int:
        if not self.fitted:
            raise ValueError("Model has not been fitted.")
        scores = {}
        for label in (0, 1):
            score = self._class_log_prior(label)
            score += sum(self._token_log_prob(token, label) for token in tokens)
            scores[label] = score
        return 1 if scores[1] >= scores[0] else 0

    def predict(self, tokenized_texts: Sequence[Sequence[str]]) -> List[int]:
        return [self.predict_one(tokens) for tokens in tokenized_texts]

    def explain_prediction(self, tokens: Sequence[str], top_k: int = 5) -> str:
        if not self.fitted:
            raise ValueError("Model has not been fitted.")
        token_impacts = defaultdict(float)
        for token in tokens:
            token_impacts[token] += self._token_log_prob(token, 1) - self._token_log_prob(
                token, 0
            )
        ranked = sorted(token_impacts.items(), key=lambda item: abs(item[1]), reverse=True)[
            :top_k
        ]
        if not ranked:
            return "No strong token-level evidence was found."
        fragments = []
        for token, impact in ranked:
            if impact > 0:
                direction = "bullish"
            elif impact < 0:
                direction = "bearish"
            else:
                direction = "neutral"
            fragments.append(f"{token}({direction},{impact:.2f})")
        return "Top evidence tokens: " + ", ".join(fragments)


class EvaluatorAgent:
    def accuracy(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        if not y_true:
            return 0.0
        return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

    def build_manual_explanation_set(
        self,
        samples: Sequence[NewsSample],
        predictions: Sequence[int],
        explanations: Sequence[str],
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        rows = []
        for sample, pred, explanation in zip(samples, predictions, explanations):
            rows.append(
                {
                    "text": sample.text,
                    "label": str(sample.label),
                    "prediction": str(pred),
                    "explanation": explanation,
                    "manual_score_1_to_5": "",
                }
            )
            if len(rows) >= limit:
                break
        return rows


class ManagerAgent:
    def __init__(
        self,
        random_seed: int = 7,
        search_space: Sequence[Dict[str, Any]] | None = None,
    ):
        self.random_seed = random_seed
        raw_space = list(search_space) if search_space is not None else list(DEFAULT_SEARCH_SPACE)
        self.search_space = []
        for candidate in raw_space:
            if not REQUIRED_SEARCH_KEYS <= set(candidate):
                raise ValueError(
                    "Each search-space candidate must define include_bigrams, "
                    "min_token_length, and smoothing."
                )
            self.search_space.append(
                {
                    "include_bigrams": bool(candidate["include_bigrams"]),
                    "min_token_length": int(candidate["min_token_length"]),
                    "smoothing": float(candidate["smoothing"]),
                }
            )
        self.evaluator = EvaluatorAgent()
        self.best_processing_agent: ProcessingAgent | None = None
        self.best_classifier_agent: ClassifierAgent | None = None
        self.best_validation_accuracy = -1.0

    def _split_data(
        self, samples: Sequence[NewsSample], train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[List[NewsSample], List[NewsSample], List[NewsSample]]:
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio and val_ratio must satisfy: train_ratio>0, val_ratio>=0, sum<1.")
        shuffled = list(samples)
        random.Random(self.random_seed).shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]
        return train, val, test

    def _train_once(
        self,
        train_samples: Sequence[NewsSample],
        val_samples: Sequence[NewsSample],
        include_bigrams: bool,
        min_token_length: int,
        smoothing: float,
    ) -> Tuple[float, ProcessingAgent, ClassifierAgent]:
        processing = ProcessingAgent(
            include_bigrams=include_bigrams, min_token_length=min_token_length
        )
        classifier = ClassifierAgent(smoothing=smoothing)
        train_tokens = [processing.tokenize(sample.text) for sample in train_samples]
        train_labels = [sample.label for sample in train_samples]
        classifier.fit(train_tokens, train_labels)
        val_tokens = [processing.tokenize(sample.text) for sample in val_samples]
        val_labels = [sample.label for sample in val_samples]
        val_pred = classifier.predict(val_tokens)
        val_acc = self.evaluator.accuracy(val_labels, val_pred)
        return val_acc, processing, classifier

    def run_iterative_training(self, samples: Sequence[NewsSample]) -> Dict[str, float]:
        if len(samples) < MIN_SAMPLES_REQUIRED:
            raise ValueError(
                f"Provide at least {MIN_SAMPLES_REQUIRED} labeled samples for training and evaluation."
            )
        train_samples, val_samples, test_samples = self._split_data(samples)

        for candidate in self.search_space:
            val_acc, processing, classifier = self._train_once(
                train_samples=train_samples,
                val_samples=val_samples,
                include_bigrams=candidate["include_bigrams"],
                min_token_length=candidate["min_token_length"],
                smoothing=candidate["smoothing"],
            )
            if val_acc > self.best_validation_accuracy:
                self.best_validation_accuracy = val_acc
                self.best_processing_agent = processing
                self.best_classifier_agent = classifier

        if self.best_processing_agent is None or self.best_classifier_agent is None:
            raise ValueError("No valid model configuration was found during iterative training.")
        test_tokens = [self.best_processing_agent.tokenize(sample.text) for sample in test_samples]
        test_labels = [sample.label for sample in test_samples]
        test_pred = self.best_classifier_agent.predict(test_tokens)
        test_accuracy = self.evaluator.accuracy(test_labels, test_pred)
        return {"validation_accuracy": self.best_validation_accuracy, "test_accuracy": test_accuracy}

    def predict_with_justification(self, text: str) -> Dict[str, str]:
        if not self.best_processing_agent or not self.best_classifier_agent:
            raise ValueError("Model is not trained. Call run_iterative_training first.")
        tokens = self.best_processing_agent.tokenize(text)
        pred = self.best_classifier_agent.predict_one(tokens)
        explanation = self.best_classifier_agent.explain_prediction(tokens)
        return {
            "prediction": "UP" if pred == 1 else "DOWN",
            "explanation": explanation,
        }

    def generate_manual_evaluation_rows(
        self, samples: Sequence[NewsSample], limit: int = 20
    ) -> List[Dict[str, str]]:
        if not self.best_processing_agent or not self.best_classifier_agent:
            raise ValueError("Model is not trained. Call run_iterative_training first.")
        tokenized = [self.best_processing_agent.tokenize(sample.text) for sample in samples]
        predictions = self.best_classifier_agent.predict(tokenized)
        explanations = [
            self.best_classifier_agent.explain_prediction(tokens) for tokens in tokenized
        ]
        return self.evaluator.build_manual_explanation_set(
            samples=samples,
            predictions=predictions,
            explanations=explanations,
            limit=limit,
        )


def _write_manual_eval_csv(rows: Iterable[Dict[str, str]], output_path: str) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-based stock prediction from financial news")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--manual-eval-output",
        default="manual_explanation_eval.csv",
        help="Output CSV for manual explanation scoring",
    )
    parser.add_argument("--example-text", default="", help="Optional text for one prediction demo")
    args = parser.parse_args()

    samples = load_financial_news_csv(args.dataset)
    manager = ManagerAgent()
    metrics = manager.run_iterative_training(samples)
    print(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    manual_rows = manager.generate_manual_evaluation_rows(samples, limit=30)
    _write_manual_eval_csv(manual_rows, args.manual_eval_output)
    print(f"Manual explanation file: {args.manual_eval_output} ({len(manual_rows)} rows)")
    if args.example_text:
        output = manager.predict_with_justification(args.example_text)
        print(f"Prediction: {output['prediction']}")
        print(f"Explanation: {output['explanation']}")


if __name__ == "__main__":
    main()
