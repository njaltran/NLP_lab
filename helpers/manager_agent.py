"""Manager agent for orchestration and iterative tuning."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

from .classifier_agent import ClassifierAgent
from .constants import DEFAULT_SEARCH_SPACE, MIN_SAMPLES_REQUIRED, REQUIRED_SEARCH_KEYS
from .evaluator_agent import EvaluatorAgent
from .processing_agent import ProcessingAgent
from .types import NewsSample


class ManagerAgent:
    """Orchestrate training, validation-based model selection, and inference."""

    def __init__(
        self,
        random_seed: int = 7,
        search_space: Sequence[Dict[str, Any]] | None = None,
    ):
        """Create manager with reproducible random seed and optional search space."""
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
        """Split samples into train/validation/test partitions."""
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
        """Train one processing/classifier configuration and return validation accuracy."""
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
        """Search candidate configurations and evaluate selected model on test split."""
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
        """Predict market direction and return an explanation grounded in text tokens."""
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
        """Create rows for manual explanation evaluation on a sample subset."""
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
