"""Tests for Nadi owning fine-tuning and per-head hyperparameter tuning
(ADR 0001). `next_head_training_params` is the pure scheduling function: given
one directional head's own prior attempts, pick the next learning rate and
focus weight. It has no notion of the OTHER head — each head's schedule is
independent (ADR 0002 decision to split the heads applies to tuning too).
"""

from agents.nadi_classifier import (
    TRAINING_LR_FLOOR,
    TRAINING_LR_START,
    TRAINING_MULTIPLIER_MAX,
    next_head_training_params,
)


def test_first_attempt_starts_at_the_base_learning_rate():
    params = next_head_training_params([], collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_START
    assert params["focus_weight_multiplier"] == 1.25


def test_first_attempt_uses_a_higher_focus_weight_when_collapsed():
    params = next_head_training_params([], collapsed=True)

    assert params["focus_weight_multiplier"] == 1.5


def test_regression_backs_off_learning_rate_and_focus_weight():
    history = [{"learning_rate": TRAINING_LR_START, "focus_weight_multiplier": 1.5,
               "regressed": True}]

    params = next_head_training_params(history, collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_START * 0.5
    assert params["focus_weight_multiplier"] == 1.25


def test_regression_floors_the_learning_rate():
    history = [{"learning_rate": TRAINING_LR_FLOOR, "focus_weight_multiplier": 1.5,
               "regressed": True}]

    params = next_head_training_params(history, collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_FLOOR


def test_no_regression_holds_learning_rate_and_steps_focus_weight_up():
    history = [{"learning_rate": TRAINING_LR_START, "focus_weight_multiplier": 1.25,
               "regressed": False}]

    params = next_head_training_params(history, collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_START
    assert params["focus_weight_multiplier"] == 1.5


def test_focus_weight_never_exceeds_the_max():
    history = [{"learning_rate": TRAINING_LR_START,
               "focus_weight_multiplier": TRAINING_MULTIPLIER_MAX, "regressed": False}]

    params = next_head_training_params(history, collapsed=False)

    assert params["focus_weight_multiplier"] == TRAINING_MULTIPLIER_MAX


def test_only_the_most_recent_attempt_matters():
    """Older attempts don't influence the next pick — only what happened last."""
    history = [
        {"learning_rate": TRAINING_LR_START, "focus_weight_multiplier": 1.25, "regressed": True},
        {"learning_rate": TRAINING_LR_START * 0.5, "focus_weight_multiplier": 1.0, "regressed": False},
    ]

    params = next_head_training_params(history, collapsed=False)

    assert params["learning_rate"] == TRAINING_LR_START * 0.5
    assert params["focus_weight_multiplier"] == 1.25
