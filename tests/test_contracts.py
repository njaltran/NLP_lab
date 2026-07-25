"""Tests for the shared handoff contract helpers."""

import csv

import pandas as pd
import pytest

import agents.contracts as contracts


def test_prediction_rows_validate_contract_columns_and_values():
    rows = contracts.read_prediction_rows("mock_data/predictions_test.csv")

    assert rows
    assert [*rows[0].keys()] == contracts.PREDICTION_COLUMNS
    contracts.validate_prediction_rows(rows)

    rows[0]["split"] = "train"
    with pytest.raises(ValueError, match="split must be val or test"):
        contracts.validate_prediction_rows(rows)


def test_explanation_sample_is_shaped_from_predictions(tmp_path):
    out = tmp_path / "sample_for_explanation.csv"

    n = contracts.write_explanation_sample(
        "mock_data/predictions_test.csv",
        out,
        sample_size=3,
    )

    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert n == 3
    assert [*rows[0].keys()] == contracts.EXPLANATION_SAMPLE_COLUMNS
    assert all(row["actual_label"] in contracts.LABELS for row in rows)
    test_ids = set(
        pd.read_csv("mock_data/predictions_test.csv")
        .query("split == 'test'")["article_id"]
    )
    assert {row["article_id"] for row in rows} <= test_ids


def test_final_results_are_shaped_from_predictions_and_explanations():
    final = contracts.build_final_results(
        pd.read_csv("mock_data/predictions_test.csv"),
        pd.read_csv("mock_data/explanations.csv"),
    )

    assert list(final.columns) == contracts.FINAL_RESULTS_COLUMNS
    assert final["explanation"].fillna("").ne("").any()
    assert set(final["label"]) <= set(contracts.LABELS)
    assert len(final) == (
        pd.read_csv("mock_data/predictions_test.csv")["split"] == "test"
    ).sum()


def test_combined_probs_always_sum_to_one():
    for p_up, p_down in [(0.0, 0.0), (1.0, 1.0), (0.62, 0.20), (0.5, 0.5), (0.9, 0.1)]:
        probs = contracts.combine_binary_probs(p_up, p_down)

        assert set(probs) == {"up", "down", "neutral"}
        assert sum(probs.values()) == pytest.approx(1.0, abs=contracts.PROBABILITY_TOLERANCE)


def test_combined_max_never_falls_below_one_third():
    """The invariant MIN_EFFECTIVE_THRESHOLD (0.34) depends on: three values
    summing to 1 cannot all be below 1/3, so the neutral fallback stays reachable."""
    for p_up in (0.0, 0.25, 0.5, 0.75, 1.0):
        for p_down in (0.0, 0.25, 0.5, 0.75, 1.0):
            probs = contracts.combine_binary_probs(p_up, p_down)

            assert max(probs.values()) >= 1 / 3 - 1e-9


def test_confident_up_head_wins():
    probs = contracts.combine_binary_probs(0.9, 0.1)

    assert max(probs, key=probs.get) == "up"


def test_confident_down_head_wins():
    probs = contracts.combine_binary_probs(0.1, 0.9)

    assert max(probs, key=probs.get) == "down"


def test_both_heads_silent_yields_neutral():
    probs = contracts.combine_binary_probs(0.1, 0.1)

    assert max(probs, key=probs.get) == "neutral"


def test_both_heads_firing_resolves_to_the_stronger_one():
    probs = contracts.combine_binary_probs(0.8, 0.7)

    assert max(probs, key=probs.get) == "up"
    assert probs["up"] > probs["down"]


def test_worked_example_from_the_spec():
    probs = contracts.combine_binary_probs(0.62, 0.20)

    assert probs["up"] == pytest.approx(0.552, abs=0.001)
    assert probs["down"] == pytest.approx(0.178, abs=0.001)
    assert probs["neutral"] == pytest.approx(0.270, abs=0.001)
