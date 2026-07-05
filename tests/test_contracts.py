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
