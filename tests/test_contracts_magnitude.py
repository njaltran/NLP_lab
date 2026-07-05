from agents.contracts import LABELS, PREDICTION_COLUMNS, EXPLANATION_SAMPLE_COLUMNS


def test_labels_are_binary_magnitude():
    assert LABELS == ("big", "small")


def test_prediction_columns_have_binary_probs():
    assert "prob_big" in PREDICTION_COLUMNS
    assert "prob_small" in PREDICTION_COLUMNS
    assert "prob_up" not in PREDICTION_COLUMNS
    assert "prob_neutral" not in PREDICTION_COLUMNS
    # order: prob_big and prob_small sit where the old three probs were
    assert PREDICTION_COLUMNS == [
        "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
        "pct_change", "label", "predicted_label", "confidence",
        "prob_big", "prob_small", "split",
    ]


def test_explanation_sample_columns_binary():
    assert EXPLANATION_SAMPLE_COLUMNS == [
        "article_id", "article_title", "predicted_label", "actual_label",
        "confidence", "prob_big", "prob_small",
    ]
