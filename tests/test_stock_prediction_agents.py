import csv

import pytest

from stock_prediction_agents import ClassifierAgent, ManagerAgent, NewsSample, ProcessingAgent, load_financial_news_csv


def _samples():
    return [
        NewsSample("Company reports strong earnings growth and profit beat", 1),
        NewsSample("Stock rises after major contract win", 1),
        NewsSample("Revenue jumps and outlook upgraded by analysts", 1),
        NewsSample("Shares climb on positive guidance", 1),
        NewsSample("Bullish momentum with rising demand and margins", 1),
        NewsSample("Company misses earnings and cuts guidance", 0),
        NewsSample("Stock drops after weak sales and layoffs", 0),
        NewsSample("Profit warning and falling revenue hit sentiment", 0),
        NewsSample("Bearish outlook after legal investigation", 0),
        NewsSample("Shares plunge on downgrade and losses", 0),
        NewsSample("Strong quarter with record cash flow", 1),
        NewsSample("Market rallies as firm announces expansion", 1),
        NewsSample("Demand slowdown triggers concerns and decline", 0),
        NewsSample("Debt issues and weak forecast pressure shares", 0),
    ]


@pytest.mark.parametrize("seed", [7, 11])
def test_end_to_end_training_and_explanation(seed):
    manager = ManagerAgent(random_seed=seed)
    metrics = manager.run_iterative_training(_samples())
    assert metrics["validation_accuracy"] >= 0.0
    assert metrics["test_accuracy"] >= 0.0

    output = manager.predict_with_justification("Analysts upgrade stock after strong demand")
    assert output["prediction"] in {"UP", "DOWN"}
    assert "Top evidence tokens:" in output["explanation"]

    manual_rows = manager.generate_manual_evaluation_rows(_samples(), limit=5)
    assert len(manual_rows) == 5
    assert "manual_score_1_to_5" in manual_rows[0]


def test_deterministic_for_same_seed():
    manager_a = ManagerAgent(random_seed=7)
    manager_b = ManagerAgent(random_seed=7)
    metrics_a = manager_a.run_iterative_training(_samples())
    metrics_b = manager_b.run_iterative_training(_samples())
    assert metrics_a == metrics_b


def test_csv_loader_detects_supported_columns(tmp_path):
    csv_path = tmp_path / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["headline", "movement"])
        writer.writeheader()
        writer.writerow({"headline": "Stock rises after guidance raise", "movement": "up"})
        writer.writerow({"headline": "Shares fall after poor quarter", "movement": "down"})

    data = load_financial_news_csv(str(csv_path))
    assert len(data) == 2
    assert data[0].label == 1
    assert data[1].label == 0


def test_input_validation_and_helper_modularity():
    with pytest.raises(ValueError):
        ProcessingAgent(min_token_length=0)
    with pytest.raises(ValueError):
        ClassifierAgent(smoothing=-0.1)

    classifier = ClassifierAgent()
    with pytest.raises(ValueError):
        classifier.fit([[], []], [0, 1])

    manager = ManagerAgent()
    with pytest.raises(ValueError):
        manager._split_data([NewsSample("x", 1)] * 10, train_ratio=0.8, val_ratio=0.3)

    assert ClassifierAgent._direction_from_impact(1.5) == "bullish"
    assert ClassifierAgent._direction_from_impact(-0.5) == "bearish"
    assert ClassifierAgent._direction_from_impact(0.0) == "neutral"


def test_token_impact_scores_are_aggregated_per_token():
    classifier = ClassifierAgent()
    classifier.fit(tokenized_texts=[["gain", "profit"], ["loss", "drop"]], labels=[1, 0])
    impacts = classifier._token_impact_scores(["gain", "gain", "loss"])
    assert "gain" in impacts
    assert "loss" in impacts
    assert impacts["gain"] > 0.0
    assert impacts["loss"] < 0.0
