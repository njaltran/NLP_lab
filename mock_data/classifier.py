"""classifier.py — MOCK example of the artifact Nadi (Classifier Agent) generates.

Handoff 2 in data_contracts.md: Nadi generates this script, runs it on the test
split, and hands both the code and `predictions_test.csv` to Sabina. Sabina reads
the code to ground her proposal (e.g. spotting the hardcoded threshold below); she
does NOT re-execute it.

Standalone: reads `processed_data.csv`, writes `predictions_test.csv` with the
columns the contract requires (predicted_label, confidence, prob_*, split).

FinBERT outputs positive/negative/neutral sentiment; this maps
positive->up, negative->down, neutral->neutral.

Run:  python classifier.py            # uses ./processed_data.csv
"""

import csv

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "ProsusAI/finbert"
MAX_LENGTH = 128
THRESHOLD = 0.5  # min top-class prob; below this -> neutral. (Sabina: hardcoded)
SENTIMENT_TO_LABEL = {"positive": "up", "negative": "down", "neutral": "neutral"}

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()
# FinBERT label order -> our label names, so probs line up with prob_up/down/neutral
ID2OURS = {i: SENTIMENT_TO_LABEL[model.config.id2label[i].lower()] for i in model.config.id2label}


def classify(title: str) -> dict:
    inputs = tokenizer(title, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    by_label = {ID2OURS[i]: round(float(p), 2) for i, p in enumerate(probs)}
    top_label = max(by_label, key=by_label.get)
    top_prob = by_label[top_label]
    predicted = top_label if top_prob >= THRESHOLD else "neutral"
    return {
        "predicted_label": predicted,
        "confidence": top_prob,
        "prob_up": by_label["up"],
        "prob_down": by_label["down"],
        "prob_neutral": by_label["neutral"],
    }


def main(src: str = "processed_data.csv", dst: str = "predictions_test.csv") -> None:
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    out_cols = list(rows[0].keys()) + [
        "predicted_label", "confidence", "prob_up", "prob_down", "prob_neutral", "split",
    ]
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for row in rows:  # mock: every row is the test split
            row.update(classify(row["article_title"]))
            row["split"] = "test"
            writer.writerow(row)


if __name__ == "__main__":
    main()
