"""Fine-tune FinBERT on our up/down/neutral move labels. Run once, by hand.

This is NOT part of the retune loop. It trains a fresh 3-class head on top of
FinBERT using the `split=="train"` rows from processed_data.csv, checks itself on
the `split=="test"` rows, and saves the trained model to outputs/finbert_finetuned/.
The classifier agent (agents/nadi_classifier.py) loads that folder automatically
when it exists.

Why a separate script: training the model and running the model are different
jobs. Training happens once and is slow; prediction happens every loop iteration
and is fast. Keeping them apart means the loop never retrains.

Run:
    uv run agents/finetune_finbert.py --data data/processed_data.csv
    uv run agents/finetune_finbert.py --data data/processed_data.csv --limit 50   # quick smoke test

Design notes: docs/study_guide.md (Classifier section) explains why we expect a
weak result here — the signal in one headline is thin.
"""

import argparse
import json
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The base model and our three labels. The order fixes the id<->label mapping,
# so id 0 = "up", id 1 = "down", id 2 = "neutral".
BASE_MODEL = "ProsusAI/finbert"
LABELS = ["up", "down", "neutral"]
LABEL_TO_ID = {"up": 0, "down": 1, "neutral": 2}

# Same .env loader main.py uses — imported here too because this script runs
# standalone, outside the pipeline entry point. Dual import: works as a package
# member and as a bare script (`uv run agents/finetune_finbert.py`).
try:
    from agents.env import load_dotenv
except ModuleNotFoundError:
    from env import load_dotenv

load_dotenv()

# Use the same HF token the classifier uses, if one is set (higher rate limits).
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class HeadlineDataset(Dataset):
    """Wraps headlines + labels so a DataLoader can hand them to the model in
    batches. Each item is a dict of tensors, which is what the model expects."""

    def __init__(self, titles, labels, tokenizer, max_length):
        # Tokenize every headline up front (pad them all to the same length).
        self.encodings = tokenizer(
            list(titles),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # Turn the string labels ("up"/"down"/"neutral") into ids (0/1/2).
        self.label_ids = torch.tensor([LABEL_TO_ID[label] for label in labels])

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
            "labels": self.label_ids[i],
        }


def pick_device():
    """Use the Mac GPU (mps) or a CUDA GPU if available, else the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_frames(df):
    """Split the data into train / validation / test DataFrames.

    All three come straight from Aurora's `split` column — she owns the one
    date-based boundary (val = last 10% of training dates), so this script and
    the retune loop validate on the same rows. Older files without `val` rows
    get the same carve computed here as a fallback."""
    if "split" not in df.columns:
        raise SystemExit(
            "processed_data.csv has no 'split' column. Regenerate it with the "
            "Processing agent first: uv run agents/aurora_processing.py"
        )

    test = df[df["split"] == "test"]

    if (df["split"] == "val").any():
        train = df[df["split"] == "train"]
        val = df[df["split"] == "val"]
        return train, val, test

    # Fallback for pre-val files: carve the LAST 10% of training DATES (not
    # random rows) so validation looks like the real future test set.
    train_all = df[df["split"] == "train"]
    val_cutoff = pd.to_datetime(train_all["date"]).quantile(0.9)
    is_val = pd.to_datetime(train_all["date"]) > val_cutoff

    train = train_all[~is_val]
    val = train_all[is_val]
    return train, val, test


def accuracy_on(model, loader, device):
    """Run the model over a DataLoader and return overall + per-class accuracy."""
    model.eval()
    correct = 0
    total = 0
    # per_class[label] = [number correct, number seen]
    per_class = {label: [0, 0] for label in LABELS}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions = logits.argmax(dim=1).cpu()

            for prediction, true_id in zip(predictions, batch["labels"]):
                true_label = LABELS[int(true_id)]
                per_class[true_label][1] += 1
                total += 1
                if int(prediction) == int(true_id):
                    per_class[true_label][0] += 1
                    correct += 1

    overall = correct / total if total else 0.0
    class_accuracy = {}
    for label in LABELS:
        hits, seen = per_class[label]
        class_accuracy[label] = round(hits / seen, 4) if seen else 0.0
    return overall, class_accuracy


def class_weights(train, device):
    """Give rarer classes more weight so the neutral-heavy data doesn't push the
    model into always predicting 'neutral'. Weight = total / (3 * count)."""
    counts = train["label"].value_counts()
    weights = []
    for label in LABELS:
        count = counts.get(label, 1)
        weights.append(len(train) / (3 * count))
    return torch.tensor(weights, dtype=torch.float).to(device)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on move labels.")
    parser.add_argument("--data", default="data/processed_data.csv")
    parser.add_argument("--out-dir", default="outputs/finbert_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows per split for a quick smoke test.")
    args = parser.parse_args()

    # Make the run repeatable.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load and split the data.
    df = pd.read_csv(args.data)
    train, val, test = split_frames(df)
    if args.limit:
        train = train.head(args.limit)
        val = val.head(args.limit)
        test = test.head(args.limit)
    print(f"[finetune] {len(train)} train / {len(val)} val / {len(test)} test rows")
    print(f"[finetune] train dates {train['date'].min()} .. {train['date'].max()}")

    device = pick_device()
    print(f"[finetune] device: {device}")

    # 2. Load FinBERT with a fresh 3-class head (its original head is 3-class
    # sentiment; we replace it with our up/down/neutral head).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        num_labels=3,
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,  # allow swapping in our own head
    ).to(device)

    # 3. Build the data loaders.
    def make_loader(frame, shuffle):
        dataset = HeadlineDataset(frame["article_title"], frame["label"],
                                  tokenizer, args.max_length)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val, shuffle=False)
    test_loader = make_loader(test, shuffle=False)

    # 4. Set up the loss (class-weighted) and optimizer.
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 5. Train. Keep the checkpoint with the best validation accuracy, not the
    # last one, because the model tends to overfit after the first epoch.
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_accuracy = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_accuracy, _ = accuracy_on(model, val_loader, device)
        history.append({"epoch": epoch,
                        "train_loss": round(train_loss, 4),
                        "val_accuracy": round(val_accuracy, 4)})
        print(f"[finetune] epoch {epoch}: train_loss {train_loss:.4f} "
              f"val_accuracy {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            print(f"[finetune] saved new best model to {args.out_dir}")

    # 6. Score the saved (best) model on the held-out test set.
    best_model = AutoModelForSequenceClassification.from_pretrained(args.out_dir).to(device)
    test_accuracy, test_class_accuracy = accuracy_on(best_model, test_loader, device)
    print(f"[finetune] test accuracy {test_accuracy:.4f}  per-class {test_class_accuracy}")

    # 7. Write a small report so the run is reproducible and reviewable.
    report = {
        "base_model": BASE_MODEL,
        "model_dir": args.out_dir,
        "test_accuracy": round(test_accuracy, 4),
        "test_class_accuracy": test_class_accuracy,
        "best_val_accuracy": round(best_val_accuracy, 4),
        "epochs": history,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "train_date_range": [str(train["date"].min()), str(train["date"].max())],
        "test_date_range": [str(test["date"].min()), str(test["date"].max())],
        "params": {"epochs": args.epochs, "batch_size": args.batch_size,
                   "lr": args.lr, "max_length": args.max_length, "seed": args.seed},
    }
    report_path = os.path.join(os.path.dirname(args.out_dir) or ".", "finetune_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[finetune] report written to {report_path}")


if __name__ == "__main__":
    main()
