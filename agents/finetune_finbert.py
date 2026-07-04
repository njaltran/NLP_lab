"""Fine-tune FinBERT on the up/down/neutral move labels (standalone, run once).

Not part of the pipeline loop. Trains on processed_data.csv's split=train rows
(the last 10% of train *dates* held out for validation), evaluates on split=test,
saves the best-validation model to outputs/finbert_finetuned/ and a
finetune_report.json reproducibility record next to it. The generated classifier
(agents/nadi_classifier.py) picks the saved model up automatically on its next run.

    uv run agents/finetune_finbert.py --data data/processed_data.csv
"""

import argparse
import json
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "ProsusAI/finbert"
LABELS = ["up", "down", "neutral"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class HeadlineDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_length):
        self.enc = tokenizer(list(titles), truncation=True, max_length=max_length,
                             padding="max_length", return_tensors="pt")
        self.labels = torch.tensor([LABEL2ID[label] for label in labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _split_frames(df: pd.DataFrame):
    """Train/val/test frames. Validation = last 10% of train dates (time-ordered,
    no random leakage into the test window)."""
    if "split" not in df.columns:
        raise SystemExit(
            "processed_data.csv has no 'split' column — regenerate it with the "
            "updated Processing agent (agents/aurora_processing.py) first."
        )
    train_all = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    val_cut = pd.to_datetime(train_all["date"]).quantile(0.9)
    is_val = pd.to_datetime(train_all["date"]) > val_cut
    return train_all[~is_val], train_all[is_val], test


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct_total = 0
    per_class = {i: [0, 0] for i in range(len(LABELS))}  # [hits, count]
    for batch in loader:
        preds = (
            model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            .logits.argmax(dim=1)
            .cpu()
        )
        for pred, true in zip(preds, batch["labels"]):
            per_class[int(true)][1] += 1
            if pred == true:
                per_class[int(true)][0] += 1
                correct_total += 1
    n = sum(count for _, count in per_class.values())
    accuracy = correct_total / n if n else 0.0
    class_acc = {
        LABELS[i]: round(hits / count, 4) if count else 0.0
        for i, (hits, count) in per_class.items()
    }
    return accuracy, class_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/processed_data.csv")
    ap.add_argument("--out-dir", default="outputs/finbert_finetuned")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per split (smoke tests)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.data)
    train, val, test = _split_frames(df)
    if args.limit:
        train = train.head(args.limit)
        val = val.head(args.limit)
        test = test.head(args.limit)
    print(
        f"[finetune] rows: {len(train)} train / {len(val)} val / {len(test)} test"
        f" | train dates {train['date'].min()}..{train['date'].max()}"
    )

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        token=HF_TOKEN,
        num_labels=len(LABELS),
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # fresh 3-way head over the sentiment one
    ).to(device)

    def make_loader(frame, shuffle=False):
        return DataLoader(
            HeadlineDataset(frame["article_title"], frame["label"], tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val)
    test_loader = make_loader(test)

    # Weighted cross-entropy so the neutral-heavy distribution doesn't collapse
    # up/down classes to noise.
    counts = train["label"].value_counts()
    weights = torch.tensor(
        [len(train) / (len(LABELS) * counts.get(name, 1)) for name in LABELS],
        dtype=torch.float,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val, history = -1.0, []
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f}")
        val_acc, _ = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(running / max(len(train_loader), 1), 4),
            "val_accuracy": round(val_acc, 4),
        })
        print(f"[finetune] epoch {epoch}: val accuracy {val_acc:.4f}")
        if val_acc > best_val:  # save best-validation weights, not last
            best_val = val_acc
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)

    # Evaluate the saved (best-val) model on the held-out test split.
    best_model = AutoModelForSequenceClassification.from_pretrained(args.out_dir).to(device)
    test_acc, test_class = evaluate(best_model, test_loader, device)
    print(f"[finetune] test accuracy {test_acc:.4f} per-class {test_class}")

    report = {
        "base_model": MODEL,
        "model_dir": args.out_dir,
        "test_accuracy": round(test_acc, 4),
        "test_class_accuracy": test_class,
        "best_val_accuracy": round(best_val, 4),
        "epochs": history,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_date_range": [str(train["date"].min()), str(train["date"].max())],
        "test_date_range": [str(test["date"].min()), str(test["date"].max())],
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "seed": args.seed,
            "limit": args.limit,
        },
    }
    report_path = os.path.join(os.path.dirname(args.out_dir) or ".", "finetune_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[finetune] report written to {report_path}")


if __name__ == "__main__":
    main()
