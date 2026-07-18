"""Reusable per-head FinBERT training engine.

Nadi (`agents/nadi_classifier.py`) owns fine-tuning directly (ADR 0001): it
calls `train_finbert()` here once per directional head it needs to retrain.
Each head is a genuine 2-class classifier — its own movement class versus
`neutral` — trained on a filtered subset of the data, not one-vs-rest
(ADR 0002): the up-head never sees `down` rows, and vice versa.
"""

import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from agents.env import load_dotenv
    from agents.finetune_finbert import split_frames
except ModuleNotFoundError:
    from env import load_dotenv
    from finetune_finbert import split_frames

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

BASE_MODEL = "ProsusAI/finbert"
HEADS = ("up", "down")


def rows_for_head(frame: pd.DataFrame, head: str) -> pd.DataFrame:
    """Filter to the rows a directional head is allowed to train on: its own
    movement class plus `neutral`. The opposite movement class is dropped
    entirely — the head never sees it during training (ADR 0002)."""
    if head not in HEADS:
        raise ValueError(f"head must be 'up' or 'down', got {head!r}")
    return frame[frame["label"].isin({head, "neutral"})]


class _HeadlineDataset(torch.utils.data.Dataset):
    """Tokenized headlines and binary label ids (0 = neutral, 1 = head)."""

    def __init__(self, titles, labels, label_to_id, tokenizer, max_length):
        self.encodings = tokenizer(
            list(titles), truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        self.label_ids = torch.tensor([label_to_id[label] for label in labels])

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.label_ids[index],
        }


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _accuracy_on(model, loader, device, id_to_label):
    model.eval()
    correct, total = 0, 0
    per_class = {label: [0, 0] for label in id_to_label.values()}
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            predictions = logits.argmax(dim=1).cpu()
            for prediction, true_id in zip(predictions, batch["labels"]):
                label = id_to_label[int(true_id)]
                per_class[label][1] += 1
                total += 1
                if int(prediction) == int(true_id):
                    per_class[label][0] += 1
                    correct += 1
    overall = correct / total if total else 0.0
    class_accuracy = {
        label: round(hits / seen, 4) if seen else 0.0
        for label, (hits, seen) in per_class.items()
    }
    return overall, class_accuracy


def _class_weights(train: pd.DataFrame, head: str, label_to_id: dict,
                    multiplier: float) -> torch.Tensor:
    """Inverse-frequency weights over the head's own binary split, with
    `multiplier` boosting the head's positive class (ADR 0001: Nadi picks
    this per head, per retune, from its own try history)."""
    counts = train["label"].value_counts()
    weights = [0.0, 0.0]
    for label, index in label_to_id.items():
        base = len(train) / (2 * max(int(counts.get(label, 0)), 1))
        weights[index] = base * multiplier if label == head else base
    return torch.tensor(weights, dtype=torch.float)


def _publish_checkpoint(temporary: Path, destination: Path) -> None:
    """Atomically swap the new checkpoint into place, keeping a backup until
    the swap succeeds so a crash mid-publish can never leave `destination`
    half-written."""
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def train_finbert(
    *,
    data_path: str,
    out_dir: str,
    head: str,
    learning_rate: float,
    focus_weight_multiplier: float = 1.0,
    epochs: int = 1,
    batch_size: int = 16,
    max_length: int = 64,
    seed: int = 42,
    parent_model_dir: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Train one directional head (`head` vs `neutral`) and atomically
    publish it to `out_dir`. Continues from `parent_model_dir` when given
    (an earlier retune of this same head), otherwise starts from pretrained
    FinBERT."""
    if head not in HEADS:
        raise ValueError(f"head must be 'up' or 'down', got {head!r}")

    random.seed(seed)
    torch.manual_seed(seed)

    frame = pd.read_csv(data_path)
    train_all, val_all, _ = split_frames(frame)
    train, val = rows_for_head(train_all, head), rows_for_head(val_all, head)
    if limit:
        train, val = train.head(limit), val.head(limit)
    if train.empty or val.empty:
        raise ValueError(f"{head}-head has no train/val rows to train on")

    label_to_id = {"neutral": 0, head: 1}
    id_to_label = {0: "neutral", 1: head}

    source = parent_model_dir or BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(source, token=HF_TOKEN)
    model_kwargs: dict[str, Any] = {"token": HF_TOKEN}
    if not parent_model_dir:
        model_kwargs.update(
            num_labels=2, id2label=id_to_label, label2id=label_to_id,
            ignore_mismatched_sizes=True,
        )
    device = _pick_device()
    model = AutoModelForSequenceClassification.from_pretrained(source, **model_kwargs).to(device)

    def make_loader(rows: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = _HeadlineDataset(rows["article_title"], rows["label"],
                                   label_to_id, tokenizer, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val, shuffle=False)

    weight_tensor = _class_weights(train, head, label_to_id, focus_weight_multiplier).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    destination = Path(out_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))

    try:
        best_val_accuracy = -1.0
        best_val_class_accuracy: dict[str, float] = {}
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
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
            val_accuracy, val_class_accuracy = _accuracy_on(model, val_loader, device, id_to_label)
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                            "val_accuracy": round(val_accuracy, 4)})
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_class_accuracy = val_class_accuracy.copy()
                model.save_pretrained(temporary)
                tokenizer.save_pretrained(temporary)

        report = {
            "head": head,
            "base_model": BASE_MODEL,
            "parent_model": parent_model_dir or BASE_MODEL,
            "model_dir": str(destination),
            "params": {
                "learning_rate": learning_rate,
                "focus_weight_multiplier": focus_weight_multiplier,
                "epochs": epochs, "batch_size": batch_size,
                "max_length": max_length, "seed": seed,
            },
            "epochs_history": history,
            "best_val_accuracy": round(best_val_accuracy, 4),
            "val_class_accuracy": best_val_class_accuracy,
            "rows": {"train": len(train), "val": len(val)},
        }
        (temporary / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _publish_checkpoint(temporary, destination)
        return {"model_dir": str(destination), "report": report}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
