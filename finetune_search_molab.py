# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.11",
#     "pandas>=2.0.0",
#     "torch>=2.2.0",
#     "transformers>=4.38.0",
# ]
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import json
    import os
    import random

    import marimo as mo
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataLoader,
        Dataset,
        itertools,
        json,
        mo,
        os,
        pd,
        random,
        torch,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # FinBERT retune search

        Recreates `agents/finetune_finbert.py`'s train/save/report logic,
        looped over a fixed seed x lr grid. No widgets — edit `CONFIG` below
        and re-run top to bottom. Ranks every combo by val accuracy, then
        retrains and saves the winner to disk with the same
        `finetune_report.json` schema the standalone script writes.
        """
    )
    return


@app.cell
def _():
    # Full grid: every combo of every list below gets trained. Grid size =
    # product of list lengths (this default is 3x3x2x2 = 36 full training
    # runs) — trim the lists for a quick pass, e.g. via row_limit first.
    CONFIG = {
        "data_path": "data/processed_data.csv",
        "seeds": [1, 2, 3],
        "lrs": [1e-5, 2e-5, 3e-5],
        "epochs_grid": [3, 6],
        "batch_sizes": [16, 32],
        "max_length": 128,
        "row_limit": 0,  # 0 = full data; set e.g. 200 for a quick smoke test
        "out_dir": "outputs/finbert_finetuned_search_best",  # separate from
        # outputs/finbert_finetuned so this never silently overwrites the
        # model the classifier agent auto-loads
    }
    return (CONFIG,)


@app.cell
def _():
    BASE_MODEL = "ProsusAI/finbert"
    LABELS = ["up", "down", "neutral"]
    LABEL_TO_ID = {"up": 0, "down": 1, "neutral": 2}
    return BASE_MODEL, LABELS, LABEL_TO_ID


@app.cell
def _(Dataset, LABEL_TO_ID, torch):
    class HeadlineDataset(Dataset):
        def __init__(self, titles, labels, tokenizer, max_length):
            self.encodings = tokenizer(
                list(titles),
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.label_ids = torch.tensor([LABEL_TO_ID[label] for label in labels])

        def __len__(self):
            return len(self.label_ids)

        def __getitem__(self, i):
            return {
                "input_ids": self.encodings["input_ids"][i],
                "attention_mask": self.encodings["attention_mask"][i],
                "labels": self.label_ids[i],
            }

    return (HeadlineDataset,)


@app.cell
def _(torch):
    def pick_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return (pick_device,)


@app.cell
def _(pd):
    def split_frames(df):
        if "split" not in df.columns:
            raise ValueError(
                "processed_data.csv has no 'split' column. Regenerate it with "
                "agents/aurora_processing.py first."
            )

        test = df[df["split"] == "test"]

        if (df["split"] == "val").any():
            train = df[df["split"] == "train"]
            val = df[df["split"] == "val"]
            return train, val, test

        train_all = df[df["split"] == "train"]
        val_cutoff = pd.to_datetime(train_all["date"]).quantile(0.9)
        is_val = pd.to_datetime(train_all["date"]) > val_cutoff
        return train_all[~is_val], train_all[is_val], test

    return (split_frames,)


@app.cell
def _(LABELS, torch):
    def class_weights(train, device):
        counts = train["label"].value_counts()
        weights = []
        for label in LABELS:
            count = counts.get(label, 1)
            weights.append(len(train) / (3 * count))
        return torch.tensor(weights, dtype=torch.float).to(device)

    def accuracy_on(model, loader, device):
        model.eval()
        correct = 0
        total = 0
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
        class_accuracy = {
            label: round(hits / seen, 4) if seen else 0.0
            for label, (hits, seen) in per_class.items()
        }
        return overall, class_accuracy

    return accuracy_on, class_weights


@app.cell
def _(
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BASE_MODEL,
    DataLoader,
    HeadlineDataset,
    LABEL_TO_ID,
    LABELS,
    accuracy_on,
    class_weights,
    json,
    os,
    pick_device,
    random,
    torch,
):
    def train_one(train, val, test, seed, lr, epochs, batch_size, max_length, save_dir=None):
        """One full train run for a single (seed, lr) combo — this IS
        agents/finetune_finbert.py's train/save/report logic, parameterized
        so a search can call it once per combo. When save_dir is given it
        writes the same artifacts the standalone script does: a saved
        model+tokenizer at save_dir, and a finetune_report.json next to it
        with the identical schema (base_model, model_dir, test_accuracy,
        test_class_accuracy, best_val_accuracy, epochs history, rows,
        date ranges, params)."""
        random.seed(seed)
        torch.manual_seed(seed)

        device = pick_device()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=3,
            id2label={i: label for i, label in enumerate(LABELS)},
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
        ).to(device)

        def make_loader(frame, shuffle):
            dataset = HeadlineDataset(frame["article_title"], frame["label"], tokenizer, max_length)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        train_loader = make_loader(train, shuffle=True)
        val_loader = make_loader(val, shuffle=False)
        test_loader = make_loader(test, shuffle=False)

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train, device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        best_val_accuracy = -1.0
        best_state = None
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
            val_accuracy, _ = accuracy_on(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_accuracy": round(val_accuracy, 4)})
            print(f"[search] seed={seed} lr={lr} epoch={epoch} train_loss={train_loss:.4f} val_accuracy={val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        test_accuracy, test_class_accuracy = accuracy_on(model, test_loader, device)

        result = {
            "seed": seed,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "best_val_accuracy": round(best_val_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "test_class_accuracy": test_class_accuracy,
            "history": history,
        }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            report = {
                "base_model": BASE_MODEL,
                "model_dir": save_dir,
                "test_accuracy": round(test_accuracy, 4),
                "test_class_accuracy": test_class_accuracy,
                "best_val_accuracy": round(best_val_accuracy, 4),
                "epochs": history,
                "rows": {"train": len(train), "val": len(val), "test": len(test)},
                "train_date_range": [str(train["date"].min()), str(train["date"].max())],
                "test_date_range": [str(test["date"].min()), str(test["date"].max())],
                "params": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                           "max_length": max_length, "seed": seed},
            }
            report_path = os.path.join(os.path.dirname(save_dir) or ".", "finetune_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            result["report_path"] = report_path

        return result

    return (train_one,)


@app.cell
def _(CONFIG, itertools, pd, split_frames, train_one):
    df = pd.read_csv(CONFIG["data_path"])
    train_df, val_df, test_df = split_frames(df)

    if CONFIG["row_limit"]:
        train_df = train_df.head(CONFIG["row_limit"])
        val_df = val_df.head(CONFIG["row_limit"])
        test_df = test_df.head(CONFIG["row_limit"])

    combos = list(itertools.product(
        CONFIG["seeds"], CONFIG["lrs"], CONFIG["epochs_grid"], CONFIG["batch_sizes"]
    ))
    print(f"[search] {len(combos)} combos on "
          f"{len(train_df)} train / {len(val_df)} val / {len(test_df)} test rows")

    results = []
    for i, (seed, lr, epochs, batch_size) in enumerate(combos, start=1):
        print(f"[search] combo {i}/{len(combos)}: seed={seed} lr={lr} epochs={epochs} batch_size={batch_size}")
        results.append(
            train_one(train_df, val_df, test_df, seed, lr, epochs, batch_size, CONFIG["max_length"])
        )

    results_df = pd.DataFrame(results).sort_values("best_val_accuracy", ascending=False).reset_index(drop=True)
    return results_df, test_df, train_df, val_df


@app.cell
def _(mo, results_df):
    mo.vstack([
        mo.md("## Results, ranked by validation accuracy"),
        mo.ui.table(results_df.drop(columns=["history"])),
    ])
    return


@app.cell
def _(CONFIG, results_df, test_df, train_df, train_one, val_df):
    best = results_df.iloc[0]
    saved = train_one(
        train_df, val_df, test_df,
        seed=int(best["seed"]), lr=float(best["lr"]), epochs=int(best["epochs"]),
        batch_size=int(best["batch_size"]), max_length=CONFIG["max_length"], save_dir=CONFIG["out_dir"],
    )
    print(f"[search] winner seed={saved['seed']} lr={saved['lr']} "
          f"best_val_accuracy={saved['best_val_accuracy']} test_accuracy={saved['test_accuracy']}")
    print(f"[search] saved model+tokenizer to {CONFIG['out_dir']}, report at {saved['report_path']}")
    return (saved,)


@app.cell
def _(CONFIG, mo, saved):
    mo.md(
        f"""
        **Winner:** seed `{saved['seed']}`, lr `{saved['lr']}`,
        best_val_accuracy `{saved['best_val_accuracy']}`,
        test_accuracy `{saved['test_accuracy']}`

        Saved to `{CONFIG['out_dir']}` with `finetune_report.json` next to it —
        same schema `agents/finetune_finbert.py` writes for a normal run.

        This does **not** touch `outputs/finbert_finetuned` (the path the
        classifier agent auto-loads) — copy `{CONFIG['out_dir']}` there
        yourself once you're happy with it.
        """
    )
    return


if __name__ == "__main__":
    app.run()
