"""Callable fine-tuner for the in-loop experiment.

`agents/finetune_finbert.py` keeps its training loop inside `main()`, so it can
only be run from the command line. The retune loop needs to call training as a
function, so this module wraps the same steps in `train_head()` and adds the one
thing the loop needs: `parent_model_dir`, which continues training from the
previous iteration's checkpoint instead of starting from pretrained FinBERT
every time.

Everything else is imported from the real fine-tuner, so the data split, the
per-class weighting and the scoring are identical to the offline script.

Exports
-------
train_head   train one round and save the checkpoint; returns a small report

See experiment_finetuning_in_loop/README.md for why this is a separate folder.
"""

import os
import random
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Run from the repo root either way: `uv run experiment_.../run_experiment.py`
# or as an imported module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.finetune_finbert import (
    BASE_MODEL,
    HF_TOKEN,
    LABELS,
    LABEL_TO_ID,
    HeadlineDataset,
    accuracy_on,
    class_weights,
    pick_device,
    split_frames,
)

# One epoch per round, not three. Every offline run in docs/finetune_runs.md
# peaked at epoch 1 and got worse after, so a loop that trains three epochs per
# retune would compound that overfitting on every pass.
EPOCHS_PER_ROUND = 1

# Lower than the 2e-5 the offline script uses. The loop trains repeatedly on the
# same rows, so each round should nudge the weights rather than move them far.
LEARNING_RATE = 5e-6


def train_head(*, data_path, out_dir, parent_model_dir=None,
               epochs=EPOCHS_PER_ROUND, learning_rate=LEARNING_RATE,
               batch_size=16, max_length=128, seed=42):
    """Train one round and save the result to `out_dir`.

    Args:
        data_path: processed_data.csv from the Processing agent.
        out_dir: where this round's checkpoint is written.
        parent_model_dir: checkpoint to continue from. None starts from
            pretrained FinBERT, which is what the first round does.
        epochs: passes over the training rows this round.

    Returns a dict with the validation accuracy and where the checkpoint landed.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Same split the offline fine-tuner and the retune loop use — Aurora owns
    # the one date-based boundary, so nothing here re-derives it.
    import pandas as pd
    df = pd.read_csv(data_path)
    train, val, _test = split_frames(df)

    device = pick_device()
    print(f"[loop-finetune] device: {device}  ({len(train):,} train / {len(val):,} val)")

    # 2. Continue from the previous round when there is one, otherwise start
    # from pretrained FinBERT with a fresh 3-class head.
    source = parent_model_dir or BASE_MODEL
    print(f"[loop-finetune] starting from: {source}")
    tokenizer = AutoTokenizer.from_pretrained(source, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        token=HF_TOKEN,
        num_labels=3,
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,  # allow swapping in our own head
    ).to(device)

    # 3. Build the data loaders.
    def make_loader(frame, shuffle):
        dataset = HeadlineDataset(frame["article_title"], frame["label"],
                                  tokenizer, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val, shuffle=False)

    # 4. Class-weighted loss — the same weighting the offline fine-tuner uses,
    # correcting for the neutral-heavy data and nothing else. Deliberately no
    # per-round reweighting: this experiment changes when training happens, not
    # what it optimises for, so the training objective is identical every round.
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 5. Train. Unlike the offline script there is no best-checkpoint choice to
    # make here: one epoch per round means one candidate, and the pipeline's own
    # `select_best` already restores the best iteration at the end.
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
        val_accuracy, val_class_accuracy = accuracy_on(model, val_loader, device)
        print(f"[loop-finetune] epoch {epoch}: train_loss {train_loss:.4f} "
              f"val_accuracy {val_accuracy:.4f}  per-class {val_class_accuracy}")

    # 6. Publish the round's checkpoint for the classifier to load.
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[loop-finetune] saved checkpoint to {out_dir}")

    return {
        "model_dir": out_dir,
        "train_loss": round(train_loss, 4),
        "val_accuracy": round(val_accuracy, 4),
        "val_class_accuracy": val_class_accuracy,
        "started_from": source,
    }
