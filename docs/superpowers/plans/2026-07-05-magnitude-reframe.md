# Magnitude Reframe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the pipeline from 3-class direction (`up/down/neutral`) to 2-class magnitude (`big/small`), classified by a fine-tuned binary FinBERT whose hyperparameters the Manager retune loop drives, clearing the gate with genuine skill.

**Architecture:** The label definition changes at the source (Aurora), the shared contract (`agents/contracts.py`) flips `LABELS` and the prob columns, and every agent that hardcodes `up/down/neutral` is updated to `big/small`. The classifier fine-tunes FinBERT (binary head) inside the classify node; the retune loop tunes learning rate + epochs instead of the old inference threshold/boost knobs.

**Tech Stack:** Python 3.13, uv, PyTorch + HuggingFace transformers (FinBERT), LangGraph, pytest.

## Global Constraints

- Label values are lowercase `big` / `small`; dates `YYYY-MM-DD`; nulls as empty string; UTF-8, comma-separated (AGENTS.md Definition of Done).
- The contract in `docs/data_contracts.md` is law: columns, types, filenames fixed. This plan changes the contract, so `docs/data_contracts.md` + all `mock_data/` files + every downstream agent are updated together (golden rule 2).
- Cut threshold default is ±1% = `--threshold 0.01`; label = `big` if `abs(pct_change) > threshold*100` else `small`.
- Gate: `target_accuracy = 0.525`; pass = `accuracy >= target AND not collapsed`; collapse floor `min_class_accuracy = 0.05` spans 2 classes.
- No AI co-author trailer in commits (user global rule).
- Run all commands from repo root so `.dlt/secrets.toml` / `.env` resolve. Use `uv run`.
- Signal floor of record: trained NB reaches val 0.564 at ±1%. If fine-tuned FinBERT lands below majority 0.525, that is surfaced in `finetune_report.json`, never hidden.

---

## Phase 1 — Contract + labels (foundation)

### Task 1: Flip `LABELS` and prob columns in the shared contract

**Files:**
- Modify: `agents/contracts.py:17` (LABELS), `:25-29` (PREDICTION_COLUMNS), `:33-36` (EXPLANATION_SAMPLE_COLUMNS), `:108-118` (build_explanation_sample)
- Test: `tests/test_contracts_magnitude.py` (create)

**Interfaces:**
- Produces: `LABELS = ("big", "small")`; `PREDICTION_COLUMNS` with `prob_big, prob_small` replacing `prob_up, prob_down, prob_neutral`; `EXPLANATION_SAMPLE_COLUMNS` with `prob_big, prob_small`.
- Consumed by: Nadi, Sabina, Freddi, Manager (all later tasks).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contracts_magnitude.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts_magnitude.py -v`
Expected: FAIL (`LABELS == ('up','down','neutral')`).

- [ ] **Step 3: Edit `agents/contracts.py`**

```python
# line 17
LABELS = ("big", "small")

# lines 25-29
PREDICTION_COLUMNS = [
    "article_id", "date", "ticker", "article_title", "price_t", "price_t1",
    "pct_change", "label", "predicted_label", "confidence",
    "prob_big", "prob_small", "split",
]

# lines 33-36
EXPLANATION_SAMPLE_COLUMNS = [
    "article_id", "article_title", "predicted_label", "actual_label",
    "confidence", "prob_big", "prob_small",
]
```

In `build_explanation_sample` (lines 111-114) replace the hardcoded prob columns:

```python
    sample = (predictions[[
        "article_id", "article_title", "predicted_label", "label",
        "confidence", "prob_big", "prob_small",
    ]].rename(columns={"label": "actual_label"}))
```

Note: `validate_prediction_rows` (lines 86) builds `prob_{label_name}` by looping `LABELS`, so it needs no edit — it follows the new labels automatically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts_magnitude.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add agents/contracts.py tests/test_contracts_magnitude.py
git commit -m "Flip contract to 2-class magnitude (big/small) + binary prob columns"
```

### Task 2: Aurora emits binary magnitude labels

**Files:**
- Modify: `agents/aurora_processing.py:66-71` (`_assign_label`), `:220-221` (help text)
- Test: `tests/test_aurora_magnitude.py` (create)

**Interfaces:**
- Consumes: `threshold` (decimal, e.g. 0.01).
- Produces: `_assign_label(pct_change, threshold) -> "big" | "small"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aurora_magnitude.py
from agents.aurora_processing import _assign_label

def test_big_when_move_exceeds_band():
    assert _assign_label(3.38, 0.01) == "big"    # +3.38% > 1%
    assert _assign_label(-2.0, 0.01) == "big"    # -2% magnitude > 1%

def test_small_when_move_within_band():
    assert _assign_label(0.6163, 0.01) == "small"  # 0.62% <= 1%
    assert _assign_label(-0.5, 0.01) == "small"

def test_band_scales_with_threshold():
    assert _assign_label(1.5, 0.02) == "small"   # 1.5% <= 2%
    assert _assign_label(2.5, 0.02) == "big"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aurora_magnitude.py -v`
Expected: FAIL (returns `up`/`down`/`neutral`).

- [ ] **Step 3: Edit `_assign_label` (lines 66-71)**

```python
def _assign_label(pct_change, threshold):
    # Magnitude reframe: predict whether tomorrow is a BIG move (|Δ| beyond the
    # band) or a SMALL/quiet move. threshold is a decimal; pct_change is percent.
    return "big" if abs(pct_change) > threshold * 100 else "small"
```

Update the CLI help (lines 220-221) to say the band separates big vs small moves:

```python
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Big/small move band in decimal form (default: 0.01 = ±1%%)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aurora_magnitude.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Regenerate `data/processed_data.csv` at ±1%**

Run: `uv run agents/aurora_processing.py --threshold 0.01`
Expected: prints `Labels: {'small': ..., 'big': ...}` (two keys only). Confirm with:
`uv run python -c "import csv,collections; r=list(csv.DictReader(open('data/processed_data.csv'))); print(collections.Counter((x['split'],x['label']) for x in r))"`
Expected: only `big`/`small` labels; val ≈ 52% small / 48% big.

- [ ] **Step 6: Commit**

```bash
git add agents/aurora_processing.py tests/test_aurora_magnitude.py
git commit -m "Aurora: label big/small magnitude instead of up/down/neutral"
```

### Task 3: Regenerate every `mock_data/` file to the 2-class shape

**Files:**
- Modify: `mock_data/processed_data.csv`, `predictions_test.csv`, `sample_for_explanation.csv`, `evaluation_report.json`, `explanations.csv`, `final_results.csv`, `final_report.json`, `decision.json`, `retune_request.json`
- Test: `tests/test_mock_data_contract.py` (create)

**Interfaces:**
- Produces: mock fixtures whose `label`/`predicted_label` ∈ {big, small} and whose prob columns are `prob_big, prob_small`, matching Task 1's contract lists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mock_data_contract.py
import csv
from agents.contracts import PREDICTION_COLUMNS, EXPLANATION_SAMPLE_COLUMNS, LABELS

def _cols(path):
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))

def test_predictions_mock_matches_contract():
    assert _cols("mock_data/predictions_test.csv") == PREDICTION_COLUMNS

def test_sample_mock_matches_contract():
    assert _cols("mock_data/sample_for_explanation.csv") == EXPLANATION_SAMPLE_COLUMNS

def test_processed_mock_labels_binary():
    with open("mock_data/processed_data.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["label"] for r in rows} <= set(LABELS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mock_data_contract.py -v`
Expected: FAIL (mock columns still `prob_up/...`).

- [ ] **Step 3: Rewrite the mock CSV/JSON files**

Edit `mock_data/processed_data.csv` — keep rows, change each `label` to `big` (|pct_change|>1) or `small` (else). For the shown row (`pct_change 3.38`) → `big`.

Edit `mock_data/predictions_test.csv` — header to `PREDICTION_COLUMNS`; replace the three prob values with two. For a `big`-predicted row use e.g. `confidence 0.62, prob_big 0.62, prob_small 0.38`; keep `confidence == max(prob_big, prob_small)` and `prob_big + prob_small == 1` (contract check in `validate_prediction_rows`). Set `label`/`predicted_label` to `big`/`small`.

Edit `mock_data/sample_for_explanation.csv` — header to `EXPLANATION_SAMPLE_COLUMNS`; prob columns `prob_big, prob_small`; labels binary.

Edit `mock_data/evaluation_report.json` — `class_accuracy` keys become `big`, `small` (drop `up/down/neutral`). Example:

```json
{
  "accuracy": 0.56,
  "below_threshold": false,
  "class_accuracy": {"big": 0.72, "small": 0.39},
  "misclassified_count": 2,
  "misclassified_ids": ["FNSPID_00002", "FNSPID_00005"],
  "proposal": {
    "recommended_action": "proceed",
    "reason": "accuracy 0.56 clears the 0.53 target",
    "focus_labels": ["small"],
    "suggested_params": {},
    "code_notes": ""
  }
}
```

Edit `mock_data/explanations.csv`, `final_results.csv` — set `predicted_label`/`actual_label`/`label` to binary values (these files' columns don't include prob_*, so only label values change).

Edit `mock_data/final_report.json` — `class_accuracy` keys to `big`/`small`.

Edit `mock_data/decision.json` — any `focus_labels` to a binary label (e.g. `["small"]`); `suggested_params` to lr/epochs form (e.g. `{"lr": 3e-5, "epochs": 3}`) to match Task 8.

Edit `mock_data/retune_request.json` — `focus_labels` binary; `suggested_params` `{"lr": 3e-5, "epochs": 3}`.

Update `mock_data/README.md` prose that names `up/down/neutral`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mock_data_contract.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add mock_data tests/test_mock_data_contract.py
git commit -m "Regenerate mock_data fixtures for 2-class magnitude contract"
```

### Task 4: Update the contract + architecture docs

**Files:**
- Modify: `docs/data_contracts.md` (Handoff 1 label values, Handoff 2 prob columns, Handoff 4 sample columns), `docs/architecture.md` (mission line), `CLAUDE.md` project file (label-values line if present)

**Interfaces:** none (docs only). Must match Task 1's column lists exactly.

- [ ] **Step 1: Edit `docs/data_contracts.md`** — every place listing `up`/`down`/`neutral` becomes `big`/`small`; every `prob_up`/`prob_down`/`prob_neutral` becomes `prob_big`/`prob_small`; add a one-line note: "Label reframed 2026-07-05 from direction to magnitude — see docs/superpowers/specs/2026-07-05-magnitude-reframe-design.md."

- [ ] **Step 2: Edit `docs/architecture.md`** — mission/flow lines: "up/down/neutral" → "big/small (next-day move magnitude)".

- [ ] **Step 3: Grep for stragglers**

Run: `grep -rn "up/down/neutral\|prob_up\|prob_neutral\|prob_down" docs/ CLAUDE.md`
Expected: no matches left except historical spec/log entries (those describe the old state and stay).

- [ ] **Step 4: Commit**

```bash
git add docs/data_contracts.md docs/architecture.md CLAUDE.md
git commit -m "Docs: reframe contract + architecture to big/small magnitude"
```

---

## Phase 2 — Binary fine-tune

### Task 5: Make `finetune_finbert.py` binary + importable

**Files:**
- Modify: `agents/finetune_finbert.py:33-35` (LABELS), `:149-157` (class_weights), `:190-200` (num_labels), `:160-272` (extract a `finetune(...)` function), keep `main()`
- Test: `tests/test_finetune_binary.py` (create)

**Interfaces:**
- Produces: `finetune(data_path, out_dir, *, epochs=3, lr=2e-5, batch_size=16, max_length=128, seed=42, limit=None) -> tuple[str, float]` returning `(out_dir, best_val_accuracy)`. Also writes `finetune_report.json`.
- `LABELS = ["big", "small"]`, `LABEL_TO_ID = {"big": 0, "small": 1}`, `num_labels=2`.
- Consumed by: Nadi (Task 7).

- [ ] **Step 1: Write the failing test** (smoke, CPU-friendly via `limit`)

```python
# tests/test_finetune_binary.py
import agents.finetune_finbert as ft

def test_labels_binary():
    assert ft.LABELS == ["big", "small"]
    assert ft.LABEL_TO_ID == {"big": 0, "small": 1}

def test_finetune_is_importable_and_returns_signature():
    # function exists with the documented signature (no training run here)
    import inspect
    sig = inspect.signature(ft.finetune)
    assert "data_path" in sig.parameters and "out_dir" in sig.parameters
    assert {"epochs", "lr"} <= set(sig.parameters)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_finetune_binary.py -v`
Expected: FAIL (`LABELS == ["up","down","neutral"]`, no `finetune`).

- [ ] **Step 3: Edit `agents/finetune_finbert.py`**

Labels (lines 33-35):

```python
BASE_MODEL = "ProsusAI/finbert"
LABELS = ["big", "small"]
LABEL_TO_ID = {"big": 0, "small": 1}
```

`class_weights` denominator (line 156): `weights.append(len(train) / (2 * count))`.

Wrap the training body (current `main` lines 160-272) into a function; `main` parses args then calls it:

```python
def finetune(data_path, out_dir, *, epochs=3, batch_size=16, lr=2e-5,
             max_length=128, seed=42, limit=None):
    """Fine-tune FinBERT (binary big/small head) and return (out_dir, best_val_acc).
    Same logic as the old main() body; parameters are now explicit so the retune
    loop can call it in-process."""
    random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(data_path)
    train, val, test = split_frames(df)
    if limit:
        train = train.head(limit); val = val.head(limit); test = test.head(limit)
    print(f"[finetune] {len(train)} train / {len(val)} val / {len(test)} test rows")

    device = pick_device()
    print(f"[finetune] device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, token=HF_TOKEN, num_labels=2,
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID, ignore_mismatched_sizes=True,
    ).to(device)

    def make_loader(frame, shuffle):
        dataset = HeadlineDataset(frame["article_title"], frame["label"], tokenizer, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train, True)
    val_loader = make_loader(val, False)
    test_loader = make_loader(test, False)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs(out_dir, exist_ok=True)
    best_val_accuracy = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward(); optimizer.step(); running_loss += loss.item()
        train_loss = running_loss / max(len(train_loader), 1)
        val_accuracy, _ = accuracy_on(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_accuracy": round(val_accuracy, 4)})
        print(f"[finetune] epoch {epoch}: train_loss {train_loss:.4f} val_accuracy {val_accuracy:.4f}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    test_accuracy, test_class_accuracy = accuracy_on(best_model, test_loader, device)
    report = {
        "base_model": BASE_MODEL, "model_dir": out_dir,
        "test_accuracy": round(test_accuracy, 4),
        "test_class_accuracy": test_class_accuracy,
        "best_val_accuracy": round(best_val_accuracy, 4),
        "epochs": history, "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "params": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                   "max_length": max_length, "seed": seed},
    }
    report_path = os.path.join(os.path.dirname(out_dir) or ".", "finetune_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[finetune] report written to {report_path}")
    return out_dir, round(best_val_accuracy, 4)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on move magnitude.")
    parser.add_argument("--data", default="data/processed_data.csv")
    parser.add_argument("--out-dir", default="outputs/finbert_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    finetune(args.data, args.out_dir, epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, max_length=args.max_length, seed=args.seed, limit=args.limit)
```

Update the module docstring (lines 1-19): 3-class → binary big/small; note `finetune()` is now importable by the loop.

- [ ] **Step 4: Run unit tests + a real smoke train**

Run: `uv run pytest tests/test_finetune_binary.py -v`
Expected: PASS (2 tests).
Run: `uv run agents/finetune_finbert.py --data data/processed_data.csv --limit 50 --epochs 1`
Expected: prints device + one epoch + writes `outputs/finbert_finetuned/` and `outputs/finetune_report.json` with `test_class_accuracy` keyed `big`/`small`.

- [ ] **Step 5: Commit**

```bash
git add agents/finetune_finbert.py tests/test_finetune_binary.py
git commit -m "finetune_finbert: binary big/small head + importable finetune()"
```

---

## Phase 3 — Classifier integration

### Task 6: Classifier template emits `prob_big`/`prob_small`

**Files:**
- Modify: `agents/nadi_classifier.py:51-156` (CLASSIFIER_TEMPLATE), specifically the `classify()` return dict and the pretrained fallback
- Test: `tests/test_classifier.py` (update existing prob-column assertions)

**Interfaces:**
- Produces: generated `classifier.py` whose `classify()` returns keys `predicted_label, confidence, prob_big, prob_small`, and whose `main()` writes `PREDICTION_COLUMNS` (from contract).
- Depends on: a fine-tuned 2-class `MODEL_DIR` (Task 5). The pretrained-FinBERT sentiment path no longer maps to magnitude, so when no fine-tuned model exists the template raises a clear error rather than emitting meaningless labels.

- [ ] **Step 1: Update the existing classifier test**

In `tests/test_classifier.py`, change any assertions referencing `prob_up/prob_down/prob_neutral` or `up/down/neutral` predicted labels to `prob_big/prob_small` and `big/small`. Add:

```python
def test_generated_classifier_outputs_binary_prob_columns(tmp_path):
    from agents.contracts import PREDICTION_COLUMNS
    assert "prob_big" in PREDICTION_COLUMNS and "prob_small" in PREDICTION_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAIL on the renamed prob columns.

- [ ] **Step 3: Edit `CLASSIFIER_TEMPLATE`**

Replace the sentiment fallback + `classify()` body (template lines ~64-115) with a binary version. The fine-tuned model's `id2label` is `{0:"big",1:"small"}`, so map straight from it; drop `SENTIMENT_TO_LABEL`, `FOCUS_LABELS`, `BOOST_FACTOR` (direction-era knobs unused by magnitude):

```python
MODEL = "ProsusAI/finbert"
MODEL_DIR = {model_dir}
MAX_LENGTH = {max_length}

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if not (MODEL_DIR and os.path.isdir(MODEL_DIR)):
    raise SystemExit(
        "magnitude classifier requires a fine-tuned 2-class model_dir; "
        "run the fine-tune step first (agents/finetune_finbert.py)"
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
ID2OURS = {{i: model.config.id2label[i].lower() for i in model.config.id2label}}
model.eval()

def classify(title: str) -> dict:
    inputs = tokenizer(title, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    by_label = {{ID2OURS[i]: round(float(p), 4) for i, p in enumerate(probs)}}
    top_label = max(by_label, key=by_label.get)
    return {{
        "predicted_label": top_label,
        "confidence": by_label[top_label],
        "prob_big": by_label.get("big", 0.0),
        "prob_small": by_label.get("small", 0.0),
    }}
```

In `main()` inside the template, the output columns line becomes:

```python
    out_cols = [c for c in rows[0].keys() if c != "split"] + [
        "predicted_label", "confidence", "prob_big", "prob_small", "split"
    ]
```

In `generate_code` (lines 294-326) drop `focus_labels`/`boost_factor` formatting (no longer in the template) and keep `model_dir` + `max_length`. The `.format(...)` call keeps only `model_dir=repr(model_dir), max_length=max_length`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/nadi_classifier.py tests/test_classifier.py
git commit -m "Nadi: binary magnitude classifier template (prob_big/prob_small)"
```

### Task 7: Fine-tune inside the classify node

**Files:**
- Modify: `agents/nadi_classifier.py` (`generate_code`/`run_classifier` or a new pre-step), `agents/pipeline_graph.py:168-176` (classify node passes hyperparams)
- Test: `tests/test_classifier.py` (add: retune params drive a fresh fine-tune)

**Interfaces:**
- Consumes: `state["retune_request"]["suggested_params"] = {"lr": float, "epochs": int}` (Task 8 shape); `state["processed_data_path"]`.
- Produces: a fine-tuned model at `outputs/finbert_finetuned` reflecting the current hyperparameters, then predictions over it. Adds `finetune_params` to the returned metadata.

- [ ] **Step 1: Write the failing test** (inject a fake finetune to avoid GPU in tests)

```python
def test_classify_node_finetunes_with_retune_hyperparams(monkeypatch, tmp_path):
    import agents.nadi_classifier as nc
    calls = {}
    def fake_finetune(data_path, out_dir, *, epochs, lr, **kw):
        calls["epochs"] = epochs; calls["lr"] = lr
        return out_dir, 0.55
    monkeypatch.setattr(nc, "finetune", fake_finetune, raising=False)
    params = {"lr": 3e-5, "epochs": 2}
    lr, epochs = nc._finetune_hyperparams({"suggested_params": params})
    assert (lr, epochs) == (3e-5, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classifier.py::test_classify_node_finetunes_with_retune_hyperparams -v`
Expected: FAIL (`_finetune_hyperparams` missing).

- [ ] **Step 3: Implement fine-tune-in-loop in `agents/nadi_classifier.py`**

Add at module top (dual import, like the others):

```python
try:
    from agents.finetune_finbert import finetune
except ModuleNotFoundError:
    from finetune_finbert import finetune

DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 3
FINETUNE_DIR = os.path.join(OUTPUT_DIR, "finbert_finetuned")


def _finetune_hyperparams(retune_req) -> tuple[float, int]:
    """Read lr/epochs from a retune request, falling back to first-pass defaults."""
    params = (retune_req or {}).get("suggested_params", {}) or {}
    return params.get("lr", DEFAULT_LR), int(params.get("epochs", DEFAULT_EPOCHS))
```

In `generate_code`, before writing the classifier, (re)fine-tune when running the real pipeline (a `state["skip_finetune"]` flag lets tests/mock runs bypass training):

```python
    retune_req = state.get("retune_request")
    if not state.get("skip_finetune"):
        lr, epochs = _finetune_hyperparams(retune_req)
        data_path = state.get("processed_data_path")
        finetune(data_path, FINETUNE_DIR, lr=lr, epochs=epochs)
        model_dir = FINETUNE_DIR
    else:
        model_dir = state.get("model_dir")
```

Keep the rest of `generate_code` (writing `classifier.py` with `model_dir`, history snapshot). `max_length` stays 128.

In `agents/pipeline_graph.py` `classify` node (lines 168-176), pass `processed_data_path` through (already present) and let Nadi own fine-tuning; no `model_dir` flag needed since the loop always fine-tunes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/nadi_classifier.py agents/pipeline_graph.py tests/test_classifier.py
git commit -m "Nadi: fine-tune FinBERT inside classify node, hyperparams from retune"
```

---

## Phase 4 — Evaluator + gate

### Task 8: Manager retunes lr/epochs; gate target 0.525

**Files:**
- Modify: `agents/jack_manager.py:102-118` (`_RETUNE_SCHEDULE`), `:141-174` (`_next_params`), `:506` (`target_accuracy=0.525`), state comment `:35`
- Modify: `agents/sabina_evaluator.py:49` (`TARGET_ACCURACY = 0.525`), `:199-211` (`_suggest_retune_params` → lr/epochs)
- Test: `tests/test_manager_retune_hyperparams.py` (create)

**Interfaces:**
- Produces: retune `suggested_params` of shape `{"lr": float, "epochs": int}`; escalation schedule walks lr up / epochs up.
- Consumed by: Nadi `_finetune_hyperparams` (Task 7).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manager_retune_hyperparams.py
from agents.jack_manager import _next_params, _RETUNE_SCHEDULE

def test_schedule_tunes_lr_and_epochs():
    for entry in _RETUNE_SCHEDULE:
        assert set(entry) == {"lr", "epochs"}

def test_next_params_escalates_within_schedule():
    first = _next_params(tried=[])
    assert first == _RETUNE_SCHEDULE[0]
    second = _next_params(tried=[_RETUNE_SCHEDULE[0]])
    assert second == _RETUNE_SCHEDULE[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manager_retune_hyperparams.py -v`
Expected: FAIL (schedule entries are `threshold`/`boost_factor`).

- [ ] **Step 3: Edit the schedule + params in `agents/jack_manager.py`**

```python
# lines 102-118 — hyperparameter escalation for the fine-tune loop.
# Levers: higher lr adapts faster (risking instability); more epochs fit harder
# (risking overfit). Each retune tries a genuinely different (lr, epochs) point.
_RETUNE_SCHEDULE = [
    {"lr": 3e-5, "epochs": 3},
    {"lr": 3e-5, "epochs": 4},
    {"lr": 5e-5, "epochs": 3},
    {"lr": 5e-5, "epochs": 4},
    {"lr": 1e-4, "epochs": 3},
]
```

Simplify `_next_params` to the schedule walk (drop the threshold/boost revert-and-perturb block, which tuned the old knobs). Keep the `_same`/schedule-exhaustion behaviour:

```python
def _next_params(tried: list, history: list = ()) -> dict:
    """Next (lr, epochs) point: first schedule entry not already tried, or the
    last entry once the schedule is exhausted (the iteration cap still bounds
    the loop)."""
    for params in _RETUNE_SCHEDULE:
        if not any(_same(params, t) for t in tried):
            return params
    return _RETUNE_SCHEDULE[-1]
```

Delete the now-unused `_REGRESSION_DELTA`, `_THRESHOLD_STEP`, `_BOOST_STEP`, `_BOOST_MAX` constants (lines 120-129) — they only fed the removed block. `_same` stays.

Set the gate default (line 506): `target_accuracy=0.525`. Update the state comment (line 35) `0.516` → `0.525`.

- [ ] **Step 4: Edit `agents/sabina_evaluator.py`**

Line 49: `TARGET_ACCURACY = 0.525`.

`_suggest_retune_params` (lines 199-211): return the first schedule point instead of reading the old `THRESHOLD` from code text:

```python
def _suggest_retune_params(code_text: str) -> dict:
    """First fine-tune retune point. The Manager escalates from here."""
    return {"lr": 3e-5, "epochs": 3}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_manager_retune_hyperparams.py tests/test_manager*.py -v`
Expected: PASS. Fix any existing manager test asserting `threshold`/`boost_factor` in suggested_params.

- [ ] **Step 6: Commit**

```bash
git add agents/jack_manager.py agents/sabina_evaluator.py tests/test_manager_retune_hyperparams.py
git commit -m "Manager+Sabina: retune lr/epochs, gate target 0.525"
```

Note: Sabina's `compute_metrics` and `_weakest_labels` iterate `LABELS`/`class_accuracy` dynamically, so they follow the new 2 classes with no edit. `_is_collapsed`/`report_score` in the Manager likewise operate over whatever classes appear.

---

## Phase 5 — Explanation

### Task 9: Freddi explains big/small, reads binary probs

**Files:**
- Modify: `agents/freddi_explanation.py:92-134` (system prompt, few-shot, `move_phrase`), `:120-133` (`prob_*` in user turn), `:170-179` (offline fallback), `:11-13` (docstring)
- Test: `tests/test_freddi_magnitude.py` (create)

**Interfaces:**
- Consumes: sample rows with `prob_big, prob_small`, `predicted_label ∈ {big, small}`.
- Produces: explanations phrased as volatility (big move vs quiet day).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_freddi_magnitude.py
from agents.freddi_explanation import _fallback_explanation

def test_fallback_phrases_big_move():
    row = {"article_title": "Company X warns on guidance", "predicted_label": "big",
           "prob_big": "0.7", "prob_small": "0.3"}
    text = _fallback_explanation(row)
    assert "big" in text.lower() or "large" in text.lower() or "volatil" in text.lower()

def test_fallback_phrases_small_move():
    row = {"article_title": "Routine filing published", "predicted_label": "small",
           "prob_big": "0.2", "prob_small": "0.8"}
    text = _fallback_explanation(row)
    assert "small" in text.lower() or "quiet" in text.lower() or "little" in text.lower()
```

(If the fallback function has a different name, match it — see `agents/freddi_explanation.py` line ~168.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_freddi_magnitude.py -v`
Expected: FAIL (fallback talks about direction).

- [ ] **Step 3: Edit `agents/freddi_explanation.py`**

`move_phrase` map (lines 123-127) and the fallback `direction` map (lines 171-175):

```python
    move_phrase = {
        "big": "a large next-day price move (in either direction)",
        "small": "little price movement — a quiet next day",
    }.get((row.get("predicted_label") or "").strip(), "the predicted next-day move size")
```

User-turn prob line (lines 132-133):

```python
        f"(probabilities — big move: {row.get('prob_big', '')}, "
        f"small move: {row.get('prob_small', '')})."
```

System prompt (line 96): "predicted a stock's next-day move (up / down / neutral)" → "predicted whether a stock will make a big or small next-day price move". Replace the few-shot example (lines 114-115) with a magnitude one:

```python
# few-shot
'Headline: "Company issues surprise profit warning". Predicted: a large next-day price move.'
'{"reasoning": "A surprise profit warning is material news that typically triggers a sharp repricing, so a large move is likely regardless of direction.", "explanation": "The surprise profit warning is material enough to drive a large price move the next day."}'
```

Offline fallback body (lines 178-179):

```python
        f'The headline "{title}" reads as {"material" if pred == "big" else "routine"} news, '
        f"which is consistent with {direction}."
```

Update the docstring input-columns line (lines 11-13) to `prob_big, prob_small`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_freddi_magnitude.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add agents/freddi_explanation.py tests/test_freddi_magnitude.py
git commit -m "Freddi: explain big/small magnitude, read binary probs"
```

---

## Phase 6 — Wiring, tests, end-to-end

### Task 10: Update pipeline/integration tests to 2-class

**Files:**
- Modify: `tests/test_pipeline_graph.py`, `tests/test_finetune_split.py`, and any remaining test with `up/down/neutral`
- Test: the suite itself

**Interfaces:** the fake classifier used in `test_pipeline_graph.py` must emit `big/small` + `prob_big/prob_small`.

- [ ] **Step 1: Grep the test tree for direction labels**

Run: `grep -rn "up\|down\|neutral\|prob_up\|prob_down\|prob_neutral" tests/`
Expected: a list of spots to update.

- [ ] **Step 2: Update fakes/fixtures** — fake classifier writes `predicted_label ∈ {big,small}`, `prob_big`, `prob_small`; fake evaluation reports use `class_accuracy={"big":..,"small":..}`. The pipeline test should set `skip_finetune=True` (Task 7) and inject a `model_dir` fake so no training runs in CI.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all green. Fix stragglers until it passes.

- [ ] **Step 4: Commit**

```bash
git add tests
git commit -m "Tests: migrate pipeline + finetune-split tests to 2-class magnitude"
```

### Task 11: Real end-to-end run + record result

**Files:**
- Modify: `docs/optimization_log.md` (append the reframe run's numbers)

**Interfaces:** none — this is the acceptance run.

- [ ] **Step 1: Run the pipeline for real (bounded)**

Run: `uv run main.py --threshold 0.01 --target-accuracy 0.525 --max-iterations 2`
Expected: completes; `outputs/decision.json` shows `accuracy_history` with the val numbers and a `proceed`.

- [ ] **Step 2: Read the honest result**

Run: `uv run python -c "import json; d=json.load(open('outputs/decision.json')); print('history', d['accuracy_history'], 'decision', d['decision'], d['final_action']); e=json.load(open('outputs/evaluation_report.json')); print('test acc', e['accuracy'], 'class', e['class_accuracy'])"`

- [ ] **Step 3: Judge against the bar**

- If val ≥ 0.525 with both class recalls > 0.05 → gate cleared with genuine skill. Record it.
- If below → record honestly (per Global Constraints); compare to NB floor 0.564; the retune loop's lr/epochs points are the tuning surface. Do NOT widen the band or abstain to clear — that's the gaming this reframe exists to avoid.

- [ ] **Step 4: Append to `docs/optimization_log.md`** a "Magnitude reframe — end-to-end result" section: val accuracy history, test accuracy, per-class recall, whether the gate cleared, and how it compares to the NB floor.

- [ ] **Step 5: Commit**

```bash
git add docs/optimization_log.md outputs/decision.json outputs/evaluation_report.json outputs/final_report.json
git commit -m "Record magnitude-reframe end-to-end result"
```

---

## Self-review notes

- **Spec coverage:** label reframe (T1-4), binary fine-tune (T5), classifier prob columns + fine-tune-in-loop (T6-7), gate 0.525 + lr/epochs retune (T8), Sabina per-class (T8 note), Freddi wording (T9), tests + e2e (T10-11), NB floor honesty (T11 step 3). All spec sections mapped.
- **Type consistency:** `finetune(...) -> (out_dir, best_val_accuracy)` defined in T5, consumed in T7; `suggested_params={"lr","epochs"}` produced in T8, read by `_finetune_hyperparams` in T7; `LABELS=("big","small")` in T1 flows to Sabina/Manager unchanged (dynamic loops).
- **Contract-rename fan-out:** `prob_up/down/neutral → prob_big/prob_small` explicitly edited in contracts (T1), mock_data (T3), Nadi template (T6), Freddi (T9); auto-followed in `validate_prediction_rows` + Sabina `compute_metrics` (loop over LABELS).
