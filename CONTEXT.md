# Stock-Move Prediction — Domain Glossary

Terms specific to the feedback-driven classifier redesign (`feature/classifier-owns-finetuning`). For the overall agent pipeline and file formats, see `AGENTS.md`.

## Language

**Directional head**:
One of the two independently fine-tuned FinBERT binary models (`up`-head, `down`-head) that predicts whether a headline signals its specific price movement versus `neutral`. Trained only on rows labeled with its own class or `neutral` — it never sees the opposite movement class during training, so its two classes are genuinely exhaustive of what it was taught.
_Avoid_: one-vs-rest head, binary head (both describe the earlier, rejected design where each head trained on every row — see [[0002-directional-heads-use-filtered-training]])

**Collapse**:
A class whose recall/accuracy has fallen to near zero — the model has effectively stopped predicting that label. Detected per-class against `MIN_CLASS_ACCURACY`.

**Gate**:
Manager's proceed-vs-retune decision. As of `feature/classifier-owns-finetuning`, the gate cannot pass while any class is collapsed, even if the score would otherwise clear the target or the iteration budget is exhausted — see [[0004-collapse-hard-blocks-the-gate]].

**Retune**:
One cycle of the feedback loop: Manager flags which directional head(s) are weak from Sabina's report, Nadi fine-tunes those head(s), Sabina re-evaluates. As of this branch every retune is a fine-tune pass — there is no other kind — see [[0003-every-retune-is-a-fine-tune-pass]].

**Baseline pass**:
Iteration 0, before any retune has happened. Classifies with pretrained `ProsusAI/finbert` (sentiment translated to up/down/neutral) at zero training cost, giving Manager a real number to compare the first fine-tuned iteration against.
