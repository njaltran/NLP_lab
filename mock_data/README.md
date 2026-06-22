# Mock data

Fake-but-valid sample for **every handoff** in [`../docs/data_contracts.md`](../docs/data_contracts.md). Lets each agent be built and tested against realistic inputs before the upstream agent exists. **Not real predictions** — hand-authored.

One consistent story runs through the files: 12 test articles; FinBERT needs **two cycles** to clear the 0.60 accuracy gate. The same `article_id`s flow stage to stage so you can trace one row end to end.

- **Cycle 1** fails at accuracy **0.54** → Sabina recommends retune → Jack approves → `retune_request.json` (iteration 1) goes back to Nadi.
- **Cycle 2** converges at accuracy **0.67** → this is the run captured in `predictions_test.csv` / `evaluation_report.json` / `decision.json` (proceed) and the `final_*` outputs.

`neutral` is the weakest class (0.33) — the realistic failure mode. Misclassified in the converged run: `FNSPID_00006`, `FNSPID_00010`, `FNSPID_00011`, `FNSPID_00012`.

| File | Handoff | Producer → consumer |
|---|---|---|
| `processed_data.csv` | 1 | Aurora → Nadi |
| `classifier.py` | 2 | Nadi → Sabina (generated code; Sabina reads, does not run) |
| `predictions_test.csv` | 2 | Nadi → Sabina (cycle 2, converged) |
| `evaluation_report.json` | 3 | Sabina → Jack (metrics + `proposal`, cycle 2) |
| `decision.json` | 3b | Jack's record — accept/override, here `proceed` |
| `retune_request.json` | 3b | Jack → Nadi — the cycle-1 retune branch (iteration 1) |
| `sample_for_explanation.csv` | 4 | Jack → Freddi |
| `explanations.csv` | 5 | Freddi → Jack |
| `final_results.csv` + `final_report.json` | 6 | Jack → final output |

Notes:
- `predictions_test.csv` carries `prob_up/prob_down/prob_neutral`; `confidence` = the max of the three, and `predicted_label` = the argmax. `sample_for_explanation.csv` forwards the same `prob_*` so Freddi can see how close each call was.
- `decision.json` (proceed) and `retune_request.json` (retune) show the **two branches** of Handoff 3b. In a single converged run only one is written; both are provided here as format examples.
- `classifier.py` is illustrative generated code — it produces output of the right shape but real FinBERT probabilities will differ from the hand-tuned values in `predictions_test.csv`.
