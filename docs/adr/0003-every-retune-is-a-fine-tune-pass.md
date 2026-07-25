# Every retune is a fine-tune pass

The previous design had two retune levers: `fine_tune` (train a new checkpoint) and `inference_only` (adjust threshold/boost_factor, or let an LLM rewrite `classify()` via `try_llm_classifier`/`CLASSIFIER_USE_OLLAMA`, subject to a runtime validation gauntlet). This branch retires `retune_mode` entirely — every retune fine-tunes the weak head(s); `classify()` is permanently deterministic template code.

**Rejected alternative:** keeping the LLM code-adaptation path as an occasional cheap lever alongside fine-tuning. Rejected because it was solving the same problem (react to evaluator feedback on weak classes) through a less reliable mechanism, and once fine-tuning became the primary lever the LLM path's complexity — prompt construction, static + runtime validation, template-swap logic — stopped earning its keep.
