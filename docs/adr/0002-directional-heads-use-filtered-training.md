# Directional heads train on filtered data, not one-vs-rest

`docs/superpowers/specs/2026-07-12-two-binary-heads-design.md` (written on `feature/feedback-driven-finbert`) chose one-vs-rest: each head trains on every row, with the opposite class collapsed into "not-this-head" alongside neutral. It explicitly accepted a fuzzy negative class as the cost of guaranteeing no input is ever out-of-distribution at inference.

This branch makes the opposite call: the `up`-head trains only on `up`/`neutral` rows, the `down`-head only on `down`/`neutral` rows — each head never sees the opposite movement class at all. The `combine_binary_probs` normalized-product formula is unaffected (its invariants hold for any `p_up, p_down ∈ [0,1]` regardless of what the heads were trained on), so this is purely a training-data decision, not an inference-math one.

**Considered and rejected:** one-vs-rest (the original spec's design) — rejected because the current baseline's weak `up`/`down` recall (0.28 / 0.39) suggested the fuzzy negative class was diluting each head's discrimination, and the goal here is sharper per-class signal even at the cost of accepting some out-of-distribution risk (a head may misfire confidently on the movement class it's never seen). If that risk proves worse in practice than the fuzzy-negative cost it replaced, one-vs-rest is the documented fallback.
