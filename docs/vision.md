# Vision

## Purpose
Arbiter is a research harness for an arXiv-oriented study. It prioritizes reproducibility and auditability over productization and is not a service. UX quality is valued but not treated as a scientific contribution.

## Thesis
A single answer is one sample. Reasoning is modeled as an induced landscape under an explicit configuration distribution `Q(c)`.

The estimand is the induced distribution `P_Q(y|x)` where `x` is the question. The empirical estimate from finite trials is `P̂^Q(y|x)`. Changing `Q(c)` changes the estimand, so `Q(c)` must be explicit.

## Formal Framing
- Configuration tuple: `c = (m, d, p, π)` where `m` is model/provider, `d` is decoding parameters, `p` is prompt/persona framing, and `π` is protocol.
- Question: `x` (the question text shown to the model).
- Trial output object: `o`, a structured free-form output.
- Embedding: `z = f(o)`.
- Mode assignment: `y := cluster(z)`.
- Each trial must emit a parseable output object `o`; modes emerge from embedding and clustering rather than a predefined label set.

## Two Uncertainties
- Decision uncertainty: dispersion of `P̂^Q(·|x)` over discovered modes (entropy, margin, or disagreement).
- Estimation uncertainty (meta-uncertainty): uncertainty in `P̂^Q` due to finite trials, expressed through per-question confidence intervals and convergence diagnostics.

Convergence refers to estimate stability, not correctness.

## Heterogeneity Ladder
The ladder isolates heterogeneity sources to understand their contribution:
- Decoding/sampling heterogeneity.
- Prompt/persona heterogeneity.
- Cross-model heterogeneity.
- Interaction protocol heterogeneity.

## Budget Commitment
The primary matched budget axis is the number of model calls. Tokens, cost, and latency are logged and reported as secondary metadata.

## Scope Boundaries
- No simulation of human populations, juries, or institutions.
- No claim that convergence implies correctness.
- No claim that debate is universally superior.
- Not a production system or commercial product.

## Implications
Concrete contracts (artifact bundle, `Q(c)` materialization, reporting discipline) are specified in `docs/spec.md`. Agent behavior and commit discipline are defined in `AGENTS.md`.
