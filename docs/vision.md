# Vision

## Purpose
Arbiter is a research harness for an arXiv-oriented study. It prioritizes reproducibility and auditability over productization and is not a service. UX quality is valued but is not treated as a scientific contribution.

## Thesis
A single answer is one sample. Reasoning is modeled as a distribution over emergent outcome modes induced by an explicit configuration distribution `Q(c)`.

The estimand is the induced distribution `P_Q(y|x)`, where `x` is the question and `y := cluster(z)` is a discovered mode. The empirical estimate from finite trials is `P̂^Q(y|x)`. Changing `Q(c)` changes the estimand, so `Q(c)` must be explicit and serialized.

## Formal Framing
- Configuration tuple: `c = (m, d, p, π)` where `m` is model/provider, `d` is decoding parameters, `p` is prompt/persona framing, and `π` is protocol.
- Question: `x` (the question text shown to the model).
- Trial output object: `o`, a structured free-form output.
- Embedding: `z = f(o)` using a locked embedding instrument.
- Mode assignment: `y := cluster(z)` using online clustering for the control loop.
- Each trial must emit a parseable output object `o`; modes emerge from embedding and clustering rather than a predefined label set.

## Two Uncertainties
- Outcome-mode uncertainty: dispersion of `P̂^Q(·|x)` over discovered clusters (entropy, margin, disagreement).
- Estimation uncertainty (meta-uncertainty): uncertainty in `P̂^Q` due to finite trials, expressed through per-question confidence intervals and convergence diagnostics.

Convergence refers to estimate stability, not correctness.

## Online vs Offline Clustering
- Online clustering provides a stable, streaming control loop for convergence and early stopping.
- Offline clustering is an analysis instrument, not part of the control loop. It must be reported separately to avoid relabeling artifacts.

## Locked Instruments
Embedding and summarization are treated as locked instruments, not experimental knobs. Their model slugs, versions, and prompt hashes are recorded for provenance.

## UX Phases
The interactive experience is intentionally simple:
1) Wizard (configuration + question capture)
2) Execution (batched trials with convergence checks)
3) Receipt (summary + artifact location)

## Budget Commitment
The primary matched budget axis is the number of model calls. Tokens, cost, and latency are logged and reported as secondary metadata.

## Scope Boundaries
- No simulation of human populations, juries, or institutions.
- No claim that convergence implies correctness.
- No claim that debate is universally superior.
- Not a production system or commercial product.

## Implications
Concrete contracts (artifact bundle, `Q(c)` materialization, reporting discipline) are specified in `docs/spec.md`. Agent behavior and commit discipline are defined in `AGENTS.md`.
