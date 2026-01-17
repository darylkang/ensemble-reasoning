# Vision

## Purpose
arbiter is a research harness for an arXiv-oriented study. The goal is a reproducible, auditable experimental pipeline, not a product or service.

## Thesis
A single answer is one sample. Reasoning should be treated as an induced decision distribution under an explicit configuration distribution `Q(c)`.

## Formal Framing
- Configuration tuple: `c = (m, d, p, π)` for model, decoding, prompt/persona, and protocol.
- Label set: `Y` (a finite set of decisions) and decision `y ∈ Y`.
- `Q(c)` defines how configurations are sampled.
- The induced distribution is `P_Q(y|x)`, and its empirical estimate is `P̂_Q(y|x)`.

## Uncertainty Types
- Decision uncertainty: dispersion of `P̂_Q(·|x)` (entropy, margin, or similar).
- Estimation uncertainty: uncertainty due to finite trials (confidence intervals, convergence diagnostics).

## Ensemble Reasoning
Ensemble reasoning means repeated trials under `c ~ Q(c)` and aggregation of the induced decision distribution. Interaction and debate are treated as ablations, not assumptions.

## Heterogeneity Ladder
We separate four heterogeneity sources:
- H0: single-shot baseline.
- H1: decoding/sampling heterogeneity (temperature, seeds).
- H2: prompt/persona heterogeneity (role, framing).
- H3: cross-model heterogeneity (multiple providers/models).
- H4: interaction heterogeneity (debate or deliberation protocols).

v0 focuses on H0–H2; H3/H4 are deferred.

## Budget Commitment
The primary matched budget axis is number of model calls. Tokens and cost are logged as secondary metadata.

## Scope Boundaries
- No simulation of human populations or juries.
- No claim that convergence implies correctness.
- No claim that debate is always superior.
- Not a production system or commercial product.

## Implications
This vision is operationalized in `docs/spec.md`. Agent behavior and commit discipline are defined in `AGENTS.md`.
