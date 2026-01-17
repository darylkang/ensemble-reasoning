# Specification (v0)

This document defines the high-level contract for arbiter. It is implementation-facing but intentionally light on mechanics.

## Notation
- Input question: `x`.
- Configuration tuple: `c = (m, d, p, π)` (model, decoding, prompt/persona, protocol).
- Configuration distribution: `Q(c)`.
- Label set: `Y` and decision `y ∈ Y`.
- Induced distribution: `P_Q(y|x)`.
- Empirical estimate: `P̂_Q(y|x)`.

## Core Objects
- Question: an immutable prompt with a stable question_id, optional choices/labels, and optional gold label.
- Trial: a single model call under a resolved configuration, storing raw output, transcript, and parsed decision.
- Run: a collection of trials with a resolved configuration distribution, artifacts, and summary metrics.

## Decision Contract
Every trial must emit a normalized categorical decision `y` from the allowed label set for the question. ABSTAIN is supported but OFF by default in v0. When enabled, ABSTAIN is an element of `Y` and should be reported explicitly; coverage-based evaluation should treat ABSTAIN as non-coverage rather than correctness.

## Operational Distribution
Decision distributions are defined with respect to an explicit configuration distribution `Q(c)`. `Q(c)` must be resolved and serialized in run artifacts so that `P_Q(y|x)` is well-defined and auditable.

## Uncertainty Types
- Decision uncertainty: dispersion of the induced decision distribution `P̂_Q(·|x)`.
- Estimation uncertainty: confidence intervals on vote shares or estimator variability due to finite trials.

Confidence intervals are NOT probabilities of correctness; they are estimation CIs on the induced distribution `P_Q(y|x)`.

## Budget Accounting
The primary budget axis is the number of model calls. Token totals and cost estimates are logged on a best-effort basis later.

## Convergence and Early Stopping
- Sampling should be batched, with convergence checks at batch boundaries.
- Stopping criteria may include CI width thresholds and/or stability of `P̂_Q(y|x)` across batches.
- Convergence signals must be recorded when used.

## Provenance and Reproducibility
- Each run must serialize resolved configuration and capture environment and git metadata.
- The manifest must include a `semantic_config_hash`, computed by hashing the resolved config with run-specific fields removed (e.g., run_id, timestamps, output_dir). This is distinct from the raw config hash.

## v0 Scope
Only H0–H2 are supported in v0. H3/H4 are deferred.

## v0 Status
- v0 currently: the wizard materializes `Q(c)` and writes `manifest.json` and `config.resolved.json`.
- v0 near-term target: execute trials and produce trials/parsed/aggregates/metrics artifacts.

## CLI UX Target
- arbiter run: interactive wizard that selects the question set, heterogeneity rung, and trial budget; writes a resolved config and executes.
- arbiter analyze: optional, deferred command for aggregate analysis.

## Artifact Bundle Contract (Run Folder)
Each run writes a self-contained directory containing:
- manifest.json (run id, timestamp, git SHA, config hash, python version)
- config.resolved.json or config.resolved.yaml (fully resolved config, including `Q(c)` weights)
- questions.jsonl (exact questions used)
- trials.jsonl (one row per trial; includes transcript, raw output, metadata)
- parsed.jsonl (normalized decision `y`, parse validity, structured fields)
- aggregates.json (per-question distributions and uncertainty summaries)
- metrics.json (run-level summaries; calibration only if gold labels exist)
- logs/ (optional)

## Single-Question v0 Note
v0 supports single-question runs. The question schema should remain extensible to multi-question sets without breaking artifacts.

## Confidence Intervals (v0)
Use Wilson intervals on the top-choice proportion and, optionally, on each per-label proportion. Keep the estimator simple and transparent.
