# Specification (v0)

This document defines the high-level contract for arbiter. It is implementation-facing but intentionally light on mechanics.

## Core Objects
- Question: an immutable prompt with a stable question_id, optional choices/labels, and optional gold label.
- Trial: a single model call under a resolved configuration, storing raw output, transcript, and parsed decision.
- Run: a collection of trials with a resolved configuration distribution, artifacts, and summary metrics.

## Decision Contract
Every trial must emit a normalized categorical decision y from the allowed label set for the question. ABSTAIN is supported but OFF by default in v0. Parsing must be strict enough to ensure y is always machine-readable.

## Operational Distribution
Decision distributions are defined with respect to an explicit configuration distribution Q(c). Q(c) must be resolved and serialized in the run artifacts so that P_Q(y|x) is well-defined and auditable.

## Uncertainty Types
- Decision uncertainty: entropy or margin over the induced decision distribution P_Q(y|x).
- Estimation uncertainty: confidence intervals on vote shares or estimator variability due to finite trials.

Confidence intervals are NOT probabilities of correctness; they are estimation CIs on the induced distribution P_Q(y|x).

## Budget Accounting
The primary budget axis is the number of model calls. Token totals and cost estimates are logged on a best-effort basis later.

## v0 Scope
Only H0–H2 are supported in v0. H3/H4 are deferred.

## CLI UX Target
- arbiter run: interactive wizard that selects the question set, heterogeneity rung, and trial budget; writes a resolved config and executes.
- arbiter analyze: optional, deferred command for aggregate analysis.

## Artifact Bundle Contract (Run Folder)
Each run writes a self-contained directory containing:
- manifest.json (run id, timestamp, git SHA, config hash, python version)
- config.resolved.json or config.resolved.yaml (fully resolved config, including Q(c) weights)
- questions.jsonl (exact questions used)
- trials.jsonl (one row per trial; includes transcript, raw output, metadata)
- parsed.jsonl (normalized decision y, parse validity, structured fields)
- aggregates.json (per-question distributions and uncertainty summaries)
- metrics.json (run-level summaries; calibration only if gold labels exist)
- logs/ (optional)

## Confidence Intervals (v0)
Use Wilson intervals on the top-choice proportion and, optionally, on each per-label proportion. Keep the estimator simple and transparent.
