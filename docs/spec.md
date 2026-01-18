# Specification

This document defines the high-level contract for arbiter. It is implementation-facing but intentionally light on mechanics.

## Notation
- Input instance: `x`.
- Configuration tuple: `c = (m, d, p, π)` (model, decoding, prompt/persona, protocol).
- Configuration distribution: `Q(c)`.
- Label set: `Y` and decision `y ∈ Y`.
- Induced distribution: `P_Q(y|x)`.
- Empirical estimate: `P̂_Q(y|x)`.

## Core Objects
- Instance: an immutable prompt with a stable instance_id, optional choices/labels, and optional gold label.
- Trial: a single model call under a resolved configuration, storing raw output, transcript, and parsed decision.
- Run: a collection of trials for one primary instance by default, with a resolved configuration distribution, artifacts, and summary metrics.

## Execution Unit
- A Run targets one primary instance by default.
- Multi-instance sets are a future extension; current artifact naming remains compatible.
- `questions.jsonl` contains one instance record in current usage (filename retained for now).

## Decision Contract
Every trial must emit a normalized categorical decision `y` from the allowed label set `Y` for the instance. ABSTAIN is supported but OFF by default. When enabled, ABSTAIN is an element of `Y` and should be reported explicitly; coverage-based evaluation should treat ABSTAIN as non-coverage rather than correctness.

## Operational Distribution
Decision distributions are defined with respect to an explicit configuration distribution `Q(c)`. `Q(c)` must be resolved and serialized in run artifacts so that `P_Q(y|x)` is well-defined and auditable.

## Resolved Config Structure
- `config.resolved.json` must separate `run` metadata from `semantic` configuration.
- `semantic` includes the heterogeneity rung, decoding settings, persona policy, `trial_budget` with `k_max`, call guardrails, and the explicit `Q(c)` atoms/weights.
- `run` includes run identifiers and output paths; timestamps may be included for provenance.

## Uncertainty Types
- Decision uncertainty: dispersion of the induced decision distribution `P̂_Q(·|x)`.
- Estimation uncertainty (meta-uncertainty): confidence intervals on vote shares or estimator variability due to finite trials.

Estimation uncertainty must be reported per instance, at minimum as a top-choice proportion CI. Aggregate bootstraps do not substitute for per-instance estimation uncertainty.

Confidence intervals are NOT probabilities of correctness; they are estimation CIs on the induced distribution `P_Q(y|x)`.

## Budget Accounting
The primary budget axis is the number of model calls. Token totals, cost estimates, and latency are logged on a best-effort basis as secondary metadata.

## Trial Budget
- `K_max` is the maximum trials cap for a run (user-specified).
- Early stopping may halt sampling at `K <= K_max`.
- The budget is interpreted as total trials for the instance, not per-atom replicates.

## Batching and Concurrency (Contract)
- Trials may execute asynchronously within each batch.
- `W` is worker concurrency, a configurable parameter for in-flight trials.
- `B` is batch size, the number of trials launched between convergence checks.
- Convergence is evaluated at batch boundaries.
- `B` and `W` are related but not identical; `B` may be <= or >= `W`, and a common default is `B = W`.

## Convergence and Early Stopping
- Sampling should be batched, with convergence checks at batch boundaries.
- Stopping criteria may include CI width thresholds and/or stability of `P̂_Q(y|x)` across batches.
- Per-batch convergence metrics must be recorded in `metrics.json` under a `convergence_trace` array.

## Provenance and Reproducibility
- Each run must serialize resolved configuration and capture environment and git metadata.
- The manifest must include a `semantic_config_hash`, computed by hashing the resolved config with run-specific fields removed (e.g., run_id, timestamps, output_dir). This is distinct from the raw config hash.

## CLI UX Target
- arbiter run: interactive wizard that collects a single prompt (instance), heterogeneity rung, and trial budget; writes a resolved config and executes.
- arbiter analyze: optional command for aggregate analysis.

## Artifact Bundle Contract (Run Folder)
Each run writes a self-contained directory containing:
- manifest.json (run id, timestamp, git SHA, config hash, `semantic_config_hash`, python version)
- config.resolved.json or config.resolved.yaml (resolved config with `run` metadata, `semantic` config, and `Q(c)` weights)
- questions.jsonl (exact instances used; one record in current usage)
- trials.jsonl (one row per trial; includes transcript, raw output, metadata)
- parsed.jsonl (normalized decision `y`, parse validity, structured fields)
- aggregates.json (per-instance distributions and uncertainty summaries)
- metrics.json (run-level summaries; includes `convergence_trace` when applicable)
- logs/ (optional)

## Confidence Intervals (Current Default)
Use Wilson intervals on the top-choice proportion and, optionally, on each per-label proportion. Report these per instance.

## Current Status (Implementation Snapshot)
This section is non-normative and expected to change.
- The wizard materializes `Q(c)` and writes `manifest.json` and `config.resolved.json`.
- Trial execution and derived artifacts (trials/parsed/aggregates/metrics) are not yet produced.
- Currently supported heterogeneity rungs are H0–H2.
