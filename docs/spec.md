# Specification

This document defines the high-level contract for Arbiter. It is implementation-facing but intentionally light on mechanics.

## Notation
- Question: `x` (the primary unit).
- Configuration tuple: `c = (m, d, p, π)` (model, decoding, prompt/persona, protocol).
- Configuration distribution: `Q(c)`.
- Trial output object: `o` (structured free-form output).
- Embedding: `z = f(o)`.
- Mode assignment: `y := cluster(z)`.
- Induced distribution: `P_Q(y|x)`.
- Empirical estimate: `P̂^Q(y|x)`.

## Core Objects
- Question: an immutable prompt with a stable question_id, question text, and optional metadata.
- Trial: a single model call under a resolved configuration, storing output object `o`, raw response, and metadata.
- Run: a collection of trials for one primary question, with a resolved configuration distribution, artifacts, and summary metrics.

## Execution Unit
- A Run targets one primary question by default.
- Multi-question sets are a future extension.
- `question.json` contains the single question record for the run.

## Trial Output Contract
Each trial must emit a structured output object `o` under a fixed schema (JSON-only):

```json
{
  "outcome": "string (required)",
  "rationale": "string (optional)",
  "trace_summary": "string (optional)"
}
```

The output object is embedded and clustered to define emergent modes; no predefined categorical label set is required.

## Parsing and Retries
- Responses are parsed as JSON and validated for required keys.
- On parse/validation failure, Arbiter may retry up to `execution.max_retries` by appending a corrective user message.
- If retries are exhausted, the run stops with a clear `stop_reason` and writes partial artifacts; invalid outputs do not contribute to embeddings or clustering.

## Operational Distribution
Mode distributions are defined with respect to an explicit configuration distribution `Q(c)`. `Q(c)` must be resolved and serialized in run artifacts so that `P_Q(y|x)` is well-defined and auditable.

## Canonical Config (`arbiter.config.json`)
Arbiter accepts a canonical, human-editable configuration file (`arbiter.config.json`) that mirrors the wizard steps. The canonical config is factorized and normalized before being resolved into explicit `Q(c)` atoms.

Top-level structure:
- `schema_version`: canonical config schema version (current: `1.1`).
- `question`: `{id?, text, metadata}` (question text can be filled by the wizard if missing).
- `q`: factorized `Q(c)` definition with weighted `models`, `personas`, and `decode` temperature policy.
- `protocol`: `{type}` (independent for now).
- `execution`: `{k_max, workers, batch_size, retries}`.
- `convergence`: `{delta_js_threshold, epsilon_new_threshold, epsilon_ci_half_width, min_trials, patience_batches}`.
- `clustering`: `{method, tau, embed_text, embedding_model}`.
- `summarizer`: `{enabled, model, prompt_version}`.
- `llm`: `{mode, model, request_defaults, routing_defaults, extra_body_defaults}`.

Each run writes a `config.input.json` copy of the canonical config used to drive that run. The repo includes a tracked template (`arbiter.config.example.json`) that can be copied to `arbiter.config.json`.

## Resolved Config Structure
- `config.resolved.json` must separate `run` metadata from `semantic` configuration.
- `schema_version` identifies the resolved config schema; breaking changes bump this value (current: `0.8`).
- `semantic` includes the heterogeneity rung, `protocol`, `clustering`, `summarizer`, decoding settings, persona policy, `trial_budget` with `k_max`, call guardrails, `execution` controls, and the explicit `Q(c)` atoms/weights.
- `execution` includes `worker_count`, `batch_size`, `max_retries`, and `convergence` thresholds (`delta_js_threshold`, `epsilon_new_threshold`, `epsilon_ci_half_width`, `min_trials`, `patience_batches`).
- `run` includes run identifiers and output paths; timestamps may be included for provenance.

## LLM Configuration
- `semantic.llm.client` is fixed to `openrouter` to reflect the sole network provider.
- `semantic.llm.mode` records `openrouter` or `mock` to reflect whether remote calls are enabled.
- `semantic.llm.model` is the OpenRouter model slug used for requests.
- `semantic.llm.request_defaults` stores OpenAI-compatible parameters (temperature, top_p, max_tokens, seed, stop, response_format, tools, tool_choice, parallel_tool_calls).
- `semantic.llm.routing_defaults` stores OpenRouter routing defaults, with `allow_fallbacks` set to `false` in measurement mode.
- `semantic.llm.extra_body_defaults` stores additional request fields for passthrough.

Routing defaults must not silently enable fallbacks. Overrides must be explicit and auditable.

## Embeddings and Clustering
- Each output object `o` is embedded with a locked embedding model `f` (model slug recorded in config and artifacts).
- Default clustering is online leader clustering with cosine threshold `τ` and stable cluster identifiers within a run.
- Embeddings and clustering state must be serialized to enable audit and recomputation.
- Offline clustering is a separate analysis artifact and must be reported distinctly from online clustering used for convergence.
Embedding model overrides may be provided via `ARBITER_EMBEDDING_MODEL`, but are not exposed as an experiment knob in the wizard.

## Summarizer Instrument
- Cluster summaries use a locked summarizer model and a fixed prompt template version/hash.
- Summaries are an analysis artifact and must not alter the online control loop.
Summarizer model overrides may be provided via `ARBITER_SUMMARIZER_MODEL`, but are not exposed as an experiment knob in the wizard.
The prompt hash and version are recorded in `cluster_summaries.json`.

## Uncertainty Types
- Outcome-mode uncertainty: dispersion of the induced mode distribution `P̂^Q(·|x)`.
- Estimation uncertainty (meta-uncertainty): confidence intervals on mode shares or estimator variability due to finite trials.

Estimation uncertainty must be reported per question, at minimum as a top-cluster share CI when enabled. Aggregate bootstraps do not substitute for per-question estimation uncertainty.

Confidence intervals are NOT probabilities of correctness; they are estimation CIs on the induced distribution `P_Q(y|x)`.

## Budget Accounting
The primary budget axis is the number of model calls. Token totals, cost estimates, and latency are logged on a best-effort basis as secondary metadata.

## Trial Budget
- `K_max` is the maximum trials cap for a run (user-specified).
- Early stopping may halt sampling at `K <= K_max`.
- The budget is interpreted as total trials for the question, not per-atom replicates.

## Batching and Concurrency (Contract)
- Trials may execute asynchronously within each batch.
- `W` is worker concurrency (`execution.worker_count`).
- `B` is batch size (`execution.batch_size`), the number of trials launched between convergence checks.
- Convergence is evaluated at batch boundaries.
- `B` and `W` are related but not identical; `B` may be <= or >= `W`, and a common default is `B = W`.

## Convergence and Early Stopping
- Sampling is batched, with convergence checks at batch boundaries.
- Convergence is defined on cluster distributions derived from online clustering.
- Online clustering must maintain stable cluster identifiers; offline clustering results must be reported separately.
- Convergence indicates estimate stability, not correctness.
- Per-batch convergence metrics must be recorded in `metrics.json` under a `convergence_trace` array.
- Runs must record `stop_reason` and `stop_at_trials` when execution halts.

Default convergence criteria include:
- JS divergence threshold (`delta_js_threshold`).
- New-cluster discovery rate threshold (`epsilon_new_threshold`).
- Optional top-cluster CI half-width (`epsilon_ci_half_width`).

## Provenance and Reproducibility
- Each run must serialize resolved configuration and capture environment and git metadata.
- The manifest must include a `semantic_config_hash`, computed by hashing the resolved config with run-specific fields removed (e.g., run_id, timestamps, output_dir). This is distinct from the raw config hash.
- Embedding and summarizer instruments must be recorded (model slugs, prompt version/hash, normalization).

## CLI UX Target
The interactive TUI is a decision tree:
- Welcome and environment check (OpenRouter key, remote vs mock).
- Config mode selection (load `arbiter.config.json` vs guided build); prompt only for missing fields if a config is loaded.
- Collect question text `x` (multi-line input or file path).
- Quick Run (defaults) or Customize (decode, personas, models, protocol, advanced execution/convergence/clustering).
- Review -> Execute -> Receipt.

Use `arbiter config validate` to validate a config file without executing a run.

## Artifact Bundle Contract (Run Folder)
Each run writes a self-contained directory containing:
- `manifest.json` (run id, timestamp, git SHA, config hash, `semantic_config_hash`, python version, call counts, instrument slugs)
- `config.input.json` (canonical config used to drive the run, including question text)
- `config.resolved.json` (resolved config with `run` metadata, `semantic` config, and `Q(c)` weights)
- `question.json` (the primary question record)
- `trials.jsonl` (one row per trial; includes atom/model/temperature/persona, effective request body, routing, overrides, timestamps, usage, and raw response)
- `parsed.jsonl` (parsed output object `o`, parse validity, retries used, optional parse error, `cluster_id`, `embedded_text`)
- `embeddings.jsonl` (embedding vectors encoded as base64 float32 with model metadata)
- `clusters_online.json` (online clustering assignments, centroids encoded as base64 float32, exemplars, and clustering parameters)
- `clusters_offline.json` (offline clustering output or `not_run` scaffold)
- `cluster_summaries.json` (summarizer output or `not_run` scaffold)
- `aggregates.json` (per-question cluster distributions, counts, uncertainty summaries, top-cluster CI, parse error rate, effective cluster count)
- `metrics.json` (run-level summaries; includes `convergence_trace`, `stop_reason`, `stop_at_trials`, and sampling seed)
- `logs/` (optional)

## Confidence Intervals (Current Default)
Use Wilson intervals on the top-cluster share and, optionally, on each per-cluster share. Report these per question.

## Current Status (Implementation Snapshot)
This section is non-normative and expected to change.
- Offline clustering and summarization are scaffolded and may be `not_run` in some runs.
- Currently supported heterogeneity rungs are H0–H2.
