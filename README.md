# Arbiter
Ensemble Reasoning Research Harness

Arbiter (repo: ensemble-reasoning) is a research harness for mapping the reasoning landscape of language models. It treats model outputs as structured objects, embeds them, and clusters them to discover emergent outcome modes rather than enforcing a predefined label set.

The key framing is explicit configuration sampling `Q(c)` and the induced distribution over discovered modes `P_Q(y|x)`, estimated by `P̂^Q(y|x)`. We prioritize reliability signals, decision stability, and meta-uncertainty (confidence intervals, convergence) rather than accuracy alone.

## Non-goals
- Not a production system or hosted service.
- Not a commercial product.
- Not a simulation of human populations or juries.

## Current status
- `arbiter` executes single-question runs with OpenRouter (or mock when no key is present), including embeddings and online clustering for convergence.
- Offline clustering and summarization artifacts are scaffolded and may be `not_run` in some runs.
- `runs/` is generated output and should remain untracked/ignored.

## Requirements
- Python >= 3.14
- OpenRouter is the only network provider; models are selected by OpenRouter slug.
- Measurement mode defaults to no fallbacks (`allow_fallbacks=false`) unless explicitly overridden.

## Configuration
Copy `.env.example` to `.env`, fill in values, and keep `.env` untracked. Arbiter reads these environment variables at runtime. Embedding and summarizer instruments are locked by default and can be overridden via `ARBITER_EMBEDDING_MODEL` and `ARBITER_SUMMARIZER_MODEL`.

Copy `arbiter.config.example.json` to `arbiter.config.json` to start from a canonical template; the wizard will load it and only prompt for missing fields.

Validate a config file before running:

```bash
arbiter config validate
```

## Install
From a local clone:

```bash
git clone https://github.com/darylkang/ensemble-reasoning.git
cd ensemble-reasoning
python -m pip install -e .
```

Direct from GitHub:

```bash
python -m pip install "git+https://github.com/darylkang/ensemble-reasoning.git@main"
```

## Run
`arbiter` launches a Textual TUI wizard (arrow keys to navigate, space to toggle, Enter to confirm):

1) Welcome + environment check (OpenRouter key, remote vs mock; Remote disabled if missing key)
2) Config mode selection (load `arbiter.config.json` or guided build)
3) Question text (multi-line TextArea)
4) Quick Run (recommended) or Customize
5) If Customize: decode params, persona mix, model mix, protocol, advanced settings
6) Review -> Execute -> Receipt

Runs write a folder under `./runs` following the artifact contract in `docs/spec.md`, including `manifest.json`, `config.input.json`, `config.resolved.json`, `question.json`, `trials.jsonl`, `parsed.jsonl`, `embeddings.*`, `clusters_online.json`, `clusters_offline.json`, `cluster_summaries.json`, `aggregates.json`, and `metrics.json`.

```bash
arbiter --help
arbiter
arbiter llm dry-run
```

## Docs
- Vision: `docs/vision.md`
- Spec: `docs/spec.md`
- Agent rules: `AGENTS.md`
