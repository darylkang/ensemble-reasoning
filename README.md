# Arbiter
Ensemble Reasoning Research Harness

Arbiter (repo: ensemble-reasoning) is a research harness for mapping the reasoning landscape of language models. It treats model outputs as structured objects, embeds them, and clusters them to discover emergent modes rather than enforcing a predefined label set.

The key framing is explicit configuration sampling `Q(c)` and the induced distribution over discovered modes `P_Q(y|x)`, estimated by `P̂^Q(y|x)`. We prioritize reliability signals, decision stability, and meta-uncertainty (confidence intervals, convergence) rather than accuracy alone.

## Non-goals
- Not a production system or hosted service.
- Not a commercial product.
- Not a simulation of human populations or juries.

## Current status
- `arbiter run` executes single-question runs with OpenRouter (or mock when no key is present).
- The artifact bundle and clustering outputs are evolving toward the full contract in `docs/spec.md`.
- `runs/` is generated output and should remain untracked/ignored.

## Requirements
- Python >= 3.14
- OpenRouter is the only network provider; models are selected by OpenRouter slug.
- Measurement mode defaults to no fallbacks (`allow_fallbacks=false`) unless explicitly overridden.

## Configuration
Copy `.env.example` to `.env`, fill in values, and keep `.env` untracked. Arbiter reads these environment variables at runtime.
You can optionally create `arbiter.config.json` in the working directory; the wizard will load it and only prompt for missing fields.

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
`arbiter run` launches a decision-tree wizard:

1) Welcome + environment check (OpenRouter key, remote vs mock)
2) Config mode selection (load `arbiter.config.json` or guided build)
3) Question text `x` (multi-line or file path)
4) Decode params `d` (fixed or ranged temperature, defaults/extras)
5) Persona mix `p`
6) Model mix `m` (OpenRouter slugs, default from `ARBITER_DEFAULT_MODEL`)
7) Protocol `π` (independent vs interaction/debate)
8) Advanced settings (execution + convergence + clustering)
9) Review -> Execute -> Receipt

Runs write a folder under `./runs` following the artifact contract in `docs/spec.md`, including `manifest.json`, `config.input.json`, `config.resolved.json`, `question.json`, `trials.jsonl`, `parsed.jsonl`, `aggregates.json`, `metrics.json`, and clustering outputs.

```bash
arbiter --help
arbiter run
arbiter llm dry-run
```

## Docs
- Vision: `docs/vision.md`
- Spec: `docs/spec.md`
- Agent rules: `AGENTS.md`
