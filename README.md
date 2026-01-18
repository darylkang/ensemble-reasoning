# Arbiter
Ensemble Reasoning Research Harness

Arbiter (repo: ensemble-reasoning) is a research harness for running ensemble reasoning experiments with language models. It treats reasoning outputs as a distribution over decisions and rationales, and emphasizes reproducibility, auditability, and statistical rigor over product features.

The key framing is explicit configuration sampling `Q(c)` and the induced decision distribution `P_Q(y|x)`, estimated by `P̂_Q(y|x)`. We prioritize reliability signals, decision stability, and meta-uncertainty (confidence intervals, convergence) rather than accuracy alone.

## Non-goals
- Not a production system or hosted service.
- Not a commercial product.
- Not a simulation of human populations or juries.

## Current status
- `arbiter run` sets up runs and writes run folder artifacts; execution/providers come next.
- `config.resolved.json` separates `run` metadata from `semantic` config (including `trial_budget.k_max`).
- `runs/` is generated output and should remain untracked/ignored.

## Requirements
- Python >= 3.14

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
`arbiter run` launches an interactive wizard that creates a run folder under `./runs` with `manifest.json` and `config.resolved.json`.

```bash
arbiter --help
arbiter run
```

## Docs
- Vision: `docs/vision.md`
- Spec: `docs/spec.md`
- Agent rules: `AGENTS.md`
