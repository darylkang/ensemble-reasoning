# ensemble-reasoning

ensemble-reasoning is a research harness for running ensemble reasoning experiments with language models. It treats reasoning outputs as a distribution over decisions and rationales, and emphasizes reproducibility, auditability, and statistical rigor over product features.

## Non-goals
- Not a production system or hosted service.
- Not a commercial product.
- Not a simulation of human populations or juries.

## Requirements
- Python >= 3.14

## Install
Editable install:

```bash
pip install -e .
```

Install from Git (SSH):

```bash
pip install "git+ssh://git@github.com/<org>/ensemble-reasoning.git"
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
