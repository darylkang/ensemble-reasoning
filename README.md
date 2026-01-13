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
The CLI is stubbed in this round:

```bash
arbiter
```

You should see:

```
arbiter: scaffold installed; implementation pending
```

## Docs
- Vision: `docs/vision.md`
- Spec: `docs/spec.md`
- Agent rules: `AGENTS.md`
