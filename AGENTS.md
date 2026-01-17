# AGENTS

This file is authoritative for Codex behavior in this repo.

## Project Principles
- This repository is research tooling; avoid overengineering.
- Prefer clear, auditable scaffolding over complex implementation.

## Commit Discipline
- Conventional Commits are REQUIRED.
- Format: type(scope): description
- Body REQUIRED as bullet points.
- Footer optional.
- Type, scope, description, and body are required; body must be bullets.
- Commit message body must use real newlines; each bullet must be on its own line. Do NOT include literal "\n" characters in the commit message.
- Commit at the end of each coding round, including scaffold-only rounds.
- Keep commits small and coherent; avoid unrelated changes in a single commit.

## Repo Hygiene
- Never commit run artifacts under `runs/`. If they are accidentally tracked, remove them from git and ensure they are ignored.
- If code changes any contract (CLI behavior, artifact schema, config fields), update `docs/spec.md` and `README.md` in the same round.
- Security hygiene: API keys only via environment variables; never log or serialize secrets.

## Round-Specific Instructions
- Round-specific instructions must be provided in the prompt, issue, or PR description and must not be added to AGENTS.md.
