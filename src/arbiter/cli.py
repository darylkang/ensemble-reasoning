"""CLI entrypoint for arbiter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import typer

from arbiter.env import load_dotenv
from arbiter.llm.client import build_request_body
from arbiter.llm.types import LLMRequest
from arbiter.ui.app import run_app
from arbiter.validation import load_and_validate_config

app = typer.Typer(add_completion=False, help="Research harness for ensemble reasoning.")
llm_app = typer.Typer(add_completion=False, help="LLM utilities and diagnostics.")
config_app = typer.Typer(add_completion=False, help="Config helpers and validation.")
app.add_typer(llm_app, name="llm")
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate(path: str = typer.Option("arbiter.config.json", "--path", "-p")) -> None:
    """Validate a canonical config file without executing a run."""
    config_path = Path(path)
    default_model = os.getenv("ARBITER_DEFAULT_MODEL", "openai/gpt-5")
    api_key_present = bool(os.getenv("OPENROUTER_API_KEY"))
    llm_mode = "remote" if api_key_present else "mock"
    result = load_and_validate_config(config_path, default_model=default_model, llm_mode=llm_mode)

    errors = [f"{issue.path}: {issue.message}" for issue in result.errors]
    warnings = [f"{issue.path}: {issue.message}" for issue in result.warnings]

    if errors:
        print("INVALID")
        for issue in errors:
            print(f"- {issue}")
        raise typer.Exit(code=1)

    if warnings:
        print("VALID (with warnings)")
        for issue in warnings:
            print(f"- {issue}")
    else:
        print("VALID")


@llm_app.command("dry-run")
def llm_dry_run() -> None:
    """Build and display an OpenRouter request body without network access."""
    default_model = os.getenv("ARBITER_DEFAULT_MODEL", "openai/gpt-5")
    print("LLM DRY-RUN")
    default_routing = {"allow_fallbacks": False}

    cases = [
        (
            "Defaults (provider_routing=None)",
            LLMRequest(
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
                model=default_model,
                temperature=0.7,
                provider_routing=None,
                metadata={"mode": "dry_run"},
            ),
        ),
        (
            "Empty provider override (provider_routing={})",
            LLMRequest(
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
                model=default_model,
                temperature=0.7,
                provider_routing={},
                metadata={"mode": "dry_run"},
            ),
        ),
        (
            "extra_body overrides provider",
            LLMRequest(
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
                model=default_model,
                temperature=0.7,
                provider_routing=None,
                extra_body={"provider": {"allow_fallbacks": True}},
                metadata={"mode": "dry_run"},
            ),
        ),
    ]

    for index, (title, request) in enumerate(cases, start=1):
        body, overrides = build_request_body(request, default_provider_routing=default_routing)
        print(f"\n[{index}/{len(cases)}] {title}")
        print(json.dumps(body, indent=2, sort_keys=True, ensure_ascii=True))
        if overrides:
            print(f"Overrides applied: {', '.join(sorted(overrides))}")
        else:
            print("No overrides detected.")


def main() -> None:
    load_dotenv()
    if len(sys.argv) == 1:
        run_app()
    else:
        app()


if __name__ == "__main__":
    main()
