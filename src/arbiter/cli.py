"""CLI entrypoint for arbiter."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import uuid

import typer

from arbiter.config import (
    BudgetGuardrail,
    LLMConfig,
    LLMRequestDefaults,
    PersonaPolicy,
    ResolvedConfig,
    TemperaturePolicy,
    TrialBudget,
    build_q_distribution,
    dedupe_preserve_order,
    load_persona_ids,
    slugify_run_name,
)
from arbiter.env import load_dotenv
from arbiter.manifest import Manifest, get_git_info, platform_info
from arbiter.storage import cleanup_run_dir, compute_hash, create_run_dir, write_json
from arbiter.llm.client import build_request_body
from arbiter.llm.types import LLMRequest
from arbiter.ui.progress import status_spinner
from arbiter.ui.render import (
    render_banner,
    render_error,
    render_gap,
    render_info,
    render_notice,
    render_step_header,
    render_success,
    render_summary_table,
    render_warning,
)

app = typer.Typer(add_completion=False, help="Research harness for ensemble reasoning.")
llm_app = typer.Typer(add_completion=False, help="LLM utilities and diagnostics.")
app.add_typer(llm_app, name="llm")


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context) -> None:
    """Arbiter research harness CLI."""
    load_dotenv()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@app.command("run")
def run_wizard() -> None:
    """Interactive wizard to create a run folder and resolved config."""
    render_banner("arbiter", "Ensemble reasoning run setup")

    started_at = datetime.now(timezone.utc)
    default_model = os.getenv("ARBITER_DEFAULT_MODEL", "openai/gpt-5")
    api_key_present = bool(os.getenv("OPENROUTER_API_KEY"))
    llm_mode = "openrouter" if api_key_present else "mock"
    if not api_key_present:
        render_notice("OpenRouter API key not found. Remote calls are disabled; using mock client.")

    step_total = 6
    render_step_header(1, step_total, "Run identity", "Label the run for traceability.")
    run_name_input = typer.prompt("Run name (optional)", default="", show_default=False)
    run_name = run_name_input.strip() or "auto"
    run_slug = slugify_run_name(run_name)
    render_gap(after_prompt=True)

    render_step_header(2, step_total, "Sampling design", "Choose rung, models, trials, and temperature policy.")
    heterogeneity_rung = _prompt_choice("Heterogeneity rung", ["H0", "H1", "H2"], "H1")
    models = _prompt_csv("Model slugs (comma-separated)", default_model)
    k_max = _prompt_int("Max trials (cap)", 16, min_value=1)

    temp_kind = _prompt_choice("Temperature policy", ["fixed", "list"], "fixed")
    if temp_kind == "fixed":
        temperatures = [_prompt_float("Temperature", 0.7)]
        temperature_policy = TemperaturePolicy(kind="fixed", temperatures=temperatures)
    else:
        temperatures = _prompt_float_list("Temperatures (comma-separated)", "0.7,1.0")
        temperature_policy = TemperaturePolicy(kind="list", temperatures=temperatures)
    render_gap(after_prompt=True)

    render_step_header(3, step_total, "Personas", "Optionally provide a persona bank and selection mode.")
    persona_policy = _prompt_persona_policy()
    render_gap(after_prompt=True)

    persona_ids_for_q = persona_policy.persona_ids if persona_policy.selection_mode == "sample_uniform" else None
    render_step_header(4, step_total, "Materialize Q(c)", "Generate the explicit configuration distribution.")
    with status_spinner("Materializing Q(c)"):
        q_distribution = build_q_distribution(models, temperatures, persona_ids_for_q)
    render_success(f"Q(c) materialized with {len(q_distribution.atoms)} atoms.")
    render_gap(after_prompt=False)

    planned_total_trials = k_max
    render_step_header(5, step_total, "Budget and output", "Set a per-instance call guardrail and output path.")
    max_calls = _prompt_int(
        "Max model calls (per instance)",
        planned_total_trials,
        min_value=1,
    )
    budget_guardrail = BudgetGuardrail(max_calls=max_calls, scope="per_instance")

    output_base_dir_input = typer.prompt("Output base directory", default="./runs")
    output_base_dir = str(Path(output_base_dir_input.strip() or "./runs").expanduser())
    render_gap(after_prompt=True)

    render_step_header(6, step_total, "Write run artifacts", "Write manifest and resolved config to disk.")
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = create_run_dir(Path(output_base_dir), timestamp, run_slug)
    with status_spinner("Writing run artifacts"):
        try:
            notes = ["Instances and trials are not executed in this round."]
            if llm_mode == "mock":
                notes.append("OpenRouter API key missing; LLM mode set to mock.")

            trial_budget = TrialBudget(k_max=k_max, scope="per_instance")
            llm_request_defaults = LLMRequestDefaults(
                temperature=temperatures[0] if temp_kind == "fixed" else None,
                top_p=None,
                max_tokens=None,
                seed=None,
                stop=None,
                response_format=None,
                tools=None,
                tool_choice=None,
                parallel_tool_calls=None,
            )
            llm_config = LLMConfig(
                client="openrouter",
                mode=llm_mode,
                model=models[0],
                request_defaults=llm_request_defaults,
                routing_defaults={"allow_fallbacks": False},
                extra_body_defaults={},
            )
            resolved_config = ResolvedConfig.build_from_wizard_inputs(
                schema_version="0.4",
                run_name=run_name,
                run_slug=run_slug,
                run_id=run_id,
                output_base_dir=output_base_dir,
                output_dir=str(run_dir),
                started_at=started_at.isoformat(),
                heterogeneity_rung=heterogeneity_rung,
                models=models,
                llm=llm_config,
                temperature_policy=temperature_policy,
                personas=persona_policy,
                trial_budget=trial_budget,
                budget_guardrail=budget_guardrail,
                q_distribution=q_distribution,
                notes=notes,
            )

            resolved = resolved_config.to_dict()
            config_hash = compute_hash(resolved)
            semantic_config_hash = compute_hash(resolved_config.semantic.to_dict())

            config_path = run_dir / "config.resolved.json"
            write_json(config_path, resolved)

            ended_at = datetime.now(timezone.utc)
            git_info = get_git_info(Path.cwd())
            manifest = Manifest(
                run_id=run_id,
                started_at=started_at.isoformat(),
                ended_at=ended_at.isoformat(),
                git_sha=git_info.sha,
                git_dirty=git_info.dirty,
                python_version=platform.python_version(),
                platform=platform_info(),
                config_hash=config_hash,
                semantic_config_hash=semantic_config_hash,
                planned_call_budget=max_calls,
                planned_call_budget_scope="per_instance",
                planned_total_trials=planned_total_trials,
                planned_total_trials_scope="per_instance",
            )
            manifest_path = run_dir / "manifest.json"
            write_json(manifest_path, manifest.to_dict())
        except Exception as exc:
            cleanup_run_dir(run_dir)
            render_error(f"Failed to write run artifacts: {exc}")
            raise typer.Exit(code=1) from exc

    render_success("Run artifacts written.")

    weight_sum = sum(atom.weight for atom in q_distribution.atoms)
    summary = {
        "Run folder": str(run_dir),
        "Rung": heterogeneity_rung,
        "Q(c) atoms": str(len(q_distribution.atoms)),
        "Weight sum": f"{weight_sum:.6f}",
        "Max trials (cap)": str(planned_total_trials),
        "Max calls/instance": str(max_calls),
    }
    render_summary_table(summary)
    render_info("Next step: execution is not implemented in this round.")


@llm_app.command("dry-run")
def llm_dry_run() -> None:
    """Build and display an OpenRouter request body without network access."""
    default_model = os.getenv("ARBITER_DEFAULT_MODEL", "openai/gpt-5")
    render_banner("arbiter", "LLM dry-run request preview")
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
        render_step_header(index, len(cases), title, "Merged request with override tracking.")
        print(json.dumps(body, indent=2, sort_keys=True, ensure_ascii=True))
        if overrides:
            render_warning(f"Overrides applied: {', '.join(sorted(overrides))}")
        else:
            render_info("No overrides detected.")
        if index < len(cases):
            render_gap(after_prompt=False)


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    normalized_choices = {choice.lower(): choice for choice in choices}
    while True:
        response = typer.prompt(f"{prompt} ({'/'.join(choices)})", default=default)
        normalized = response.strip().lower()
        if normalized in normalized_choices:
            return normalized_choices[normalized]
        render_warning(f"Invalid choice: {response}. Choose from {', '.join(choices)}.")


def _prompt_persona_policy() -> PersonaPolicy:
    while True:
        persona_bank_input = typer.prompt("Persona bank path (optional)", default="", show_default=False)
        persona_bank_path = persona_bank_input.strip() or None
        selection_mode = "none"
        persona_ids: list[str] = []
        persona_loaded = False

        if persona_bank_path:
            selection_mode = _prompt_choice(
                "Persona selection mode",
                ["none", "sample_uniform"],
                "sample_uniform",
            )
            if selection_mode == "sample_uniform":
                load_result = load_persona_ids(Path(persona_bank_path))
                if load_result.loaded:
                    persona_ids = load_result.persona_ids
                    persona_loaded = True
                else:
                    render_error(f"{load_result.error}. Provide a valid persona bank or leave empty.")
                    continue

        return PersonaPolicy(
            persona_bank_path=persona_bank_path,
            selection_mode=selection_mode,
            persona_ids=persona_ids,
            loaded=persona_loaded,
        )


def _prompt_csv(prompt: str, default: str) -> list[str]:
    while True:
        response = typer.prompt(prompt, default=default)
        items = [item.strip() for item in response.split(",") if item.strip()]
        items = dedupe_preserve_order(items)
        if items:
            return items
        render_warning("Please provide at least one value.")


def _prompt_int(prompt: str, default: int, min_value: int = 1) -> int:
    while True:
        response = typer.prompt(prompt, default=str(default))
        try:
            value = int(response)
        except ValueError:
            render_warning("Please enter an integer.")
            continue
        if value < min_value:
            render_warning(f"Value must be >= {min_value}.")
            continue
        return value


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        response = typer.prompt(prompt, default=str(default))
        try:
            return float(response)
        except ValueError:
            render_warning("Please enter a number.")


def _prompt_float_list(prompt: str, default: str) -> list[float]:
    while True:
        response = typer.prompt(prompt, default=default)
        parts = [part.strip() for part in response.split(",") if part.strip()]
        if not parts:
            render_warning("Please enter one or more numbers.")
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError:
            render_warning("Invalid list; use comma-separated numbers.")
            continue
        return values


def main() -> None:
    app()


if __name__ == "__main__":
    main()
