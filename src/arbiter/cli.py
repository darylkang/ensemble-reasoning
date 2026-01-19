"""CLI entrypoint for arbiter."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import uuid

import typer

from arbiter.config import (
    BudgetGuardrail,
    ConvergenceConfig,
    ExecutionConfig,
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
from arbiter.engine import execute_trials
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

    step_total = 9
    render_step_header(1, step_total, "Run identity", "Label the run for traceability.")
    run_name_input = typer.prompt("Run name (optional)", default="", show_default=False)
    run_name = run_name_input.strip() or "auto"
    run_slug = slugify_run_name(run_name)
    render_gap(after_prompt=True)

    render_step_header(2, step_total, "Instance and decision contract", "Provide the prompt and allowed labels.")
    prompt_text = _prompt_text("Prompt text (single line; use \\n for newlines)")
    labels = _prompt_csv("Decision labels (comma-separated)", "YES,NO")
    enable_abstain = _prompt_bool("Enable ABSTAIN?", default=False)
    if enable_abstain and "ABSTAIN" not in labels:
        labels.append("ABSTAIN")
    gold_label = _prompt_optional_label("Gold label (optional)", labels)
    instance_id = _build_instance_id(prompt_text)
    instance_record = {
        "instance_id": instance_id,
        "prompt": prompt_text,
        "labels": labels,
        "gold": gold_label,
        "metadata": {"abstain_enabled": enable_abstain},
    }
    render_gap(after_prompt=True)

    render_step_header(3, step_total, "Sampling design", "Choose rung, models, trial cap, and temperature policy.")
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

    render_step_header(4, step_total, "Personas", "Optionally provide a persona bank and selection mode.")
    persona_policy = _prompt_persona_policy()
    render_gap(after_prompt=True)

    persona_ids_for_q = persona_policy.persona_ids if persona_policy.selection_mode == "sample_uniform" else None
    render_step_header(5, step_total, "Execution controls", "Set concurrency and convergence thresholds.")
    worker_count = _prompt_int("Worker concurrency (W)", 8, min_value=1)
    batch_size = _prompt_int("Batch size (B)", worker_count, min_value=1)
    epsilon = _prompt_float("CI half-width threshold (epsilon)", 0.05)
    min_trials_default = max(batch_size, 16)
    min_trials = _prompt_int("Minimum valid trials before early stop", min_trials_default, min_value=1)
    patience_batches = _prompt_int("Patience batches", 2, min_value=1)
    max_retries = _prompt_int("Max parse retries per trial", 2, min_value=0)
    convergence_config = ConvergenceConfig(
        epsilon_ci_half_width=epsilon,
        min_trials=min_trials,
        patience_batches=patience_batches,
    )
    execution_config = ExecutionConfig(
        worker_count=worker_count,
        batch_size=batch_size,
        max_retries=max_retries,
        convergence=convergence_config,
    )
    render_gap(after_prompt=True)

    render_step_header(6, step_total, "Materialize Q(c)", "Generate the explicit configuration distribution.")
    with status_spinner("Materializing Q(c)"):
        q_distribution = build_q_distribution(models, temperatures, persona_ids_for_q)
    render_success(f"Q(c) materialized with {len(q_distribution.atoms)} atoms.")
    render_gap(after_prompt=False)

    planned_total_trials = k_max
    render_step_header(7, step_total, "Budget and output", "Set a per-instance call guardrail and output path.")
    max_calls = _prompt_int(
        "Max model calls (per instance)",
        planned_total_trials,
        min_value=1,
    )
    budget_guardrail = BudgetGuardrail(max_calls=max_calls, scope="per_instance")

    output_base_dir_input = typer.prompt("Output base directory", default="./runs")
    output_base_dir = str(Path(output_base_dir_input.strip() or "./runs").expanduser())
    render_gap(after_prompt=True)

    render_step_header(8, step_total, "Write run artifacts", "Write resolved config to disk.")
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = create_run_dir(Path(output_base_dir), timestamp, run_slug)
    with status_spinner("Writing resolved config"):
        try:
            notes = []
            if llm_mode == "mock":
                notes.append("OpenRouter API key missing; LLM mode set to mock.")

            planned_total_trials = min(k_max, max_calls)
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
                schema_version="0.5",
                run_name=run_name,
                run_slug=run_slug,
                run_id=run_id,
                output_base_dir=output_base_dir,
                output_dir=str(run_dir),
                started_at=started_at.isoformat(),
                heterogeneity_rung=heterogeneity_rung,
                models=models,
                llm=llm_config,
                execution=execution_config,
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
        except Exception as exc:
            cleanup_run_dir(run_dir)
            render_error(f"Failed to write run artifacts: {exc}")
            raise typer.Exit(code=1) from exc

    render_success("Run config written.")
    render_gap(after_prompt=False)

    render_step_header(9, step_total, "Execute trials", "Run batched trials with convergence checks.")
    execution_result = None
    execution_error = None
    try:
        execution_result = asyncio.run(
            execute_trials(run_dir=run_dir, resolved_config=resolved_config, instance=instance_record)
        )
    except Exception as exc:
        execution_error = exc

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

    if execution_error is not None:
        render_error(f"Execution failed: {execution_error}")
        raise typer.Exit(code=1) from execution_error

    weight_sum = sum(atom.weight for atom in q_distribution.atoms)
    ci_summary = "n/a"
    if execution_result.top_ci_low is not None and execution_result.top_ci_high is not None:
        ci_summary = f"[{execution_result.top_ci_low:.3f}, {execution_result.top_ci_high:.3f}]"

    top_summary = "n/a"
    if execution_result.top_label and execution_result.top_p is not None:
        top_summary = f"{execution_result.top_label} ({execution_result.top_p:.3f}, CI {ci_summary})"

    summary = {
        "Run folder": str(run_dir),
        "Instance": instance_id,
        "Rung": heterogeneity_rung,
        "Q(c) atoms": str(len(q_distribution.atoms)),
        "Weight sum": f"{weight_sum:.6f}",
        "Max trials (cap)": str(planned_total_trials),
        "Workers / batch": f"{worker_count} / {batch_size}",
        "Trials executed": str(execution_result.stop_at_trials),
        "Valid trials": str(execution_result.valid_trials),
        "Batches": str(execution_result.batches_completed),
        "Converged": "yes" if execution_result.converged else "no",
        "Stop reason": execution_result.stop_reason,
        "Top label": top_summary,
    }
    render_summary_table(summary)
    render_info(f"Artifacts written to {run_dir}")

    if execution_result.stop_reason in {"parse_failure", "llm_error", "budget_exhausted"}:
        render_error("Execution stopped due to errors. Review metrics.json for details.")
        raise typer.Exit(code=1)


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


def _prompt_text(prompt: str) -> str:
    while True:
        response = typer.prompt(prompt, default="", show_default=False)
        if response.strip():
            return response
        render_warning("Prompt text is required.")


def _prompt_bool(prompt: str, *, default: bool) -> bool:
    default_value = "y" if default else "n"
    while True:
        response = typer.prompt(f"{prompt} (y/n)", default=default_value)
        normalized = response.strip().lower()
        if normalized in {"y", "yes"}:
            return True
        if normalized in {"n", "no"}:
            return False
        render_warning("Please enter y or n.")


def _prompt_optional_label(prompt: str, labels: list[str]) -> str | None:
    while True:
        response = typer.prompt(prompt, default="", show_default=False)
        cleaned = response.strip()
        if not cleaned:
            return None
        if cleaned in labels:
            return cleaned
        render_warning(f"Gold label must be one of: {', '.join(labels)}.")


def _build_instance_id(prompt_text: str) -> str:
    digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return f"instance_{digest[:10]}"


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
