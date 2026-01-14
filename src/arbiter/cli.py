"""CLI entrypoint for arbiter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import uuid

import typer

from arbiter.config import (
    BudgetGuardrail,
    PersonaPolicy,
    RunConfig,
    TemperaturePolicy,
    build_q_distribution,
    dedupe_preserve_order,
    load_persona_ids,
    slugify_run_name,
)
from arbiter.manifest import Manifest, get_git_info, platform_info
from arbiter.storage import compute_hash, create_run_dir, write_json
from arbiter.ui.console import get_console
from arbiter.ui.progress import status_spinner
from arbiter.ui.render import (
    render_banner,
    render_info,
    render_step_header,
    render_success,
    render_summary_table,
    render_warning,
)

app = typer.Typer(add_completion=False, help="Research harness for ensemble reasoning.")


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    normalized_choices = {choice.lower(): choice for choice in choices}
    while True:
        response = typer.prompt(f"{prompt} ({'/'.join(choices)})", default=default)
        normalized = response.strip().lower()
        if normalized in normalized_choices:
            return normalized_choices[normalized]
        render_warning(f"Invalid choice: {response}. Choose from {', '.join(choices)}.")


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


@app.command()
def run() -> None:
    """Interactive wizard to create a run folder and resolved config."""
    get_console()
    render_banner("arbiter", "Ensemble reasoning run setup")

    started_at = datetime.now(timezone.utc)

    step_total = 6
    render_step_header(1, step_total, "Run identity")
    run_name_input = typer.prompt("Run name (optional)", default="", show_default=False)
    run_name = run_name_input.strip() or "auto"
    run_slug = slugify_run_name(run_name)

    render_step_header(2, step_total, "Sampling design")
    heterogeneity_rung = _prompt_choice("Heterogeneity rung", ["H0", "H1", "H2"], "H1")
    models = _prompt_csv("Model identifiers (comma-separated)", "model-1")
    trials_per_question = _prompt_int("Trials per question", 16, min_value=1)

    temp_kind = _prompt_choice("Temperature policy", ["fixed", "list"], "fixed")
    if temp_kind == "fixed":
        temperatures = [_prompt_float("Temperature", 0.7)]
        temperature_policy = TemperaturePolicy(kind="fixed", temperatures=temperatures)
    else:
        temperatures = _prompt_float_list("Temperatures (comma-separated)", "0.7,1.0")
        temperature_policy = TemperaturePolicy(kind="list", temperatures=temperatures)

    render_step_header(3, step_total, "Personas")
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
                render_warning(f"{load_result.error}; using default persona.")

    persona_policy = PersonaPolicy(
        persona_bank_path=persona_bank_path,
        selection_mode=selection_mode,
        persona_ids=persona_ids,
        loaded=persona_loaded,
    )

    persona_ids_for_q = persona_ids if selection_mode == "sample_uniform" else None
    render_step_header(4, step_total, "Materialize Q(c)")
    with status_spinner("Materializing configuration distribution"):
        q_distribution = build_q_distribution(models, temperatures, persona_ids_for_q)
    render_success(f"Materialized Q(c) with {len(q_distribution.atoms)} atoms.")

    planned_total_trials = len(q_distribution.atoms) * trials_per_question
    render_step_header(5, step_total, "Budget and output")
    max_calls = _prompt_int(
        "Max model calls (per question)",
        planned_total_trials,
        min_value=1,
    )
    budget_guardrail = BudgetGuardrail(max_calls=max_calls, scope="per_question")

    output_base_dir_input = typer.prompt("Output base directory", default="./runs")
    output_base_dir = str(Path(output_base_dir_input.strip() or "./runs").expanduser())

    render_step_header(6, step_total, "Write run artifacts")
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    with status_spinner("Creating run folder and writing artifacts"):
        run_dir = create_run_dir(Path(output_base_dir), timestamp, run_slug)

        notes = ["Questions and trials are not executed in this round."]
        if persona_bank_path and not persona_loaded and selection_mode == "sample_uniform":
            notes.append("Persona bank not loaded; default persona used for Q(c).")

        run_config = RunConfig(
            name=run_name,
            slug=run_slug,
            run_id=run_id,
            heterogeneity_rung=heterogeneity_rung,
            trials_per_question=trials_per_question,
            output_base_dir=output_base_dir,
            output_dir=str(run_dir),
        )

        resolved = {
            "schema_version": "0.1",
            "run": run_config.to_dict(),
            "models": models,
            "temperature_policy": temperature_policy.to_dict(),
            "personas": persona_policy.to_dict(),
            "budget_guardrail": budget_guardrail.to_dict(),
            "q_distribution": q_distribution.to_dict(),
            "notes": notes,
        }

        config_hash = compute_hash(resolved)

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
            planned_call_budget=max_calls,
            planned_call_budget_scope="per_question",
            planned_total_trials=planned_total_trials,
            planned_total_trials_scope="per_question",
        )
        manifest_path = run_dir / "manifest.json"
        write_json(manifest_path, manifest.to_dict())

    render_success("Wrote config.resolved.json and manifest.json.")

    weight_sum = sum(atom.weight for atom in q_distribution.atoms)
    summary = {
        "Run folder": str(run_dir),
        "Rung": heterogeneity_rung,
        "Q(c) atoms": str(len(q_distribution.atoms)),
        "Weight sum": f"{weight_sum:.6f}",
        "Planned trials (per question)": str(planned_total_trials),
        "Max calls (per question)": str(max_calls),
    }
    render_summary_table(summary)
    render_info("Next step: trial execution is not implemented in this round.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
