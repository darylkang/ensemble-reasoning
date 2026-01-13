"""CLI entrypoint for arbiter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import uuid

import typer
from rich.console import Console

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

app = typer.Typer(add_completion=False, help="Research harness for ensemble reasoning.")


def _prompt_choice(console: Console, prompt: str, choices: list[str], default: str) -> str:
    normalized_choices = {choice.lower(): choice for choice in choices}
    while True:
        response = typer.prompt(f"{prompt} ({'/'.join(choices)})", default=default)
        normalized = response.strip().lower()
        if normalized in normalized_choices:
            return normalized_choices[normalized]
        console.print(f"[red]Invalid choice:[/red] {response}. Choose from {', '.join(choices)}.")


def _prompt_csv(console: Console, prompt: str, default: str) -> list[str]:
    while True:
        response = typer.prompt(prompt, default=default)
        items = [item.strip() for item in response.split(",") if item.strip()]
        items = dedupe_preserve_order(items)
        if items:
            return items
        console.print("[red]Please provide at least one value.[/red]")


def _prompt_int(console: Console, prompt: str, default: int, min_value: int = 1) -> int:
    while True:
        response = typer.prompt(prompt, default=str(default))
        try:
            value = int(response)
        except ValueError:
            console.print("[red]Please enter an integer.[/red]")
            continue
        if value < min_value:
            console.print(f"[red]Value must be >= {min_value}.[/red]")
            continue
        return value


def _prompt_float(console: Console, prompt: str, default: float) -> float:
    while True:
        response = typer.prompt(prompt, default=str(default))
        try:
            return float(response)
        except ValueError:
            console.print("[red]Please enter a number.[/red]")


def _prompt_float_list(console: Console, prompt: str, default: str) -> list[float]:
    while True:
        response = typer.prompt(prompt, default=default)
        parts = [part.strip() for part in response.split(",") if part.strip()]
        if not parts:
            console.print("[red]Please enter one or more numbers.[/red]")
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError:
            console.print("[red]Invalid list; use comma-separated numbers.[/red]")
            continue
        return values


@app.command()
def run() -> None:
    """Interactive wizard to create a run folder and resolved config."""
    console = Console()
    console.print("[bold]arbiter run[/bold] - configuration wizard")

    started_at = datetime.now(timezone.utc)

    run_name_input = typer.prompt("Run name (optional)", default="", show_default=False)
    run_name = run_name_input.strip() or "auto"
    run_slug = slugify_run_name(run_name)

    heterogeneity_rung = _prompt_choice(console, "Heterogeneity rung", ["H0", "H1", "H2"], "H1")
    models = _prompt_csv(console, "Model identifiers (comma-separated)", "model-1")
    trials_per_question = _prompt_int(console, "Trials per question", 16, min_value=1)

    temp_kind = _prompt_choice(console, "Temperature policy", ["fixed", "list"], "fixed")
    if temp_kind == "fixed":
        temperatures = [_prompt_float(console, "Temperature", 0.7)]
        temperature_policy = TemperaturePolicy(kind="fixed", temperatures=temperatures)
    else:
        temperatures = _prompt_float_list(console, "Temperatures (comma-separated)", "0.7,1.0")
        temperature_policy = TemperaturePolicy(kind="list", temperatures=temperatures)

    persona_bank_input = typer.prompt("Persona bank path (optional)", default="", show_default=False)
    persona_bank_path = persona_bank_input.strip() or None
    selection_mode = "none"
    persona_ids: list[str] = []
    persona_loaded = False

    if persona_bank_path:
        selection_mode = _prompt_choice(
            console,
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
                console.print(f"[yellow]Warning:[/yellow] {load_result.error}; using default persona.")

    persona_policy = PersonaPolicy(
        persona_bank_path=persona_bank_path,
        selection_mode=selection_mode,
        persona_ids=persona_ids,
        loaded=persona_loaded,
    )

    persona_ids_for_q = persona_ids if selection_mode == "sample_uniform" else None
    q_distribution = build_q_distribution(models, temperatures, persona_ids_for_q)

    planned_total_trials = len(q_distribution.atoms) * trials_per_question
    max_calls = _prompt_int(
        console,
        "Max model calls (per question)",
        planned_total_trials,
        min_value=1,
    )
    budget_guardrail = BudgetGuardrail(max_calls=max_calls, scope="per_question")

    output_base_dir_input = typer.prompt("Output base directory", default="./runs")
    output_base_dir = str(Path(output_base_dir_input.strip() or "./runs").expanduser())

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
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

    console.print(f"Run folder: {run_dir}")
    console.print("Wrote config.resolved.json and manifest.json")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
