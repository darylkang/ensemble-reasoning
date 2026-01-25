"""Wizard steps and registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import typer

from arbiter.config import default_canonical_config
from arbiter.ui.render import (
    render_error,
    render_info,
    render_notice,
    render_step_header,
    render_summary_table,
    render_warning,
)
from arbiter.storage import write_json
from arbiter.validation import load_and_validate_config
from arbiter.wizard.state import WizardState


@dataclass(frozen=True)
class Step:
    step_id: str
    title: str
    description: str
    handler: Callable[[WizardState], None]


_REGISTRY: dict[str, Step] = {}


def register_step(step: Step) -> None:
    _REGISTRY[step.step_id] = step


def get_step(step_id: str) -> Step:
    return _REGISTRY[step_id]


def run_step(step_id: str, state: WizardState) -> None:
    step = get_step(step_id)
    index, total = state.step_index(step.step_id)
    render_step_header(index, total, step.title, step.description)
    step.handler(state)


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    normalized_choices = {choice.lower(): choice for choice in choices}
    while True:
        response = typer.prompt(f"{prompt} ({'/'.join(choices)})", default=default)
        normalized = response.strip().lower()
        if normalized in normalized_choices:
            return normalized_choices[normalized]
        render_warning(f"Invalid choice: {response}. Choose from {', '.join(choices)}.")


def _prompt_text(prompt: str) -> str:
    while True:
        response = typer.prompt(prompt, default="", show_default=False)
        if response.strip():
            return response
        render_warning("Question text is required.")


def _prompt_csv(prompt: str, default: str, *, allow_empty: bool = False) -> list[str]:
    while True:
        response = typer.prompt(prompt, default=default)
        items = [item.strip() for item in response.split(",") if item.strip()]
        if items or allow_empty:
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


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_value = "y" if default else "n"
    while True:
        response = typer.prompt(f"{prompt} (y/n)", default=default_value)
        normalized = response.strip().lower()
        if normalized in {"y", "yes"}:
            return True
        if normalized in {"n", "no"}:
            return False
        render_warning("Please enter y or n.")


def _load_config_file(path: Path, state: WizardState) -> dict[str, Any]:
    result = load_and_validate_config(
        path,
        default_model=state.default_model,
        llm_mode=_default_llm_mode(state),
    )
    if result.errors:
        message_lines = [f"{issue.path}: {issue.message}" for issue in result.errors]
        raise ValueError("Invalid config\n" + "\n".join(message_lines))
    if result.warnings:
        warning_lines = [f"{issue.path}: {issue.message}" for issue in result.warnings]
        render_warning("Config warnings:\n" + "\n".join(warning_lines))
    if result.config is None:
        raise ValueError("Failed to load config.")
    return result.config


def _default_llm_mode(state: WizardState) -> str:
    return "remote" if state.api_key_present else "mock"


def step_welcome(state: WizardState) -> None:
    if not state.api_key_present:
        render_notice("OpenRouter API key not found. Remote calls are disabled; using mock client.")
    else:
        render_info("OpenRouter API key detected. Remote calls are enabled.")


def step_config_mode(state: WizardState) -> None:
    config_exists = state.config_path.exists()
    choices = ["load", "guided", "create"] if config_exists else ["guided", "create"]
    default_choice = "load" if config_exists else "guided"
    selection = _prompt_choice("Config mode", choices, default_choice).lower()

    if selection == "load":
        try:
            state.input_config = _load_config_file(state.config_path, state)
        except ValueError as exc:
            render_error(str(exc))
            action = _prompt_choice("Config invalid. Choose next step", ["guided", "exit"], "guided")
            if action == "exit":
                raise typer.Exit(code=1)
            state.input_config = default_canonical_config(
                default_model=state.default_model,
                llm_mode=_default_llm_mode(state),
            )
            state.config_mode = "guided"
        else:
            state.config_mode = "load"
    elif selection == "create":
        example_path = Path("arbiter.config.example.json")
        if example_path.exists():
            if state.config_path.exists():
                overwrite = _prompt_yes_no("arbiter.config.json exists. Overwrite?", default=False)
                if not overwrite:
                    return step_config_mode(state)
            state.config_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
            render_info(f"Wrote template config to {state.config_path}. Edit it, then rerun `arbiter run` and choose load.")
            raise typer.Exit(code=0)
        else:
            render_warning("arbiter.config.example.json not found; using default template.")
            state.input_config = default_canonical_config(
                default_model=state.default_model,
                llm_mode=_default_llm_mode(state),
            )
            write_json(state.config_path, state.input_config)
            render_info(f"Wrote template config to {state.config_path}. Edit it, then rerun `arbiter run` and choose load.")
            raise typer.Exit(code=0)
    else:
        state.input_config = default_canonical_config(
            default_model=state.default_model,
            llm_mode=_default_llm_mode(state),
        )
        state.config_mode = "guided"

    if not state.api_key_present:
        state.input_config.setdefault("llm", {})
        state.input_config["llm"]["mode"] = "mock"

    state.compute_step_order()


def step_question(state: WizardState) -> None:
    question = state.input_config.setdefault("question", {})
    if question.get("text"):
        render_info("Question text loaded from config.")
        return
    text = _prompt_text("Question text (single line; use \\n for newlines)")
    question["text"] = text


def step_decode(state: WizardState) -> None:
    q = state.input_config.setdefault("q", {})
    decode = q.setdefault("decode", {})
    temp_type = _prompt_choice("Temperature policy", ["fixed", "range"], "fixed")
    if temp_type == "fixed":
        value = _prompt_float("Temperature", 0.7)
        decode["temperature"] = {"type": "fixed", "value": value}
    else:
        temp_min = _prompt_float("Temperature min", 0.2)
        temp_max = _prompt_float("Temperature max", 1.0)
        if temp_max < temp_min:
            temp_min, temp_max = temp_max, temp_min
        decode["temperature"] = {"type": "range", "min": temp_min, "max": temp_max}
    decode.setdefault("extra", {})


def step_personas(state: WizardState) -> None:
    q = state.input_config.setdefault("q", {})
    personas = q.setdefault("personas", {})
    items = _prompt_csv("Persona ids (comma-separated, blank for none)", "", allow_empty=True)
    persona_items = []
    for item in items:
        if item:
            persona_items.append({"id": item, "weight": 1.0})
    personas["items"] = persona_items
    personas.setdefault("default_behavior", "neutral_if_empty")


def step_models(state: WizardState) -> None:
    q = state.input_config.setdefault("q", {})
    models = q.setdefault("models", {})
    items = _prompt_csv("Model slugs (comma-separated)", state.default_model)
    model_items = []
    for item in items:
        if item:
            model_items.append({"slug": item, "weight": 1.0})
    models["items"] = model_items


def step_protocol(state: WizardState) -> None:
    protocol = state.input_config.setdefault("protocol", {})
    protocol_type = _prompt_choice("Protocol", ["independent"], "independent")
    protocol["type"] = protocol_type


def step_advanced(state: WizardState) -> None:
    execution = state.input_config.setdefault("execution", {})
    convergence = state.input_config.setdefault("convergence", {})
    clustering = state.input_config.setdefault("clustering", {})

    execution["k_max"] = _prompt_int("Max trials (K_max)", int(execution.get("k_max", 1000)), min_value=1)
    execution["workers"] = _prompt_int("Worker concurrency (W)", int(execution.get("workers", 8)), min_value=1)
    execution["batch_size"] = _prompt_int("Batch size (B)", int(execution.get("batch_size", execution["workers"])), min_value=1)
    execution["retries"] = _prompt_int("Max parse retries", int(execution.get("retries", 2)), min_value=0)

    convergence["delta_js_threshold"] = _prompt_float(
        "JS divergence threshold (delta_js)", float(convergence.get("delta_js_threshold", 0.02))
    )
    convergence["epsilon_new_threshold"] = _prompt_float(
        "New mode rate threshold (epsilon_new)", float(convergence.get("epsilon_new_threshold", 0.01))
    )
    convergence["epsilon_ci_half_width"] = _prompt_float(
        "Top-mode CI half-width threshold (epsilon_ci)", float(convergence.get("epsilon_ci_half_width", 0.05))
    )
    convergence["min_trials"] = _prompt_int(
        "Minimum valid trials before early stop", int(convergence.get("min_trials", 64)), min_value=1
    )
    convergence["patience_batches"] = _prompt_int(
        "Patience batches", int(convergence.get("patience_batches", 2)), min_value=1
    )

    clustering["method"] = _prompt_choice("Clustering method", ["hash_baseline", "leader"], clustering.get("method", "hash_baseline"))
    clustering["tau"] = _prompt_float("Clustering threshold (tau)", float(clustering.get("tau", 0.85)))
    clustering["embed_text"] = _prompt_choice(
        "Embed text", ["outcome", "outcome+rationale"], clustering.get("embed_text", "outcome+rationale")
    )


def step_review(state: WizardState) -> None:
    question = state.input_config.get("question", {})
    q = state.input_config.get("q", {})
    models = q.get("models", {}).get("items", [])
    personas = q.get("personas", {}).get("items", [])
    execution = state.input_config.get("execution", {})
    convergence = state.input_config.get("convergence", {})
    clustering = state.input_config.get("clustering", {})

    summary = {
        "Question id": question.get("id") or "(auto)",
        "Question text": (question.get("text") or "").strip()[:60],
        "Models": ", ".join(item.get("slug") for item in models) or "(none)",
        "Personas": ", ".join(item.get("id") for item in personas) or "(none)",
        "K_max": str(execution.get("k_max")),
        "Workers / batch": f"{execution.get('workers')} / {execution.get('batch_size')}",
        "delta_js": str(convergence.get("delta_js_threshold")),
        "epsilon_new": str(convergence.get("epsilon_new_threshold")),
        "epsilon_ci": str(convergence.get("epsilon_ci_half_width")),
        "clustering": str(clustering.get("method")),
    }
    render_summary_table(summary, title="Review")

    if state.config_mode == "guided":
        save = _prompt_yes_no("Write arbiter.config.json to the working directory?", default=True)
        if save:
            if state.config_path.exists():
                overwrite = _prompt_yes_no("arbiter.config.json exists. Overwrite?", default=False)
                if not overwrite:
                    return
            write_json(state.config_path, state.input_config)
            render_info(f"Wrote {state.config_path}.")


def step_run_setup(state: WizardState) -> None:
    run_name_input = typer.prompt("Run name (optional)", default="", show_default=False)
    state.run_name = run_name_input.strip() or "auto"
    output_base_dir_input = typer.prompt("Output base directory", default=str(state.output_base_dir))
    state.output_base_dir = Path(output_base_dir_input.strip() or "./runs").expanduser()


register_step(Step("welcome", "Welcome", "Environment check and setup.", step_welcome))
register_step(Step("config_mode", "Config mode", "Load a config or run the guided wizard.", step_config_mode))
register_step(Step("question", "Question", "Provide the question text.", step_question))
register_step(Step("decode", "Decode params", "Configure decoding parameters.", step_decode))
register_step(Step("personas", "Persona mix", "Configure persona mix.", step_personas))
register_step(Step("models", "Model mix", "Configure model mix.", step_models))
register_step(Step("protocol", "Protocol", "Select the protocol type.", step_protocol))
register_step(Step("advanced", "Advanced settings", "Execution, convergence, and clustering.", step_advanced))
register_step(Step("review", "Review", "Confirm the configuration.", step_review))
register_step(Step("run_setup", "Run setup", "Set run name and output path.", step_run_setup))
