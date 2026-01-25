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
    ClusteringConfig,
    ConvergenceConfig,
    ExecutionConfig,
    LLMConfig,
    LLMRequestDefaults,
    ProtocolConfig,
    ResolvedConfig,
    TrialBudget,
    materialize_q_distribution,
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
    render_step_header,
    render_success,
    render_summary_table,
    render_warning,
    render_validation_panel,
)
from arbiter.wizard import WizardState, run_wizard as run_wizard_flow
from arbiter.validation import load_and_validate_config

app = typer.Typer(add_completion=False, help="Research harness for ensemble reasoning.")
llm_app = typer.Typer(add_completion=False, help="LLM utilities and diagnostics.")
config_app = typer.Typer(add_completion=False, help="Config helpers and validation.")
app.add_typer(llm_app, name="llm")
app.add_typer(config_app, name="config")


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
    state = WizardState(
        default_model=default_model,
        api_key_present=api_key_present,
        config_path=Path("arbiter.config.json"),
    )
    state = run_wizard_flow(state)

    input_config = state.input_config
    question_block = input_config.get("question", {}) or {}
    question_text = str(question_block.get("text", "")).strip()
    if not question_text:
        render_error("Question text is required.")
        raise typer.Exit(code=1)
    question_id = question_block.get("id") or _build_question_id(question_text)
    question_block["id"] = question_id
    question_block.setdefault("metadata", {})
    question_record = {
        "question_id": question_id,
        "question_text": question_text,
        "metadata": question_block.get("metadata") or {},
    }

    llm_config, llm_mode_label, llm_notes = _build_llm_config(
        input_config=input_config,
        default_model=default_model,
        api_key_present=api_key_present,
    )

    step_total = len(state.step_order) + 2
    render_step_header(
        len(state.step_order) + 1,
        step_total,
        "Resolve configuration",
        "Materialize Q(c) and write run artifacts.",
    )
    with status_spinner("Materializing Q(c)"):
        q_distribution, temperature_policy, persona_policy, model_slugs = materialize_q_distribution(input_config)
    render_success(f"Q(c) materialized with {len(q_distribution.atoms)} atoms.")
    render_gap(after_prompt=False)

    protocol_config = _build_protocol_config(input_config)
    clustering_config = _build_clustering_config(input_config)
    execution_config = _build_execution_config(input_config)
    worker_count = execution_config.worker_count
    batch_size = execution_config.batch_size
    k_max = int(input_config.get("execution", {}).get("k_max", 1000))
    trial_budget = TrialBudget(k_max=k_max, scope="per_question")
    max_calls = k_max
    planned_total_trials = min(k_max, max_calls)
    budget_guardrail = BudgetGuardrail(max_calls=max_calls, scope="per_question")
    heterogeneity_rung = _infer_rung(
        model_slugs,
        persona_policy.persona_ids,
        temperature_policy.kind,
        protocol_config.type,
    )

    run_name = state.run_name.strip() or "auto"
    run_slug = slugify_run_name(run_name)
    output_base_dir = str(state.output_base_dir)

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = create_run_dir(Path(output_base_dir), timestamp, run_slug)
    with status_spinner("Writing run artifacts"):
        try:
            resolved_config = ResolvedConfig.build_from_wizard_inputs(
                schema_version="0.7",
                run_name=run_name,
                run_slug=run_slug,
                run_id=run_id,
                output_base_dir=output_base_dir,
                output_dir=str(run_dir),
                started_at=started_at.isoformat(),
                heterogeneity_rung=heterogeneity_rung,
                models=model_slugs,
                llm=llm_config,
                protocol=protocol_config,
                clustering=clustering_config,
                execution=execution_config,
                temperature_policy=temperature_policy,
                personas=persona_policy,
                trial_budget=trial_budget,
                budget_guardrail=budget_guardrail,
                q_distribution=q_distribution,
                notes=llm_notes,
            )

            resolved = resolved_config.to_dict()
            config_hash = compute_hash(resolved)
            semantic_config_hash = compute_hash(resolved_config.semantic.to_dict())

            input_config.setdefault("llm", {})
            input_config["llm"]["mode"] = llm_mode_label
            input_config["llm"].setdefault("model", llm_config.model)
            input_config.setdefault("question", {})["id"] = question_id

            write_json(run_dir / "config.input.json", input_config)
            write_json(run_dir / "config.resolved.json", resolved)
        except Exception as exc:
            cleanup_run_dir(run_dir)
            render_error(f"Failed to write run artifacts: {exc}")
            raise typer.Exit(code=1) from exc

    render_success("Run config written.")
    render_gap(after_prompt=False)

    render_step_header(
        len(state.step_order) + 2,
        step_total,
        "Execute trials",
        "Run batched trials with convergence checks.",
    )
    execution_result = None
    execution_error = None
    try:
        execution_result = asyncio.run(
            execute_trials(run_dir=run_dir, resolved_config=resolved_config, question=question_record)
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
        planned_call_budget_scope="per_question",
        planned_total_trials=planned_total_trials,
        planned_total_trials_scope="per_question",
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
    if execution_result.top_mode_id and execution_result.top_p is not None:
        top_summary = f"{execution_result.top_mode_id} ({execution_result.top_p:.3f}, CI {ci_summary})"

    summary = {
        "Run folder": str(run_dir),
        "Question": question_id,
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
        "Top mode": top_summary,
    }
    render_summary_table(summary)
    render_info(f"Artifacts written to {run_dir}")

    if execution_result.stop_reason in {"parse_failure", "llm_error", "budget_exhausted"}:
        render_error("Execution stopped due to errors. Review metrics.json for details.")
        raise typer.Exit(code=1)


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
        render_validation_panel("INVALID", errors, style="error")
        raise typer.Exit(code=1)

    if warnings:
        render_validation_panel("VALID (with warnings)", warnings, style="warning")
    else:
        render_validation_panel("VALID", ["No issues found."], style="success")


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


def _build_question_id(question_text: str) -> str:
    digest = hashlib.sha256(question_text.encode("utf-8")).hexdigest()
    return f"question_{digest[:10]}"


def _infer_rung(
    models: list[str],
    persona_ids: list[str],
    temperature_kind: str,
    protocol_type: str,
) -> str:
    if protocol_type != "independent":
        return "H4"
    if len(models) > 1:
        return "H3"
    if persona_ids:
        return "H2"
    if temperature_kind != "fixed":
        return "H1"
    return "H0"


def _build_protocol_config(input_config: dict[str, object]) -> ProtocolConfig:
    protocol = input_config.get("protocol", {}) or {}
    protocol_type = str(protocol.get("type", "independent"))
    return ProtocolConfig(type=protocol_type)


def _build_clustering_config(input_config: dict[str, object]) -> ClusteringConfig:
    clustering = input_config.get("clustering", {}) or {}
    method = str(clustering.get("method", "hash_baseline"))
    tau = float(clustering.get("tau", 0.85))
    embed_text = str(clustering.get("embed_text", "outcome+rationale"))
    return ClusteringConfig(method=method, tau=tau, embed_text=embed_text)


def _build_execution_config(input_config: dict[str, object]) -> ExecutionConfig:
    execution = input_config.get("execution", {}) or {}
    convergence = input_config.get("convergence", {}) or {}

    worker_count = int(execution.get("workers", 8))
    batch_size = int(execution.get("batch_size", worker_count))
    max_retries = int(execution.get("retries", 2))

    epsilon_ci = convergence.get("epsilon_ci_half_width", 0.05)
    if epsilon_ci is not None:
        epsilon_ci = float(epsilon_ci)
        if epsilon_ci <= 0:
            epsilon_ci = None

    convergence_config = ConvergenceConfig(
        delta_js_threshold=float(convergence.get("delta_js_threshold", 0.02)),
        epsilon_new_threshold=float(convergence.get("epsilon_new_threshold", 0.01)),
        epsilon_ci_half_width=epsilon_ci,
        min_trials=int(convergence.get("min_trials", 64)),
        patience_batches=int(convergence.get("patience_batches", 2)),
    )
    return ExecutionConfig(
        worker_count=max(1, worker_count),
        batch_size=max(1, batch_size),
        max_retries=max(0, max_retries),
        convergence=convergence_config,
    )


def _build_llm_config(
    *,
    input_config: dict[str, object],
    default_model: str,
    api_key_present: bool,
) -> tuple[LLMConfig, str, list[str]]:
    notes: list[str] = []
    llm = input_config.setdefault("llm", {})
    mode_input = str(llm.get("mode") or ("remote" if api_key_present else "mock")).lower()
    if mode_input in {"remote", "openrouter"}:
        resolved_mode = "openrouter"
        mode_label = "remote"
    elif mode_input == "mock":
        resolved_mode = "mock"
        mode_label = "mock"
    else:
        resolved_mode = "mock"
        mode_label = "mock"

    if resolved_mode == "openrouter" and not api_key_present:
        resolved_mode = "mock"
        mode_label = "mock"
        notes.append("OpenRouter API key missing; LLM mode set to mock.")

    model = str(llm.get("model") or default_model)
    llm["model"] = model
    llm["mode"] = mode_label

    request_defaults = llm.get("request_defaults") or {}
    routing_defaults = llm.get("routing_defaults")
    if routing_defaults is None:
        routing_defaults = {"allow_fallbacks": False}
    extra_body_defaults = llm.get("extra_body_defaults") or {}

    llm["request_defaults"] = request_defaults
    llm["routing_defaults"] = routing_defaults
    llm["extra_body_defaults"] = extra_body_defaults

    request_defaults_obj = LLMRequestDefaults(
        temperature=request_defaults.get("temperature"),
        top_p=request_defaults.get("top_p"),
        max_tokens=request_defaults.get("max_tokens"),
        seed=request_defaults.get("seed"),
        stop=request_defaults.get("stop"),
        response_format=request_defaults.get("response_format"),
        tools=request_defaults.get("tools"),
        tool_choice=request_defaults.get("tool_choice"),
        parallel_tool_calls=request_defaults.get("parallel_tool_calls"),
    )
    llm_config = LLMConfig(
        client="openrouter",
        mode=resolved_mode,
        model=model,
        request_defaults=request_defaults_obj,
        routing_defaults=dict(routing_defaults),
        extra_body_defaults=dict(extra_body_defaults),
    )
    return llm_config, mode_label, notes


def main() -> None:
    app()


if __name__ == "__main__":
    main()
