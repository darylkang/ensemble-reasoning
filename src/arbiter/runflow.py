"""Shared run setup helpers for Arbiter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid
import random

from arbiter.config import (
    BudgetGuardrail,
    ClusteringConfig,
    ConvergenceConfig,
    ExecutionConfig,
    LLMConfig,
    LLMRequestDefaults,
    ProtocolConfig,
    ResolvedConfig,
    SummarizerConfig,
    TrialBudget,
    SUMMARIZER_PROMPT_VERSION,
    default_embedding_model,
    default_summarizer_model,
    materialize_q_distribution,
    slugify_run_name,
)
import platform

from arbiter.manifest import Manifest, get_git_info, platform_info
from arbiter.storage import compute_hash, create_run_dir, write_json


@dataclass(frozen=True)
class RunSetup:
    run_dir: Path
    resolved_config: ResolvedConfig
    question_record: dict[str, Any]
    config_hash: str
    semantic_config_hash: str
    planned_total_trials: int
    planned_call_budget: int
    run_id: str
    started_at: datetime
    run_name: str
    output_base_dir: str
    notes: list[str]


def prepare_run(
    *,
    input_config: dict[str, Any],
    run_name: str,
    output_base_dir: Path,
    default_model: str,
    api_key_present: bool,
    selected_mode: str,
) -> RunSetup:
    started_at = datetime.now(timezone.utc)

    question_record = _build_question_record(input_config)
    llm_config, llm_mode_label, llm_notes = _build_llm_config(
        input_config=input_config,
        default_model=default_model,
        api_key_present=api_key_present,
        selected_mode=selected_mode,
    )

    q_distribution, temperature_policy, persona_policy, model_slugs = materialize_q_distribution(input_config)

    protocol_config = _build_protocol_config(input_config)
    clustering_config = _build_clustering_config(input_config)
    summarizer_config = _build_summarizer_config(input_config)
    execution_config = _build_execution_config(input_config)

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

    run_slug = slugify_run_name(run_name or "auto")
    output_base_dir_str = str(output_base_dir)

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = create_run_dir(Path(output_base_dir_str), timestamp, run_slug)

    resolved_config = ResolvedConfig.build_from_wizard_inputs(
        schema_version="0.9",
        run_name=run_name or "auto",
        run_slug=run_slug,
        run_id=run_id,
        output_base_dir=output_base_dir_str,
        output_dir=str(run_dir),
        started_at=started_at.isoformat(),
        heterogeneity_rung=heterogeneity_rung,
        models=model_slugs,
        llm=llm_config,
        protocol=protocol_config,
        clustering=clustering_config,
        summarizer=summarizer_config,
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
    input_config.setdefault("question", {})["id"] = question_record["question_id"]
    input_config.setdefault("execution", {})
    input_config["execution"].setdefault("seed", execution_config.seed)
    input_config["execution"].setdefault("parse_failure_policy", execution_config.parse_failure_policy)

    write_json(run_dir / "config.input.json", input_config)
    write_json(run_dir / "config.resolved.json", resolved)

    return RunSetup(
        run_dir=run_dir,
        resolved_config=resolved_config,
        question_record=question_record,
        config_hash=config_hash,
        semantic_config_hash=semantic_config_hash,
        planned_total_trials=planned_total_trials,
        planned_call_budget=max_calls,
        run_id=run_id,
        started_at=started_at,
        run_name=run_name or "auto",
        output_base_dir=output_base_dir_str,
        notes=llm_notes,
    )


def write_manifest(*, setup: RunSetup, ended_at: datetime, execution_result: Any | None = None) -> None:
    git_info = get_git_info(Path.cwd())
    llm_call_count = int(getattr(execution_result, "llm_call_count", 0) or 0)
    embedding_call_count = int(getattr(execution_result, "embedding_call_count", 0) or 0)
    summarizer_call_count = int(getattr(execution_result, "summarizer_call_count", 0) or 0)
    semantic = setup.resolved_config.semantic
    manifest = Manifest(
        run_id=setup.run_id,
        started_at=setup.started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        git_sha=git_info.sha,
        git_dirty=git_info.dirty,
        python_version=platform.python_version(),
        platform=platform_info(),
        config_hash=setup.config_hash,
        semantic_config_hash=setup.semantic_config_hash,
        embedding_model=semantic.clustering.embedding_model,
        summarizer_model=semantic.summarizer.model,
        summarizer_prompt_version=semantic.summarizer.prompt_version,
        planned_call_budget=setup.planned_call_budget,
        planned_call_budget_scope="per_question",
        planned_total_trials=setup.planned_total_trials,
        planned_total_trials_scope="per_question",
        llm_call_count=llm_call_count,
        embedding_call_count=embedding_call_count,
        summarizer_call_count=summarizer_call_count,
        execution_seed=semantic.execution.seed,
    )
    write_json(setup.run_dir / "manifest.json", manifest.to_dict())


def collect_top_clusters(run_dir: Path, top_n: int = 3) -> list[tuple[str, float, str]]:
    aggregates_path = run_dir / "aggregates.json"
    parsed_path = run_dir / "parsed.jsonl"
    if not aggregates_path.exists() or not parsed_path.exists():
        return []
    try:
        aggregates = json.loads(aggregates_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    distribution = aggregates.get("distribution_by_cluster_id") or {}
    if not isinstance(distribution, dict) or not distribution:
        return []

    sorted_clusters = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:top_n]
    cluster_ids = [cluster_id for cluster_id, _ in sorted_clusters]
    exemplars: dict[str, str] = {}
    try:
        for line in parsed_path.read_text(encoding="utf-8").splitlines():
            if len(exemplars) == len(cluster_ids):
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            cluster_id = payload.get("cluster_id")
            outcome = payload.get("outcome")
            if cluster_id in cluster_ids and cluster_id not in exemplars and isinstance(outcome, str):
                exemplars[cluster_id] = _truncate_text(outcome, 60)
    except OSError:
        return []

    result: list[tuple[str, float, str]] = []
    for cluster_id, share in sorted_clusters:
        exemplar = exemplars.get(cluster_id, "n/a")
        result.append((cluster_id, float(share), exemplar))
    return result


def collect_last_checkpoints(run_dir: Path, limit: int = 3) -> list[dict[str, str]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    trace = metrics.get("convergence_trace") or []
    if not isinstance(trace, list) or not trace:
        return []
    rows: list[dict[str, str]] = []
    for entry in trace[-limit:]:
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "Batch": str(entry.get("batch_index")),
                "Trials": str(entry.get("trials_completed_total")),
                "Clusters": str(len(entry.get("counts_by_cluster_id") or {})),
                "JS": f"{entry.get('js_divergence'):.3f}" if entry.get("js_divergence") is not None else "n/a",
                "New": f"{entry.get('new_cluster_rate'):.3f}" if entry.get("new_cluster_rate") is not None else "n/a",
                "CI HW": f"{entry.get('top_ci_half_width'):.3f}" if entry.get("top_ci_half_width") is not None else "n/a",
            }
        )
    return rows


def _build_question_record(input_config: dict[str, Any]) -> dict[str, Any]:
    question_block = input_config.get("question", {}) or {}
    question_text = str(question_block.get("text", "")).strip()
    if not question_text:
        raise ValueError("Question text is required.")
    question_id = question_block.get("id") or _build_question_id(question_text)
    question_block["id"] = question_id
    question_block.setdefault("metadata", {})
    return {
        "question_id": question_id,
        "question_text": question_text,
        "metadata": question_block.get("metadata") or {},
    }


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
    method = str(clustering.get("method", "leader"))
    tau = float(clustering.get("tau", 0.85))
    embed_text = str(clustering.get("embed_text", "outcome"))
    embedding_model = str(clustering.get("embedding_model") or default_embedding_model())
    return ClusteringConfig(
        method=method,
        tau=tau,
        embed_text=embed_text,
        embedding_model=embedding_model,
    )


def _build_summarizer_config(input_config: dict[str, object]) -> SummarizerConfig:
    summarizer = input_config.get("summarizer", {}) or {}
    enabled = bool(summarizer.get("enabled", False))
    model = str(summarizer.get("model") or default_summarizer_model())
    prompt_version = str(summarizer.get("prompt_version") or SUMMARIZER_PROMPT_VERSION)
    return SummarizerConfig(enabled=enabled, model=model, prompt_version=prompt_version)


def _build_execution_config(input_config: dict[str, object]) -> ExecutionConfig:
    execution = input_config.get("execution", {}) or {}
    convergence = input_config.get("convergence", {}) or {}

    worker_count = int(execution.get("workers", 8))
    batch_size = int(execution.get("batch_size", worker_count))
    max_retries = int(execution.get("retries", 2))
    seed_value = execution.get("seed")
    if seed_value is None:
        seed = random.SystemRandom().randrange(2**32)
    else:
        seed = int(seed_value)
    parse_failure_policy = str(execution.get("parse_failure_policy", "continue")).lower()
    if parse_failure_policy not in {"continue", "halt"}:
        parse_failure_policy = "continue"

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
        seed=seed,
        parse_failure_policy=parse_failure_policy,
        convergence=convergence_config,
    )


def _build_llm_config(
    *,
    input_config: dict[str, object],
    default_model: str,
    api_key_present: bool,
    selected_mode: str,
) -> tuple[LLMConfig, str, list[str]]:
    notes: list[str] = []
    llm = input_config.setdefault("llm", {})
    mode_input = str(llm.get("mode") or selected_mode or ("remote" if api_key_present else "mock")).lower()
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


def _truncate_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"
