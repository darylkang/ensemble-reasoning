"""Async execution engine for arbiter trials."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable
from array import array

from arbiter.config import LLMConfig, QAtom, ResolvedConfig
from arbiter.embeddings import create_embeddings_client
from arbiter.clustering import OnlineLeaderClustering
from arbiter.llm.client import build_request_body, create_client
from arbiter.llm.types import LLMRequest
from arbiter.summarizer import summarize_clusters
from arbiter.storage import append_jsonl, write_json


@dataclass(slots=True)
class WorkItem:
    trial_id: str
    batch_index: int
    retries_used: int
    atom: QAtom
    temperature: float
    temperature_policy: dict[str, Any]
    request_seed: int
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class PendingRetry:
    trial_id: str
    retries_used: int
    atom: QAtom
    temperature: float
    temperature_policy: dict[str, Any]
    request_seed: int
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class TrialOutcome:
    trial_record: dict[str, Any]
    parsed_record: dict[str, Any]
    parse_valid: bool
    embedded_text: str | None
    outcome_text: str | None
    rationale: str | None
    trace_summary: str | None
    retry_messages: list[dict[str, Any]] | None
    retries_used: int
    atom: QAtom
    error: str | None


@dataclass(slots=True)
class ExecutionResult:
    stop_reason: str
    stop_at_trials: int
    valid_trials: int
    parse_error_count: int
    batches_completed: int
    converged: bool
    top_cluster_id: str | None
    top_p: float | None
    top_ci_low: float | None
    top_ci_high: float | None
    top_ci_half_width: float | None
    entropy: float | None
    margin: float | None
    disagreement_rate: float | None
    llm_call_count: int
    embedding_call_count: int
    summarizer_call_count: int


async def execute_trials(
    *,
    run_dir: Path,
    resolved_config: ResolvedConfig,
    question: dict[str, Any],
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> ExecutionResult:
    question_text = str(question.get("question_text", ""))
    if not question_text.strip():
        raise ValueError("Question text is required for execution.")
    question_id = str(question.get("question_id", "question"))
    question_hash = _hash_text(question_text)

    question_path = run_dir / "question.json"
    trials_path = run_dir / "trials.jsonl"
    parsed_path = run_dir / "parsed.jsonl"
    embeddings_path = run_dir / "embeddings.jsonl"
    aggregates_path = run_dir / "aggregates.json"
    metrics_path = run_dir / "metrics.json"
    clusters_online_path = run_dir / "clusters_online.json"
    clusters_offline_path = run_dir / "clusters_offline.json"
    cluster_summaries_path = run_dir / "cluster_summaries.json"

    write_json(question_path, question)

    semantic = resolved_config.semantic
    execution = semantic.execution
    trial_budget = semantic.trial_budget
    budget_guardrail = semantic.budget_guardrail
    llm_config = semantic.llm
    if semantic.clustering.method != "leader":
        raise ValueError("Unsupported clustering method for execution.")
    call_cap = min(trial_budget.k_max, budget_guardrail.max_calls)
    worker_count = max(1, execution.worker_count)
    batch_size = max(1, execution.batch_size)
    max_retries = max(0, execution.max_retries)
    convergence = execution.convergence
    temperature_policy_payload = _temperature_policy_payload(semantic.temperature_policy)

    def _emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    rng_seed = int(execution.seed)
    rng = random.Random(rng_seed)

    counts_by_cluster: dict[str, int] = {}
    total_trials = 0
    valid_trials = 0
    parse_valid_count = 0
    parse_error_count = 0
    parse_failure_trial_ids: list[str] = []
    llm_error_count = 0
    embedding_error_count = 0
    embedding_call_count = 0
    summarizer_call_count = 0
    llm_call_count = 0
    convergence_trace: list[dict[str, Any]] = []
    stop_reason: str | None = None
    stop_error: str | None = None

    pending_retries: list[PendingRetry] = []
    trial_counter = 0
    consecutive_converged = 0
    batches_completed = 0
    prev_distribution: dict[str, float] | None = None

    clusterer = OnlineLeaderClustering(tau=semantic.clustering.tau)
    embedding_model = semantic.clustering.embedding_model
    embed_text_policy = semantic.clustering.embed_text
    embeddings_client = create_embeddings_client(
        semantic.llm.mode,
        model=embedding_model,
        base_url=None,
    )

    worker_counts = [0 for _ in range(worker_count)]

    client = create_client(llm_config.mode, default_routing=llm_config.routing_defaults)

    _emit(
        {
            "type": "execution_started",
            "run_id": resolved_config.run.run_id,
            "question_id": question_id,
            "call_cap": call_cap,
            "worker_count": worker_count,
            "batch_size": batch_size,
        }
    )

    async def worker_loop(
        worker_id: int,
        queue: asyncio.Queue[WorkItem | None],
        results: asyncio.Queue[TrialOutcome],
    ) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            _emit(
                {
                    "type": "trial_started",
                    "worker_id": worker_id,
                    "trial_id": item.trial_id,
                    "atom_id": item.atom.atom_id,
                    "model": item.atom.model,
                    "persona_id": item.atom.persona_id,
                    "temperature": item.temperature,
                }
            )
            outcome = await _execute_one_trial(
                item=item,
                worker_id=worker_id,
                run_id=resolved_config.run.run_id,
                question_id=question_id,
                question_hash=question_hash,
                embed_text_policy=embed_text_policy,
                llm_config=llm_config,
                client=client,
            )
            worker_counts[worker_id] += 1
            _emit(
                {
                    "type": "trial_finished",
                    "worker_id": worker_id,
                    "trial_id": item.trial_id,
                    "parse_valid": outcome.parse_valid,
                    "error": outcome.error,
                    "completed": worker_counts[worker_id],
                }
            )
            await results.put(outcome)
            queue.task_done()

    async def run_loop() -> None:
        nonlocal total_trials, valid_trials, parse_valid_count, parse_error_count, llm_error_count
        nonlocal embedding_call_count, embedding_error_count, llm_call_count
        nonlocal stop_reason, stop_error, batches_completed, consecutive_converged
        nonlocal trial_counter, prev_distribution

        queue: asyncio.Queue[WorkItem | None] = asyncio.Queue()
        results: asyncio.Queue[TrialOutcome] = asyncio.Queue()
        workers = [asyncio.create_task(worker_loop(i, queue, results)) for i in range(worker_count)]

        try:
            while total_trials < call_cap and stop_reason is None:
                batches_completed += 1
                batch_index = batches_completed
                remaining = call_cap - total_trials
                batch_target = min(batch_size, remaining)
                batch_items: list[WorkItem] = []

                while len(batch_items) < batch_target:
                    if pending_retries:
                        retry = pending_retries.pop(0)
                        batch_items.append(
                            WorkItem(
                                trial_id=retry.trial_id,
                                batch_index=batch_index,
                                retries_used=retry.retries_used,
                                atom=retry.atom,
                                temperature=retry.temperature,
                                temperature_policy=retry.temperature_policy,
                                request_seed=retry.request_seed,
                                messages=retry.messages,
                            )
                        )
                        continue
                    trial_counter += 1
                    atom = _sample_atom(rng, semantic.q_distribution.atoms)
                    temperature = _sample_temperature(rng, semantic.temperature_policy, atom.temperature)
                    messages = build_messages(question_text, atom.persona_id)
                    request_seed = (rng_seed + trial_counter) % (2**32)
                    batch_items.append(
                        WorkItem(
                            trial_id=f"trial_{trial_counter:06d}",
                            batch_index=batch_index,
                            retries_used=0,
                            atom=atom,
                            temperature=temperature,
                            temperature_policy=temperature_policy_payload,
                            request_seed=request_seed,
                            messages=messages,
                        )
                    )

                for item in batch_items:
                    await queue.put(item)

                batch_results: list[TrialOutcome] = []
                valid_batch: list[TrialOutcome] = []
                for _ in range(len(batch_items)):
                    outcome = await results.get()
                    batch_results.append(outcome)

                batch_results_sorted = sorted(batch_results, key=lambda item: item.trial_record["trial_id"])
                for outcome in batch_results_sorted:
                    total_trials += 1
                    llm_call_count += 1
                    append_jsonl(trials_path, outcome.trial_record)

                    if outcome.parse_valid and outcome.embedded_text and outcome.outcome_text:
                        parse_valid_count += 1
                        valid_batch.append(outcome)
                    else:
                        parse_error_count += 1
                        parse_failure_trial_ids.append(outcome.trial_record["trial_id"])
                        append_jsonl(parsed_path, outcome.parsed_record)

                        if outcome.error:
                            llm_error_count += 1
                            stop_reason = "llm_error"
                            stop_error = outcome.error
                        elif execution.parse_failure_policy == "halt":
                            stop_reason = "parse_failure"
                        elif outcome.retry_messages and outcome.retries_used < max_retries:
                            if total_trials < call_cap:
                                trial_counter += 1
                                request_seed = (rng_seed + trial_counter) % (2**32)
                                pending_retries.append(
                                    PendingRetry(
                                        trial_id=f"trial_{trial_counter:06d}",
                                        retries_used=outcome.retries_used + 1,
                                        atom=outcome.atom,
                                        temperature=outcome.trial_record.get("temperature", outcome.atom.temperature),
                                        temperature_policy=outcome.trial_record.get("temperature_policy", temperature_policy_payload),
                                        request_seed=request_seed,
                                        messages=outcome.retry_messages,
                                    )
                                )
                            else:
                                stop_reason = "budget_exhausted"

                    _emit(
                        {
                            "type": "progress",
                            "completed": total_trials,
                            "valid_trials": valid_trials,
                            "call_cap": call_cap,
                        }
                    )

                _emit(
                    {
                        "type": "batch_complete",
                        "batch_index": batch_index,
                        "trials_completed": total_trials,
                        "valid_trials": valid_trials,
                    }
                )
                new_clusters_in_batch = 0
                if valid_batch:
                    valid_batch_sorted = sorted(valid_batch, key=lambda item: item.trial_record["trial_id"])
                    texts = [item.embedded_text or "" for item in valid_batch_sorted]
                    try:
                        embedding_results = await embeddings_client.embed(texts)
                        embedding_call_count += 1
                    except Exception as exc:  # noqa: BLE001
                        embedding_error_count += 1
                        stop_reason = "embedding_error"
                        stop_error = str(exc)
                        embedding_results = []
                    if stop_reason is None and len(embedding_results) != len(valid_batch_sorted):
                        stop_reason = "embedding_error"
                        stop_error = "Embedding result count mismatch."
                    if stop_reason is None:
                        for outcome, embedding in zip(valid_batch_sorted, embedding_results):
                            cluster_id, is_new = clusterer.assign(
                                embedding.embedding,
                                trial_id=outcome.trial_record["trial_id"],
                                outcome=outcome.outcome_text or "",
                                rationale=outcome.rationale,
                            )
                            if is_new:
                                new_clusters_in_batch += 1
                            outcome.parsed_record["cluster_id"] = cluster_id
                            append_jsonl(parsed_path, outcome.parsed_record)
                            counts_by_cluster[cluster_id] = counts_by_cluster.get(cluster_id, 0) + 1
                            valid_trials += 1
                            embedding_record = _embedding_record(
                                trial_id=outcome.trial_record["trial_id"],
                                text=outcome.embedded_text or "",
                                embedding=embedding.embedding,
                                dims=embedding.dims,
                                model=embedding.model,
                                raw=embedding.raw,
                            )
                            append_jsonl(embeddings_path, embedding_record)

                new_cluster_rate = (
                    new_clusters_in_batch / len(valid_batch) if valid_batch else 0.0
                )

                distribution = _distribution(counts_by_cluster, valid_trials)
                js_divergence = _js_divergence(prev_distribution, distribution)
                top_cluster_id = _top_cluster_id(counts_by_cluster)
                top_p = distribution.get(top_cluster_id) if top_cluster_id else None
                top_ci = _wilson_ci(counts_by_cluster.get(top_cluster_id, 0), valid_trials) if top_cluster_id else None
                entry = _convergence_entry(
                    batch_index=batch_index,
                    total_trials=total_trials,
                    valid_trials=valid_trials,
                    counts_by_cluster=counts_by_cluster,
                    distribution_by_cluster=distribution,
                    top_cluster_id=top_cluster_id,
                    top_p=top_p,
                    top_ci=top_ci,
                    js_divergence=js_divergence,
                    new_cluster_rate=new_cluster_rate,
                )
                converged_candidate = _converged_candidate(
                    entry=entry,
                    min_trials=convergence.min_trials,
                    delta_js_threshold=convergence.delta_js_threshold,
                    epsilon_new_threshold=convergence.epsilon_new_threshold,
                    epsilon_ci_half_width=convergence.epsilon_ci_half_width,
                )

                if stop_reason is None:
                    if converged_candidate:
                        consecutive_converged += 1
                    else:
                        consecutive_converged = 0

                    if (
                        valid_trials >= convergence.min_trials
                        and converged_candidate
                        and consecutive_converged >= convergence.patience_batches
                        and not pending_retries
                    ):
                        stop_reason = "converged"
                    elif total_trials >= call_cap:
                        stop_reason = "max_trials_reached"

                convergence_trace.append(entry)
                checkpoint_row = _checkpoint_row(entry, stop_reason)
                _emit(
                    {
                        "type": "batch_checkpoint",
                        "batch_index": batch_index,
                        "row": checkpoint_row,
                        "stop_reason": stop_reason,
                        "entry": entry,
                    }
                )

                if stop_reason is not None:
                    break

                prev_distribution = distribution
        finally:
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers, return_exceptions=True)

    try:
        await run_loop()
    finally:
        await client.aclose()
        await embeddings_client.aclose()

    if stop_reason is None:
        stop_reason = "max_trials_reached"

    clusters_payload = _clusters_online_payload(
        clusterer=clusterer,
        method=semantic.clustering.method,
        tau=semantic.clustering.tau,
        embedding_model=embedding_model,
        embed_text=embed_text_policy,
    )
    write_json(clusters_online_path, clusters_payload)
    write_json(
        clusters_offline_path,
        {"status": "not_run", "reason": "not_implemented"},
    )

    if semantic.summarizer.enabled:
        summaries_payload, summarizer_call_count = await summarize_clusters(
            clusters=clusters_payload.get("clusters", []),
            llm_mode=semantic.llm.mode,
            model_slug=semantic.summarizer.model,
            prompt_version=semantic.summarizer.prompt_version,
        )
    else:
        summaries_payload = {"status": "not_run", "reason": "disabled"}
    write_json(cluster_summaries_path, summaries_payload)

    distribution = _distribution(counts_by_cluster, valid_trials)
    top_cluster_id = _top_cluster_id(counts_by_cluster)
    top_p = distribution.get(top_cluster_id) if top_cluster_id else None
    top_ci = _wilson_ci(counts_by_cluster.get(top_cluster_id, 0), valid_trials) if top_cluster_id else None
    entropy = _entropy(distribution) if valid_trials > 0 else None
    margin = _margin(distribution) if valid_trials > 0 else None
    disagreement_rate = 1.0 - top_p if top_p is not None else None
    eff_num_clusters = _effective_num_clusters(distribution) if valid_trials > 0 else None

    aggregates_payload = {
        "question_id": question_id,
        "discovered_cluster_count": len(counts_by_cluster),
        "counts_by_cluster_id": counts_by_cluster,
        "distribution_by_cluster_id": distribution,
        "valid_trials": valid_trials,
        "total_trials": total_trials,
        "parse_error_rate": _safe_divide(parse_error_count, total_trials),
        "top_cluster_id": top_cluster_id,
        "top_p": top_p,
        "top_ci_low": top_ci[0] if top_ci else None,
        "top_ci_high": top_ci[1] if top_ci else None,
        "top_ci_half_width": top_ci[2] if top_ci else None,
        "entropy": entropy,
        "effective_num_clusters": eff_num_clusters,
        "margin": margin,
        "disagreement_rate": disagreement_rate,
    }
    write_json(aggregates_path, aggregates_payload)

    metrics_payload = {
        "run_id": resolved_config.run.run_id,
        "question_id": question_id,
        "stop_reason": stop_reason,
        "stop_at_trials": total_trials,
        "valid_trials_total": valid_trials,
        "parse_valid_total": parse_valid_count,
        "parse_error_count": parse_error_count,
        "parse_failure_trial_ids": parse_failure_trial_ids,
        "llm_error_count": llm_error_count,
        "embedding_error_count": embedding_error_count,
        "embedding_call_count": embedding_call_count,
        "summarizer_call_count": summarizer_call_count,
        "execution_seed": rng_seed,
        "parse_failure_policy": execution.parse_failure_policy,
        "sampling_seed": rng_seed,
        "converged": stop_reason == "converged",
        "batches_completed": batches_completed,
        "convergence_trace": convergence_trace,
        "timing": {
            "started_at": resolved_config.run.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    if stop_error:
        metrics_payload["stop_error"] = stop_error
    write_json(metrics_path, metrics_payload)

    result = ExecutionResult(
        stop_reason=stop_reason,
        stop_at_trials=total_trials,
        valid_trials=valid_trials,
        parse_error_count=parse_error_count,
        batches_completed=batches_completed,
        converged=stop_reason == "converged",
        top_cluster_id=top_cluster_id,
        top_p=top_p,
        top_ci_low=top_ci[0] if top_ci else None,
        top_ci_high=top_ci[1] if top_ci else None,
        top_ci_half_width=top_ci[2] if top_ci else None,
        entropy=entropy,
        margin=margin,
        disagreement_rate=disagreement_rate,
        llm_call_count=llm_call_count,
        embedding_call_count=embedding_call_count,
        summarizer_call_count=summarizer_call_count,
    )
    _emit(
        {
            "type": "execution_finished",
            "stop_reason": result.stop_reason,
            "stop_at_trials": result.stop_at_trials,
            "valid_trials": result.valid_trials,
            "batches_completed": result.batches_completed,
        }
    )
    return result


async def _execute_one_trial(
    *,
    item: WorkItem,
    worker_id: int,
    run_id: str,
    question_id: str | None = None,
    question_hash: str,
    embed_text_policy: str,
    llm_config: LLMConfig,
    client: Any,
) -> TrialOutcome:
    default_routing = llm_config.routing_defaults
    request_defaults = llm_config.request_defaults
    request = LLMRequest(
        messages=item.messages,
        model=item.atom.model,
        temperature=item.temperature,
        top_p=request_defaults.top_p,
        max_tokens=request_defaults.max_tokens,
        seed=request_defaults.seed if request_defaults.seed is not None else item.request_seed,
        stop=request_defaults.stop,
        response_format=request_defaults.response_format,
        tools=request_defaults.tools,
        tool_choice=request_defaults.tool_choice,
        parallel_tool_calls=request_defaults.parallel_tool_calls,
        provider_routing=None,
        extra_body=dict(llm_config.extra_body_defaults),
        metadata={
            "run_id": run_id,
            "trial_id": item.trial_id,
            "atom_id": item.atom.atom_id,
            "question_hash": question_hash,
        },
    )
    request_body, overrides = build_request_body(request, default_provider_routing=default_routing)

    started_at = datetime.now(timezone.utc)
    start = time.monotonic()
    response_text = ""
    response_raw = None
    usage = None
    request_id = None
    model_returned = None
    routing = None
    latency_ms = None
    parse_valid = False
    outcome_text = None
    rationale = None
    trace_summary = None
    parse_error = None
    retry_messages = None
    error_text = None
    embedded_text = None

    try:
        response = await client.generate(request)
        response_text = response.text
        response_raw = response.raw
        usage = response.usage
        request_id = response.request_id
        model_returned = response.model_returned
        routing = response.routing
        latency_ms = response.latency_ms
        parse_valid, outcome_text, rationale, trace_summary, parse_error = _parse_output(response_text)
        if parse_valid and outcome_text is not None:
            embedded_text = _embedded_text(outcome_text, rationale, embed_text_policy)
        else:
            retry_messages = build_retry_messages(item.messages, response_text)
    except Exception as exc:
        error_text = str(exc)
    duration_ms = int((time.monotonic() - start) * 1000)
    ended_at = datetime.now(timezone.utc)

    trial_record = {
        "trial_id": item.trial_id,
        "run_id": run_id,
        "question_id": question_id,
        "batch_index": item.batch_index,
        "worker_id": worker_id + 1,
        "atom_id": item.atom.atom_id,
        "model": item.atom.model,
        "temperature": item.temperature,
        "temperature_used": item.temperature,
        "temperature_policy": item.temperature_policy,
        "persona_id": item.atom.persona_id,
        "question_hash": question_hash,
        "retries_used": item.retries_used,
        "request": {
            "body": request_body,
            "overrides": overrides,
        },
        "response": {
            "text": response_text,
            "raw": response_raw,
            "usage": usage,
            "request_id": request_id,
            "model_returned": model_returned,
            "routing": routing,
            "latency_ms": latency_ms,
        },
        "timestamps": {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
        },
        "duration_ms": duration_ms,
    }
    if error_text:
        trial_record["error"] = {
            "message": error_text,
        }

    parsed_record = {
        "trial_id": item.trial_id,
        "outcome": outcome_text,
        "rationale": rationale,
        "trace_summary": trace_summary,
        "embedded_text": embedded_text,
        "cluster_id": None,
        "parse_valid": parse_valid,
        "retries_used": item.retries_used,
    }
    if parse_error:
        parsed_record["parse_error"] = parse_error
    if error_text:
        parsed_record["error"] = error_text

    return TrialOutcome(
        trial_record=trial_record,
        parsed_record=parsed_record,
        parse_valid=parse_valid,
        embedded_text=embedded_text,
        outcome_text=outcome_text,
        rationale=rationale,
        trace_summary=trace_summary,
        retry_messages=retry_messages,
        retries_used=item.retries_used,
        atom=item.atom,
        error=error_text,
    )


def build_messages(question_text: str, persona_id: str | None) -> list[dict[str, Any]]:
    parts = []
    if persona_id:
        parts.append(f"Persona: {persona_id}.")
    parts.append("Respond ONLY with valid JSON.")
    parts.append('The JSON object must use keys "outcome", "rationale", and "trace_summary".')
    parts.append('"outcome" is required; "rationale" and "trace_summary" are optional strings.')
    system_prompt = " ".join(parts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_text},
    ]


def build_retry_messages(
    messages: list[dict[str, Any]],
    invalid_output: str,
) -> list[dict[str, Any]]:
    corrective = (
        "Your output was invalid. Output ONLY valid JSON of the form "
        '{"outcome": "<string>", "rationale": "<string>", "trace_summary": "<string>"} '
        "with outcome required and the other fields optional."
    )
    return list(messages) + [
        {"role": "assistant", "content": invalid_output},
        {"role": "user", "content": corrective},
    ]


def _parse_output(
    text: str,
) -> tuple[bool, str | None, str | None, str | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, None, None, None, f"invalid_json: {exc.msg}"
    if not isinstance(payload, dict):
        return False, None, None, None, "invalid_json: not an object"
    outcome = payload.get("outcome")
    if not isinstance(outcome, str) or not outcome.strip():
        return False, None, None, None, "invalid_outcome: required string"
    rationale = payload.get("rationale")
    trace_summary = payload.get("trace_summary")
    if rationale is not None and not isinstance(rationale, str):
        return False, None, None, None, "invalid_rationale: not a string"
    if trace_summary is not None and not isinstance(trace_summary, str):
        return False, None, None, None, "invalid_trace_summary: not a string"
    return True, outcome.strip(), rationale, trace_summary, None


def _embedded_text(outcome: str, rationale: str | None, policy: str) -> str:
    if policy == "outcome":
        return outcome.strip()
    if rationale:
        return f"{outcome.strip()}\n{rationale.strip()}"
    return outcome.strip()


def _embedding_record(
    *,
    trial_id: str,
    text: str,
    embedding: list[float],
    dims: int,
    model: str,
    raw: dict[str, Any],
) -> dict[str, Any]:
    embed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    vector_b64 = _encode_embedding(embedding)
    return {
        "trial_id": trial_id,
        "embed_text_hash": embed_hash,
        "dims": dims,
        "dtype": "float32",
        "encoding": "base64",
        "vector_b64": vector_b64,
        "model": model,
        "raw": raw,
    }


def _encode_embedding(embedding: list[float]) -> str:
    arr = array("f", embedding)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _clusters_online_payload(
    *,
    clusterer: OnlineLeaderClustering,
    method: str,
    tau: float,
    embedding_model: str,
    embed_text: str,
) -> dict[str, Any]:
    clusters = clusterer.export()
    clusters_payload = []
    for cluster in clusters:
        centroid_b64 = _encode_embedding(cluster.centroid)
        clusters_payload.append(
            {
                "cluster_id": cluster.cluster_id,
                "count": cluster.count,
                "centroid": {
                    "dtype": "float32",
                    "encoding": "base64",
                    "data_b64": centroid_b64,
                },
                "exemplars": cluster.exemplars,
            }
        )
    dims = len(clusters[0].centroid) if clusters else None
    status = "complete" if clusters_payload else "empty"
    return {
        "status": status,
        "method": method,
        "tau": tau,
        "embedding_model": embedding_model,
        "embedding_dims": dims,
        "normalization": "l2",
        "embed_text": embed_text,
        "clusters": clusters_payload,
    }


def _converged_candidate(
    *,
    entry: dict[str, Any],
    min_trials: int,
    delta_js_threshold: float,
    epsilon_new_threshold: float,
    epsilon_ci_half_width: float | None,
) -> bool:
    if entry["valid_trials_total"] < min_trials:
        return False
    js_divergence = entry.get("js_divergence")
    if js_divergence is None or js_divergence > delta_js_threshold:
        return False
    if entry.get("new_cluster_rate") is None or entry["new_cluster_rate"] > epsilon_new_threshold:
        return False
    if epsilon_ci_half_width is None:
        return True
    top_ci_half_width = entry.get("top_ci_half_width")
    if top_ci_half_width is None:
        return False
    return top_ci_half_width <= epsilon_ci_half_width


def _convergence_entry(
    *,
    batch_index: int,
    total_trials: int,
    valid_trials: int,
    counts_by_cluster: dict[str, int],
    distribution_by_cluster: dict[str, float],
    top_cluster_id: str | None,
    top_p: float | None,
    top_ci: tuple[float, float, float] | None,
    js_divergence: float | None,
    new_cluster_rate: float,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "trials_completed_total": total_trials,
        "valid_trials_total": valid_trials,
        "counts_by_cluster_id": dict(counts_by_cluster),
        "distribution_by_cluster_id": distribution_by_cluster,
        "top_cluster_id": top_cluster_id,
        "top_p": top_p,
        "top_ci_low": top_ci[0] if top_ci else None,
        "top_ci_high": top_ci[1] if top_ci else None,
        "top_ci_half_width": top_ci[2] if top_ci else None,
        "js_divergence": js_divergence,
        "new_cluster_rate": new_cluster_rate,
    }


def _distribution(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {label: count / total for label, count in counts.items()}


def _top_cluster_id(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return max(counts, key=counts.get)


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float] | None:
    if total <= 0:
        return None
    p_hat = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = p_hat + z2 / (2 * total)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * total)) / total)
    low = (center - margin) / denom
    high = (center + margin) / denom
    low = max(0.0, min(1.0, low))
    high = max(0.0, min(1.0, high))
    return low, high, (high - low) / 2


def _entropy(distribution: dict[str, float]) -> float:
    total = 0.0
    for value in distribution.values():
        if value > 0:
            total -= value * math.log2(value)
    return total


def _margin(distribution: dict[str, float]) -> float:
    if not distribution:
        return 0.0
    sorted_values = sorted(distribution.values(), reverse=True)
    if len(sorted_values) == 1:
        return sorted_values[0]
    return sorted_values[0] - sorted_values[1]


def _effective_num_clusters(distribution: dict[str, float]) -> float:
    if not distribution:
        return 0.0
    denom = sum(value * value for value in distribution.values())
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _js_divergence(
    prev: dict[str, float] | None,
    current: dict[str, float],
) -> float | None:
    if prev is None:
        return None
    keys = set(prev) | set(current)
    if not keys:
        return 0.0
    p = [prev.get(key, 0.0) for key in keys]
    q = [current.get(key, 0.0) for key in keys]
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _kl_divergence(p: list[float], q: list[float]) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0 or qi <= 0:
            continue
        total += pi * math.log2(pi / qi)
    return total


def _sample_atom(rng: random.Random, atoms: list[QAtom]) -> QAtom:
    weights = [atom.weight for atom in atoms]
    return rng.choices(atoms, weights=weights, k=1)[0]


def _sample_temperature(rng: random.Random, policy: Any, fallback: float) -> float:
    kind = getattr(policy, "kind", "fixed")
    temperatures = list(getattr(policy, "temperatures", []) or [])
    if kind == "range" and len(temperatures) >= 2:
        low, high = temperatures[0], temperatures[1]
        if high < low:
            low, high = high, low
        return rng.uniform(low, high)
    if temperatures:
        if kind == "list":
            return rng.choice(temperatures)
        return temperatures[0]
    return fallback


def _temperature_policy_payload(policy: Any) -> dict[str, Any]:
    kind = getattr(policy, "kind", "fixed")
    temperatures = list(getattr(policy, "temperatures", []) or [])
    if kind == "range" and len(temperatures) >= 2:
        low, high = temperatures[0], temperatures[1]
        if high < low:
            low, high = high, low
        return {"type": "range", "min": low, "max": high}
    if kind == "list":
        return {"type": "list", "values": temperatures}
    value = temperatures[0] if temperatures else None
    return {"type": "fixed", "value": value}


def _truncate_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _checkpoint_row(entry: dict[str, Any], stop_reason: str | None) -> dict[str, str]:
    clusters_count = len(entry.get("counts_by_cluster_id", {}) or {})
    js_divergence = entry.get("js_divergence")
    new_cluster_rate = entry.get("new_cluster_rate")
    ci_half = entry.get("top_ci_half_width")
    stop = "yes" if stop_reason is not None else "no"
    return {
        "Batch": str(entry.get("batch_index")),
        "Trials": str(entry.get("trials_completed_total")),
        "Clusters": str(clusters_count),
        "JS": f"{js_divergence:.3f}" if js_divergence is not None else "n/a",
        "New": f"{new_cluster_rate:.3f}" if new_cluster_rate is not None else "n/a",
        "CI HW": f"{ci_half:.3f}" if ci_half is not None else "n/a",
        "Stop": stop,
    }
