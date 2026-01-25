"""Async execution engine for arbiter trials."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from arbiter.config import LLMConfig, QAtom, ResolvedConfig
from arbiter.llm.client import build_request_body, create_client
from arbiter.llm.types import LLMRequest
from arbiter.storage import append_jsonl, write_json
from rich.console import Group
from rich.live import Live

from arbiter.ui.console import get_console
from arbiter.ui.progress import build_execution_progress
from arbiter.ui.render import (
    build_batch_checkpoint,
    build_execution_header,
    render_execution_header,
    render_info,
)


@dataclass(slots=True)
class WorkItem:
    trial_id: str
    batch_index: int
    retries_used: int
    atom: QAtom
    temperature: float
    temperature_policy: dict[str, Any]
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class PendingRetry:
    trial_id: str
    retries_used: int
    atom: QAtom
    temperature: float
    temperature_policy: dict[str, Any]
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class TrialOutcome:
    trial_record: dict[str, Any]
    parsed_record: dict[str, Any]
    parse_valid: bool
    mode_id: str | None
    embedded_text: str | None
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
    top_mode_id: str | None
    top_p: float | None
    top_ci_low: float | None
    top_ci_high: float | None
    top_ci_half_width: float | None
    entropy: float | None
    margin: float | None
    disagreement_rate: float | None


async def execute_trials(
    *,
    run_dir: Path,
    resolved_config: ResolvedConfig,
    question: dict[str, Any],
) -> ExecutionResult:
    question_text = str(question.get("question_text", ""))
    if not question_text.strip():
        raise ValueError("Question text is required for execution.")
    question_id = str(question.get("question_id", "question"))
    question_hash = _hash_text(question_text)

    question_path = run_dir / "question.json"
    trials_path = run_dir / "trials.jsonl"
    parsed_path = run_dir / "parsed.jsonl"
    aggregates_path = run_dir / "aggregates.json"
    metrics_path = run_dir / "metrics.json"

    write_json(question_path, question)

    semantic = resolved_config.semantic
    execution = semantic.execution
    trial_budget = semantic.trial_budget
    budget_guardrail = semantic.budget_guardrail
    llm_config = semantic.llm
    call_cap = min(trial_budget.k_max, budget_guardrail.max_calls)
    worker_count = max(1, execution.worker_count)
    batch_size = max(1, execution.batch_size)
    max_retries = max(0, execution.max_retries)
    convergence = execution.convergence
    temperature_policy_payload = _temperature_policy_payload(semantic.temperature_policy)

    rng_seed = random.randrange(2**32)
    rng = random.Random(rng_seed)

    counts_by_mode: dict[str, int] = {}
    seen_modes: set[str] = set()
    total_trials = 0
    valid_trials = 0
    parse_error_count = 0
    llm_error_count = 0
    convergence_trace: list[dict[str, Any]] = []
    stop_reason: str | None = None
    stop_error: str | None = None

    pending_retries: list[PendingRetry] = []
    trial_counter = 0
    consecutive_converged = 0
    batches_completed = 0
    prev_distribution: dict[str, float] | None = None

    console = get_console()
    use_live = console.is_terminal
    progress = None
    overall_task_id = None
    worker_task_ids: list[int] = []
    worker_counts = [0 for _ in range(worker_count)]

    client = create_client(llm_config.mode, default_routing=llm_config.routing_defaults)

    header_summary = _execution_header_summary(
        resolved_config=resolved_config,
        question_text=question_text,
        call_cap=call_cap,
        worker_count=worker_count,
        batch_size=batch_size,
    )
    header_panel = None
    checkpoint_panel = None
    live: Live | None = None
    if use_live:
        header_panel = build_execution_header(header_summary)
        checkpoint_panel = build_batch_checkpoint(None)
    else:
        render_execution_header(header_summary)

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
            if progress and overall_task_id is not None:
                progress.update(
                    worker_task_ids[worker_id],
                    description=_worker_description(worker_id, worker_counts[worker_id], item.atom),
                    completed=worker_counts[worker_id],
                )
            outcome = await _execute_one_trial(
                item=item,
                worker_id=worker_id,
                run_id=resolved_config.run.run_id,
                question_id=question_id,
                question_hash=question_hash,
                llm_config=llm_config,
                client=client,
            )
            worker_counts[worker_id] += 1
            if progress and overall_task_id is not None:
                progress.update(
                    worker_task_ids[worker_id],
                    description=_worker_description(worker_id, worker_counts[worker_id], None),
                    completed=worker_counts[worker_id],
                )
            await results.put(outcome)
            queue.task_done()

    async def run_loop() -> None:
        nonlocal total_trials, valid_trials, parse_error_count, llm_error_count
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
                                messages=retry.messages,
                            )
                        )
                        continue
                    trial_counter += 1
                    atom = _sample_atom(rng, semantic.q_distribution.atoms)
                    temperature = _sample_temperature(rng, semantic.temperature_policy, atom.temperature)
                    messages = build_messages(question_text, atom.persona_id)
                    batch_items.append(
                        WorkItem(
                            trial_id=f"trial_{trial_counter:06d}",
                            batch_index=batch_index,
                            retries_used=0,
                            atom=atom,
                            temperature=temperature,
                            temperature_policy=temperature_policy_payload,
                            messages=messages,
                        )
                    )

                for item in batch_items:
                    await queue.put(item)

                batch_results: list[TrialOutcome] = []
                for _ in range(len(batch_items)):
                    outcome = await results.get()
                    batch_results.append(outcome)
                    total_trials += 1
                    append_jsonl(trials_path, outcome.trial_record)
                    append_jsonl(parsed_path, outcome.parsed_record)

                    if outcome.parse_valid and outcome.mode_id:
                        valid_trials += 1
                        counts_by_mode[outcome.mode_id] = counts_by_mode.get(outcome.mode_id, 0) + 1
                    else:
                        parse_error_count += 1

                        if outcome.error:
                            llm_error_count += 1
                            stop_reason = "llm_error"
                            stop_error = outcome.error
                        elif outcome.retry_messages and outcome.retries_used < max_retries:
                            if total_trials < call_cap:
                                trial_counter += 1
                                pending_retries.append(
                                    PendingRetry(
                                        trial_id=f"trial_{trial_counter:06d}",
                                        retries_used=outcome.retries_used + 1,
                                        atom=outcome.atom,
                                        temperature=outcome.trial_record.get("temperature", outcome.atom.temperature),
                                        temperature_policy=outcome.trial_record.get("temperature_policy", temperature_policy_payload),
                                        messages=outcome.retry_messages,
                                    )
                                )
                            else:
                                stop_reason = "budget_exhausted"
                        else:
                            stop_reason = "parse_failure"

                    if progress and overall_task_id is not None:
                        progress.update(overall_task_id, completed=total_trials)

                if not console.is_terminal:
                    render_info(
                        f"Batch {batch_index} complete: {total_trials}/{call_cap} trials, "
                        f"{valid_trials} valid."
                    )

                new_mode_rate, seen_modes_after = _new_mode_rate(batch_results, seen_modes, len(batch_items))
                seen_modes.clear()
                seen_modes.update(seen_modes_after)

                distribution = _distribution(counts_by_mode, valid_trials)
                js_divergence = _js_divergence(prev_distribution, distribution)
                top_mode_id = _top_mode_id(counts_by_mode)
                top_p = distribution.get(top_mode_id) if top_mode_id else None
                top_ci = _wilson_ci(counts_by_mode.get(top_mode_id, 0), valid_trials) if top_mode_id else None
                entry = _convergence_entry(
                    batch_index=batch_index,
                    total_trials=total_trials,
                    valid_trials=valid_trials,
                    counts_by_mode=counts_by_mode,
                    distribution_by_mode=distribution,
                    top_mode_id=top_mode_id,
                    top_p=top_p,
                    top_ci=top_ci,
                    js_divergence=js_divergence,
                    new_mode_rate=new_mode_rate,
                )
                convergence_trace.append(entry)
                if console.is_terminal and live and header_panel:
                    checkpoint_panel = build_batch_checkpoint(_checkpoint_row(entry, stop_reason))
                    live.update(Group(header_panel, progress, checkpoint_panel), refresh=True)

                if stop_reason is not None:
                    break

                converged_candidate = _converged_candidate(
                    entry=entry,
                    min_trials=convergence.min_trials,
                    delta_js_threshold=convergence.delta_js_threshold,
                    epsilon_new_threshold=convergence.epsilon_new_threshold,
                    epsilon_ci_half_width=convergence.epsilon_ci_half_width,
                )

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
                    break

                if total_trials >= call_cap:
                    stop_reason = "max_trials_reached"
                    break

                prev_distribution = distribution
        finally:
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers, return_exceptions=True)

    try:
        if use_live:
            progress, overall_task_id, worker_task_ids = build_execution_progress(worker_count, call_cap)
            layout = Group(header_panel, progress, checkpoint_panel)
            with Live(layout, console=console, refresh_per_second=10) as live:
                progress.start()
                await run_loop()
                progress.stop()
        else:
            await run_loop()
    finally:
        await client.aclose()

    if stop_reason is None:
        stop_reason = "max_trials_reached"

    distribution = _distribution(counts_by_mode, valid_trials)
    top_mode_id = _top_mode_id(counts_by_mode)
    top_p = distribution.get(top_mode_id) if top_mode_id else None
    top_ci = _wilson_ci(counts_by_mode.get(top_mode_id, 0), valid_trials) if top_mode_id else None
    entropy = _entropy(distribution) if valid_trials > 0 else None
    margin = _margin(distribution) if valid_trials > 0 else None
    disagreement_rate = 1.0 - top_p if top_p is not None else None

    aggregates_payload = {
        "question_id": question_id,
        "discovered_mode_count": len(counts_by_mode),
        "counts_by_mode_id": counts_by_mode,
        "distribution_by_mode_id": distribution,
        "valid_trials": valid_trials,
        "total_trials": total_trials,
        "parse_error_rate": _safe_divide(parse_error_count, total_trials),
        "top_mode_id": top_mode_id,
        "top_p": top_p,
        "top_ci_low": top_ci[0] if top_ci else None,
        "top_ci_high": top_ci[1] if top_ci else None,
        "top_ci_half_width": top_ci[2] if top_ci else None,
        "entropy": entropy,
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
        "parse_error_count": parse_error_count,
        "llm_error_count": llm_error_count,
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

    return ExecutionResult(
        stop_reason=stop_reason,
        stop_at_trials=total_trials,
        valid_trials=valid_trials,
        parse_error_count=parse_error_count,
        batches_completed=batches_completed,
        converged=stop_reason == "converged",
        top_mode_id=top_mode_id,
        top_p=top_p,
        top_ci_low=top_ci[0] if top_ci else None,
        top_ci_high=top_ci[1] if top_ci else None,
        top_ci_half_width=top_ci[2] if top_ci else None,
        entropy=entropy,
        margin=margin,
        disagreement_rate=disagreement_rate,
    )


async def _execute_one_trial(
    *,
    item: WorkItem,
    worker_id: int,
    run_id: str,
    question_id: str | None = None,
    question_hash: str,
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
        seed=request_defaults.seed,
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
    mode_id = None

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
            embedded_text = _embedded_text(outcome_text, rationale)
            mode_id = _mode_id(embedded_text)
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
        "mode_id": mode_id,
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
        mode_id=mode_id,
        embedded_text=embedded_text,
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


def _embedded_text(outcome: str, rationale: str | None) -> str:
    parts = [_normalize_text(outcome)]
    if rationale:
        parts.append(_normalize_text(rationale))
    return "\n".join(parts)


def _mode_id(embedded_text: str) -> str:
    digest = hashlib.sha256(embedded_text.encode("utf-8")).hexdigest()
    return f"mode_{digest[:12]}"


def _normalize_text(text: str) -> str:
    collapsed = " ".join(text.strip().split())
    return collapsed.lower()


def _new_mode_rate(
    batch_results: list[TrialOutcome],
    seen_modes: set[str],
    batch_size: int,
) -> tuple[float, set[str]]:
    if batch_size <= 0:
        return 0.0, set(seen_modes)
    updated = set(seen_modes)
    new_mode_count = 0
    for outcome in batch_results:
        if outcome.parse_valid and outcome.mode_id:
            if outcome.mode_id not in updated:
                new_mode_count += 1
                updated.add(outcome.mode_id)
    return new_mode_count / batch_size, updated


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
    if entry.get("new_mode_rate") is None or entry["new_mode_rate"] > epsilon_new_threshold:
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
    counts_by_mode: dict[str, int],
    distribution_by_mode: dict[str, float],
    top_mode_id: str | None,
    top_p: float | None,
    top_ci: tuple[float, float, float] | None,
    js_divergence: float | None,
    new_mode_rate: float,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "trials_completed_total": total_trials,
        "valid_trials_total": valid_trials,
        "counts_by_mode_id": dict(counts_by_mode),
        "distribution_by_mode_id": distribution_by_mode,
        "top_mode_id": top_mode_id,
        "top_p": top_p,
        "top_ci_low": top_ci[0] if top_ci else None,
        "top_ci_high": top_ci[1] if top_ci else None,
        "top_ci_half_width": top_ci[2] if top_ci else None,
        "js_divergence": js_divergence,
        "new_mode_rate": new_mode_rate,
    }


def _distribution(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {label: count / total for label, count in counts.items()}


def _top_mode_id(counts: dict[str, int]) -> str | None:
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


def _worker_description(worker_id: int, completed: int, atom: QAtom | None) -> str:
    status = "RUNNING" if atom else "IDLE"
    worker_label = f"W{worker_id + 1:02d}"
    done = f"{completed:>4}"
    model = _clip(atom.model if atom else "—", 22)
    atom_id = _clip(atom.atom_id if atom else "—", 12)
    persona = _clip(atom.persona_id if atom and atom.persona_id else "none", 10)
    return (
        f"{worker_label}  {status:<7}  done:{done}  "
        f"model:{model:<22}  atom:{atom_id:<12}  persona:{persona:<10}"
    )


def _execution_header_summary(
    *,
    resolved_config: ResolvedConfig,
    question_text: str,
    call_cap: int,
    worker_count: int,
    batch_size: int,
) -> None:
    semantic = resolved_config.semantic
    protocol = semantic.protocol.type
    models = ", ".join(semantic.models) if semantic.models else "n/a"
    personas = ", ".join(semantic.personas.persona_ids) if semantic.personas.persona_ids else "none"
    temperature = _format_temperature_policy(semantic.temperature_policy)
    convergence = semantic.execution.convergence
    epsilon_ci = (
        f"{convergence.epsilon_ci_half_width:.3f}"
        if convergence.epsilon_ci_half_width is not None
        else "off"
    )
    mode_label = _mode_label(semantic.llm.mode)
    summary = {
        "Run ID": resolved_config.run.run_id,
        "Started": resolved_config.run.started_at,
        "Question": _truncate_text(question_text, 80),
        "Mode": mode_label,
        "Protocol": protocol,
        "Models": models,
        "Personas": personas,
        "Temperature": temperature,
        "K_max": str(call_cap),
        "Workers / batch": f"{worker_count} / {batch_size}",
        "delta_js": f"{convergence.delta_js_threshold:.3f}",
        "epsilon_new": f"{convergence.epsilon_new_threshold:.3f}",
        "epsilon_ci": epsilon_ci,
    }
    return summary


def _format_temperature_policy(policy: Any) -> str:
    kind = getattr(policy, "kind", "fixed")
    values = list(getattr(policy, "temperatures", []) or [])
    if kind == "range" and len(values) >= 2:
        low, high = values[0], values[1]
        return f"range {low:.2f}–{high:.2f}"
    if kind == "list" and values:
        return "list [" + ", ".join(f"{value:.2f}" for value in values) + "]"
    if values:
        return f"fixed {values[0]:.2f}"
    return "fixed"


def _mode_label(mode: str) -> str:
    if mode == "mock":
        return "mock (no network calls)"
    return "remote (OpenRouter)"


def _truncate_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _checkpoint_row(entry: dict[str, Any], stop_reason: str | None) -> dict[str, str]:
    modes = len(entry.get("counts_by_mode_id", {}) or {})
    js_divergence = entry.get("js_divergence")
    new_mode_rate = entry.get("new_mode_rate")
    ci_half = entry.get("top_ci_half_width")
    stop = "yes" if stop_reason is not None else "no"
    return {
        "Batch": str(entry.get("batch_index")),
        "Trials": str(entry.get("trials_completed_total")),
        "Modes": str(modes),
        "JS": f"{js_divergence:.3f}" if js_divergence is not None else "n/a",
        "New": f"{new_mode_rate:.3f}" if new_mode_rate is not None else "n/a",
        "CI HW": f"{ci_half:.3f}" if ci_half is not None else "n/a",
        "Stop": stop,
    }


def _clip(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"
