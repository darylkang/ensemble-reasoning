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
from arbiter.ui.console import get_console
from arbiter.ui.progress import build_execution_progress
from arbiter.ui.render import render_info


@dataclass(slots=True)
class WorkItem:
    trial_id: str
    batch_index: int
    retries_used: int
    atom: QAtom
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class PendingRetry:
    trial_id: str
    retries_used: int
    atom: QAtom
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class TrialOutcome:
    trial_record: dict[str, Any]
    parsed_record: dict[str, Any]
    parse_valid: bool
    decision: str | None
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
    top_label: str | None
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
    instance: dict[str, Any],
) -> ExecutionResult:
    prompt_text = str(instance.get("prompt", ""))
    labels = [str(label) for label in instance.get("labels", [])]
    if not prompt_text.strip():
        raise ValueError("Instance prompt text is required for execution.")
    if not labels:
        raise ValueError("Instance labels are required for execution.")
    instance_id = str(instance.get("instance_id", "instance"))
    prompt_hash = _hash_text(prompt_text)

    questions_path = run_dir / "questions.jsonl"
    trials_path = run_dir / "trials.jsonl"
    parsed_path = run_dir / "parsed.jsonl"
    aggregates_path = run_dir / "aggregates.json"
    metrics_path = run_dir / "metrics.json"

    append_jsonl(questions_path, instance)

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

    rng_seed = random.randrange(2**32)
    rng = random.Random(rng_seed)

    counts_by_label = {label: 0 for label in labels}
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

    console = get_console()
    use_live = console.is_terminal
    progress = None
    overall_task_id = None
    worker_task_ids: list[int] = []
    worker_counts = [0 for _ in range(worker_count)]

    client = create_client(llm_config.mode, default_routing=llm_config.routing_defaults)

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
                instance_id=instance_id,
                prompt_hash=prompt_hash,
                labels=labels,
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
        nonlocal trial_counter

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
                                messages=retry.messages,
                            )
                        )
                        continue
                    trial_counter += 1
                    atom = _sample_atom(rng, semantic.q_distribution.atoms)
                    messages = build_messages(prompt_text, labels, atom.persona_id)
                    batch_items.append(
                        WorkItem(
                            trial_id=f"trial_{trial_counter:06d}",
                            batch_index=batch_index,
                            retries_used=0,
                            atom=atom,
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

                    if outcome.parse_valid and outcome.decision:
                        valid_trials += 1
                        counts_by_label[outcome.decision] += 1
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

                entry = _convergence_entry(
                    batch_index=batch_index,
                    total_trials=total_trials,
                    valid_trials=valid_trials,
                    counts_by_label=counts_by_label,
                    labels=labels,
                    epsilon=convergence.epsilon_ci_half_width,
                    min_trials=convergence.min_trials,
                )
                convergence_trace.append(entry)

                if stop_reason is not None:
                    break

                if entry["converged_candidate"]:
                    consecutive_converged += 1
                else:
                    consecutive_converged = 0

                if (
                    valid_trials >= convergence.min_trials
                    and entry["converged_candidate"]
                    and consecutive_converged >= convergence.patience_batches
                    and not pending_retries
                ):
                    stop_reason = "converged"
                    break

                if total_trials >= call_cap:
                    stop_reason = "max_trials_reached"
                    break
        finally:
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers, return_exceptions=True)

    try:
        if use_live:
            progress, overall_task_id, worker_task_ids = build_execution_progress(worker_count, call_cap)
            with progress:
                await run_loop()
        else:
            await run_loop()
    finally:
        await client.aclose()

    if stop_reason is None:
        stop_reason = "max_trials_reached"

    distribution = _distribution(counts_by_label, valid_trials)
    top_label = _top_label(labels, counts_by_label, valid_trials)
    top_p = distribution.get(top_label) if top_label else None
    top_ci = _wilson_ci(counts_by_label.get(top_label, 0), valid_trials) if top_label else None
    entropy = _entropy(distribution) if valid_trials > 0 else None
    margin = _margin(distribution, labels) if valid_trials > 0 else None
    disagreement_rate = 1.0 - top_p if top_p is not None else None

    aggregates_payload = {
        "instance_id": instance_id,
        "labels": labels,
        "counts": counts_by_label,
        "distribution": distribution,
        "valid_trials": valid_trials,
        "total_trials": total_trials,
        "parse_error_rate": _safe_divide(parse_error_count, total_trials),
        "top_label": top_label,
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
        "instance_id": instance_id,
        "stop_reason": stop_reason,
        "stop_at_trials": total_trials,
        "valid_trials_total": valid_trials,
        "parse_error_count": parse_error_count,
        "llm_error_count": llm_error_count,
        "sampling_seed": rng_seed,
        "converged": stop_reason == "converged",
        "batches_completed": batches_completed,
        "convergence_trace": convergence_trace,
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
        top_label=top_label,
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
    instance_id: str | None = None,
    prompt_hash: str,
    labels: list[str],
    llm_config: LLMConfig,
    client: Any,
) -> TrialOutcome:
    default_routing = llm_config.routing_defaults
    request_defaults = llm_config.request_defaults
    request = LLMRequest(
        messages=item.messages,
        model=item.atom.model,
        temperature=item.atom.temperature,
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
            "prompt_hash": prompt_hash,
            "labels": labels,
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
    decision = None
    rationale = None
    parse_error = None
    retry_messages = None
    error_text = None

    try:
        response = await client.generate(request)
        response_text = response.text
        response_raw = response.raw
        usage = response.usage
        request_id = response.request_id
        model_returned = response.model_returned
        routing = response.routing
        latency_ms = response.latency_ms
        parse_valid, decision, rationale, parse_error = _parse_decision(response_text, labels)
        if not parse_valid:
            retry_messages = build_retry_messages(item.messages, response_text, labels)
    except Exception as exc:
        error_text = str(exc)
    duration_ms = int((time.monotonic() - start) * 1000)
    ended_at = datetime.now(timezone.utc)

    trial_record = {
        "trial_id": item.trial_id,
        "run_id": run_id,
        "batch_index": item.batch_index,
        "worker_id": worker_id + 1,
        "atom_id": item.atom.atom_id,
        "model": item.atom.model,
        "temperature": item.atom.temperature,
        "persona_id": item.atom.persona_id,
        "instance_id": instance_id,
        "prompt_hash": prompt_hash,
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
        "decision": decision,
        "rationale": rationale,
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
        decision=decision,
        retry_messages=retry_messages,
        retries_used=item.retries_used,
        atom=item.atom,
        error=error_text,
    )


def build_messages(prompt_text: str, labels: list[str], persona_id: str | None) -> list[dict[str, Any]]:
    label_list = ", ".join(labels)
    parts = []
    if persona_id:
        parts.append(f"Persona: {persona_id}.")
    parts.append("Respond ONLY with valid JSON.")
    parts.append('The JSON object must use keys "decision" and "rationale".')
    parts.append(f'Decision must be one of: {label_list}.')
    system_prompt = " ".join(parts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]


def build_retry_messages(
    messages: list[dict[str, Any]],
    invalid_output: str,
    labels: list[str],
) -> list[dict[str, Any]]:
    label_list = ", ".join(labels)
    corrective = (
        "Your output was invalid. Output ONLY valid JSON of the form "
        '{"decision": "<label>", "rationale": "<string>"} '
        f"where decision is one of: {label_list}."
    )
    return list(messages) + [
        {"role": "assistant", "content": invalid_output},
        {"role": "user", "content": corrective},
    ]


def _parse_decision(
    text: str,
    labels: list[str],
) -> tuple[bool, str | None, str | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, None, None, f"invalid_json: {exc.msg}"
    if not isinstance(payload, dict):
        return False, None, None, "invalid_json: not an object"
    decision = payload.get("decision")
    if not isinstance(decision, str):
        return False, None, None, "invalid_decision: not a string"
    if decision not in labels:
        return False, None, None, "invalid_decision: label not allowed"
    rationale = payload.get("rationale")
    if rationale is None:
        rationale = ""
    if not isinstance(rationale, str):
        return False, None, None, "invalid_rationale: not a string"
    return True, decision, rationale, None


def _convergence_entry(
    *,
    batch_index: int,
    total_trials: int,
    valid_trials: int,
    counts_by_label: dict[str, int],
    labels: list[str],
    epsilon: float,
    min_trials: int,
) -> dict[str, Any]:
    distribution = _distribution(counts_by_label, valid_trials)
    top_label = _top_label(labels, counts_by_label, valid_trials)
    top_p = distribution.get(top_label) if top_label else None
    ci = _wilson_ci(counts_by_label.get(top_label, 0), valid_trials) if top_label else None
    converged_candidate = False
    if ci is not None and valid_trials >= min_trials:
        half_width = ci[2]
        converged_candidate = half_width is not None and half_width <= epsilon

    return {
        "batch_index": batch_index,
        "trials_completed_total": total_trials,
        "valid_trials_total": valid_trials,
        "counts_by_label": dict(counts_by_label),
        "distribution_by_label": distribution,
        "top_label": top_label,
        "top_p": top_p,
        "top_ci_low": ci[0] if ci else None,
        "top_ci_high": ci[1] if ci else None,
        "top_ci_half_width": ci[2] if ci else None,
        "converged_candidate": converged_candidate,
    }


def _distribution(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {label: 0.0 for label in counts}
    return {label: count / total for label, count in counts.items()}


def _top_label(labels: list[str], counts: dict[str, int], total: int) -> str | None:
    if total <= 0:
        return None
    return max(labels, key=lambda label: counts.get(label, 0))


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


def _margin(distribution: dict[str, float], labels: list[str]) -> float:
    if not labels:
        return 0.0
    sorted_values = sorted(
        (distribution.get(label, 0.0) for label in labels),
        reverse=True,
    )
    if len(sorted_values) == 1:
        return sorted_values[0]
    return sorted_values[0] - sorted_values[1]


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sample_atom(rng: random.Random, atoms: list[QAtom]) -> QAtom:
    weights = [atom.weight for atom in atoms]
    return rng.choices(atoms, weights=weights, k=1)[0]


def _worker_description(worker_id: int, completed: int, atom: QAtom | None) -> str:
    label = f"worker {worker_id + 1}"
    suffix = f"done {completed}"
    if atom:
        return f"{label} · {atom.atom_id} · {atom.model} · {suffix}"
    return f"{label} · idle · {suffix}"
