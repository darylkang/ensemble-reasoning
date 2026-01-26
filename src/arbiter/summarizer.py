"""Cluster summarizer (locked instrument)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from arbiter.llm.client import create_client
from arbiter.llm.types import LLMRequest


PROMPT_TEMPLATE = (
    "You summarize outcome clusters for an analysis report.\n"
    "Return JSON: {\"label\": \"...\", \"summary\": \"...\"}.\n"
    "Label must be <= 6 words. Summary should be 1 sentence.\n\n"
    "Exemplars:\n{exemplars}\n"
)
PROMPT_VERSION = "v1"
PROMPT_HASH = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def _mock_summary(cluster_id: str) -> tuple[str, str]:
    seed = hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:6]
    label = f"Mock summary {seed}"
    summary = "Mock summary for cluster exemplars."
    return label, summary


def _parse_summary(text: str) -> tuple[str, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        trimmed = text.strip()
        return trimmed[:60] or "Summary", trimmed or ""
    if isinstance(payload, dict):
        label = payload.get("label")
        summary = payload.get("summary")
        if isinstance(label, str) and isinstance(summary, str):
            return label.strip()[:60], summary.strip()
    trimmed = text.strip()
    return trimmed[:60] or "Summary", trimmed or ""


async def summarize_clusters(
    *,
    clusters: list[dict[str, Any]],
    llm_mode: str,
    model_slug: str,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[dict[str, Any], int]:
    if not clusters:
        return (
            {
                "status": "empty",
                "model": model_slug,
                "prompt_version": prompt_version,
                "prompt_hash": PROMPT_HASH,
                "summaries": [],
            },
            0,
        )

    if llm_mode == "mock":
        summaries = []
        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id"))
            label, summary = _mock_summary(cluster_id)
            summaries.append(
                {
                    "cluster_id": cluster_id,
                    "label": label,
                    "summary": summary,
                }
            )
        return (
            {
                "status": "mock",
                "model": model_slug,
                "prompt_version": prompt_version,
                "prompt_hash": PROMPT_HASH,
                "summaries": summaries,
            },
            0,
        )

    if llm_mode != "openrouter":
        return (
            {
                "status": "not_run",
                "reason": "unsupported_mode",
                "model": model_slug,
                "prompt_version": prompt_version,
                "prompt_hash": PROMPT_HASH,
            },
            0,
        )

    client = create_client(llm_mode, default_routing={"allow_fallbacks": False})
    summaries: list[dict[str, Any]] = []
    call_count = 0
    try:
        for cluster in clusters:
            exemplars = cluster.get("exemplars") or []
            exemplar_text = "\n".join(
                f"- {item.get('outcome', '')}\n  {item.get('rationale', '')}" for item in exemplars
            )
            prompt = PROMPT_TEMPLATE.format(exemplars=exemplar_text)
            request = LLMRequest(
                messages=[{"role": "system", "content": prompt}],
                model=model_slug,
                temperature=0.2,
                provider_routing=None,
                metadata={"summarizer": True, "cluster_id": cluster.get("cluster_id")},
            )
            response = await client.generate(request)
            call_count += 1
            label, summary = _parse_summary(response.text)
            summaries.append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "label": label,
                    "summary": summary,
                }
            )
    finally:
        await client.aclose()

    return (
        {
            "status": "complete",
            "model": model_slug,
            "prompt_version": prompt_version,
            "prompt_hash": PROMPT_HASH,
            "summaries": summaries,
        },
        call_count,
    )
