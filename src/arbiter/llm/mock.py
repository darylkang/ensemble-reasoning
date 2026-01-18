"""Mock client for offline testing."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from arbiter.llm.types import LLMRequest, LLMResponse


class MockClient:
    def __init__(self, *, latency_ms: int = 15) -> None:
        self._latency_ms = latency_ms

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        text = _mock_text(request)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        latency_ms = max(self._latency_ms, elapsed_ms)

        raw = {
            "id": "mock",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        }
        usage = _mock_usage(request)
        return LLMResponse(
            text=text,
            raw=raw,
            usage=usage,
            model_requested=request.model,
            model_returned=request.model,
            routing=request.provider_routing,
            latency_ms=latency_ms,
            request_id=None,
        )

    def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [],
        }


def _mock_text(request: LLMRequest) -> str:
    seed_bytes = _stable_seed(request)
    choice = seed_bytes[0] % 2
    label = "YES" if choice == 0 else "NO"
    return f"Decision: {label}\nRationale: mock response for {request.model}."


def _stable_seed(request: LLMRequest) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(request.model.encode("utf-8"))
    for message in request.messages:
        hasher.update(str(message).encode("utf-8"))
    if request.seed is not None:
        hasher.update(str(request.seed).encode("utf-8"))
    return hasher.digest()


def _mock_usage(request: LLMRequest) -> dict[str, Any]:
    prompt_text = " ".join(str(message) for message in request.messages)
    prompt_tokens = max(1, len(prompt_text) // 4)
    completion_tokens = max(1, len(request.model) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
