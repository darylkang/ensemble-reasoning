"""Mock client for offline testing."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from arbiter.llm.client import build_request_body
from arbiter.llm.types import LLMRequest, LLMResponse


class MockClient:
    def __init__(
        self,
        *,
        latency_ms: int = 15,
        jitter_ms: int = 10,
        error_rate: float = 0.0,
        default_routing: dict[str, Any] | None = None,
    ) -> None:
        self._latency_ms = latency_ms
        self._jitter_ms = jitter_ms
        self._error_rate = error_rate
        self._default_routing = default_routing or {"allow_fallbacks": False}

    async def generate(self, request: LLMRequest) -> LLMResponse:
        body, overrides = build_request_body(
            request,
            default_provider_routing=self._default_routing,
        )
        if overrides:
            request.metadata.setdefault("overrides", overrides)

        start = time.monotonic()
        if self._error_rate > 0 and _should_error(request, self._error_rate):
            raise RuntimeError("MockClient simulated transient error.")
        await asyncio.sleep(_simulated_delay_s(request, self._latency_ms, self._jitter_ms))
        text = _mock_text(request, body.get("model", request.model))
        latency_ms = int((time.monotonic() - start) * 1000)

        raw = {
            "id": "mock",
            "object": "chat.completion",
            "model": body.get("model", request.model),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        }
        usage = _mock_usage(request, text)
        return LLMResponse(
            text=text,
            raw=raw,
            usage=usage,
            model_requested=request.model,
            model_returned=body.get("model", request.model),
            routing=body.get("provider"),
            latency_ms=latency_ms,
            request_id=None,
        )

    async def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [],
        }

    async def aclose(self) -> None:
        return


def _mock_text(request: LLMRequest, model: str) -> str:
    seed_bytes = _stable_seed(request, model)
    choice = seed_bytes[0] % 2
    label = "YES" if choice == 0 else "NO"
    return f"Decision: {label}\nRationale: mock response for {model}."


def _stable_seed(request: LLMRequest, model: str) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    for message in request.messages:
        hasher.update(str(message).encode("utf-8"))
    if request.seed is not None:
        hasher.update(str(request.seed).encode("utf-8"))
    return hasher.digest()


def _mock_usage(request: LLMRequest, text: str) -> dict[str, Any]:
    prompt_text = " ".join(str(message) for message in request.messages)
    prompt_tokens = max(1, len(prompt_text) // 4)
    completion_tokens = max(1, len(text) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _simulated_delay_s(request: LLMRequest, latency_ms: int, jitter_ms: int) -> float:
    seed_bytes = _stable_seed(request, request.model)
    jitter = seed_bytes[1] % max(1, jitter_ms + 1)
    return (latency_ms + jitter) / 1000.0


def _should_error(request: LLMRequest, error_rate: float) -> bool:
    seed_bytes = _stable_seed(request, request.model)
    threshold = int(error_rate * 255)
    return seed_bytes[2] < threshold
