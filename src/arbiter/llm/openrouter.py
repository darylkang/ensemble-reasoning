"""OpenRouter-backed client implementation."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any

import httpx

from arbiter.llm.client import build_request_body
from arbiter.llm.types import LLMRequest, LLMResponse


@dataclass
class OpenRouterError(RuntimeError):
    status_code: int
    response_text: str
    request_id: str | None
    request: dict[str, Any]

    def __str__(self) -> str:
        return f"OpenRouterError(status={self.status_code}, request_id={self.request_id})"


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        default_routing: dict[str, Any] | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterClient.")
        self._api_key = resolved_key
        self._base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout_s)
        self._default_routing = default_routing or {"allow_fallbacks": False}

    async def generate(self, request: LLMRequest) -> LLMResponse:
        body, overrides = build_request_body(
            request,
            default_provider_routing=self._default_routing,
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for key, value in request.extra_headers.items():
            if key.lower() in {"authorization", "content-type"}:
                continue
            headers[key] = value

        start = time.monotonic()
        response = await self._client.post("/chat/completions", json=body, headers=headers)
        latency_ms = int((time.monotonic() - start) * 1000)
        request_id = _extract_request_id(response.headers)

        if response.status_code < 200 or response.status_code >= 300:
            raise OpenRouterError(
                status_code=response.status_code,
                response_text=response.text,
                request_id=request_id,
                request=_sanitize_request(body, headers),
            )

        data = response.json()
        text = _extract_text(data)
        usage = data.get("usage")
        model_returned = data.get("model")
        routing = data.get("provider") or data.get("routing")

        if overrides:
            request.metadata.setdefault("overrides", overrides)

        return LLMResponse(
            text=text,
            raw=data,
            usage=usage,
            model_requested=request.model,
            model_returned=model_returned,
            routing=routing,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    async def list_models(self) -> dict[str, Any]:
        response = await self._client.get("/models", headers=_auth_headers(self._api_key))
        request_id = _extract_request_id(response.headers)
        if response.status_code < 200 or response.status_code >= 300:
            raise OpenRouterError(
                status_code=response.status_code,
                response_text=response.text,
                request_id=request_id,
                request={"endpoint": "/models"},
            )
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


def _extract_request_id(headers: httpx.Headers) -> str | None:
    return headers.get("x-request-id") or headers.get("openrouter-request-id")


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    text = message.get("content")
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return str(text)


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
    }


def _sanitize_request(body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    scrubbed_headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
    return {
        "body": body,
        "headers": scrubbed_headers,
    }
