"""Client interface and shared helpers for LLM access."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from arbiter.llm.types import LLMRequest, LLMResponse


@runtime_checkable
class LLMClient(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute a single completion request."""

    async def list_models(self) -> dict[str, Any]:
        """Return available models metadata."""

    async def aclose(self) -> None:
        """Release any underlying client resources."""


def create_client(mode: str, **kwargs: Any) -> LLMClient:
    if mode == "mock":
        from arbiter.llm.mock import MockClient

        return MockClient(**kwargs)
    if mode == "openrouter":
        from arbiter.llm.openrouter import OpenRouterClient

        return OpenRouterClient(**kwargs)
    raise ValueError(f"Unsupported LLM mode: {mode}")


def build_request_body(
    request: LLMRequest,
    *,
    default_provider_routing: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    overrides: list[str] = []
    body: dict[str, Any] = {
        "model": request.model,
        "messages": request.messages,
    }

    def add_optional(key: str, value: Any) -> None:
        if value is not None:
            body[key] = value

    add_optional("temperature", request.temperature)
    add_optional("top_p", request.top_p)
    add_optional("max_tokens", request.max_tokens)
    add_optional("seed", request.seed)
    add_optional("stop", request.stop)
    add_optional("response_format", request.response_format)
    add_optional("tools", request.tools)
    add_optional("tool_choice", request.tool_choice)
    add_optional("parallel_tool_calls", request.parallel_tool_calls)

    provider = request.provider_routing or default_provider_routing
    if provider is not None:
        body["provider"] = provider

    for key, value in request.extra_body.items():
        if key in body:
            overrides.append(key)
        body[key] = value

    return body, overrides


def build_call_log_record(
    request: LLMRequest,
    response: LLMResponse,
    *,
    default_provider_routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body, overrides = build_request_body(request, default_provider_routing=default_provider_routing)
    return {
        "run_id": request.metadata.get("run_id"),
        "trial_id": request.metadata.get("trial_id"),
        "atom_id": request.metadata.get("atom_id"),
        "prompt_hash": request.metadata.get("prompt_hash"),
        "model_requested": response.model_requested,
        "model_returned": response.model_returned,
        "effective_routing": body.get("provider"),
        "sampling_params": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "stop": request.stop,
            "response_format": request.response_format,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            "parallel_tool_calls": request.parallel_tool_calls,
        },
        "latency_ms": response.latency_ms,
        "usage": response.usage,
        "request_id": response.request_id,
        "overrides": overrides,
        "raw_response": response.raw,
    }
