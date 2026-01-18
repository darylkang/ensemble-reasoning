"""Core request/response types for LLM clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | str | None = None
    response_format: dict[str, Any] | str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None
    parallel_tool_calls: bool | None = None
    provider_routing: dict[str, Any] | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]
    usage: dict[str, Any] | None
    model_requested: str
    model_returned: str | None
    routing: dict[str, Any] | None
    latency_ms: int
    request_id: str | None
