"""Canonical config validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from arbiter.config import CANONICAL_SCHEMA_VERSION, normalize_canonical_config


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    config: dict[str, Any] | None
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]


def load_and_validate_config(
    path: Path,
    *,
    default_model: str,
    llm_mode: str,
) -> ValidationResult:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ValidationResult(
            config=None,
            errors=[ValidationIssue("config", f"Config not found: {path}")],
            warnings=[],
        )
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationResult(
            config=None,
            errors=[ValidationIssue("config", f"Invalid JSON: {exc}")],
            warnings=[],
        )
    if not isinstance(raw, dict):
        return ValidationResult(
            config=None,
            errors=[ValidationIssue("config", "Config must be a JSON object.")],
            warnings=[],
        )
    return validate_config(raw, default_model=default_model, llm_mode=llm_mode)


def validate_config(
    data: dict[str, Any],
    *,
    default_model: str,
    llm_mode: str,
) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    schema_version = data.get("schema_version")
    if not schema_version:
        errors.append(ValidationIssue("schema_version", "Missing schema_version."))
    elif not isinstance(schema_version, str):
        errors.append(ValidationIssue("schema_version", "schema_version must be a string."))
    elif schema_version != CANONICAL_SCHEMA_VERSION:
        warnings.append(
            ValidationIssue(
                "schema_version",
                f"Unexpected schema_version '{schema_version}'. Expected {CANONICAL_SCHEMA_VERSION}.",
            )
        )

    question = data.get("question")
    if question is None or not isinstance(question, dict):
        warnings.append(ValidationIssue("question", "Missing question block; wizard will prompt."))
    else:
        text = question.get("text", "")
        if not isinstance(text, str) or not text.strip():
            warnings.append(ValidationIssue("question.text", "Question text missing; wizard will prompt."))

    q = data.get("q")
    if not isinstance(q, dict):
        errors.append(ValidationIssue("q", "Missing q block."))
    else:
        _validate_decode(q.get("decode"), errors)
        _validate_weighted_items(
            q.get("models", {}).get("items"),
            "slug",
            "q.models.items",
            errors,
            warnings,
            allow_empty=False,
        )
        _validate_weighted_items(
            q.get("personas", {}).get("items"),
            "id",
            "q.personas.items",
            errors,
            warnings,
            allow_empty=True,
        )

    protocol = data.get("protocol")
    if not isinstance(protocol, dict):
        errors.append(ValidationIssue("protocol", "Missing protocol block."))
    else:
        protocol_type = protocol.get("type")
        if protocol_type not in {"independent"}:
            errors.append(ValidationIssue("protocol.type", "Unsupported protocol type."))

    execution = data.get("execution")
    if not isinstance(execution, dict):
        errors.append(ValidationIssue("execution", "Missing execution block."))
    else:
        _require_positive_int(execution.get("k_max"), "execution.k_max", errors)
        _require_positive_int(execution.get("workers"), "execution.workers", errors)
        _require_positive_int(execution.get("batch_size"), "execution.batch_size", errors)
        _require_non_negative_int(execution.get("retries"), "execution.retries", errors)

    convergence = data.get("convergence")
    if not isinstance(convergence, dict):
        errors.append(ValidationIssue("convergence", "Missing convergence block."))
    else:
        _require_non_negative_number(convergence.get("delta_js_threshold"), "convergence.delta_js_threshold", errors)
        _require_non_negative_number(convergence.get("epsilon_new_threshold"), "convergence.epsilon_new_threshold", errors)
        epsilon_ci = convergence.get("epsilon_ci_half_width")
        if epsilon_ci is not None:
            _require_non_negative_number(epsilon_ci, "convergence.epsilon_ci_half_width", errors)
        _require_positive_int(convergence.get("min_trials"), "convergence.min_trials", errors)
        _require_positive_int(convergence.get("patience_batches"), "convergence.patience_batches", errors)

    clustering = data.get("clustering")
    if not isinstance(clustering, dict):
        errors.append(ValidationIssue("clustering", "Missing clustering block."))
    else:
        method = clustering.get("method")
        if method not in {"hash_baseline", "leader"}:
            errors.append(ValidationIssue("clustering.method", "Unsupported clustering method."))
        tau = clustering.get("tau")
        if tau is None:
            errors.append(ValidationIssue("clustering.tau", "Missing clustering threshold."))
        else:
            _require_positive_number(tau, "clustering.tau", errors)
        embed_text = clustering.get("embed_text")
        if embed_text not in {"outcome", "outcome+rationale"}:
            errors.append(ValidationIssue("clustering.embed_text", "Unsupported embed_text option."))

    llm = data.get("llm")
    if not isinstance(llm, dict):
        errors.append(ValidationIssue("llm", "Missing llm block."))
    else:
        mode = llm.get("mode")
        if mode not in {"mock", "remote", "openrouter"}:
            errors.append(ValidationIssue("llm.mode", "Unsupported llm mode."))
        model = llm.get("model")
        if not isinstance(model, str) or not model.strip():
            errors.append(ValidationIssue("llm.model", "Model slug is required."))
        _require_dict(llm.get("request_defaults"), "llm.request_defaults", errors)
        _require_dict(llm.get("routing_defaults"), "llm.routing_defaults", errors)
        _require_dict(llm.get("extra_body_defaults"), "llm.extra_body_defaults", errors)

    config = None
    if not errors:
        config = normalize_canonical_config(data, default_model=default_model, llm_mode=llm_mode)
    return ValidationResult(config=config, errors=errors, warnings=warnings)


def _validate_decode(decode: Any, errors: list[ValidationIssue]) -> None:
    if not isinstance(decode, dict):
        errors.append(ValidationIssue("q.decode", "Missing decode block."))
        return
    temperature = decode.get("temperature")
    if not isinstance(temperature, dict):
        errors.append(ValidationIssue("q.decode.temperature", "Missing temperature policy."))
        return
    temp_type = temperature.get("type")
    if temp_type not in {"fixed", "range"}:
        errors.append(ValidationIssue("q.decode.temperature.type", "Unsupported temperature policy type."))
        return
    if temp_type == "fixed":
        value = temperature.get("value")
        if value is None:
            errors.append(ValidationIssue("q.decode.temperature.value", "Missing fixed temperature value."))
        else:
            _require_number(value, "q.decode.temperature.value", errors)
    if temp_type == "range":
        min_val = temperature.get("min")
        max_val = temperature.get("max")
        if min_val is None or max_val is None:
            errors.append(ValidationIssue("q.decode.temperature", "Range policy requires min and max."))
            return
        _require_number(min_val, "q.decode.temperature.min", errors)
        _require_number(max_val, "q.decode.temperature.max", errors)
        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
            if max_val < min_val:
                errors.append(ValidationIssue("q.decode.temperature", "Range max must be >= min."))


def _validate_weighted_items(
    items: Any,
    key: str,
    path_prefix: str,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
    *,
    allow_empty: bool,
) -> None:
    if not isinstance(items, list):
        errors.append(ValidationIssue(path_prefix, "Must be a list."))
        return
    if not items:
        if allow_empty:
            warnings.append(ValidationIssue(path_prefix, "Empty list; defaults may be applied."))
            return
        errors.append(ValidationIssue(path_prefix, "Must be a non-empty list."))
        return
    weight_sum = 0.0
    weights_present = False
    for index, item in enumerate(items):
        item_path = f"{path_prefix}[{index}]"
        if isinstance(item, str):
            warnings.append(ValidationIssue(item_path, "Weight missing; defaulting to uniform."))
            continue
        if not isinstance(item, dict):
            errors.append(ValidationIssue(item_path, "Item must be an object or string."))
            continue
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(ValidationIssue(f"{item_path}.{key}", "Missing identifier."))
        if "weight" in item:
            weights_present = True
            weight = item.get("weight")
            if not isinstance(weight, (int, float)):
                errors.append(ValidationIssue(f"{item_path}.weight", "Weight must be numeric."))
                continue
            if weight <= 0:
                errors.append(ValidationIssue(f"{item_path}.weight", "Weight must be > 0."))
                continue
            weight_sum += float(weight)
        else:
            warnings.append(ValidationIssue(f"{item_path}.weight", "Weight missing; defaulting to uniform."))

    if weights_present:
        if weight_sum <= 0:
            errors.append(ValidationIssue(path_prefix, "Weights must sum to a positive value."))
        elif abs(weight_sum - 1.0) > 0.01:
            warnings.append(
                ValidationIssue(
                    path_prefix,
                    f"Weights sum to {weight_sum:.3f}; values will be normalized.",
                )
            )


def _require_dict(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if value is None:
        errors.append(ValidationIssue(path, "Missing required object."))
        return
    if not isinstance(value, dict):
        errors.append(ValidationIssue(path, "Must be an object."))


def _require_number(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if not isinstance(value, (int, float)):
        errors.append(ValidationIssue(path, "Must be a number."))


def _require_positive_number(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        errors.append(ValidationIssue(path, "Must be a positive number."))


def _require_non_negative_number(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if not isinstance(value, (int, float)) or value < 0:
        errors.append(ValidationIssue(path, "Must be a non-negative number."))


def _require_positive_int(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if not isinstance(value, int) or value < 1:
        errors.append(ValidationIssue(path, "Must be an integer >= 1."))


def _require_non_negative_int(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if not isinstance(value, int) or value < 0:
        errors.append(ValidationIssue(path, "Must be an integer >= 0."))
