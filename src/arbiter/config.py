"""Configuration models and resolution helpers for arbiter."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable
import os


@dataclass(frozen=True)
class TemperaturePolicy:
    kind: str
    temperatures: list[float]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "temperatures": self.temperatures,
        }


@dataclass(frozen=True)
class PersonaPolicy:
    persona_bank_path: str | None
    selection_mode: str
    persona_ids: list[str]
    loaded: bool

    def to_dict(self) -> dict:
        return {
            "persona_bank_path": self.persona_bank_path,
            "selection_mode": self.selection_mode,
            "persona_ids": self.persona_ids,
            "loaded": self.loaded,
        }


@dataclass(frozen=True)
class BudgetGuardrail:
    max_calls: int
    scope: str

    def to_dict(self) -> dict:
        return {
            "max_calls": self.max_calls,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class LLMRequestDefaults:
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    seed: int | None
    stop: list[str] | str | None
    response_format: dict[str, Any] | str | None
    tools: list[dict[str, Any]] | None
    tool_choice: dict[str, Any] | str | None
    parallel_tool_calls: bool | None

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": self.stop,
            "response_format": self.response_format,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
        }


@dataclass(frozen=True)
class LLMConfig:
    client: str
    mode: str
    model: str
    request_defaults: LLMRequestDefaults
    routing_defaults: dict[str, Any]
    extra_body_defaults: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "client": self.client,
            "mode": self.mode,
            "model": self.model,
            "request_defaults": self.request_defaults.to_dict(),
            "routing_defaults": self.routing_defaults,
            "extra_body_defaults": self.extra_body_defaults,
        }


@dataclass(frozen=True)
class ProtocolConfig:
    type: str

    def to_dict(self) -> dict:
        return {"type": self.type}


@dataclass(frozen=True)
class ClusteringConfig:
    method: str
    tau: float
    embed_text: str
    embedding_model: str

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "tau": self.tau,
            "embed_text": self.embed_text,
            "embedding_model": self.embedding_model,
        }


@dataclass(frozen=True)
class ConvergenceConfig:
    delta_js_threshold: float
    epsilon_new_threshold: float
    epsilon_ci_half_width: float | None
    min_trials: int
    patience_batches: int

    def to_dict(self) -> dict:
        return {
            "delta_js_threshold": self.delta_js_threshold,
            "epsilon_new_threshold": self.epsilon_new_threshold,
            "epsilon_ci_half_width": self.epsilon_ci_half_width,
            "min_trials": self.min_trials,
            "patience_batches": self.patience_batches,
        }


@dataclass(frozen=True)
class ExecutionConfig:
    worker_count: int
    batch_size: int
    max_retries: int
    seed: int
    parse_failure_policy: str
    convergence: ConvergenceConfig

    def to_dict(self) -> dict:
        return {
            "worker_count": self.worker_count,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "seed": self.seed,
            "parse_failure_policy": self.parse_failure_policy,
            "convergence": self.convergence.to_dict(),
        }

@dataclass(frozen=True)
class SummarizerConfig:
    enabled: bool
    model: str
    prompt_version: str

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


@dataclass(frozen=True)
class TrialBudget:
    k_max: int
    scope: str

    def to_dict(self) -> dict:
        return {
            "k_max": self.k_max,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class RunMetadata:
    name: str
    slug: str
    run_id: str
    output_base_dir: str
    output_dir: str
    started_at: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "run_id": self.run_id,
            "output_base_dir": self.output_base_dir,
            "output_dir": self.output_dir,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class QAtom:
    atom_id: str
    model: str
    temperature: float
    persona_id: str | None
    weight: float

    def to_dict(self) -> dict:
        return {
            "atom_id": self.atom_id,
            "model": self.model,
            "temperature": self.temperature,
            "persona_id": self.persona_id,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class QDistribution:
    strategy: dict
    atoms: list[QAtom]
    weights: dict

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "weights": self.weights,
        }


@dataclass(frozen=True)
class ResolvedConfig:
    schema_version: str
    run: RunMetadata
    semantic: "SemanticConfig"
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "semantic": self.semantic.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def build_from_wizard_inputs(
        cls,
        *,
        schema_version: str,
        run_name: str,
        run_slug: str,
        run_id: str,
        output_base_dir: str,
        output_dir: str,
        started_at: str,
        heterogeneity_rung: str,
        models: list[str],
        llm: "LLMConfig",
        protocol: "ProtocolConfig",
        clustering: "ClusteringConfig",
        summarizer: "SummarizerConfig",
        execution: "ExecutionConfig",
        temperature_policy: TemperaturePolicy,
        personas: PersonaPolicy,
        trial_budget: TrialBudget,
        budget_guardrail: BudgetGuardrail,
        q_distribution: QDistribution,
        notes: list[str],
    ) -> "ResolvedConfig":
        run = RunMetadata(
            name=run_name,
            slug=run_slug,
            run_id=run_id,
            output_base_dir=output_base_dir,
            output_dir=output_dir,
            started_at=started_at,
        )
        semantic = SemanticConfig(
            heterogeneity_rung=heterogeneity_rung,
            models=models,
            llm=llm,
            protocol=protocol,
            clustering=clustering,
            summarizer=summarizer,
            execution=execution,
            temperature_policy=temperature_policy,
            personas=personas,
            trial_budget=trial_budget,
            budget_guardrail=budget_guardrail,
            q_distribution=q_distribution,
        )
        return cls(schema_version=schema_version, run=run, semantic=semantic, notes=notes)


@dataclass(frozen=True)
class SemanticConfig:
    heterogeneity_rung: str
    models: list[str]
    llm: LLMConfig
    protocol: ProtocolConfig
    clustering: ClusteringConfig
    summarizer: SummarizerConfig
    execution: ExecutionConfig
    temperature_policy: TemperaturePolicy
    personas: PersonaPolicy
    trial_budget: TrialBudget
    budget_guardrail: BudgetGuardrail
    q_distribution: QDistribution

    def to_dict(self) -> dict:
        return {
            "heterogeneity_rung": self.heterogeneity_rung,
            "models": self.models,
            "llm": self.llm.to_dict(),
            "protocol": self.protocol.to_dict(),
            "clustering": self.clustering.to_dict(),
            "summarizer": self.summarizer.to_dict(),
            "execution": self.execution.to_dict(),
            "temperature_policy": self.temperature_policy.to_dict(),
            "personas": self.personas.to_dict(),
            "trial_budget": self.trial_budget.to_dict(),
            "budget_guardrail": self.budget_guardrail.to_dict(),
            "q_distribution": self.q_distribution.to_dict(),
        }

@dataclass(frozen=True)
class PersonaLoadResult:
    persona_ids: list[str]
    loaded: bool
    error: str | None


CANONICAL_SCHEMA_VERSION = "1.2"
SUMMARIZER_PROMPT_VERSION = "v1"


def default_embedding_model() -> str:
    return os.getenv("ARBITER_EMBEDDING_MODEL", "openai/text-embedding-3-large")


def default_summarizer_model() -> str:
    return os.getenv("ARBITER_SUMMARIZER_MODEL", "openai/gpt-5")


def default_canonical_config(
    *,
    default_model: str,
    llm_mode: str,
    embedding_model: str | None = None,
    summarizer_model: str | None = None,
) -> dict[str, Any]:
    embedding_model = embedding_model or default_embedding_model()
    summarizer_model = summarizer_model or default_summarizer_model()
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "question": {
            "id": None,
            "text": "",
            "metadata": {},
        },
        "q": {
            "decode": {
                "temperature": {"type": "fixed", "value": 0.7},
                "extra": {},
            },
            "personas": {
                "items": [],
                "default_behavior": "neutral_if_empty",
            },
            "models": {
                "items": [{"slug": default_model, "weight": 1.0}],
            },
        },
        "protocol": {"type": "independent"},
        "execution": {
            "k_max": 1000,
            "workers": 8,
            "batch_size": 8,
            "retries": 2,
            "seed": None,
            "parse_failure_policy": "continue",
        },
        "convergence": {
            "delta_js_threshold": 0.02,
            "epsilon_new_threshold": 0.01,
            "epsilon_ci_half_width": 0.05,
            "min_trials": 64,
            "patience_batches": 2,
        },
        "clustering": {
            "method": "leader",
            "tau": 0.85,
            "embed_text": "outcome",
            "embedding_model": embedding_model,
        },
        "summarizer": {
            "enabled": False,
            "model": summarizer_model,
            "prompt_version": SUMMARIZER_PROMPT_VERSION,
        },
        "llm": {
            "mode": llm_mode,
            "model": default_model,
            "request_defaults": {},
            "routing_defaults": {"allow_fallbacks": False},
            "extra_body_defaults": {},
        },
    }


def normalize_canonical_config(
    data: dict[str, Any],
    *,
    default_model: str,
    llm_mode: str,
    embedding_model: str | None = None,
    summarizer_model: str | None = None,
) -> dict[str, Any]:
    base = default_canonical_config(
        default_model=default_model,
        llm_mode=llm_mode,
        embedding_model=embedding_model,
        summarizer_model=summarizer_model,
    )
    merged = _merge_dicts(base, data)
    question = merged.setdefault("question", {})
    question.setdefault("id", None)
    question.setdefault("text", "")
    question.setdefault("metadata", {})

    q = merged.setdefault("q", {})
    decode = q.setdefault("decode", {})
    temperature = decode.setdefault("temperature", {"type": "fixed", "value": 0.7})
    decode.setdefault("extra", {})
    if temperature.get("type") not in {"fixed", "range"}:
        temperature["type"] = "fixed"
        temperature["value"] = 0.7

    personas = q.setdefault("personas", {})
    personas.setdefault("items", [])
    personas.setdefault("default_behavior", "neutral_if_empty")

    models = q.setdefault("models", {})
    models.setdefault("items", [{"slug": default_model, "weight": 1.0}])

    merged.setdefault("protocol", {"type": "independent"})
    merged.setdefault("execution", base["execution"])
    merged.setdefault("convergence", base["convergence"])
    merged.setdefault("clustering", base["clustering"])
    merged.setdefault("summarizer", base["summarizer"])
    merged.setdefault("llm", base["llm"])

    execution = merged.setdefault("execution", base["execution"])
    execution.setdefault("seed", None)
    execution.setdefault("parse_failure_policy", "continue")

    merged["q"]["models"]["items"] = _normalize_weighted_items(models["items"], key="slug")
    merged["q"]["personas"]["items"] = _normalize_weighted_items(personas["items"], key="id")

    if question.get("text") and not question.get("id"):
        question["id"] = f"question_{_hash_text(question['text'])[:10]}"

    return merged


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        result[key] = value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_weighted_items(items: list[Any], *, key: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({key: item, "weight": 1.0})
        elif isinstance(item, dict):
            value = item.get(key)
            if value:
                normalized.append({key: value, "weight": float(item.get("weight", 1.0))})
    if not normalized:
        return []
    total = sum(item["weight"] for item in normalized if item["weight"] > 0)
    if total <= 0:
        total = float(len(normalized))
        for item in normalized:
            item["weight"] = 1.0
    for item in normalized:
        item["weight"] = item["weight"] / total
    normalized.sort(key=lambda item: str(item.get(key)))
    return normalized


def materialize_q_distribution(config: dict[str, Any]) -> tuple[QDistribution, TemperaturePolicy, PersonaPolicy, list[str]]:
    q = config["q"]
    decode = q["decode"]
    temperature_config = decode["temperature"]
    temp_type = temperature_config.get("type", "fixed")
    temperatures: list[float]
    if temp_type == "range":
        temp_min = float(temperature_config.get("min", 0.2))
        temp_max = float(temperature_config.get("max", 1.0))
        if temp_max < temp_min:
            temp_min, temp_max = temp_max, temp_min
        temperatures = [temp_min, temp_max]
        temperature_policy = TemperaturePolicy(kind="range", temperatures=temperatures)
        temp_values = [round((temp_min + temp_max) / 2, 6)]
    else:
        temp_value = float(temperature_config.get("value", 0.7))
        temperatures = [temp_value]
        temperature_policy = TemperaturePolicy(kind="fixed", temperatures=temperatures)
        temp_values = temperatures

    models = q["models"]["items"]
    personas = q["personas"]["items"]
    default_behavior = q["personas"].get("default_behavior", "neutral_if_empty")

    model_items = _normalize_weighted_items(models, key="slug")
    persona_items = _normalize_weighted_items(personas, key="id")
    if not persona_items and default_behavior == "neutral_if_empty":
        persona_items = [{"id": None, "weight": 1.0}]

    atoms: list[QAtom] = []
    combos = list(product(model_items, persona_items, temp_values))
    weights: list[float] = []
    for model_item, persona_item, temperature in combos:
        weight = model_item["weight"] * persona_item["weight"]
        weights.append(weight)
        atom_id = _stable_atom_id(model_item["slug"], persona_item.get("id"), temperature)
        atoms.append(
            QAtom(
                atom_id=atom_id,
                model=model_item["slug"],
                temperature=temperature,
                persona_id=persona_item.get("id"),
                weight=weight,
            )
        )
    total_weight = sum(weights) if weights else 1.0
    atoms = [
        QAtom(
            atom_id=atom.atom_id,
            model=atom.model,
            temperature=atom.temperature,
            persona_id=atom.persona_id,
            weight=atom.weight / total_weight,
        )
        for atom in atoms
    ]

    strategy = {
        "name": "factorized",
        "description": "Weighted cartesian product over models, personas, and decoding policy.",
        "inputs": {
            "models": model_items,
            "personas": persona_items,
            "temperature": temperature_config,
            "decode_extra": decode.get("extra", {}),
        },
    }
    weights_info = {"type": "normalized", "sum": round(sum(atom.weight for atom in atoms), 6)}
    q_distribution = QDistribution(strategy=strategy, atoms=atoms, weights=weights_info)

    persona_policy = PersonaPolicy(
        persona_bank_path=None,
        selection_mode="weighted" if persona_items else "none",
        persona_ids=[item.get("id") for item in persona_items if item.get("id")],
        loaded=bool(persona_items),
    )
    model_slugs = [item["slug"] for item in model_items]
    return q_distribution, temperature_policy, persona_policy, model_slugs


def _stable_atom_id(model_slug: str, persona_id: str | None, temperature: float) -> str:
    payload = f"{model_slug}|{persona_id or ''}|{temperature:.6f}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"atom_{digest[:12]}"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify_run_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "auto"


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _extract_ids_from_json(data: object) -> list[str]:
    ids: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    ids.append(candidate)
            elif isinstance(item, dict):
                candidate = item.get("id") or item.get("persona_id")
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate.strip())
    elif isinstance(data, dict):
        personas = data.get("personas")
        if isinstance(personas, list):
            ids.extend(_extract_ids_from_json(personas))
        else:
            candidate = data.get("id") or data.get("persona_id")
            if isinstance(candidate, str) and candidate.strip():
                ids.append(candidate.strip())
    elif isinstance(data, str):
        candidate = data.strip()
        if candidate:
            ids.append(candidate)
    return ids


def _extract_ids_from_lines(text: str) -> list[str]:
    ids: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if cleaned.startswith("{"):
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            ids.extend(_extract_ids_from_json(data))
        else:
            ids.append(cleaned)
    return ids


def load_persona_ids(path: Path) -> PersonaLoadResult:
    if not path.exists():
        return PersonaLoadResult([], False, f"persona bank not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return PersonaLoadResult([], False, f"persona bank unreadable: {exc}")

    if not text.strip():
        return PersonaLoadResult([], False, "persona bank is empty")

    ids: list[str] = []
    try:
        data = json.loads(text)
        ids = _extract_ids_from_json(data)
    except json.JSONDecodeError:
        ids = _extract_ids_from_lines(text)

    ids = dedupe_preserve_order([value for value in ids if value])
    if not ids:
        return PersonaLoadResult([], False, "persona bank contains no ids")

    return PersonaLoadResult(ids, True, None)


def build_q_distribution(
    models: list[str],
    temperatures: list[float],
    persona_ids: list[str] | None,
) -> QDistribution:
    normalized_personas = persona_ids or [None]
    atoms: list[QAtom] = []
    combos = list(product(models, temperatures, normalized_personas))
    weight = 1.0 / len(combos)
    for index, (model, temperature, persona_id) in enumerate(combos, start=1):
        atoms.append(
            QAtom(
                atom_id=f"atom_{index:04d}",
                model=model,
                temperature=temperature,
                persona_id=persona_id,
                weight=weight,
            )
        )

    strategy = {
        "name": "cartesian_uniform",
        "description": "Uniform weights over cartesian product of models, temperatures, and personas.",
        "inputs": {
            "models": models,
            "temperatures": temperatures,
            "personas": normalized_personas,
        },
    }
    weights = {
        "type": "uniform",
        "sum": round(weight * len(atoms), 6),
    }
    return QDistribution(strategy=strategy, atoms=atoms, weights=weights)
