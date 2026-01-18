"""Configuration models and resolution helpers for arbiter."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import re
from typing import Iterable


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
    temperature_policy: TemperaturePolicy
    personas: PersonaPolicy
    trial_budget: TrialBudget
    budget_guardrail: BudgetGuardrail
    q_distribution: QDistribution

    def to_dict(self) -> dict:
        return {
            "heterogeneity_rung": self.heterogeneity_rung,
            "models": self.models,
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
