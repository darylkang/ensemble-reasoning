from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.utils import run_mock


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_deterministic_clusters(tmp_path: Path) -> None:
    setup_a, _ = await run_mock(tmp_path / "a", seed=42, run_name="det-a")
    setup_b, _ = await run_mock(tmp_path / "b", seed=42, run_name="det-b")

    parsed_a = {
        record["trial_id"]: record.get("cluster_id")
        for record in _load_jsonl(setup_a.run_dir / "parsed.jsonl")
        if record.get("parse_valid")
    }
    parsed_b = {
        record["trial_id"]: record.get("cluster_id")
        for record in _load_jsonl(setup_b.run_dir / "parsed.jsonl")
        if record.get("parse_valid")
    }
    assert parsed_a == parsed_b

    aggregates_a = json.loads((setup_a.run_dir / "aggregates.json").read_text(encoding="utf-8"))
    aggregates_b = json.loads((setup_b.run_dir / "aggregates.json").read_text(encoding="utf-8"))
    assert aggregates_a.get("counts_by_cluster_id") == aggregates_b.get("counts_by_cluster_id")
