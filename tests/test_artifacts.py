from __future__ import annotations

from pathlib import Path


def test_artifacts_present(mock_run) -> None:
    setup, _result = mock_run
    run_dir: Path = setup.run_dir
    expected = {
        "question.json",
        "trials.jsonl",
        "parsed.jsonl",
        "embeddings.jsonl",
        "clusters_online.json",
        "clusters_offline.json",
        "cluster_summaries.json",
        "metrics.json",
        "aggregates.json",
        "manifest.json",
        "config.input.json",
        "config.resolved.json",
    }
    found = {path.name for path in run_dir.iterdir()}
    missing = expected - found
    assert not missing, f"Missing artifacts: {sorted(missing)}"
