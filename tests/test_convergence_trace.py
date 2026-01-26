from __future__ import annotations

import json
from pathlib import Path


def test_convergence_trace_invariants(mock_run) -> None:
    setup, _result = mock_run
    metrics_path = Path(setup.run_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trace = metrics.get("convergence_trace") or []
    assert isinstance(trace, list)
    prev_trials = 0
    expected_batch = 1
    for entry in trace:
        assert entry.get("batch_index") == expected_batch
        expected_batch += 1
        trials = int(entry.get("trials_completed_total", 0))
        assert trials >= prev_trials
        prev_trials = trials
        distribution = entry.get("distribution_by_cluster_id") or {}
        if distribution:
            total = sum(float(value) for value in distribution.values())
            assert abs(total - 1.0) < 1e-6
            counts = entry.get("counts_by_cluster_id") or {}
            if counts:
                assert sum(int(value) for value in counts.values()) == int(entry.get("valid_trials_total", 0))
