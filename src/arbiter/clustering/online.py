"""Online leader clustering (cosine similarity)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vec)) or 1.0
    return [value / norm for value in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Cluster:
    cluster_id: str
    centroid: list[float]
    count: int = 0
    exemplars: list[dict[str, Any]] = field(default_factory=list)


class OnlineLeaderClustering:
    def __init__(self, tau: float, max_exemplars: int = 3) -> None:
        self.tau = tau
        self.max_exemplars = max_exemplars
        self.clusters: list[Cluster] = []

    def assign(
        self,
        embedding: list[float],
        *,
        trial_id: str,
        outcome: str,
        rationale: str | None,
    ) -> tuple[str, bool]:
        embedding = _normalize(embedding)
        if not self.clusters:
            cluster = self._create_cluster(embedding, trial_id, outcome, rationale)
            return cluster.cluster_id, True

        best_idx = -1
        best_sim = -1.0
        for idx, cluster in enumerate(self.clusters):
            sim = _cosine(embedding, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= self.tau and best_idx >= 0:
            cluster = self.clusters[best_idx]
            self._update_cluster(cluster, embedding, trial_id, outcome, rationale)
            return cluster.cluster_id, False

        cluster = self._create_cluster(embedding, trial_id, outcome, rationale)
        return cluster.cluster_id, True

    def _create_cluster(
        self,
        embedding: list[float],
        trial_id: str,
        outcome: str,
        rationale: str | None,
    ) -> Cluster:
        cluster_id = f"cluster_{len(self.clusters) + 1:04d}"
        cluster = Cluster(cluster_id=cluster_id, centroid=embedding, count=1)
        cluster.exemplars.append({"trial_id": trial_id, "outcome": outcome, "rationale": rationale})
        self.clusters.append(cluster)
        return cluster

    def _update_cluster(
        self,
        cluster: Cluster,
        embedding: list[float],
        trial_id: str,
        outcome: str,
        rationale: str | None,
    ) -> None:
        cluster.count += 1
        updated = [
            (cluster.centroid[i] * (cluster.count - 1) + embedding[i]) / cluster.count
            for i in range(len(embedding))
        ]
        cluster.centroid = _normalize(updated)
        if len(cluster.exemplars) < self.max_exemplars:
            cluster.exemplars.append({"trial_id": trial_id, "outcome": outcome, "rationale": rationale})

    def export(self) -> list[Cluster]:
        return list(self.clusters)
