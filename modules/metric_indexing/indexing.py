"""Coordinate and external-topology implementations of ``BaseMetricIndex``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from core.base import BaseMetricIndex, MetricRelation
from data.phase2_graph import external_graph_search


class MetricIndex(BaseMetricIndex):
    """A small, production-facing adapter over the project's two ANN backends.

    ``faiss_hnsw`` builds a native multi-layer HNSW. ``external_graph`` searches
    a stored level-0 topology.  The latter deliberately does not claim to be a
    physical HNSW merge.
    """

    BACKENDS = {"faiss_hnsw", "external_graph"}

    def __init__(
        self,
        backend: str = "faiss_hnsw",
        *,
        graph: dict[str, Any] | None = None,
        item_ids: list[str] | None = None,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"Unsupported backend {backend!r}; expected one of {sorted(self.BACKENDS)}.")
        if min(int(m), int(ef_construction), int(ef_search)) <= 0:
            raise ValueError("m, ef_construction, and ef_search must be positive.")
        self.backend = backend
        self.graph = graph
        self._configured_item_ids = None if item_ids is None else [str(value) for value in item_ids]
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._index: faiss.Index | None = None
        self._vectors: np.ndarray | None = None
        self._item_ids: list[str] | None = None
        self._global_to_item: dict[int, str] = {}
        self._external_row: dict[int, int] | None = None
        self._external_corpus64: np.ndarray | None = None
        self.last_search_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _relation_ids(data: MetricRelation) -> list[str]:
        if hasattr(data, "ids"):
            ids = [str(value) for value in data.ids]
        elif hasattr(data, "metadata"):
            ids = [str(row["id"]) for row in data.metadata]
        else:
            raise ValueError("MetricRelation must expose row-aligned 'ids' or metadata with an 'id' field.")
        if not ids:
            raise ValueError("Cannot build an index over an empty relation.")
        if len(set(ids)) != len(ids):
            raise ValueError("MetricRelation contains duplicate item IDs.")
        return ids

    def build(self, data: MetricRelation) -> None:
        ids = self._relation_ids(data)
        vectors = np.asarray(data.get_vectors(ids), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0:
            raise ValueError("MetricRelation vectors must be a non-empty, row-aligned 2-D array.")
        if not np.isfinite(vectors).all():
            raise ValueError("MetricRelation vectors contain non-finite values.")
        vectors = np.ascontiguousarray(vectors)
        configured = self._configured_item_ids
        if configured is not None:
            if len(configured) != len(ids):
                raise ValueError("item_ids and relation vectors are not row-aligned.")
            if len(set(configured)) != len(configured):
                raise ValueError("item_ids contains duplicates.")
            ids = configured

        if self.backend == "faiss_hnsw":
            raw = faiss.IndexHNSWFlat(vectors.shape[1], self.m, faiss.METRIC_L2)
            raw.hnsw.efConstruction = self.ef_construction
            raw.hnsw.efSearch = self.ef_search
            index = faiss.IndexIDMap2(raw)
            index.add_with_ids(vectors, np.arange(len(ids), dtype=np.int64))
            self._index = index
        else:
            if self.graph is None:
                raise ValueError("external_graph backend requires graph=... at construction.")
            graph_ids = np.asarray(self.graph.get("node_global_ids", []), dtype=np.int64)
            if graph_ids.ndim != 1 or len(graph_ids) != len(vectors):
                raise ValueError("graph, vectors, and item IDs are not row-aligned.")
            if len(np.unique(graph_ids)) != len(graph_ids):
                raise ValueError("External graph contains duplicate global IDs.")
            nodes = set(int(value) for value in graph_ids)
            neighbor_nodes = {int(key) for key in self.graph.get("neighbors", {})}
            targets = {int(value) for values in self.graph.get("neighbors", {}).values() for value in values}
            if neighbor_nodes != nodes or not targets.issubset(nodes):
                raise ValueError("External graph adjacency references missing or extra global IDs.")
            self._global_to_item = {int(global_id): ids[row] for row, global_id in enumerate(graph_ids)}
            self._external_row = {
                int(global_id): row for row, global_id in enumerate(graph_ids)
            }
            self._external_corpus64 = vectors.astype(np.float64)

        self._vectors = vectors
        self._item_ids = ids
        self.last_search_diagnostics = {}

    @property
    def is_built(self) -> bool:
        return self._vectors is not None

    def _validate_query(self, query_vector: np.ndarray, k: int) -> np.ndarray:
        if not self.is_built or self._item_ids is None or self._vectors is None:
            raise RuntimeError("MetricIndex.search() called before build().")
        if not isinstance(k, (int, np.integer)) or k <= 0 or k > len(self._item_ids):
            raise ValueError(f"k must be between 1 and {len(self._item_ids)}.")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._vectors.shape[1]:
            raise ValueError(f"query_vector must have shape ({self._vectors.shape[1]},).")
        if not np.isfinite(query).all():
            raise ValueError("query_vector contains non-finite values.")
        return np.ascontiguousarray(query[None, :])

    @staticmethod
    def _reset_hnsw_stats() -> Any | None:
        stats = getattr(faiss.cvar, "hnsw_stats", None)
        if stats is not None and hasattr(stats, "reset"):
            stats.reset()
        return stats

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[str]:
        query = self._validate_query(query_vector, k)
        assert self._item_ids is not None and self._vectors is not None
        if self.backend == "faiss_hnsw":
            assert self._index is not None
            stats = self._reset_hnsw_stats()
            _, labels = self._index.search(query, int(k))
            self.last_search_diagnostics = {
                "backend": self.backend,
                "ndis": int(getattr(stats, "ndis", 0)) if stats is not None else None,
                "nhops": int(getattr(stats, "nhops", 0)) if stats is not None else None,
            }
            return [self._item_ids[int(label)] for label in labels[0] if label >= 0]

        assert self.graph is not None
        labels, _, diagnostics = external_graph_search(
            self.graph,
            self._vectors,
            query,
            k=int(k),
            ef_search=max(self.ef_search, int(k)),
            return_diagnostics=True,
            row_lookup=self._external_row,
            corpus_float64=self._external_corpus64,
        )
        self.last_search_diagnostics = {
            "backend": self.backend,
            "distance_evaluations": int(diagnostics["distance_evaluation_count"]),
            "visited_nodes": int(diagnostics["visited_node_count"]),
        }
        return [self._global_to_item[int(label)] for label in labels[0] if label >= 0]

    def serialized_size_bytes(self, path: str | Path) -> int:
        """Serialize the native index or external graph proxy and return its size."""
        if not self.is_built:
            raise RuntimeError("Cannot serialize before build().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss_hnsw":
            assert self._index is not None
            faiss.write_index(self._index, str(path))
        else:
            import json
            assert self.graph is not None
            payload = {
                "node_global_ids": [int(value) for value in self.graph["node_global_ids"]],
                "neighbors": {str(key): [int(value) for value in values] for key, values in self.graph["neighbors"].items()},
                "item_ids": self._item_ids,
            }
            path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        return path.stat().st_size

    def capacity_metrics(self, artifact_size_bytes: int | None = None) -> dict[str, int | None]:
        if not self.is_built or self._vectors is None or self._item_ids is None:
            raise RuntimeError("Capacity metrics require a built index.")
        id_bytes = sum(len(value.encode("utf-8")) for value in self._item_ids)
        if self.backend == "external_graph":
            assert self.graph is not None
            directed_edges = sum(len(values) for values in self.graph["neighbors"].values())
        else:
            raw = faiss.downcast_index(self._index.index)  # type: ignore[union-attr]
            directed_edges = int(np.sum(np.asarray(faiss.vector_to_array(raw.hnsw.neighbors)) >= 0))
        return {
            "vector_bytes": int(self._vectors.nbytes),
            "id_bytes_utf8": int(id_bytes),
            "directed_edge_count": int(directed_edges),
            "compact_adjacency_proxy_bytes": int(directed_edges * 8 + (len(self._item_ids) + 1) * 8),
            "serialized_artifact_bytes": None if artifact_size_bytes is None else int(artifact_size_bytes),
        }
