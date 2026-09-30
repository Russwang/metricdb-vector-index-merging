"""External HNSW graph extraction, repair, search, and diagnostics."""

from __future__ import annotations

import heapq
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np


def unwrap_hnsw(index: faiss.Index) -> tuple[faiss.IndexHNSW, np.ndarray]:
    """Return the underlying HNSW index and row-aligned external IDs."""
    raw = faiss.downcast_index(index.index) if hasattr(index, "index") else index
    external_ids = (
        np.asarray(faiss.vector_to_array(index.id_map), dtype=np.int64)
        if hasattr(index, "id_map")
        else np.arange(index.ntotal, dtype=np.int64)
    )
    if not isinstance(raw, faiss.IndexHNSW):
        raise TypeError(f"Expected HNSW, got {type(raw).__name__}.")
    if len(external_ids) != raw.ntotal:
        raise ValueError("HNSW row count and external ID count differ.")
    return raw, external_ids


def _edge_distances(
    node_ids: np.ndarray,
    neighbors: dict[int, list[int]],
    vectors: np.ndarray,
) -> dict[int, list[float]]:
    row = {int(node): index for index, node in enumerate(node_ids)}
    result: dict[int, list[float]] = {}
    for node, targets in neighbors.items():
        source = vectors[row[node]].astype(np.float64)
        result[node] = [
            float(np.sum((source - vectors[row[target]].astype(np.float64)) ** 2))
            for target in targets
        ]
    return result


def extract_raw_level0_graph(
    index: faiss.Index,
    vectors: np.ndarray,
    source_subset: str,
    degree_budget_policy: str = "original_degree",
) -> dict[str, Any]:
    """Read actual FAISS level-0 links without mutating the index."""
    raw, external_ids = unwrap_hnsw(index)
    hnsw = raw.hnsw
    levels = np.asarray(faiss.vector_to_array(hnsw.levels), dtype=np.int64)
    flat_neighbors = np.asarray(
        faiss.vector_to_array(hnsw.neighbors), dtype=np.int64
    )
    neighbors: dict[int, list[int]] = {}
    upper: dict[int, list[int]] = {}
    for internal_id, level_count in enumerate(levels):
        node = int(external_ids[internal_id])
        begin = np.zeros(1, dtype=np.uint64)
        end = np.zeros(1, dtype=np.uint64)
        hnsw.neighbor_range(
            internal_id, 0, faiss.swig_ptr(begin), faiss.swig_ptr(end)
        )
        internal_targets = flat_neighbors[int(begin[0]) : int(end[0])]
        neighbors[node] = [
            int(external_ids[target])
            for target in internal_targets
            if target >= 0 and target != internal_id
        ]
        for layer in range(1, int(level_count)):
            upper.setdefault(layer, []).append(node)

    original_degrees = np.array(
        [len(neighbors[int(node)]) for node in external_ids], dtype=np.int64
    )
    max_m0 = int(hnsw.nb_neighbors(0))
    degree_budget = (
        original_degrees.copy()
        if degree_budget_policy == "original_degree"
        else np.full(len(external_ids), max_m0, dtype=np.int64)
    )
    entry_internal = int(hnsw.entry_point)
    entry_global = (
        int(external_ids[entry_internal]) if entry_internal >= 0 else -1
    )
    return {
        "node_global_ids": external_ids,
        "neighbors": neighbors,
        "edge_distances": _edge_distances(external_ids, neighbors, vectors),
        "source_subset": source_subset,
        "graph_kind": "raw_level0",
        "directed": True,
        "node_levels": levels,
        "upper_layer_nodes_by_level": [
            upper.get(layer, []) for layer in range(1, int(levels.max()))
        ],
        "original_entry_point": entry_global,
        "hnsw_M": int(hnsw.nb_neighbors(1)) if int(levels.max()) > 1 else max_m0 // 2,
        "maxM0": max_m0,
        "original_degrees": original_degrees,
        "degree_budget": degree_budget,
    }


def extract_query_derived_graph(
    index: faiss.Index,
    local_vectors: np.ndarray,
    transformed_vectors: np.ndarray,
    source_subset: str,
    k: int,
    degree_budget_policy: str = "original_degree",
) -> dict[str, Any]:
    """Build the explicitly labelled search-result graph fallback."""
    raw, external_ids = unwrap_hnsw(index)
    effective_k = min(max(1, k), len(local_vectors) - 1)
    _, result_ids = index.search(
        np.ascontiguousarray(local_vectors, dtype=np.float32), effective_k + 1
    )
    neighbors: dict[int, list[int]] = {}
    for node, candidates in zip(external_ids, result_ids):
        neighbors[int(node)] = [
            int(target)
            for target in candidates
            if target >= 0 and target != node
        ][:effective_k]
    levels = np.asarray(faiss.vector_to_array(raw.hnsw.levels), dtype=np.int64)
    original_degrees = np.full(len(external_ids), effective_k, dtype=np.int64)
    max_m0 = int(raw.hnsw.nb_neighbors(0))
    degree_budget = (
        original_degrees.copy()
        if degree_budget_policy == "original_degree"
        else np.full(len(external_ids), max_m0, dtype=np.int64)
    )
    entry = int(raw.hnsw.entry_point)
    return {
        "node_global_ids": external_ids,
        "neighbors": neighbors,
        "edge_distances": _edge_distances(
            external_ids, neighbors, transformed_vectors
        ),
        "source_subset": source_subset,
        "graph_kind": "query_derived",
        "directed": True,
        "node_levels": levels,
        "upper_layer_nodes_by_level": [],
        "original_entry_point": int(external_ids[entry]) if entry >= 0 else -1,
        "hnsw_M": int(raw.hnsw.nb_neighbors(1)),
        "maxM0": max_m0,
        "original_degrees": original_degrees,
        "degree_budget": degree_budget,
    }


def weak_components(graph: dict[str, Any]) -> list[list[int]]:
    """Return weakly connected components of a directed graph."""
    nodes = [int(value) for value in graph["node_global_ids"]]
    undirected = {node: set() for node in nodes}
    for source, targets in graph["neighbors"].items():
        for target in targets:
            undirected[source].add(target)
            undirected[target].add(source)
    components: list[list[int]] = []
    remaining = set(nodes)
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for target in undirected[node]:
                if target in remaining:
                    remaining.remove(target)
                    queue.append(target)
        components.append(sorted(component))
    return components


def directed_reachable_from_entry(graph: dict[str, Any]) -> set[int]:
    """Return nodes reachable by directed outgoing links from the entry point."""
    nodes = {int(value) for value in graph["node_global_ids"]}
    entry = int(graph["original_entry_point"])
    if entry not in nodes:
        entry = min(nodes) if nodes else -1
    if entry < 0:
        return set()
    reachable = {entry}
    queue = deque([entry])
    while queue:
        node = queue.popleft()
        for target in graph["neighbors"].get(node, []):
            if target in nodes and target not in reachable:
                reachable.add(target)
                queue.append(target)
    return reachable


def repair_graph(
    graph: dict[str, Any],
    local_index: faiss.Index,
    local_vectors: np.ndarray,
    transformed_vectors: np.ndarray,
    candidate_search_k: int,
    force_connectivity: bool = True,
    selected_node_ids: set[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Repair all nodes or a selected subset using local-HNSW candidates."""
    started = perf_counter()
    node_ids = np.asarray(graph["node_global_ids"], dtype=np.int64)
    row = {int(node): index for index, node in enumerate(node_ids)}
    budgets = {
        int(node): int(graph["degree_budget"][index])
        for index, node in enumerate(node_ids)
    }
    candidate_started = perf_counter()
    search_k = min(len(node_ids), candidate_search_k + 1)
    if selected_node_ids is None:
        selected = [int(node) for node in node_ids]
    else:
        unknown = set(int(node) for node in selected_node_ids) - set(row)
        if unknown:
            raise ValueError(f"Selected repair nodes are not in graph: {sorted(unknown)[:5]}")
        selected = sorted(set(int(node) for node in selected_node_ids))
    selected_rows = [row[node] for node in selected]
    _, search_ids = local_index.search(
        np.ascontiguousarray(local_vectors[selected_rows], dtype=np.float32), search_k
    )
    new_neighbors: dict[int, list[int]] = {
        int(node): list(targets) for node, targets in graph["neighbors"].items()
    }
    distance_count = 0
    for search_index, node in enumerate(selected):
        index = row[node]
        candidates = set(graph["neighbors"][node])
        for first in graph["neighbors"][node]:
            candidates.update(graph["neighbors"].get(first, []))
        candidates.update(int(value) for value in search_ids[search_index] if value >= 0)
        candidates.discard(node)
        candidates.intersection_update(row)
        source = transformed_vectors[index].astype(np.float64)
        ranked = []
        for candidate in candidates:
            distance = float(
                np.sum(
                    (
                        source
                        - transformed_vectors[row[candidate]].astype(np.float64)
                    )
                    ** 2
                )
            )
            ranked.append((distance, candidate))
        distance_count += len(ranked)
        ranked.sort()
        new_neighbors[node] = [
            candidate for _, candidate in ranked[: budgets[node]]
        ]
    candidate_generation_seconds = perf_counter() - candidate_started

    repaired = dict(graph)
    repaired["neighbors"] = new_neighbors
    repaired["edge_distances"] = _edge_distances(
        node_ids, new_neighbors, transformed_vectors
    )
    repaired["variant"] = "practical_repair"
    protected: list[tuple[int, int]] = []
    if force_connectivity:
        components = weak_components(repaired)
        while len(components) > 1:
            base = components[0]
            best: tuple[float, int, int] | None = None
            for other in components[1:]:
                for source in base:
                    delta = (
                        transformed_vectors[row[source]].astype(np.float64)[None, :]
                        - transformed_vectors[
                            [row[target] for target in other]
                        ].astype(np.float64)
                    )
                    distances = np.sum(delta**2, axis=1)
                    distance_count += len(other)
                    target_index = int(np.argmin(distances))
                    candidate = (
                        float(distances[target_index]),
                        source,
                        other[target_index],
                    )
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                break
            _, source, target = best
            if target not in repaired["neighbors"][source]:
                repaired["neighbors"][source].append(target)
                protected.append((source, target))
            components = weak_components(repaired)
        reachable = directed_reachable_from_entry(repaired)
        missing = set(int(node) for node in node_ids) - reachable
        while missing and reachable:
            best: tuple[float, int, int] | None = None
            missing_list = sorted(missing)
            for source in sorted(reachable):
                delta = (
                    transformed_vectors[row[source]].astype(np.float64)[None, :]
                    - transformed_vectors[
                        [row[target] for target in missing_list]
                    ].astype(np.float64)
                )
                distances = np.sum(delta**2, axis=1)
                distance_count += len(missing_list)
                target_index = int(np.argmin(distances))
                candidate = (
                    float(distances[target_index]),
                    source,
                    missing_list[target_index],
                )
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                break
            _, source, target = best
            if target not in repaired["neighbors"][source]:
                repaired["neighbors"][source].append(target)
                protected.append((source, target))
            reachable = directed_reachable_from_entry(repaired)
            missing = set(int(node) for node in node_ids) - reachable
        repaired["edge_distances"] = _edge_distances(
            node_ids, repaired["neighbors"], transformed_vectors
        )

    exceptions = sum(
        len(repaired["neighbors"][node]) > budgets[node] for node in budgets
    )
    diagnostics = graph_diagnostics(graph, repaired)
    diagnostics.update(
        {
            "candidate_distance_computations": distance_count,
            "candidate_generation_seconds": candidate_generation_seconds,
            "protected_bridge_edges": [list(edge) for edge in protected],
            "degree_budget_exceptions": exceptions,
            "unresolved_component_count": max(0, len(weak_components(repaired)) - 1),
            "directed_unreachable_from_entry_count": (
                len(node_ids) - len(directed_reachable_from_entry(repaired))
            ),
            "processed_node_ratio": len(selected) / len(node_ids),
            "processed_node_count": len(selected),
            "repair_seconds": perf_counter() - started,
        }
    )
    return repaired, diagnostics


def structural_oracle_graph(
    graph: dict[str, Any],
    transformed_vectors: np.ndarray,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Construct a batched exact-kNN structural oracle with matching budgets."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    node_ids = np.asarray(graph["node_global_ids"], dtype=np.int64)
    vectors = transformed_vectors.astype(np.float64)
    budgets = np.asarray(graph["degree_budget"], dtype=np.int64)
    max_budget = min(int(np.max(budgets, initial=0)), len(node_ids) - 1)
    neighbors: dict[int, list[int]] = {}
    if max_budget <= 0:
        neighbors = {int(node): [] for node in node_ids}
    else:
        corpus_norms = np.sum(vectors**2, axis=1)
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            batch = vectors[start:end]
            distances = (
                np.sum(batch**2, axis=1)[:, None]
                + corpus_norms[None, :]
                - 2.0 * (batch @ vectors.T)
            )
            distances = np.maximum(distances, 0.0)
            distances[np.arange(end - start), np.arange(start, end)] = np.inf
            candidate_indices = np.argpartition(
                distances, kth=max_budget - 1, axis=1
            )[:, :max_budget]
            for row_offset, candidates in enumerate(candidate_indices):
                index = start + row_offset
                budget = min(int(budgets[index]), len(node_ids) - 1)
                if budget <= 0:
                    neighbors[int(node_ids[index])] = []
                    continue
                candidate_distances = distances[row_offset, candidates]
                candidate_ids = node_ids[candidates]
                order = np.lexsort((candidate_ids, candidate_distances))
                selected = candidates[order[:budget]]
                neighbors[int(node_ids[index])] = [
                    int(node_ids[target]) for target in selected
                ]
    result = dict(graph)
    result["graph_kind"] = "exact_knn"
    result["variant"] = "structural_oracle"
    result["neighbors"] = neighbors
    result["edge_distances"] = _edge_distances(
        node_ids, neighbors, transformed_vectors
    )
    return result


def graph_with_updated_distances(
    graph: dict[str, Any], transformed_vectors: np.ndarray, variant: str
) -> dict[str, Any]:
    """Copy a topology and recompute all edge distances in the current space."""
    result = dict(graph)
    result["neighbors"] = {
        int(node): list(targets) for node, targets in graph["neighbors"].items()
    }
    result["edge_distances"] = _edge_distances(
        np.asarray(graph["node_global_ids"], dtype=np.int64),
        result["neighbors"],
        transformed_vectors,
    )
    result["variant"] = variant
    return result


def external_graph_search(
    graph: dict[str, Any],
    corpus_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    ef_search: int,
    exclude_global_ids: np.ndarray | None = None,
    distance_batch_size: int = 256,
    return_diagnostics: bool = False,
    row_lookup: dict[int, int] | None = None,
    corpus_float64: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run deterministic bounded best-first traversal on an external graph.

    Distances are evaluated lazily as nodes are visited.  Earlier versions
    materialised a query-by-corpus distance matrix before graph traversal,
    which made the reported external-search latency and work quadratic in the
    corpus size even when ``ef_search`` was small.
    """
    if distance_batch_size <= 0:
        raise ValueError("distance_batch_size must be positive.")
    node_ids = np.asarray(graph["node_global_ids"], dtype=np.int64)
    row = (
        {int(node): index for index, node in enumerate(node_ids)}
        if row_lookup is None
        else row_lookup
    )
    if len(row) != len(node_ids):
        raise ValueError("row_lookup is not aligned with graph node IDs.")
    entry = int(graph["original_entry_point"])
    if entry not in row:
        entry = int(node_ids[0])
    effective_k = min(k, len(node_ids) - (1 if exclude_global_ids is not None else 0))
    result_ids = np.full((len(query_vectors), effective_k), -1, dtype=np.int64)
    result_distances = np.full(
        (len(query_vectors), effective_k), np.inf, dtype=np.float32
    )
    corpus = (
        corpus_vectors.astype(np.float64)
        if corpus_float64 is None
        else np.asarray(corpus_float64, dtype=np.float64)
    )
    if corpus.shape != corpus_vectors.shape:
        raise ValueError("corpus_float64 is not aligned with corpus_vectors.")
    queries = query_vectors.astype(np.float64)
    total_distance_evaluations = 0
    total_visited_nodes = 0
    per_query_distance_evaluations = np.zeros(len(queries), dtype=np.int64)
    per_query_visited_nodes = np.zeros(len(queries), dtype=np.int64)
    for start in range(0, len(queries), distance_batch_size):
        end = min(start + distance_batch_size, len(queries))
        batch = queries[start:end]
        for offset, query in enumerate(batch):
            query_index = start + offset

            distance_cache: dict[int, float] = {}

            def distance(node: int) -> float:
                if node not in distance_cache:
                    delta = corpus[row[node]] - query
                    distance_cache[node] = float(np.dot(delta, delta))
                return distance_cache[node]

            candidates = [(distance(entry), entry)]
            visited = {entry}
            evaluated: list[tuple[float, int]] = []
            while candidates and len(evaluated) < max(ef_search, effective_k):
                current_distance, node = heapq.heappop(candidates)
                evaluated.append((current_distance, node))
                unseen = [
                    int(target)
                    for target in graph["neighbors"].get(node, [])
                    if target not in visited
                ]
                if unseen:
                    # A graph expansion commonly exposes tens of neighbours.
                    # Compute that frontier as one NumPy operation while retaining
                    # exactly the same lazy distance cache and work accounting.
                    unseen_rows = np.fromiter(
                        (row[target] for target in unseen), dtype=np.int64
                    )
                    delta = corpus[unseen_rows] - query
                    frontier_distances = np.einsum("ij,ij->i", delta, delta)
                    for target, target_distance in zip(unseen, frontier_distances):
                        visited.add(target)
                        distance_cache[target] = float(target_distance)
                        heapq.heappush(candidates, (float(target_distance), target))
            per_query_distance_evaluations[query_index] = len(distance_cache)
            per_query_visited_nodes[query_index] = len(evaluated)
            total_distance_evaluations += len(distance_cache)
            total_visited_nodes += len(evaluated)
            excluded = (
                int(exclude_global_ids[query_index])
                if exclude_global_ids is not None
                else None
            )
            selected = sorted(
                (distance_value, node)
                for distance_value, node in evaluated
                if node != excluded
            )[:effective_k]
            for rank, (distance_value, node) in enumerate(selected):
                result_ids[query_index, rank] = node
                result_distances[query_index, rank] = distance_value
    if not return_diagnostics:
        return result_ids, result_distances
    diagnostics = {
        "distance_evaluation_count": int(total_distance_evaluations),
        "visited_node_count": int(total_visited_nodes),
        "per_query_distance_evaluations": per_query_distance_evaluations,
        "per_query_visited_nodes": per_query_visited_nodes,
        "distance_mode": "lazy_visited_nodes",
    }
    return result_ids, result_distances, diagnostics


def graph_diagnostics(
    original: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Compare directed edge sets and report structural diagnostics."""
    old = {
        (source, target)
        for source, targets in original["neighbors"].items()
        for target in targets
    }
    new = {
        (source, target)
        for source, targets in current["neighbors"].items()
        for target in targets
    }
    reciprocal = sum((target, source) in new for source, target in new)
    degrees = [len(current["neighbors"][int(node)]) for node in current["node_global_ids"]]
    components = weak_components(current)
    reachable = directed_reachable_from_entry(current)
    return {
        "retained_edge_ratio": len(old & new) / len(old) if old else 1.0,
        "removed_edge_ratio": len(old - new) / len(old) if old else 0.0,
        "added_edge_ratio": len(new - old) / len(old) if old else float(len(new)),
        "reciprocal_edge_ratio": reciprocal / len(new) if new else 1.0,
        "average_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": max(degrees, default=0),
        "connected_component_count": len(components),
        "largest_component_ratio": (
            max(map(len, components)) / len(current["node_global_ids"])
            if components
            else 0.0
        ),
        "directed_reachable_from_entry_count": len(reachable),
        "directed_reachable_from_entry_ratio": (
            len(reachable) / len(current["node_global_ids"])
            if len(current["node_global_ids"])
            else 0.0
        ),
        "directed_unreachable_from_entry_count": (
            len(current["node_global_ids"]) - len(reachable)
        ),
    }


def load_index(path: Path) -> faiss.Index:
    """Read a FAISS index from disk."""
    return faiss.read_index(str(path))
