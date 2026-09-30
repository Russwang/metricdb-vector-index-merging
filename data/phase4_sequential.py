"""Bounded second-step graph integration for Cycle 24."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from data.phase2_graph import directed_reachable_from_entry, weak_components


def _copy_adjacency(graph: dict[str, Any]) -> dict[int, list[int]]:
    return {int(node): [int(value) for value in targets] for node, targets in graph["neighbors"].items()}


def sequential_bounded_merge(
    current_graph: dict[str, Any],
    new_graph: dict[str, Any],
    current_vectors: np.ndarray,
    new_vectors: np.ndarray,
    *,
    current_subset_by_node: dict[int, str] | None = None,
    new_subset: str,
    portal_budget: int = 8,
    cross_k: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add bounded cross bridges while preserving every previously stored edge."""
    total_started = perf_counter()
    old_ids = np.asarray(current_graph["node_global_ids"], dtype=np.int64)
    new_ids = np.asarray(new_graph["node_global_ids"], dtype=np.int64)
    if set(old_ids.tolist()) & set(new_ids.tolist()):
        raise ValueError("Sequential merge inputs contain duplicate global IDs.")
    old_vectors = np.asarray(current_vectors, dtype=np.float32)
    added_vectors = np.asarray(new_vectors, dtype=np.float32)
    if old_vectors.ndim != 2 or added_vectors.ndim != 2 or old_vectors.shape[1] != added_vectors.shape[1]:
        raise ValueError("Sequential merge vectors must have matching dimensions.")
    if len(old_vectors) != len(old_ids) or len(added_vectors) != len(new_ids):
        raise ValueError("Graph IDs and vectors must be row-aligned.")
    if portal_budget <= 0 or cross_k <= 0:
        raise ValueError("portal_budget and cross_k must be positive.")

    selection_started = perf_counter()
    old_count = min(portal_budget, len(old_ids))
    new_count = min(portal_budget, len(new_ids))
    old_rows = np.linspace(0, len(old_ids) - 1, old_count, dtype=np.int64)
    new_rows = np.linspace(0, len(new_ids) - 1, new_count, dtype=np.int64)
    selection_seconds = perf_counter() - selection_started

    candidate_started = perf_counter()
    delta = old_vectors[old_rows, None, :].astype(np.float64) - added_vectors[new_rows][None, :, :].astype(np.float64)
    distances = np.sum(delta * delta, axis=2)
    neighbors = _copy_adjacency(current_graph)
    original_edges = {(s, t) for s, targets in neighbors.items() for t in targets}
    neighbors.update(_copy_adjacency(new_graph))
    bridge_edges: list[tuple[int, int]] = []
    for local_row, old_row in enumerate(old_rows):
        order = np.lexsort((new_ids[new_rows], distances[local_row]))[: min(cross_k, new_count)]
        for target_offset in order:
            bridge_edges.append((int(old_ids[old_row]), int(new_ids[new_rows[target_offset]])))
    for local_row, new_row in enumerate(new_rows):
        order = np.lexsort((old_ids[old_rows], distances[:, local_row]))[: min(cross_k, old_count)]
        for target_offset in order:
            bridge_edges.append((int(new_ids[new_row]), int(old_ids[old_rows[target_offset]])))
    for source, target in bridge_edges:
        if target not in neighbors[source]:
            neighbors[source].append(target)
    candidate_seconds = perf_counter() - candidate_started

    subset_by_node = dict(current_subset_by_node or current_graph.get("source_subset_by_node", {}))
    if not subset_by_node:
        subset_by_node = {int(value): str(current_graph.get("source_subset", "merged")) for value in old_ids}
    subset_by_node.update({int(value): new_subset for value in new_ids})
    all_ids = np.concatenate([old_ids, new_ids])
    all_vectors = np.vstack([old_vectors, added_vectors]).astype(np.float64)
    row_by_id = {int(node): row for row, node in enumerate(all_ids)}
    edge_distances = {}
    for source, targets in neighbors.items():
        source_vector = all_vectors[row_by_id[source]]
        edge_distances[source] = [
            float(np.sum((source_vector - all_vectors[row_by_id[target]]) ** 2))
            for target in targets
        ]
    old_levels = np.asarray(current_graph.get("node_levels", np.ones(len(old_ids))), dtype=np.int64)
    new_levels = np.asarray(new_graph.get("node_levels", np.ones(len(new_ids))), dtype=np.int64)
    old_original_degrees = np.asarray(current_graph.get("original_degrees", [len(current_graph["neighbors"][int(node)]) for node in old_ids]), dtype=np.int64)
    new_original_degrees = np.asarray(new_graph.get("original_degrees", [len(new_graph["neighbors"][int(node)]) for node in new_ids]), dtype=np.int64)
    old_budgets = np.asarray(current_graph.get("degree_budget", old_original_degrees), dtype=np.int64)
    new_budgets = np.asarray(new_graph.get("degree_budget", new_original_degrees), dtype=np.int64)
    result = {
        **current_graph,
        "node_global_ids": all_ids,
        "neighbors": neighbors,
        "edge_distances": edge_distances,
        "node_levels": np.concatenate([old_levels, new_levels]),
        "original_degrees": np.concatenate([old_original_degrees, new_original_degrees]),
        "degree_budget": np.concatenate([old_budgets, new_budgets]),
        "source_subset": "merged",
        "source_subset_by_node": subset_by_node,
        "original_entry_point": int(current_graph.get("original_entry_point", old_ids[0])),
        "variant": "sequential_bounded_merge",
    }
    new_edges = {(s, t) for s, targets in result["neighbors"].items() for t in targets}
    if not original_edges.issubset(new_edges):
        raise AssertionError("Sequential merge removed a previously stored edge.")
    diagnostics = {
        "schema_version": "metricdb.sequential-merge.v1",
        "selection_seconds": float(selection_seconds),
        "candidate_seconds": float(candidate_seconds),
        "propagation_seconds": 0.0,
        "integration_seconds": float(perf_counter() - total_started),
        "candidate_distance_count": int(old_count * new_count),
        "candidate_budget_upper_bound": int(portal_budget * portal_budget),
        "bridge_edge_count": int(len(new_edges - original_edges - {(s, t) for s, targets in new_graph["neighbors"].items() for t in targets})),
        "preserved_prior_edge_count": int(len(original_edges)),
        "prior_edges_preserved": True,
        "weak_component_count": len(weak_components(result)),
        "directed_reachable_count": len(directed_reachable_from_entry(result)),
        "node_count": int(len(result["node_global_ids"])),
    }
    return result, diagnostics
