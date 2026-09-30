"""Shared Phase 3 retrieval, cost, and graph metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from data.phase2_graph import directed_reachable_from_entry, weak_components


def recall_at_k(
    predicted: np.ndarray,
    truth: np.ndarray,
    requested_k: list[int] | tuple[int, ...],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute mean and per-query Recall@k for already-ranked global IDs."""
    if predicted.ndim != 2 or truth.ndim != 2:
        raise ValueError("predicted and truth must be rank-2 arrays.")
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must contain the same query count.")
    means: dict[str, float] = {}
    per_query: dict[str, np.ndarray] = {}
    available = min(predicted.shape[1], truth.shape[1])
    for requested in requested_k:
        k = min(int(requested), available)
        values = np.zeros(len(predicted), dtype=np.float64)
        for index, (row_predicted, row_truth) in enumerate(
            zip(predicted[:, :k], truth[:, :k])
        ):
            valid_truth = [int(value) for value in row_truth if int(value) >= 0]
            if not valid_truth:
                values[index] = 0.0
                continue
            values[index] = len(
                set(int(value) for value in row_predicted if int(value) >= 0)
                & set(valid_truth)
            ) / len(valid_truth)
        means[f"recall_at_{requested}"] = float(np.mean(values)) if len(values) else 0.0
        means[f"effective_k_{requested}"] = int(k)
        per_query[str(requested)] = values
    return means, per_query


def subset_lookup(subset_to_ids: dict[str, np.ndarray]) -> dict[int, str]:
    """Build a global-ID to subset-name lookup and reject overlaps."""
    lookup: dict[int, str] = {}
    for subset, ids in subset_to_ids.items():
        for value in np.asarray(ids, dtype=np.int64):
            item = int(value)
            if item in lookup:
                raise ValueError(f"Global ID {item} appears in multiple subsets.")
            lookup[item] = subset
    return lookup


def source_target_recall_at_k(
    predicted: np.ndarray,
    truth: np.ndarray,
    query_subsets: list[str] | tuple[str, ...],
    id_to_subset: dict[int, str],
    requested_k: list[int] | tuple[int, ...],
) -> dict[str, dict[str, float]]:
    """Compute Recall@k split by same-subset and cross-subset truth items."""
    if len(query_subsets) != len(predicted):
        raise ValueError("query_subsets length must match query count.")
    result: dict[str, dict[str, float]] = {}
    for requested in requested_k:
        k = min(int(requested), predicted.shape[1], truth.shape[1])
        source_values = []
        target_values = []
        for query_index, source_subset in enumerate(query_subsets):
            predicted_set = {
                int(value) for value in predicted[query_index, :k] if int(value) >= 0
            }
            truth_ids = [int(value) for value in truth[query_index, :k] if int(value) >= 0]
            source_truth = [
                value for value in truth_ids if id_to_subset.get(value) == source_subset
            ]
            target_truth = [
                value for value in truth_ids if id_to_subset.get(value) != source_subset
            ]
            if source_truth:
                source_values.append(len(predicted_set & set(source_truth)) / len(source_truth))
            if target_truth:
                target_values.append(len(predicted_set & set(target_truth)) / len(target_truth))
        result[f"recall_at_{requested}"] = {
            "source_side": float(np.mean(source_values)) if source_values else 0.0,
            "target_side": float(np.mean(target_values)) if target_values else 0.0,
            "cross_subset": float(np.mean(target_values)) if target_values else 0.0,
            "source_query_count": int(len(source_values)),
            "target_query_count": int(len(target_values)),
        }
    return result


def graph_statistics(graph: dict[str, Any]) -> dict[str, Any]:
    """Return shared graph-size and reachability statistics."""
    node_count = int(len(graph["node_global_ids"]))
    edge_count = int(sum(len(values) for values in graph["neighbors"].values()))
    components = weak_components(graph)
    reachable = directed_reachable_from_entry(graph)
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "average_out_degree": edge_count / node_count if node_count else 0.0,
        "weak_connected_component_count": len(components),
        "largest_component_ratio": (
            max(map(len, components)) / node_count if node_count and components else 0.0
        ),
        "directed_reachable_from_entry_count": len(reachable),
        "directed_reachable_from_entry_ratio": (
            len(reachable) / node_count if node_count else 0.0
        ),
    }


def sampled_cross_knn_cost(
    size_a: int,
    size_b: int,
    sampled_a: int,
    sampled_b: int,
) -> dict[str, float | int]:
    """Report the Stage 1 cost definitions for Sampled Cross-kNN."""
    if min(size_a, size_b, sampled_a, sampled_b) < 0:
        raise ValueError("Sizes and sample counts must be non-negative.")
    if sampled_a > size_a or sampled_b > size_b:
        raise ValueError("Sample counts cannot exceed subset sizes.")
    total = size_a + size_b
    full_bidirectional = 2 * size_a * size_b
    distance_count = sampled_a * size_b + sampled_b * size_a
    return {
        "sample_ratio": (sampled_a + sampled_b) / total if total else 0.0,
        "sample_ratio_A": sampled_a / size_a if size_a else 0.0,
        "sample_ratio_B": sampled_b / size_b if size_b else 0.0,
        "cross_candidate_distance_count": int(distance_count),
        "cross_scan_work_ratio": (
            distance_count / full_bidirectional if full_bidirectional else 0.0
        ),
    }


def random_cross_link_cost(
    edge_count: int,
    edge_weight_distance_count: int = 0,
    endpoint_selection_distance_count: int = 0,
) -> dict[str, int]:
    """Record random-cross-link cost without mixing it with cross-kNN search."""
    if min(edge_count, edge_weight_distance_count, endpoint_selection_distance_count) < 0:
        raise ValueError("Random cross-link cost counts must be non-negative.")
    return {
        "random_cross_link_edge_count": int(edge_count),
        "endpoint_selection_distance_count": int(endpoint_selection_distance_count),
        "edge_weight_distance_count": int(edge_weight_distance_count),
    }


__all__ = [
    "graph_statistics",
    "random_cross_link_cost",
    "recall_at_k",
    "sampled_cross_knn_cost",
    "source_target_recall_at_k",
    "subset_lookup",
]
