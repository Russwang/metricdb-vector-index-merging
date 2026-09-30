"""Phase 3 external-graph merge helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from data.phase2_graph import directed_reachable_from_entry, weak_components
from data.phase3_io import Phase2SubsetInput


def _directed_edges(graph: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, targets in graph["neighbors"].items()
        for target in targets
    }


def count_cross_subset_edges(graph: dict[str, Any]) -> int:
    """Count directed edges whose endpoints belong to different subsets."""
    subset_by_node = {
        int(node): subset
        for node, subset in graph.get("source_subset_by_node", {}).items()
    }
    if not subset_by_node:
        source_subset = graph.get("source_subset")
        subset_by_node = {
            int(node): source_subset for node in graph["node_global_ids"]
        }
    count = 0
    for source, targets in graph["neighbors"].items():
        source_subset = subset_by_node[int(source)]
        for target in targets:
            if subset_by_node[int(target)] != source_subset:
                count += 1
    return count


def no_cross_union_graph(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Concatenate two repaired local graphs without adding cross-subset links."""
    if first.graph["graph_kind"] != second.graph["graph_kind"]:
        raise ValueError("Input graph kinds differ.")
    if first.graph["variant"] != second.graph["variant"]:
        raise ValueError("Input graph variants differ.")
    overlap = set(first.global_ids.tolist()) & set(second.global_ids.tolist())
    if overlap:
        raise ValueError(f"Pairwise graph inputs overlap: {sorted(overlap)[:5]}")

    first_ids = np.asarray(first.graph["node_global_ids"], dtype=np.int64)
    second_ids = np.asarray(second.graph["node_global_ids"], dtype=np.int64)
    np.testing.assert_array_equal(first_ids, first.global_ids.astype(np.int64))
    np.testing.assert_array_equal(second_ids, second.global_ids.astype(np.int64))

    neighbors = {
        int(node): [int(target) for target in targets]
        for node, targets in first.graph["neighbors"].items()
    }
    neighbors.update(
        {
            int(node): [int(target) for target in targets]
            for node, targets in second.graph["neighbors"].items()
        }
    )
    edge_distances = {
        int(node): [float(value) for value in values]
        for node, values in first.graph["edge_distances"].items()
    }
    edge_distances.update(
        {
            int(node): [float(value) for value in values]
            for node, values in second.graph["edge_distances"].items()
        }
    )

    node_global_ids = np.concatenate([first_ids, second_ids]).astype(np.int64)
    first_edges = _directed_edges(first.graph)
    second_edges = _directed_edges(second.graph)
    union_edges = {
        (int(source), int(target))
        for source, targets in neighbors.items()
        for target in targets
    }
    input_edge_count = len(first_edges) + len(second_edges)
    source_subset_by_node = {
        **{int(node): first.subset for node in first_ids},
        **{int(node): second.subset for node in second_ids},
    }
    graph = {
        "node_global_ids": node_global_ids,
        "neighbors": neighbors,
        "edge_distances": edge_distances,
        "source_subset": f"{first.subset}+{second.subset}",
        "source_subset_by_node": source_subset_by_node,
        "graph_kind": "no_cross_merged",
        "input_graph_kind": first.graph["graph_kind"],
        "input_variant": first.graph["variant"],
        "variant": "no_cross",
        "directed": True,
        "node_levels": np.concatenate(
            [
                np.asarray(first.graph["node_levels"], dtype=np.int64),
                np.asarray(second.graph["node_levels"], dtype=np.int64),
            ]
        ),
        "upper_layer_nodes_by_level": [],
        "original_entry_point": int(first.graph["original_entry_point"]),
        "hnsw_M": max(int(first.graph["hnsw_M"]), int(second.graph["hnsw_M"])),
        "maxM0": max(int(first.graph["maxM0"]), int(second.graph["maxM0"])),
        "original_degrees": np.concatenate(
            [
                np.asarray(first.graph["original_degrees"], dtype=np.int64),
                np.asarray(second.graph["original_degrees"], dtype=np.int64),
            ]
        ),
        "degree_budget": np.concatenate(
            [
                np.asarray(first.graph["degree_budget"], dtype=np.int64),
                np.asarray(second.graph["degree_budget"], dtype=np.int64),
            ]
        ),
        "added_cross_link_count": 0,
        "cross_link_strategy": "none",
    }
    retained_edges = len(union_edges & (first_edges | second_edges))
    diagnostics = {
        "input_edge_count": int(input_edge_count),
        "reused_within_subset_edge_count": int(retained_edges),
        "reused_within_subset_edge_ratio": (
            retained_edges / input_edge_count if input_edge_count else 1.0
        ),
        "added_cross_link_count": 0,
        "cross_subset_edge_count": count_cross_subset_edges(graph),
        "entry_point_source_subset": first.subset,
    }
    if diagnostics["cross_subset_edge_count"] != 0:
        raise ValueError("No-cross union graph unexpectedly contains cross edges.")
    if set(union_edges) != first_edges | second_edges:
        raise ValueError("No-cross union graph changed within-subset edges.")
    return graph, diagnostics


def _squared_l2(source: np.ndarray, target: np.ndarray) -> float:
    delta = source.astype(np.float64) - target.astype(np.float64)
    return float(np.sum(delta**2))


def _add_directed_edge(
    graph: dict[str, Any],
    vectors_by_id: dict[int, np.ndarray],
    source: int,
    target: int,
    distance: float | None = None,
) -> bool:
    if source == target:
        return False
    neighbors = graph["neighbors"].setdefault(source, [])
    if target in neighbors:
        return False
    neighbors.append(target)
    graph["edge_distances"].setdefault(source, []).append(
        float(distance)
        if distance is not None
        else _squared_l2(vectors_by_id[source], vectors_by_id[target])
    )
    return True


def add_random_cross_links(
    graph: dict[str, Any],
    base_diagnostics: dict[str, Any],
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    sample_ratio: float,
    cross_k: int,
    random_seed: int,
    bidirectional: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add random cross-subset links to an existing no-cross union graph."""
    if not 0.0 <= float(sample_ratio) <= 1.0:
        raise ValueError("sample_ratio must be in [0, 1].")
    if int(cross_k) <= 0:
        raise ValueError("cross_k must be positive.")
    graph["graph_kind"] = "random_cross_links"
    graph["variant"] = "random_cross_links"
    graph["cross_link_strategy"] = "random"
    graph["sample_ratio"] = float(sample_ratio)
    graph["cross_k"] = int(cross_k)
    graph["random_seed"] = int(random_seed)
    graph["bidirectional_cross_links"] = bool(bidirectional)

    first_ids = np.asarray(first.global_ids, dtype=np.int64)
    second_ids = np.asarray(second.global_ids, dtype=np.int64)
    sample_count_a = min(
        len(first_ids), int(np.ceil(float(sample_ratio) * len(first_ids)))
    )
    sample_count_b = min(
        len(second_ids), int(np.ceil(float(sample_ratio) * len(second_ids)))
    )
    rng = np.random.default_rng(int(random_seed))
    sampled_a = (
        rng.choice(first_ids, size=sample_count_a, replace=False)
        if sample_count_a
        else np.asarray([], dtype=np.int64)
    )
    sampled_b = (
        rng.choice(second_ids, size=sample_count_b, replace=False)
        if sample_count_b
        else np.asarray([], dtype=np.int64)
    )

    vectors_by_id = {
        int(global_id): first.transformed_vectors[index]
        for index, global_id in enumerate(first.global_ids)
    }
    vectors_by_id.update(
        {
            int(global_id): second.transformed_vectors[index]
            for index, global_id in enumerate(second.global_ids)
        }
    )
    added_edges: list[tuple[int, int]] = []

    def add_edge(source: int, target: int) -> None:
        if _add_directed_edge(graph, vectors_by_id, int(source), int(target)):
            added_edges.append((int(source), int(target)))

    target_count_b = min(int(cross_k), len(second_ids))
    for source in sampled_a:
        targets = rng.choice(second_ids, size=target_count_b, replace=False)
        for target in targets:
            add_edge(int(source), int(target))
            if bidirectional:
                add_edge(int(target), int(source))

    target_count_a = min(int(cross_k), len(first_ids))
    for source in sampled_b:
        targets = rng.choice(first_ids, size=target_count_a, replace=False)
        for target in targets:
            add_edge(int(source), int(target))
            if bidirectional:
                add_edge(int(target), int(source))

    graph["added_cross_link_count"] = len(added_edges)
    cross_subset_edge_count = count_cross_subset_edges(graph)
    diagnostics = {
        **base_diagnostics,
        "sample_ratio": float(sample_ratio),
        "sample_ratio_A": sample_count_a / len(first_ids) if len(first_ids) else 0.0,
        "sample_ratio_B": sample_count_b / len(second_ids) if len(second_ids) else 0.0,
        "sampled_node_count_A": int(sample_count_a),
        "sampled_node_count_B": int(sample_count_b),
        "sampled_node_ids_A": [int(value) for value in sampled_a.tolist()],
        "sampled_node_ids_B": [int(value) for value in sampled_b.tolist()],
        "sampled_bridge_node_count": int(sample_count_a + sample_count_b),
        "sampled_bridge_ratio": (
            (sample_count_a + sample_count_b) / (len(first_ids) + len(second_ids))
            if len(first_ids) + len(second_ids)
            else 0.0
        ),
        "cross_k": int(cross_k),
        "random_seed": int(random_seed),
        "bidirectional": bool(bidirectional),
        "added_cross_link_count": int(len(added_edges)),
        "cross_subset_edge_count": int(cross_subset_edge_count),
        "endpoint_selection_distance_count": 0,
        "edge_weight_distance_count": int(len(added_edges)),
        "reused_within_subset_edge_count": base_diagnostics[
            "reused_within_subset_edge_count"
        ],
        "reused_within_subset_edge_ratio": base_diagnostics[
            "reused_within_subset_edge_ratio"
        ],
    }
    if diagnostics["cross_subset_edge_count"] != diagnostics["added_cross_link_count"]:
        raise ValueError("Random graph has unexpected non-random cross edges.")
    return graph, diagnostics


def random_cross_link_union_graph(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    sample_ratio: float,
    cross_k: int,
    random_seed: int,
    bidirectional: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a no-cross union graph and add random cross-subset links."""
    graph, base_diagnostics = no_cross_union_graph(first, second)
    return add_random_cross_links(
        graph,
        base_diagnostics,
        first,
        second,
        sample_ratio,
        cross_k,
        random_seed,
        bidirectional=bidirectional,
    )


def add_sampled_cross_knn_links(
    graph: dict[str, Any],
    base_diagnostics: dict[str, Any],
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    sample_ratio: float,
    cross_k: int,
    random_seed: int,
    bidirectional: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add exact cross-kNN links for sampled bridge nodes."""
    if not 0.0 <= float(sample_ratio) <= 1.0:
        raise ValueError("sample_ratio must be in [0, 1].")
    if int(cross_k) <= 0:
        raise ValueError("cross_k must be positive.")
    graph["graph_kind"] = "sampled_cross_knn"
    graph["variant"] = "sampled_cross_knn"
    graph["cross_link_strategy"] = "sampled_cross_knn"
    graph["sample_ratio"] = float(sample_ratio)
    graph["cross_k"] = int(cross_k)
    graph["random_seed"] = int(random_seed)
    graph["bidirectional_cross_links"] = bool(bidirectional)
    graph["degree_pruning"] = "none"
    graph["connectivity_repair"] = "none"

    first_ids = np.asarray(first.global_ids, dtype=np.int64)
    second_ids = np.asarray(second.global_ids, dtype=np.int64)
    sample_count_a = min(
        len(first_ids), int(np.ceil(float(sample_ratio) * len(first_ids)))
    )
    sample_count_b = min(
        len(second_ids), int(np.ceil(float(sample_ratio) * len(second_ids)))
    )
    rng = np.random.default_rng(int(random_seed))
    sampled_a = (
        rng.choice(first_ids, size=sample_count_a, replace=False)
        if sample_count_a
        else np.asarray([], dtype=np.int64)
    )
    sampled_b = (
        rng.choice(second_ids, size=sample_count_b, replace=False)
        if sample_count_b
        else np.asarray([], dtype=np.int64)
    )

    vectors_by_id = {
        int(global_id): first.transformed_vectors[index]
        for index, global_id in enumerate(first.global_ids)
    }
    vectors_by_id.update(
        {
            int(global_id): second.transformed_vectors[index]
            for index, global_id in enumerate(second.global_ids)
        }
    )
    first_row = {int(global_id): index for index, global_id in enumerate(first_ids)}
    second_row = {int(global_id): index for index, global_id in enumerate(second_ids)}
    added_edges: list[tuple[int, int]] = []

    def add_edge(source: int, target: int, distance: float) -> None:
        if _add_directed_edge(
            graph,
            vectors_by_id,
            int(source),
            int(target),
            distance=float(distance),
        ):
            added_edges.append((int(source), int(target)))

    def top_cross_neighbors(
        source_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> list[tuple[int, float]]:
        if len(candidate_ids) == 0:
            return []
        delta = candidate_vectors.astype(np.float64) - source_vector.astype(np.float64)
        distances = np.sum(delta**2, axis=1)
        selected_count = min(int(cross_k), len(candidate_ids))
        order = np.lexsort((candidate_ids, distances))[:selected_count]
        return [
            (int(candidate_ids[index]), float(distances[index]))
            for index in order
        ]

    for source in sampled_a:
        source_vector = first.transformed_vectors[first_row[int(source)]]
        for target, distance in top_cross_neighbors(
            source_vector,
            second.transformed_vectors,
            second_ids,
        ):
            add_edge(int(source), target, distance)
            if bidirectional:
                add_edge(target, int(source), distance)

    for source in sampled_b:
        source_vector = second.transformed_vectors[second_row[int(source)]]
        for target, distance in top_cross_neighbors(
            source_vector,
            first.transformed_vectors,
            first_ids,
        ):
            add_edge(int(source), target, distance)
            if bidirectional:
                add_edge(target, int(source), distance)

    graph["added_cross_link_count"] = len(added_edges)
    cross_subset_edge_count = count_cross_subset_edges(graph)
    total = len(first_ids) + len(second_ids)
    full_bidirectional = 2 * len(first_ids) * len(second_ids)
    candidate_distance_count = sample_count_a * len(second_ids) + sample_count_b * len(first_ids)
    diagnostics = {
        **base_diagnostics,
        "sample_ratio": float(sample_ratio),
        "sample_ratio_A": sample_count_a / len(first_ids) if len(first_ids) else 0.0,
        "sample_ratio_B": sample_count_b / len(second_ids) if len(second_ids) else 0.0,
        "sampled_node_count_A": int(sample_count_a),
        "sampled_node_count_B": int(sample_count_b),
        "sampled_node_ids_A": [int(value) for value in sampled_a.tolist()],
        "sampled_node_ids_B": [int(value) for value in sampled_b.tolist()],
        "sampled_bridge_node_count": int(sample_count_a + sample_count_b),
        "sampled_bridge_ratio": (
            (sample_count_a + sample_count_b) / total if total else 0.0
        ),
        "cross_k": int(cross_k),
        "random_seed": int(random_seed),
        "bidirectional": bool(bidirectional),
        "added_cross_link_count": int(len(added_edges)),
        "cross_subset_edge_count": int(cross_subset_edge_count),
        "cross_candidate_distance_count": int(candidate_distance_count),
        "cross_scan_work_ratio": (
            candidate_distance_count / full_bidirectional
            if full_bidirectional
            else 0.0
        ),
        "edge_weight_distance_count": 0,
        "edge_weight_distance_source": "reused_from_cross_candidate_search",
        "reused_within_subset_edge_count": base_diagnostics[
            "reused_within_subset_edge_count"
        ],
        "reused_within_subset_edge_ratio": base_diagnostics[
            "reused_within_subset_edge_ratio"
        ],
    }
    if diagnostics["cross_subset_edge_count"] != diagnostics["added_cross_link_count"]:
        raise ValueError("Sampled cross-kNN graph has unexpected cross edges.")
    return graph, diagnostics


def _bridge_count(size: int, cross_link_config: dict[str, Any]) -> int:
    """Return the deterministic bridge budget for one subset."""
    budget = cross_link_config.get("bridge_budget", {})
    mode = str(budget.get("mode", "ratio"))
    if mode == "sqrt":
        multiplier = float(budget.get("sqrt_multiplier", 2.0))
        if multiplier <= 0.0:
            raise ValueError("bridge_budget.sqrt_multiplier must be positive.")
        return min(size, max(1, int(np.ceil(multiplier * np.sqrt(size)))))
    if mode == "fixed":
        count = int(budget.get("count", 0))
        if count <= 0:
            raise ValueError("bridge_budget.count must be positive for fixed mode.")
        return min(size, count)
    if mode == "ratio":
        ratio = float(budget.get("ratio", cross_link_config.get("smoke_default", {}).get("sample_ratio", 0.05)))
        if not 0.0 < ratio <= 1.0:
            raise ValueError("bridge_budget.ratio must be in (0, 1].")
        return min(size, max(1, int(np.ceil(ratio * size))))
    raise ValueError(f"Unsupported bridge_budget.mode: {mode!r}")


def select_bridge_nodes(
    first_ids: np.ndarray,
    second_ids: np.ndarray,
    cross_link_config: dict[str, Any],
    random_seed: int,
    first_graph: dict[str, Any] | None = None,
    second_graph: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select reproducible bridge IDs for both sides of one pairwise merge."""
    rng = np.random.default_rng(int(random_seed))
    first = np.asarray(first_ids, dtype=np.int64)
    second = np.asarray(second_ids, dtype=np.int64)
    strategy = str(cross_link_config.get("bridge_selection", "random"))

    def select(ids: np.ndarray, graph: dict[str, Any] | None) -> np.ndarray:
        count = _bridge_count(len(ids), cross_link_config)
        if strategy == "random":
            return rng.choice(ids, size=count, replace=False)
        if graph is None:
            raise ValueError(f"bridge_selection={strategy} requires a local graph.")
        degrees = {int(node): len(graph["neighbors"].get(int(node), [])) for node in ids}
        if strategy == "high_degree":
            ordered = sorted((int(node) for node in ids), key=lambda node: (-degrees[node], node))
        elif strategy in {"upper_layer", "hybrid"}:
            levels = {
                int(node): int(level)
                for node, level in zip(graph["node_global_ids"], graph["node_levels"])
            }
            if strategy == "upper_layer":
                ordered = sorted((int(node) for node in ids), key=lambda node: (-levels[node], node))
            else:
                ordered = sorted(
                    (int(node) for node in ids),
                    key=lambda node: (-levels[node], -degrees[node], node),
                )
        else:
            raise ValueError(f"Unsupported bridge_selection: {strategy!r}")
        return np.asarray(ordered[:count], dtype=np.int64)

    return (
        select(first, first_graph),
        select(second, second_graph),
    )


def _mapping_metadata(target: Phase2SubsetInput) -> dict[str, Any]:
    """Load affine metadata used to query a target local HNSW from shared space."""
    metadata = target.condition_metadata
    if "matrix" in metadata:
        return metadata
    base = metadata.get("base")
    if base != "rotation":
        raise ValueError(
            "Indexed cross-kNN requires affine metadata or a supported base condition."
        )
    path = target.compiled_dir.parent.parent / "rotation" / target.subset / "condition.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing base rotation metadata for noisy condition: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _shared_to_target_local(
    shared_vectors: np.ndarray,
    target: Phase2SubsetInput,
) -> np.ndarray:
    """Map shared-space vectors into a target index's clean local coordinate system."""
    metadata = _mapping_metadata(target)
    matrix = np.asarray(metadata["matrix"], dtype=np.float64)
    translation = np.asarray(metadata.get("translation", np.zeros(matrix.shape[1])), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Target transformation matrix must be square.")
    if len(translation) != matrix.shape[1]:
        raise ValueError("Target transformation translation has the wrong dimension.")
    inverse = np.linalg.inv(matrix)
    return np.ascontiguousarray(
        (np.asarray(shared_vectors, dtype=np.float64) - translation) @ inverse,
        dtype=np.float32,
    )


def _add_candidate(
    candidates: dict[int, dict[int, float]], source: int, target: int, distance: float
) -> None:
    """Keep the closest proposed cross target for a directed source."""
    if source == target:
        return
    current = candidates.setdefault(int(source), {})
    previous = current.get(int(target))
    if previous is None or distance < previous:
        current[int(target)] = float(distance)


def _apply_cross_candidates(
    graph: dict[str, Any],
    vectors_by_id: dict[int, np.ndarray],
    candidates: dict[int, dict[int, float]],
    degree_cap: int,
) -> tuple[list[tuple[int, int]], int]:
    """Prune cross candidates by distance and append the retained edges."""
    if degree_cap <= 0:
        raise ValueError("cross_degree_cap must be positive.")
    added: list[tuple[int, int]] = []
    pruned = 0
    for source in sorted(candidates):
        ranked = sorted(
            ((distance, target) for target, distance in candidates[source].items()),
            key=lambda item: (item[0], item[1]),
        )
        pruned += max(0, len(ranked) - degree_cap)
        for distance, target in ranked[:degree_cap]:
            if _add_directed_edge(graph, vectors_by_id, source, target, distance):
                added.append((source, target))
    return added, pruned


def _connect_with_candidate_pool(
    graph: dict[str, Any],
    vectors_by_id: dict[int, np.ndarray],
    candidates: dict[int, dict[int, float]],
) -> list[tuple[int, int]]:
    """Use already generated candidate edges to repair weak/directed gaps."""
    repairs: list[tuple[int, int]] = []
    while len(weak_components(graph)) > 1:
        component_by_node = {
            int(node): index
            for index, component in enumerate(weak_components(graph))
            for node in component
        }
        options = sorted(
            (distance, source, target)
            for source, targets in candidates.items()
            for target, distance in targets.items()
            if component_by_node.get(source) != component_by_node.get(target)
        )
        if not options:
            break
        distance, source, target = options[0]
        if _add_directed_edge(graph, vectors_by_id, source, target, distance):
            repairs.append((source, target))
        else:
            break
    nodes = set(int(node) for node in graph["node_global_ids"])
    while True:
        reachable = directed_reachable_from_entry(graph)
        missing = nodes - reachable
        if not missing:
            break
        options = sorted(
            (distance, source, target)
            for source in reachable
            for target, distance in candidates.get(source, {}).items()
            if target in missing
        )
        if not options:
            break
        distance, source, target = options[0]
        if _add_directed_edge(graph, vectors_by_id, source, target, distance):
            repairs.append((source, target))
        else:
            break
    return repairs


def add_indexed_cross_knn_links(
    graph: dict[str, Any],
    base_diagnostics: dict[str, Any],
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    cross_k: int,
    random_seed: int,
    cross_link_config: dict[str, Any],
    bidirectional: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add cross-links from existing target local HNSW candidate searches.

    Candidate generation queries the target's pre-existing local HNSW.  Every
    returned candidate is then reranked with its transformed shared-space
    vector, so the added edge weights retain the Phase 3 shared-space metric.
    """
    if int(cross_k) <= 0:
        raise ValueError("cross_k must be positive.")
    if first.local_index_path is None or second.local_index_path is None:
        raise ValueError("Indexed cross-kNN requires saved Phase 2 local HNSW paths.")
    candidate_k = int(cross_link_config.get("candidate_k", 64))
    candidate_ef_search = int(cross_link_config.get("candidate_ef_search", 128))
    if candidate_k < int(cross_k) or candidate_ef_search <= 0:
        raise ValueError("candidate_k must be >= cross_k and candidate_ef_search positive.")

    graph["graph_kind"] = "indexed_cross_knn"
    graph["variant"] = "indexed_cross_knn"
    graph["cross_link_strategy"] = "indexed_local_hnsw_rerank"
    graph["cross_k"] = int(cross_k)
    graph["random_seed"] = int(random_seed)
    graph["bidirectional_cross_links"] = bool(bidirectional)
    degree_cap = int(cross_link_config.get("cross_degree_cap", cross_k))
    propagation_hops = int(cross_link_config.get("propagation_hops", 0))
    if propagation_hops not in {0, 1}:
        raise ValueError("propagation_hops must be 0 or 1.")
    bounded_propagation = any(
        field in cross_link_config
        for field in ("propagation_neighbor_budget", "propagation_candidate_cap")
    )
    propagation_neighbor_budget = (
        int(cross_link_config.get("propagation_neighbor_budget", 2**31 - 1))
        if bounded_propagation else None
    )
    propagation_candidate_cap = (
        int(cross_link_config.get("propagation_candidate_cap", 2**31 - 1))
        if bounded_propagation else None
    )
    if bounded_propagation and propagation_neighbor_budget <= 0:
        raise ValueError("propagation_neighbor_budget must be positive.")
    if bounded_propagation and propagation_candidate_cap <= 0:
        raise ValueError("propagation_candidate_cap must be positive.")
    graph["degree_pruning"] = f"cross_degree_cap_{degree_cap}"
    graph["connectivity_repair"] = "candidate_pool_if_needed"

    first_ids = np.asarray(first.global_ids, dtype=np.int64)
    second_ids = np.asarray(second.global_ids, dtype=np.int64)
    sampled_a, sampled_b = select_bridge_nodes(
        first_ids, second_ids, cross_link_config, random_seed, first.graph, second.graph
    )
    count_a = len(sampled_a)
    count_b = len(sampled_b)
    first_row = {int(value): index for index, value in enumerate(first_ids)}
    second_row = {int(value): index for index, value in enumerate(second_ids)}
    vectors_by_id = {
        int(global_id): first.transformed_vectors[index]
        for index, global_id in enumerate(first_ids)
    }
    vectors_by_id.update(
        {
            int(global_id): second.transformed_vectors[index]
            for index, global_id in enumerate(second_ids)
        }
    )
    index_a = faiss.read_index(str(first.local_index_path))
    index_b = faiss.read_index(str(second.local_index_path))
    for index in (index_a, index_b):
        raw = faiss.downcast_index(index.index) if hasattr(index, "index") else index
        if hasattr(raw, "hnsw"):
            raw.hnsw.efSearch = candidate_ef_search

    cross_candidates: dict[int, dict[int, float]] = {}
    candidate_calls = 0
    rerank_distance_count = 0
    propagation_distance_count = 0
    propagated_neighbor_ids: set[int] = set()
    propagated_candidate_counts: dict[int, int] = {}

    def link_sources(
        source_ids: np.ndarray,
        source_vectors: np.ndarray,
        target: Phase2SubsetInput,
        target_index: faiss.Index,
        target_rows: dict[int, int],
    ) -> None:
        nonlocal candidate_calls, rerank_distance_count
        local_queries = _shared_to_target_local(source_vectors, target)
        _, candidate_ids = target_index.search(local_queries, candidate_k)
        candidate_calls += len(source_ids)
        for source, source_vector, row_ids in zip(source_ids, source_vectors, candidate_ids):
            valid = [int(value) for value in row_ids if int(value) in target_rows]
            valid = list(dict.fromkeys(valid))
            rerank_distance_count += len(valid)
            ranked = sorted(
                (
                    _squared_l2(source_vector, target.transformed_vectors[target_rows[target_id]]),
                    target_id,
                )
                for target_id in valid
            )[: min(int(cross_k), len(valid))]
            for distance, target_id in ranked:
                _add_candidate(cross_candidates, int(source), target_id, distance)
                if bidirectional:
                    _add_candidate(cross_candidates, target_id, int(source), distance)

    link_sources(
        sampled_a,
        np.asarray([first.transformed_vectors[first_row[int(value)]] for value in sampled_a]),
        second,
        index_b,
        second_row,
    )
    link_sources(
        sampled_b,
        np.asarray([second.transformed_vectors[second_row[int(value)]] for value in sampled_b]),
        first,
        index_a,
        first_row,
    )
    if propagation_hops:
        subset_by_node = graph["source_subset_by_node"]
        # Propagate only from the selected portals.  Propagating from reciprocal
        # target entries would recursively fan out through every popular target
        # and defeat the bounded join-set budget.
        portals = [int(value) for value in np.concatenate([sampled_a, sampled_b])]
        if not bounded_propagation:
            # Preserve the Stage 16 unlimited join1 endpoint exactly.  The live
            # candidate dictionaries allow later portals to see reciprocal
            # candidates created by earlier portals.
            for source in portals:
                targets = cross_candidates.get(source, {})
                local_neighbors = [
                    int(node) for node in graph["neighbors"].get(source, [])
                    if subset_by_node[int(node)] == subset_by_node[source]
                ]
                for neighbor in local_neighbors:
                    propagated_neighbor_ids.add(neighbor)
                    for target in targets:
                        distance = _squared_l2(vectors_by_id[neighbor], vectors_by_id[target])
                        propagation_distance_count += 1
                        _add_candidate(cross_candidates, neighbor, target, distance)
                        if bidirectional:
                            _add_candidate(cross_candidates, target, neighbor, distance)
        else:
            propagation_pool: dict[int, list[tuple[float, int, int]]] = {}
            for source in portals:
                targets = dict(cross_candidates.get(source, {}))
                neighbor_ids = graph["neighbors"].get(source, [])
                edge_distances = graph.get("edge_distances", {}).get(source, [])
                local_neighbors = sorted(
                    (
                        (
                            float(edge_distances[index])
                            if index < len(edge_distances)
                            else float("inf"),
                            int(node),
                        )
                        for index, node in enumerate(neighbor_ids)
                        if subset_by_node[int(node)] == subset_by_node[source]
                    ),
                    key=lambda item: (item[0], item[1]),
                )[:propagation_neighbor_budget]
                for _, neighbor in local_neighbors:
                    propagated_neighbor_ids.add(neighbor)
                    for target in targets:
                        distance = _squared_l2(vectors_by_id[neighbor], vectors_by_id[target])
                        propagation_distance_count += 1
                        propagation_pool.setdefault(neighbor, []).append(
                            (distance, int(target), source)
                        )
            for neighbor in sorted(propagation_pool):
                ranked = sorted(
                    propagation_pool[neighbor], key=lambda item: (item[0], item[1], item[2])
                )
                selected: list[tuple[float, int, int]] = []
                seen_targets: set[int] = set()
                for candidate in ranked:
                    if candidate[1] in seen_targets:
                        continue
                    selected.append(candidate)
                    seen_targets.add(candidate[1])
                    if len(selected) >= propagation_candidate_cap:
                        break
                propagated_candidate_counts[neighbor] = len(selected)
                for distance, target, _ in selected:
                    _add_candidate(cross_candidates, neighbor, target, distance)
                    if bidirectional:
                        _add_candidate(cross_candidates, target, neighbor, distance)
    added_edges, pruned_cross_candidates = _apply_cross_candidates(
        graph, vectors_by_id, cross_candidates, degree_cap
    )
    connectivity_repairs = _connect_with_candidate_pool(
        graph, vectors_by_id, cross_candidates
    )
    graph["added_cross_link_count"] = len(added_edges) + len(connectivity_repairs)
    total = len(first_ids) + len(second_ids)
    diagnostics = {
        **base_diagnostics,
        "bridge_budget_mode": str(cross_link_config.get("bridge_budget", {}).get("mode", "ratio")),
        "bridge_selection": str(cross_link_config.get("bridge_selection", "random")),
        "sampled_node_count_A": int(count_a),
        "sampled_node_count_B": int(count_b),
        "sampled_node_ids_A": [int(value) for value in sampled_a.tolist()],
        "sampled_node_ids_B": [int(value) for value in sampled_b.tolist()],
        "sampled_bridge_node_count": int(count_a + count_b),
        "sampled_bridge_ratio": (count_a + count_b) / total if total else 0.0,
        "cross_k": int(cross_k),
        "candidate_k": candidate_k,
        "candidate_ef_search": candidate_ef_search,
        "candidate_backend": "target_local_hnsw_shared_rerank",
        "candidate_hnsw_call_count": int(candidate_calls),
        "cross_candidate_distance_count": int(rerank_distance_count + propagation_distance_count),
        "direct_candidate_distance_count": int(rerank_distance_count),
        "propagation_distance_count": int(propagation_distance_count),
        "propagation_hops": propagation_hops,
        "bounded_propagation": bounded_propagation,
        "propagation_neighbor_budget": propagation_neighbor_budget,
        "propagation_candidate_cap": propagation_candidate_cap,
        "propagated_neighbor_count": int(len(propagated_neighbor_ids)),
        "propagated_neighbor_ids": sorted(propagated_neighbor_ids),
        "max_propagated_candidates_per_neighbor": int(
            max(propagated_candidate_counts.values(), default=0)
        ),
        "cross_degree_cap": degree_cap,
        "pruned_cross_candidate_count": int(pruned_cross_candidates),
        "connectivity_repair_edge_count": int(len(connectivity_repairs)),
        "edge_weight_distance_count": 0,
        "edge_weight_distance_source": "reused_from_shared_space_rerank",
        "random_seed": int(random_seed),
        "bidirectional": bool(bidirectional),
        "added_cross_link_count": int(len(added_edges) + len(connectivity_repairs)),
        "cross_subset_edge_count": int(count_cross_subset_edges(graph)),
        "reused_within_subset_edge_count": base_diagnostics["reused_within_subset_edge_count"],
        "reused_within_subset_edge_ratio": base_diagnostics["reused_within_subset_edge_ratio"],
    }
    if diagnostics["cross_subset_edge_count"] != diagnostics["added_cross_link_count"]:
        raise ValueError("Indexed cross-kNN graph has unexpected cross edges.")
    return graph, diagnostics


def sampled_cross_knn_union_graph(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    sample_ratio: float,
    cross_k: int,
    random_seed: int,
    bidirectional: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a no-cross union graph and add sampled exact cross-kNN links."""
    graph, base_diagnostics = no_cross_union_graph(first, second)
    return add_sampled_cross_knn_links(
        graph,
        base_diagnostics,
        first,
        second,
        sample_ratio,
        cross_k,
        random_seed,
        bidirectional=bidirectional,
    )


__all__ = [
    "add_random_cross_links",
    "add_indexed_cross_knn_links",
    "add_sampled_cross_knn_links",
    "count_cross_subset_edges",
    "no_cross_union_graph",
    "random_cross_link_union_graph",
    "sampled_cross_knn_union_graph",
    "select_bridge_nodes",
]
