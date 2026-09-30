"""Phase 3 brute-force ground truth and first-round baselines."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np

from data.phase2_conditions import build_hnsw
from data.phase2_graph import extract_raw_level0_graph, external_graph_search, repair_graph
from data.phase3_io import Phase2SubsetInput
from data.phase3_metrics import (
    graph_statistics,
    random_cross_link_cost,
    recall_at_k,
    sampled_cross_knn_cost,
    source_target_recall_at_k,
    subset_lookup,
)
from data.phase3_merge import (
    add_indexed_cross_knn_links,
    add_random_cross_links,
    add_sampled_cross_knn_links,
    no_cross_union_graph,
    select_bridge_nodes,
)
from data.phase2_common import save_graph, save_json
from data.run_phase1_bruteforce_baseline import exact_knn_squared_l2


def union_phase2_pair(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
) -> dict[str, Any]:
    """Build row-aligned union vectors, IDs, held-out queries, and query labels."""
    if first.transformed_vectors.shape[1] != second.transformed_vectors.shape[1]:
        raise ValueError("Pairwise inputs have different vector dimensions.")
    vectors = np.ascontiguousarray(
        np.vstack([first.transformed_vectors, second.transformed_vectors]),
        dtype=np.float32,
    )
    global_ids = np.concatenate([first.global_ids, second.global_ids]).astype(np.int64)
    if len(np.unique(global_ids)) != len(global_ids):
        raise ValueError("Union corpus contains duplicate global IDs.")
    queries = np.ascontiguousarray(
        np.vstack([first.held_out_queries, second.held_out_queries]),
        dtype=np.float32,
    )
    query_subsets = [first.subset] * len(first.held_out_queries) + [
        second.subset
    ] * len(second.held_out_queries)
    subset_to_ids = {
        first.subset: first.global_ids,
        second.subset: second.global_ids,
    }
    return {
        "vectors": vectors,
        "global_ids": global_ids,
        "queries": queries,
        "query_subsets": query_subsets,
        "subset_to_ids": subset_to_ids,
        "subsets": [first.subset, second.subset],
        "sizes": {
            first.subset: int(len(first.global_ids)),
            second.subset: int(len(second.global_ids)),
        },
    }


def _save_arrays(directory: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, value in arrays.items():
        path = directory / f"{name}.npy"
        np.save(path, value)
        paths[name] = str(path)
    return paths


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    """Return the versioned integration-timing schema used from Stage 18."""
    if not samples:
        raise ValueError("Integration timing requires at least one sample.")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "integration_repetitions": int(len(samples)),
        "integration_seconds_samples": [float(value) for value in samples],
        "integration_seconds_median": float(np.median(values)),
        "integration_seconds_p25": float(np.percentile(values, 25)),
        "integration_seconds_p75": float(np.percentile(values, 75)),
    }


def _validate_integration_repetitions(value: int) -> int:
    repetitions = int(value)
    if repetitions <= 0:
        raise ValueError("integration_repetitions must be positive.")
    return repetitions


def _bridge_repair_nodes(graph: dict[str, Any], bridge_ids: np.ndarray, radius: int) -> set[int]:
    """Return bridges plus a deterministic local out-neighbourhood."""
    selected = set(int(value) for value in bridge_ids)
    frontier = set(selected)
    for _ in range(int(radius)):
        next_frontier = {
            int(target)
            for source in frontier
            for target in graph["neighbors"].get(int(source), [])
        }
        selected.update(next_frontier)
        frontier = next_frontier
    return selected


def _indexed_repair_inputs(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    cross_link_config: dict[str, Any],
    random_seed: int,
) -> tuple[Phase2SubsetInput, Phase2SubsetInput, dict[str, Any]]:
    """Select the Stage 15 local-repair variant before indexed merging."""
    policy = str(cross_link_config.get("repair_policy", "input_graph"))
    if policy == "input_graph":
        return first, second, {"repair_policy": policy, "repair_seconds": 0.0}
    if first.raw_no_repair_graph is None or second.raw_no_repair_graph is None:
        raise ValueError("Selected repair policy requires raw no-repair Phase 2 graphs.")
    raw_first = replace(first, graph=first.raw_no_repair_graph)
    raw_second = replace(second, graph=second.raw_no_repair_graph)
    if policy == "no_repair":
        return raw_first, raw_second, {"repair_policy": policy, "repair_seconds": 0.0}
    if policy != "bridge_local":
        raise ValueError(f"Unsupported indexed repair policy: {policy!r}")
    if first.local_vectors is None or second.local_vectors is None:
        raise ValueError("Bridge-local repair requires local vectors.")
    if first.local_index_path is None or second.local_index_path is None:
        raise ValueError("Bridge-local repair requires local HNSW paths.")
    bridges_a, bridges_b = select_bridge_nodes(
        first.global_ids,
        second.global_ids,
        cross_link_config,
        random_seed,
        raw_first.graph,
        raw_second.graph,
    )
    radius = int(cross_link_config.get("repair_radius", 1))
    candidate_k = int(cross_link_config.get("repair_candidate_k", 64))
    repaired: list[Phase2SubsetInput] = []
    details: dict[str, Any] = {"repair_policy": policy, "repair_radius": radius}
    for label, item, bridges in (("A", raw_first, bridges_a), ("B", raw_second, bridges_b)):
        selected = _bridge_repair_nodes(item.graph, bridges, radius)
        graph, diagnostics = repair_graph(
            item.graph,
            faiss.read_index(str(item.local_index_path)),
            np.asarray(item.local_vectors, dtype=np.float32),
            item.transformed_vectors,
            candidate_k,
            force_connectivity=False,
            selected_node_ids=selected,
        )
        graph["variant"] = "bridge_local_repair"
        repaired.append(replace(item, graph=graph))
        details[f"subset_{label}"] = diagnostics
    details["repair_seconds"] = sum(
        float(details[f"subset_{label}"]["repair_seconds"]) for label in ("A", "B")
    )
    details["repair_candidate_distance_count"] = sum(
        int(details[f"subset_{label}"]["candidate_distance_computations"])
        for label in ("A", "B")
    )
    details["repair_processed_node_count"] = sum(
        int(details[f"subset_{label}"]["processed_node_count"]) for label in ("A", "B")
    )
    return repaired[0], repaired[1], details


def run_bruteforce_ground_truth(
    union: dict[str, Any],
    output_dir: Path,
    evaluation_k: list[int],
    batch_size: int,
) -> dict[str, Any]:
    """Compute exact squared-L2 ground truth over the transformed union corpus."""
    started = perf_counter()
    max_k = min(max(evaluation_k), len(union["global_ids"]))
    neighbor_ids, neighbor_distances = exact_knn_squared_l2(
        union["queries"],
        union["vectors"],
        union["global_ids"],
        max_k,
        exclude_query_global_ids=None,
        batch_size=batch_size,
    )
    seconds = perf_counter() - started
    arrays = _save_arrays(
        output_dir,
        {
            "query_vectors": union["queries"],
            "union_transformed_vectors": union["vectors"],
            "union_global_ids": union["global_ids"],
            "neighbor_global_ids": neighbor_ids,
            "neighbor_distances": neighbor_distances,
        },
    )
    query_subsets_path = output_dir / "query_subsets.npy"
    np.save(query_subsets_path, np.asarray(union["query_subsets"], dtype=object))
    manifest = {
        "method": "bruteforce_ground_truth",
        "distance_metric": "squared_l2",
        "corpus_size": int(len(union["global_ids"])),
        "query_count": int(len(union["queries"])),
        "dimension": int(union["vectors"].shape[1]),
        "evaluation_k": [int(value) for value in evaluation_k],
        "cost": {
            "exact_distance_count": int(len(union["queries"]) * len(union["global_ids"])),
            "total_seconds": seconds,
        },
        "arrays": {**arrays, "query_subsets": str(query_subsets_path)},
    }
    save_json(output_dir / "metrics.json", manifest)
    return manifest


def run_full_rebuild_hnsw(
    union: dict[str, Any],
    ground_truth_ids: np.ndarray,
    output_dir: Path,
    search_config: dict[str, Any],
    integration_repetitions: int = 1,
) -> dict[str, Any]:
    """Build one HNSW from the transformed union vectors and evaluate held-out Recall."""
    output_dir.mkdir(parents=True, exist_ok=True)
    build_samples: list[float] = []
    index = None
    for _ in range(_validate_integration_repetitions(integration_repetitions)):
        build_started = perf_counter()
        index = build_hnsw(
            union["vectors"],
            union["global_ids"],
            int(search_config["m"]),
            int(search_config["ef_construction"]),
            int(search_config["ef_search"]),
        )
        build_samples.append(perf_counter() - build_started)
    assert index is not None
    timing = _timing_summary(build_samples)
    build_seconds = timing["integration_seconds_median"]

    max_k = min(max(int(value) for value in search_config["evaluation_k"]), len(union["global_ids"]))
    search_started = perf_counter()
    distances, predicted_ids = index.search(union["queries"], max_k)
    search_seconds = perf_counter() - search_started

    recall, per_query = recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        [int(value) for value in search_config["evaluation_k"]],
    )
    id_to_subset = subset_lookup(union["subset_to_ids"])
    split_recall = source_target_recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        union["query_subsets"],
        id_to_subset,
        [int(value) for value in search_config["evaluation_k"]],
    )
    graph_extract_started = perf_counter()
    graph = extract_raw_level0_graph(
        index,
        union["vectors"],
        "D_A+D_B",
        "original_degree",
    )
    graph["graph_kind"] = "rebuilt_union"
    graph["variant"] = "full_rebuild_hnsw"
    graph_extract_seconds = perf_counter() - graph_extract_started

    index_path = output_dir / "full_rebuild_hnsw.faiss"
    faiss.write_index(index, str(index_path))
    arrays = _save_arrays(
        output_dir,
        {
            "neighbor_global_ids": predicted_ids,
            "neighbor_distances": distances,
            **{f"per_query_recall_{k}": values for k, values in per_query.items()},
        },
    )
    metrics = {
        "method": "full_rebuild_hnsw",
        "distance_metric": "squared_l2",
        "corpus_size": int(len(union["global_ids"])),
        "query_count": int(len(union["queries"])),
        "dimension": int(union["vectors"].shape[1]),
        "hnsw": {
            "m": int(search_config["m"]),
            "ef_construction": int(search_config["ef_construction"]),
            "ef_search": int(search_config["ef_search"]),
        },
        "quality": {
            "overall": recall,
            "source_target": split_recall,
        },
        "cost": {
            "build_seconds": build_seconds,
            "graph_extraction_seconds": graph_extract_seconds,
            "integration_seconds": build_seconds,
            "total_integration_seconds": build_seconds,
            **timing,
            "integration_scope": "native_hnsw_build_only",
            "excluded_from_integration": [
                "graph_extraction",
                "result_serialization",
                "query_search",
            ],
            "total_search_seconds": search_seconds,
            "average_query_ms": search_seconds * 1000.0 / len(union["queries"]),
            "processed_vector_ratio": 1.0,
        },
        "graph_statistics": graph_statistics(graph),
        "index_path": str(index_path),
        "index_size_bytes": int(index_path.stat().st_size),
        "arrays": arrays,
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def _direction_label(base_subset: str, inserted_subset: str) -> str:
    """Return the compact A<-B style label used in Phase 3 manifests."""
    base = base_subset.removeprefix("D_")
    inserted = inserted_subset.removeprefix("D_")
    return f"{base}<-{inserted}"


def _direction_slug(direction: str) -> str:
    return direction.replace("<-", "_from_")


def _index_id_set(index: faiss.IndexIDMap2) -> set[int]:
    return set(int(value) for value in faiss.vector_to_array(index.id_map))


def _run_incremental_direction(
    base: Phase2SubsetInput,
    inserted: Phase2SubsetInput,
    union: dict[str, Any],
    ground_truth_ids: np.ndarray,
    output_dir: Path,
    search_config: dict[str, Any],
    integration_repetitions: int = 1,
) -> dict[str, Any]:
    """Build one native FAISS incremental-insertion direction and evaluate it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    direction = _direction_label(base.subset, inserted.subset)
    total_samples: list[float] = []
    prepare_samples: list[float] = []
    insert_samples: list[float] = []
    index = None
    for _ in range(_validate_integration_repetitions(integration_repetitions)):
        prepare_started = perf_counter()
        index = build_hnsw(
            base.transformed_vectors,
            base.global_ids,
            int(search_config["m"]),
            int(search_config["ef_construction"]),
            int(search_config["ef_search"]),
        )
        prepare_seconds = perf_counter() - prepare_started
        insert_started = perf_counter()
        index.add_with_ids(
            np.ascontiguousarray(inserted.transformed_vectors, dtype=np.float32),
            np.asarray(inserted.global_ids, dtype=np.int64),
        )
        insert_seconds = perf_counter() - insert_started
        prepare_samples.append(prepare_seconds)
        insert_samples.append(insert_seconds)
        total_samples.append(prepare_seconds + insert_seconds)
    assert index is not None
    timing = _timing_summary(total_samples)
    base_index_build_seconds = float(np.median(prepare_samples))
    base_index_prepare_seconds = base_index_build_seconds
    inserted_add_seconds = float(np.median(insert_samples))
    total_integration_seconds = timing["integration_seconds_median"]

    expected_ids = set(int(value) for value in union["global_ids"])
    final_ids = _index_id_set(index)
    if final_ids != expected_ids:
        missing = sorted(expected_ids - final_ids)[:5]
        extra = sorted(final_ids - expected_ids)[:5]
        raise ValueError(
            f"Incremental index ID mismatch for {direction}: "
            f"missing={missing} extra={extra}"
        )

    max_k = min(
        max(int(value) for value in search_config["evaluation_k"]),
        len(union["global_ids"]),
    )
    search_started = perf_counter()
    distances, predicted_ids = index.search(union["queries"], max_k)
    search_seconds = perf_counter() - search_started

    recall, per_query = recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        [int(value) for value in search_config["evaluation_k"]],
    )
    id_to_subset = subset_lookup(union["subset_to_ids"])
    split_recall = source_target_recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        union["query_subsets"],
        id_to_subset,
        [int(value) for value in search_config["evaluation_k"]],
    )
    index_path = output_dir / "incremental_hnsw.faiss"
    faiss.write_index(index, str(index_path))
    arrays = _save_arrays(
        output_dir,
        {
            "neighbor_global_ids": predicted_ids,
            "neighbor_distances": distances,
            **{f"per_query_recall_{k}": values for k, values in per_query.items()},
        },
    )
    query_count = len(union["queries"])
    inserted_count = int(len(inserted.global_ids))
    corpus_size = int(len(union["global_ids"]))
    return {
        "direction": direction,
        "base_subset": base.subset,
        "inserted_subset": inserted.subset,
        "method": "incremental_insertion",
        "baseline_type": "strong_incremental_insertion_baseline",
        "method_type": "baseline",
        "backend": "native_faiss_hnsw",
        "constructs_single_union_graph": True,
        "constructs_single_union_index": True,
        "distance_metric": "squared_l2",
        "corpus_size": corpus_size,
        "query_count": int(query_count),
        "dimension": int(union["vectors"].shape[1]),
        "input": {
            "graph_kind": base.graph["graph_kind"],
            "within_subset_variant": base.graph["variant"],
            "subsets": [base.subset, inserted.subset],
        },
        "hnsw": {
            "m": int(search_config["m"]),
            "ef_construction": int(search_config["ef_construction"]),
            "ef_search": int(search_config["ef_search"]),
        },
        "reuse": {
            "base_index_source": "rebuilt_native_hnsw_from_base_transformed_vectors",
            "base_index_source_note": (
                "First version does not load a saved Phase 2 local_hnsw.faiss; "
                "base cost is recorded instead of treated as free physical reuse."
            ),
            "logical_base_reuse": True,
            "physical_base_loaded": False,
            "reuses_base_topology": False,
            "reuses_inserted_topology": False,
            "inserted_topology_reuse_note": (
                "Inserted subset vectors are added by standard FAISS HNSW insertion; "
                "their stored raw_level0/practical_repair topology is not reused."
            ),
        },
        "quality": {
            "overall": recall,
            "source_target": split_recall,
        },
        "cost": {
            "base_index_prepare_seconds": base_index_prepare_seconds,
            "base_index_build_seconds": base_index_build_seconds,
            "inserted_add_seconds": inserted_add_seconds,
            "total_integration_seconds": total_integration_seconds,
            "integration_seconds": total_integration_seconds,
            **timing,
            "base_index_prepare_seconds_samples": prepare_samples,
            "inserted_add_seconds_samples": insert_samples,
            "integration_scope": "base_hnsw_build_plus_incremental_add",
            "excluded_from_integration": ["result_serialization", "query_search"],
            "total_search_seconds": search_seconds,
            "average_query_ms": search_seconds * 1000.0 / query_count
            if query_count
            else 0.0,
            "processed_inserted_vector_count": inserted_count,
            "processed_inserted_vector_ratio": (
                inserted_count / corpus_size if corpus_size else 0.0
            ),
        },
        "diagnostics": {
            "global_id_output": True,
            "final_index_id_count": int(len(final_ids)),
            "final_index_contains_all_union_ids": True,
            "base_vector_count": int(len(base.global_ids)),
            "inserted_vector_count": inserted_count,
        },
        "index_path": str(index_path),
        "index_size_bytes": int(index_path.stat().st_size),
        "arrays": arrays,
    }


def run_incremental_insertion_baseline(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    union: dict[str, Any],
    ground_truth_ids: np.ndarray,
    output_dir: Path,
    search_config: dict[str, Any],
    integration_repetitions: int = 1,
) -> dict[str, Any]:
    """Run the Metric Algebra/Lucene-style incremental insertion baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    directions: dict[str, Any] = {}
    for base, inserted in ((first, second), (second, first)):
        direction = _direction_label(base.subset, inserted.subset)
        directions[direction] = _run_incremental_direction(
            base,
            inserted,
            union,
            ground_truth_ids,
            output_dir / _direction_slug(direction),
            search_config,
            integration_repetitions,
        )
    metrics = {
        "method": "incremental_insertion",
        "baseline_type": "strong_incremental_insertion_baseline",
        "method_type": "baseline",
        "backend": "native_faiss_hnsw",
        "constructs_single_union_graph": True,
        "constructs_single_union_index": True,
        "distance_metric": "squared_l2",
        "corpus_size": int(len(union["global_ids"])),
        "query_count": int(len(union["queries"])),
        "dimension": int(union["vectors"].shape[1]),
        "input": {
            "graph_kind": first.graph["graph_kind"],
            "within_subset_variant": first.graph["variant"],
            "subsets": [first.subset, second.subset],
        },
        "directions": directions,
        "direction_count": int(len(directions)),
        "direction_labels": list(directions.keys()),
        "reuse_summary": {
            "logical_base_reuse": True,
            "physical_base_loaded": False,
            "reuses_inserted_topology": False,
            "not_proposed_topology_reuse_method": True,
        },
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def _rank_union_candidates(
    query: np.ndarray,
    candidates: list[int],
    id_to_vector: dict[int, np.ndarray],
    k: int,
) -> tuple[list[int], list[float], int]:
    """Deduplicate global-ID candidates and rerank by shared-space squared-L2."""
    unique_candidates = sorted(set(int(value) for value in candidates if int(value) >= 0))
    distances = []
    query64 = query.astype(np.float64)
    for candidate in unique_candidates:
        delta = query64 - id_to_vector[candidate].astype(np.float64)
        distances.append((float(np.sum(delta**2)), candidate))
    distances.sort()
    selected = distances[:k]
    return (
        [candidate for _, candidate in selected],
        [distance for distance, _ in selected],
        len(unique_candidates),
    )


def dual_index_search_external(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    queries: np.ndarray,
    k: int,
    candidate_count_per_subset: int,
    ef_search: int,
    distance_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Search two local external graphs independently and globally rerank candidates."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if candidate_count_per_subset <= 0:
        raise ValueError("candidate_count_per_subset must be positive.")
    if first.transformed_vectors.shape[1] != second.transformed_vectors.shape[1]:
        raise ValueError("Pairwise inputs have different vector dimensions.")
    max_candidates_a = min(candidate_count_per_subset, len(first.global_ids))
    max_candidates_b = min(candidate_count_per_subset, len(second.global_ids))

    search_started = perf_counter()
    ids_a, _ = external_graph_search(
        first.graph,
        first.transformed_vectors,
        queries,
        max_candidates_a,
        ef_search,
        distance_batch_size=distance_batch_size,
    )
    ids_b, _ = external_graph_search(
        second.graph,
        second.transformed_vectors,
        queries,
        max_candidates_b,
        ef_search,
        distance_batch_size=distance_batch_size,
    )
    local_search_seconds = perf_counter() - search_started

    id_to_vector = {
        int(global_id): first.transformed_vectors[index]
        for index, global_id in enumerate(first.global_ids)
    }
    id_to_vector.update(
        {
            int(global_id): second.transformed_vectors[index]
            for index, global_id in enumerate(second.global_ids)
        }
    )
    result_ids = np.full((len(queries), k), -1, dtype=np.int64)
    result_distances = np.full((len(queries), k), np.inf, dtype=np.float32)
    rerank_started = perf_counter()
    rerank_distance_count = 0
    unique_candidate_counts = np.zeros(len(queries), dtype=np.int64)
    for query_index, query in enumerate(queries):
        selected_ids, selected_distances, unique_count = _rank_union_candidates(
            query,
            [*ids_a[query_index].tolist(), *ids_b[query_index].tolist()],
            id_to_vector,
            k,
        )
        rerank_distance_count += unique_count
        unique_candidate_counts[query_index] = unique_count
        for rank, (global_id, distance) in enumerate(zip(selected_ids, selected_distances)):
            result_ids[query_index, rank] = global_id
            result_distances[query_index, rank] = distance
    rerank_seconds = perf_counter() - rerank_started
    diagnostics = {
        "backend": "external_graph_search",
        "constructs_single_union_graph": False,
        "candidate_count_per_subset": {
            first.subset: int(max_candidates_a),
            second.subset: int(max_candidates_b),
        },
        "local_search_seconds": local_search_seconds,
        "rerank_seconds": rerank_seconds,
        "rerank_distance_count": int(rerank_distance_count),
        "unique_candidate_count_mean": float(np.mean(unique_candidate_counts))
        if len(unique_candidate_counts)
        else 0.0,
        "unique_candidate_count_max": int(np.max(unique_candidate_counts))
        if len(unique_candidate_counts)
        else 0,
    }
    return result_ids, result_distances, diagnostics


def run_dual_index_baseline(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    union: dict[str, Any],
    ground_truth_ids: np.ndarray,
    output_dir: Path,
    search_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the dual-index reuse baseline without constructing a merged graph."""
    output_dir.mkdir(parents=True, exist_ok=True)
    integration_started = perf_counter()
    candidate_count = min(
        max(int(value) for value in search_config["evaluation_k"]),
        len(first.global_ids),
        len(second.global_ids),
    )
    integration_seconds = perf_counter() - integration_started
    max_k = min(max(int(value) for value in search_config["evaluation_k"]), len(union["global_ids"]))
    predicted_ids, predicted_distances, diagnostics = dual_index_search_external(
        first,
        second,
        union["queries"],
        max_k,
        candidate_count,
        int(search_config["ef_search"]),
        int(search_config["brute_force_batch_size"]),
    )
    recall, per_query = recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        [int(value) for value in search_config["evaluation_k"]],
    )
    id_to_subset = subset_lookup(union["subset_to_ids"])
    split_recall = source_target_recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        union["query_subsets"],
        id_to_subset,
        [int(value) for value in search_config["evaluation_k"]],
    )
    arrays = _save_arrays(
        output_dir,
        {
            "neighbor_global_ids": predicted_ids,
            "neighbor_distances": predicted_distances,
            **{f"per_query_recall_{k}": values for k, values in per_query.items()},
        },
    )
    query_count = len(union["queries"])
    total_search_seconds = diagnostics["local_search_seconds"] + diagnostics["rerank_seconds"]
    metrics = {
        "method": "dual_index",
        "baseline_type": "reuse_query_time_baseline",
        "backend": diagnostics["backend"],
        "constructs_single_union_graph": False,
        "distance_metric": "squared_l2",
        "corpus_size": int(len(union["global_ids"])),
        "query_count": int(query_count),
        "dimension": int(union["vectors"].shape[1]),
        "input": {
            "graph_kind": first.graph["graph_kind"],
            "within_subset_variant": first.graph["variant"],
            "subsets": [first.subset, second.subset],
        },
        "quality": {
            "overall": recall,
            "source_target": split_recall,
        },
        "cost": {
            "integration_seconds": integration_seconds,
            "integration_cost_note": "no-build baseline; integration cost is input selection only",
            "total_search_seconds": total_search_seconds,
            "average_query_ms": total_search_seconds * 1000.0 / query_count
            if query_count
            else 0.0,
            "local_search_seconds": diagnostics["local_search_seconds"],
            "rerank_seconds": diagnostics["rerank_seconds"],
            "candidate_count_per_subset": diagnostics["candidate_count_per_subset"],
            "rerank_distance_count": diagnostics["rerank_distance_count"],
            "processed_vector_ratio": 0.0,
        },
        "diagnostics": {
            "unique_candidate_count_mean": diagnostics["unique_candidate_count_mean"],
            "unique_candidate_count_max": diagnostics["unique_candidate_count_max"],
            "global_id_output": True,
        },
        "arrays": arrays,
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def run_no_cross_merged_baseline(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    union: dict[str, Any],
    ground_truth_ids: np.ndarray,
    output_dir: Path,
    search_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the no-cross single-graph negative control."""
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_started = perf_counter()
    graph, merge_diagnostics = no_cross_union_graph(first, second)
    graph_merge_seconds = perf_counter() - merge_started

    max_k = min(
        max(int(value) for value in search_config["evaluation_k"]),
        len(union["global_ids"]),
    )
    search_started = perf_counter()
    predicted_ids, predicted_distances, search_diagnostics = external_graph_search(
        graph,
        union["vectors"],
        union["queries"],
        max_k,
        int(search_config["ef_search"]),
        distance_batch_size=int(search_config["brute_force_batch_size"]),
        return_diagnostics=True,
    )
    search_seconds = perf_counter() - search_started

    recall, per_query = recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        [int(value) for value in search_config["evaluation_k"]],
    )
    id_to_subset = subset_lookup(union["subset_to_ids"])
    split_recall = source_target_recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        union["query_subsets"],
        id_to_subset,
        [int(value) for value in search_config["evaluation_k"]],
    )
    graph_path = output_dir / "graph.json"
    save_graph(graph_path, graph)
    arrays = _save_arrays(
        output_dir,
        {
            "neighbor_global_ids": predicted_ids,
            "neighbor_distances": predicted_distances,
            **{f"per_query_recall_{k}": values for k, values in per_query.items()},
        },
    )
    query_count = len(union["queries"])
    statistics = graph_statistics(graph)
    metrics = {
        "method": "no_cross_merged",
        "baseline_type": "negative_control_single_union_graph",
        "backend": "external_graph_search",
        "constructs_single_union_graph": True,
        "distance_metric": "squared_l2",
        "corpus_size": int(len(union["global_ids"])),
        "query_count": int(query_count),
        "dimension": int(union["vectors"].shape[1]),
        "input": {
            "graph_kind": graph["input_graph_kind"],
            "within_subset_variant": graph["input_variant"],
            "subsets": [first.subset, second.subset],
        },
        "graph": {
            "graph_kind": graph["graph_kind"],
            "input_graph_kind": graph["input_graph_kind"],
            "input_variant": graph["input_variant"],
            "cross_link_strategy": graph["cross_link_strategy"],
            "added_cross_link_count": graph["added_cross_link_count"],
            "cross_subset_edge_count": merge_diagnostics["cross_subset_edge_count"],
            "reused_within_subset_edge_count": merge_diagnostics[
                "reused_within_subset_edge_count"
            ],
            "reused_within_subset_edge_ratio": merge_diagnostics[
                "reused_within_subset_edge_ratio"
            ],
            "entry_point_source_subset": merge_diagnostics[
                "entry_point_source_subset"
            ],
        },
        "quality": {
            "overall": recall,
            "source_target": split_recall,
        },
        "cost": {
            "integration_seconds": graph_merge_seconds,
            "graph_merge_seconds": graph_merge_seconds,
            "total_search_seconds": search_seconds,
            "average_query_ms": search_seconds * 1000.0 / query_count
            if query_count
            else 0.0,
            "processed_vector_ratio": 0.0,
        },
        "graph_statistics": statistics,
        "diagnostics": {
            "global_id_output": True,
            "input_edge_count": merge_diagnostics["input_edge_count"],
        },
        "graph_path": str(graph_path),
        "arrays": arrays,
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def _select_cross_link_parameters(
    cross_link_config: dict[str, Any],
) -> tuple[float, int, int]:
    smoke_defaults = cross_link_config.get("smoke_default", {})
    sample_ratio = float(smoke_defaults.get("sample_ratio", 0.05))
    cross_k = int(smoke_defaults.get("cross_k", 4))
    if sample_ratio not in [float(value) for value in cross_link_config["sample_ratios"]]:
        sample_ratio = float(cross_link_config["sample_ratios"][0])
    if cross_k not in [int(value) for value in cross_link_config["cross_k"]]:
        cross_k = int(cross_link_config["cross_k"][0])
    return sample_ratio, cross_k, int(cross_link_config["random_seed"])


def run_random_cross_links_baseline(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    union: dict[str, Any],
    ground_truth_ids: np.ndarray,
    output_dir: Path,
    search_config: dict[str, Any],
    cross_link_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the weak random-cross-link single-graph baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ratio, cross_k, random_seed = _select_cross_link_parameters(
        cross_link_config
    )
    merge_started = perf_counter()
    graph, base_diagnostics = no_cross_union_graph(first, second)
    graph_merge_seconds = perf_counter() - merge_started
    cross_link_started = perf_counter()
    graph, diagnostics = add_random_cross_links(
        graph,
        base_diagnostics,
        first,
        second,
        sample_ratio=sample_ratio,
        cross_k=cross_k,
        random_seed=random_seed,
        bidirectional=True,
    )
    cross_link_construction_seconds = perf_counter() - cross_link_started
    integration_seconds = graph_merge_seconds + cross_link_construction_seconds
    cross_link_cost = random_cross_link_cost(
        diagnostics["added_cross_link_count"],
        edge_weight_distance_count=diagnostics["edge_weight_distance_count"],
        endpoint_selection_distance_count=diagnostics[
            "endpoint_selection_distance_count"
        ],
    )

    max_k = min(
        max(int(value) for value in search_config["evaluation_k"]),
        len(union["global_ids"]),
    )
    search_started = perf_counter()
    predicted_ids, predicted_distances, search_diagnostics = external_graph_search(
        graph,
        union["vectors"],
        union["queries"],
        max_k,
        int(search_config["ef_search"]),
        distance_batch_size=int(search_config["brute_force_batch_size"]),
        return_diagnostics=True,
    )
    search_seconds = perf_counter() - search_started

    recall, per_query = recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        [int(value) for value in search_config["evaluation_k"]],
    )
    id_to_subset = subset_lookup(union["subset_to_ids"])
    split_recall = source_target_recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        union["query_subsets"],
        id_to_subset,
        [int(value) for value in search_config["evaluation_k"]],
    )
    graph_path = output_dir / "graph.json"
    save_graph(graph_path, graph)
    arrays = _save_arrays(
        output_dir,
        {
            "neighbor_global_ids": predicted_ids,
            "neighbor_distances": predicted_distances,
            **{f"per_query_recall_{k}": values for k, values in per_query.items()},
        },
    )
    query_count = len(union["queries"])
    statistics = graph_statistics(graph)
    metrics = {
        "method": "random_cross_links",
        "baseline_type": "weak_random_cross_link_baseline",
        "backend": "external_graph_search",
        "constructs_single_union_graph": True,
        "distance_metric": "squared_l2",
        "corpus_size": int(len(union["global_ids"])),
        "query_count": int(query_count),
        "dimension": int(union["vectors"].shape[1]),
        "input": {
            "graph_kind": graph["input_graph_kind"],
            "within_subset_variant": graph["input_variant"],
            "subsets": [first.subset, second.subset],
        },
        "graph": {
            "graph_kind": graph["graph_kind"],
            "input_graph_kind": graph["input_graph_kind"],
            "input_variant": graph["input_variant"],
            "cross_link_strategy": graph["cross_link_strategy"],
            "sample_ratio": sample_ratio,
            "sample_ratio_A": diagnostics["sample_ratio_A"],
            "sample_ratio_B": diagnostics["sample_ratio_B"],
            "sampled_node_count_A": diagnostics["sampled_node_count_A"],
            "sampled_node_count_B": diagnostics["sampled_node_count_B"],
            "sampled_node_ids_A": diagnostics["sampled_node_ids_A"],
            "sampled_node_ids_B": diagnostics["sampled_node_ids_B"],
            "sampled_bridge_node_count": diagnostics["sampled_bridge_node_count"],
            "sampled_bridge_ratio": diagnostics["sampled_bridge_ratio"],
            "cross_k": cross_k,
            "random_seed": random_seed,
            "bidirectional": diagnostics["bidirectional"],
            "added_cross_link_count": diagnostics["added_cross_link_count"],
            "cross_subset_edge_count": diagnostics["cross_subset_edge_count"],
            "reused_within_subset_edge_count": diagnostics[
                "reused_within_subset_edge_count"
            ],
            "reused_within_subset_edge_ratio": diagnostics[
                "reused_within_subset_edge_ratio"
            ],
            "entry_point_source_subset": diagnostics[
                "entry_point_source_subset"
            ],
        },
        "quality": {
            "overall": recall,
            "source_target": split_recall,
        },
        "cost": {
            "integration_seconds": integration_seconds,
            "graph_merge_seconds": graph_merge_seconds,
            "cross_link_construction_seconds": cross_link_construction_seconds,
            "total_search_seconds": search_seconds,
            "average_query_ms": search_seconds * 1000.0 / query_count
            if query_count
            else 0.0,
            "processed_vector_ratio": diagnostics["sampled_bridge_ratio"],
            "sampled_bridge_ratio": diagnostics["sampled_bridge_ratio"],
            **cross_link_cost,
        },
        "graph_statistics": statistics,
        "diagnostics": {
            "global_id_output": True,
            "input_edge_count": diagnostics["input_edge_count"],
        },
        "graph_path": str(graph_path),
        "arrays": arrays,
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def run_sampled_cross_knn_baseline(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    union: dict[str, Any],
    ground_truth_ids: np.ndarray,
    output_dir: Path,
    search_config: dict[str, Any],
    cross_link_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the first proposed sampled cross-kNN topology-reuse method."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ratio, cross_k, random_seed = _select_cross_link_parameters(
        cross_link_config
    )
    merge_started = perf_counter()
    graph, base_diagnostics = no_cross_union_graph(first, second)
    graph_merge_seconds = perf_counter() - merge_started
    candidate_started = perf_counter()
    graph, diagnostics = add_sampled_cross_knn_links(
        graph,
        base_diagnostics,
        first,
        second,
        sample_ratio=sample_ratio,
        cross_k=cross_k,
        random_seed=random_seed,
        bidirectional=True,
    )
    cross_candidate_search_seconds = perf_counter() - candidate_started
    cross_link_construction_seconds = cross_candidate_search_seconds
    integration_seconds = graph_merge_seconds + cross_link_construction_seconds
    cost_definition = sampled_cross_knn_cost(
        len(first.global_ids),
        len(second.global_ids),
        diagnostics["sampled_node_count_A"],
        diagnostics["sampled_node_count_B"],
    )
    if (
        cost_definition["cross_candidate_distance_count"]
        != diagnostics["cross_candidate_distance_count"]
    ):
        raise ValueError("Sampled cross-kNN cost definition mismatch.")

    max_k = min(
        max(int(value) for value in search_config["evaluation_k"]),
        len(union["global_ids"]),
    )
    search_started = perf_counter()
    predicted_ids, predicted_distances, search_diagnostics = external_graph_search(
        graph,
        union["vectors"],
        union["queries"],
        max_k,
        int(search_config["ef_search"]),
        distance_batch_size=int(search_config["brute_force_batch_size"]),
        return_diagnostics=True,
    )
    search_seconds = perf_counter() - search_started

    recall, per_query = recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        [int(value) for value in search_config["evaluation_k"]],
    )
    id_to_subset = subset_lookup(union["subset_to_ids"])
    split_recall = source_target_recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        union["query_subsets"],
        id_to_subset,
        [int(value) for value in search_config["evaluation_k"]],
    )
    graph_path = output_dir / "graph.json"
    save_graph(graph_path, graph)
    arrays = _save_arrays(
        output_dir,
        {
            "neighbor_global_ids": predicted_ids,
            "neighbor_distances": predicted_distances,
            **{f"per_query_recall_{k}": values for k, values in per_query.items()},
        },
    )
    query_count = len(union["queries"])
    statistics = graph_statistics(graph)
    metrics = {
        "method": "sampled_cross_knn",
        "method_type": "proposed_topology_reuse",
        "baseline_type": "proposed_topology_reuse",
        "backend": "external_graph_search",
        "constructs_single_union_graph": True,
        "distance_metric": "squared_l2",
        "corpus_size": int(len(union["global_ids"])),
        "query_count": int(query_count),
        "dimension": int(union["vectors"].shape[1]),
        "input": {
            "graph_kind": graph["input_graph_kind"],
            "within_subset_variant": graph["input_variant"],
            "subsets": [first.subset, second.subset],
        },
        "graph": {
            "graph_kind": graph["graph_kind"],
            "input_graph_kind": graph["input_graph_kind"],
            "input_variant": graph["input_variant"],
            "cross_link_strategy": graph["cross_link_strategy"],
            "sample_ratio": sample_ratio,
            "sample_ratio_A": diagnostics["sample_ratio_A"],
            "sample_ratio_B": diagnostics["sample_ratio_B"],
            "sampled_node_count_A": diagnostics["sampled_node_count_A"],
            "sampled_node_count_B": diagnostics["sampled_node_count_B"],
            "sampled_node_ids_A": diagnostics["sampled_node_ids_A"],
            "sampled_node_ids_B": diagnostics["sampled_node_ids_B"],
            "sampled_bridge_node_count": diagnostics["sampled_bridge_node_count"],
            "sampled_bridge_ratio": diagnostics["sampled_bridge_ratio"],
            "cross_k": cross_k,
            "random_seed": random_seed,
            "bidirectional": diagnostics["bidirectional"],
            "degree_pruning": graph["degree_pruning"],
            "connectivity_repair": graph["connectivity_repair"],
            "added_cross_link_count": diagnostics["added_cross_link_count"],
            "cross_subset_edge_count": diagnostics["cross_subset_edge_count"],
            "reused_within_subset_edge_count": diagnostics[
                "reused_within_subset_edge_count"
            ],
            "reused_within_subset_edge_ratio": diagnostics[
                "reused_within_subset_edge_ratio"
            ],
            "entry_point_source_subset": diagnostics[
                "entry_point_source_subset"
            ],
        },
        "quality": {
            "overall": recall,
            "source_target": split_recall,
        },
        "cost": {
            "integration_seconds": integration_seconds,
            "graph_merge_seconds": graph_merge_seconds,
            "cross_candidate_search_seconds": cross_candidate_search_seconds,
            "cross_link_construction_seconds": cross_link_construction_seconds,
            "total_search_seconds": search_seconds,
            "average_query_ms": search_seconds * 1000.0 / query_count
            if query_count
            else 0.0,
            "processed_vector_ratio": diagnostics["sampled_bridge_ratio"],
            "sampled_bridge_ratio": diagnostics["sampled_bridge_ratio"],
            "cross_candidate_distance_count": diagnostics[
                "cross_candidate_distance_count"
            ],
            "cross_scan_work_ratio": diagnostics["cross_scan_work_ratio"],
            "edge_weight_distance_count": diagnostics[
                "edge_weight_distance_count"
            ],
            "edge_weight_distance_source": diagnostics[
                "edge_weight_distance_source"
            ],
            "query_distance_evaluation_count": search_diagnostics[
                "distance_evaluation_count"
            ],
            "query_visited_node_count": search_diagnostics["visited_node_count"],
            "query_distance_mode": search_diagnostics["distance_mode"],
        },
        "graph_statistics": statistics,
        "diagnostics": {
            "global_id_output": True,
            "input_edge_count": diagnostics["input_edge_count"],
        },
        "graph_path": str(graph_path),
        "arrays": arrays,
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def run_indexed_cross_knn_baseline(
    first: Phase2SubsetInput,
    second: Phase2SubsetInput,
    union: dict[str, Any],
    ground_truth_ids: np.ndarray,
    output_dir: Path,
    search_config: dict[str, Any],
    cross_link_config: dict[str, Any],
    integration_repetitions: int = 1,
) -> dict[str, Any]:
    """Run topology reuse with target local-HNSW cross-candidate generation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _, cross_k, random_seed = _select_cross_link_parameters(cross_link_config)
    first, second, repair_diagnostics = _indexed_repair_inputs(
        first, second, cross_link_config, random_seed
    )
    total_samples: list[float] = []
    merge_samples: list[float] = []
    candidate_samples: list[float] = []
    graph = None
    diagnostics = None
    for _ in range(_validate_integration_repetitions(integration_repetitions)):
        merge_started = perf_counter()
        graph, base_diagnostics = no_cross_union_graph(first, second)
        graph_merge_seconds = perf_counter() - merge_started
        candidate_started = perf_counter()
        graph, diagnostics = add_indexed_cross_knn_links(
            graph,
            base_diagnostics,
            first,
            second,
            cross_k=cross_k,
            random_seed=random_seed,
            cross_link_config=cross_link_config,
            bidirectional=True,
        )
        cross_candidate_search_seconds = perf_counter() - candidate_started
        merge_samples.append(graph_merge_seconds)
        candidate_samples.append(cross_candidate_search_seconds)
        total_samples.append(graph_merge_seconds + cross_candidate_search_seconds)
    assert graph is not None and diagnostics is not None
    timing = _timing_summary(total_samples)
    graph_merge_seconds = float(np.median(merge_samples))
    cross_candidate_search_seconds = float(np.median(candidate_samples))
    integration_seconds = timing["integration_seconds_median"]
    max_k = min(
        max(int(value) for value in search_config["evaluation_k"]),
        len(union["global_ids"]),
    )
    search_started = perf_counter()
    predicted_ids, predicted_distances, search_diagnostics = external_graph_search(
        graph,
        union["vectors"],
        union["queries"],
        max_k,
        int(search_config["ef_search"]),
        distance_batch_size=int(search_config["brute_force_batch_size"]),
        return_diagnostics=True,
    )
    search_seconds = perf_counter() - search_started
    recall, per_query = recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        [int(value) for value in search_config["evaluation_k"]],
    )
    id_to_subset = subset_lookup(union["subset_to_ids"])
    split_recall = source_target_recall_at_k(
        predicted_ids,
        ground_truth_ids[:, :max_k],
        union["query_subsets"],
        id_to_subset,
        [int(value) for value in search_config["evaluation_k"]],
    )
    graph_path = output_dir / "graph.json"
    save_graph(graph_path, graph)
    arrays = _save_arrays(
        output_dir,
        {
            "neighbor_global_ids": predicted_ids,
            "neighbor_distances": predicted_distances,
            **{f"per_query_recall_{k}": values for k, values in per_query.items()},
        },
    )
    query_count = len(union["queries"])
    metrics = {
        "method": "indexed_cross_knn",
        "method_type": "proposed_topology_reuse",
        "baseline_type": "indexed_cross_candidate_topology_reuse",
        "backend": "external_graph_search",
        "constructs_single_union_graph": True,
        "distance_metric": "squared_l2",
        "corpus_size": int(len(union["global_ids"])),
        "query_count": int(query_count),
        "dimension": int(union["vectors"].shape[1]),
        "input": {
            "graph_kind": graph["input_graph_kind"],
            "within_subset_variant": graph["input_variant"],
            "subsets": [first.subset, second.subset],
        },
        "graph": {
            "graph_kind": graph["graph_kind"],
            "cross_link_strategy": graph["cross_link_strategy"],
            "bridge_budget_mode": diagnostics["bridge_budget_mode"],
            "bridge_selection": diagnostics["bridge_selection"],
            "repair_policy": repair_diagnostics["repair_policy"],
            "sampled_bridge_node_count": diagnostics["sampled_bridge_node_count"],
            "sampled_bridge_ratio": diagnostics["sampled_bridge_ratio"],
            "sampled_node_count_A": diagnostics["sampled_node_count_A"],
            "sampled_node_count_B": diagnostics["sampled_node_count_B"],
            "sampled_node_ids_A": diagnostics["sampled_node_ids_A"],
            "sampled_node_ids_B": diagnostics["sampled_node_ids_B"],
            "cross_k": cross_k,
            "candidate_k": diagnostics["candidate_k"],
            "candidate_ef_search": diagnostics["candidate_ef_search"],
            "candidate_backend": diagnostics["candidate_backend"],
            "propagation_hops": diagnostics["propagation_hops"],
            "propagation_neighbor_budget": diagnostics["propagation_neighbor_budget"],
            "propagation_candidate_cap": diagnostics["propagation_candidate_cap"],
            "propagated_neighbor_count": diagnostics["propagated_neighbor_count"],
            "max_propagated_candidates_per_neighbor": diagnostics[
                "max_propagated_candidates_per_neighbor"
            ],
            "cross_degree_cap": diagnostics["cross_degree_cap"],
            "pruned_cross_candidate_count": diagnostics["pruned_cross_candidate_count"],
            "connectivity_repair_edge_count": diagnostics["connectivity_repair_edge_count"],
            "random_seed": random_seed,
            "added_cross_link_count": diagnostics["added_cross_link_count"],
            "cross_subset_edge_count": diagnostics["cross_subset_edge_count"],
            "degree_pruning": graph["degree_pruning"],
            "connectivity_repair": graph["connectivity_repair"],
        },
        "quality": {"overall": recall, "source_target": split_recall},
        "cost": {
            "integration_seconds": integration_seconds,
            **timing,
            "graph_merge_seconds_samples": merge_samples,
            "cross_candidate_search_seconds_samples": candidate_samples,
            "integration_scope": "selection_merge_candidate_rerank_propagation",
            "excluded_from_integration": ["result_serialization", "query_search"],
            "repair_seconds": repair_diagnostics["repair_seconds"],
            "cold_integration_seconds": integration_seconds + repair_diagnostics["repair_seconds"],
            "graph_merge_seconds": graph_merge_seconds,
            "cross_candidate_search_seconds": cross_candidate_search_seconds,
            "cross_link_construction_seconds": cross_candidate_search_seconds,
            "total_search_seconds": search_seconds,
            "average_query_ms": search_seconds * 1000.0 / query_count if query_count else 0.0,
            "processed_vector_ratio": diagnostics["sampled_bridge_ratio"],
            "candidate_hnsw_call_count": diagnostics["candidate_hnsw_call_count"],
            "cross_candidate_distance_count": diagnostics["cross_candidate_distance_count"],
            "direct_candidate_distance_count": diagnostics["direct_candidate_distance_count"],
            "propagation_distance_count": diagnostics["propagation_distance_count"],
            "edge_weight_distance_count": diagnostics["edge_weight_distance_count"],
            "edge_weight_distance_source": diagnostics["edge_weight_distance_source"],
            "query_distance_evaluation_count": search_diagnostics["distance_evaluation_count"],
            "query_visited_node_count": search_diagnostics["visited_node_count"],
            "query_distance_mode": search_diagnostics["distance_mode"],
            "repair_candidate_distance_count": repair_diagnostics.get(
                "repair_candidate_distance_count", 0
            ),
            "repair_processed_node_count": repair_diagnostics.get(
                "repair_processed_node_count", 0
            ),
        },
        "graph_statistics": graph_statistics(graph),
        "diagnostics": {
            "global_id_output": True,
            "input_edge_count": diagnostics["input_edge_count"],
            "repair": repair_diagnostics,
        },
        "graph_path": str(graph_path),
        "arrays": arrays,
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


__all__ = [
    "dual_index_search_external",
    "run_bruteforce_ground_truth",
    "run_dual_index_baseline",
    "run_full_rebuild_hnsw",
    "run_incremental_insertion_baseline",
    "run_indexed_cross_knn_baseline",
    "run_no_cross_merged_baseline",
    "run_random_cross_links_baseline",
    "run_sampled_cross_knn_baseline",
    "union_phase2_pair",
]
