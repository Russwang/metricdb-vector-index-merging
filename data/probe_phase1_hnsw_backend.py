"""Probe FAISS HNSW graph access and Phase 1 graph reuse feasibility.

This is intentionally a backend probe rather than a merge implementation. It
tests Python access to HNSW layers and neighbor tables, performs one controlled
raw-link mutation on a disposable index, verifies search and serialization,
and measures local/shared kNN overlap plus level-0 graph-edge retention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase1_common import add_run_metadata, start_run
from data.prepare_phase1_transformations import SUBSET_NAMES
from data.run_phase1_bruteforce_baseline import exact_knn_squared_l2


def _write_json(path: Path, value: Any) -> None:
    """Write JSON with stable, readable formatting."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def _unwrap_hnsw_index(
    index: faiss.Index,
) -> tuple[faiss.IndexHNSW, np.ndarray]:
    """Return the underlying HNSW index and its row-to-external-ID mapping."""
    if hasattr(index, "index") and hasattr(index, "id_map"):
        raw_index = faiss.downcast_index(index.index)
        external_ids = np.asarray(faiss.vector_to_array(index.id_map), dtype=np.int64)
    else:
        raw_index = faiss.downcast_index(index)
        external_ids = np.arange(index.ntotal, dtype=np.int64)

    if not isinstance(raw_index, faiss.IndexHNSW):
        raise TypeError(f"Expected an HNSW index, got {type(raw_index).__name__}.")
    if len(external_ids) != raw_index.ntotal:
        raise ValueError("HNSW row count and external ID map length differ.")
    return raw_index, external_ids


def extract_hnsw_layers(index: faiss.Index) -> dict[int, dict[int, list[int]]]:
    """Extract every node's links at every layer using external IDs."""
    raw_index, external_ids = _unwrap_hnsw_index(index)
    hnsw = raw_index.hnsw
    levels = np.asarray(faiss.vector_to_array(hnsw.levels), dtype=np.int64)
    neighbors = np.asarray(faiss.vector_to_array(hnsw.neighbors), dtype=np.int64)

    graph: dict[int, dict[int, list[int]]] = {}
    for internal_id, level_count in enumerate(levels):
        node_layers: dict[int, list[int]] = {}
        for layer in range(int(level_count)):
            begin = np.zeros(1, dtype=np.uint64)
            end = np.zeros(1, dtype=np.uint64)
            hnsw.neighbor_range(
                internal_id,
                layer,
                faiss.swig_ptr(begin),
                faiss.swig_ptr(end),
            )
            internal_neighbors = neighbors[int(begin[0]) : int(end[0])]
            valid_neighbors = internal_neighbors[internal_neighbors >= 0]
            node_layers[layer] = [
                int(external_ids[neighbor]) for neighbor in valid_neighbors
            ]
        graph[int(external_ids[internal_id])] = node_layers
    return graph


def _level_zero_edge_set(index: faiss.Index) -> set[tuple[int, int]]:
    """Return directed level-0 edges in internal row-ID space."""
    raw_index, _ = _unwrap_hnsw_index(index)
    hnsw = raw_index.hnsw
    levels = np.asarray(faiss.vector_to_array(hnsw.levels), dtype=np.int64)
    neighbors = np.asarray(faiss.vector_to_array(hnsw.neighbors), dtype=np.int64)
    edges: set[tuple[int, int]] = set()

    for source, level_count in enumerate(levels):
        if level_count < 1:
            continue
        begin = np.zeros(1, dtype=np.uint64)
        end = np.zeros(1, dtype=np.uint64)
        hnsw.neighbor_range(source, 0, faiss.swig_ptr(begin), faiss.swig_ptr(end))
        for target in neighbors[int(begin[0]) : int(end[0])]:
            if target >= 0 and target != source:
                edges.add((source, int(target)))
    return edges


def _query_derived_graph(
    index: faiss.Index,
    vectors: np.ndarray,
    global_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    """Construct an approximate kNN graph only through public search()."""
    search_k = min(len(vectors), k + 1)
    _, raw_ids = index.search(
        np.ascontiguousarray(vectors, dtype=np.float32), search_k
    )
    graph = np.empty((len(vectors), k), dtype=np.int64)
    for row, global_id in enumerate(global_ids):
        candidates = raw_ids[row][raw_ids[row] != global_id]
        candidates = candidates[candidates >= 0]
        if len(candidates) < k:
            raise RuntimeError(f"Query-derived graph row {row} has fewer than {k} links.")
        graph[row] = candidates[:k]
    return graph


def _mean_knn_overlap(
    local_vectors: np.ndarray,
    shared_vectors: np.ndarray,
    global_ids: np.ndarray,
    k: int,
) -> float:
    """Compare exact within-subset kNN before and after transformation."""
    local_ids, _ = exact_knn_squared_l2(
        local_vectors,
        local_vectors,
        global_ids,
        k,
        exclude_query_global_ids=global_ids,
    )
    shared_ids, _ = exact_knn_squared_l2(
        shared_vectors,
        shared_vectors,
        global_ids,
        k,
        exclude_query_global_ids=global_ids,
    )
    overlaps = [
        len(set(local_row.tolist()) & set(shared_row.tolist())) / k
        for local_row, shared_row in zip(local_ids, shared_ids)
    ]
    return float(np.mean(overlaps))


def run_faiss_capability_probe(output_dir: Path) -> dict[str, Any]:
    """Run a disposable graph-read, raw-mutation, search, and reload probe."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    vectors = rng.normal(size=(48, 8)).astype(np.float32)
    index = faiss.IndexHNSWFlat(8, 4, faiss.METRIC_L2)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 32
    index.add(vectors)

    graph_before = extract_hnsw_layers(index)
    _, search_before = index.search(vectors[:8], 5)
    neighbors = np.asarray(
        faiss.vector_to_array(index.hnsw.neighbors), dtype=np.int32
    )
    valid_slots = np.flatnonzero(neighbors >= 0)
    if len(valid_slots) == 0:
        raise RuntimeError("Probe HNSW unexpectedly has no graph links.")

    mutation_slot = int(valid_slots[0])
    old_target = int(neighbors[mutation_slot])
    candidate_targets = [
        candidate
        for candidate in range(index.ntotal)
        if candidate != old_target
    ]
    new_target = candidate_targets[0]
    neighbors[mutation_slot] = new_target
    faiss.copy_array_to_vector(neighbors, index.hnsw.neighbors)

    persisted_neighbors = np.asarray(
        faiss.vector_to_array(index.hnsw.neighbors), dtype=np.int32
    )
    mutation_visible = int(persisted_neighbors[mutation_slot]) == new_target
    _, search_after = index.search(vectors[:8], 5)

    modified_index_path = output_dir / "modified_probe_hnsw.faiss"
    faiss.write_index(index, str(modified_index_path))
    reloaded = faiss.read_index(str(modified_index_path))
    reloaded_neighbors = np.asarray(
        faiss.vector_to_array(reloaded.hnsw.neighbors), dtype=np.int32
    )
    _, search_reloaded = reloaded.search(vectors[:8], 5)
    reload_preserved_mutation = (
        int(reloaded_neighbors[mutation_slot]) == new_target
    )

    graph_path = output_dir / "all_layer_links_before_mutation.json"
    _write_json(graph_path, graph_before)
    return {
        "faiss_version": faiss.__version__,
        "python_api": {
            "levels_exposed": hasattr(index.hnsw, "levels"),
            "offsets_exposed": hasattr(index.hnsw, "offsets"),
            "neighbors_exposed": hasattr(index.hnsw, "neighbors"),
            "neighbor_range_exposed": hasattr(index.hnsw, "neighbor_range"),
            "copy_array_to_vector_available": hasattr(
                faiss, "copy_array_to_vector"
            ),
            "init_level_0_from_knngraph_exposed": hasattr(
                index, "init_level_0_from_knngraph"
            ),
        },
        "graph_read": {
            "node_count": len(graph_before),
            "max_layer": int(index.hnsw.max_level),
            "entry_point": int(index.hnsw.entry_point),
            "all_layer_links": str(graph_path),
        },
        "raw_mutation_probe": {
            "slot": mutation_slot,
            "old_internal_target": old_target,
            "new_internal_target": new_target,
            "mutation_visible_in_memory": mutation_visible,
            "search_completed_after_mutation": search_after.shape == search_before.shape,
            "search_results_changed": not np.array_equal(
                search_before, search_after
            ),
            "reload_preserved_mutation": reload_preserved_mutation,
            "search_completed_after_reload": search_reloaded.shape
            == search_before.shape,
            "modified_index": str(modified_index_path),
        },
        "safety_conclusion": {
            "raw_link_overwrite_mechanically_possible": True,
            "safe_high_level_link_edit_api": False,
            "invariants_automatically_maintained": False,
            "note": (
                "Direct neighbor-vector writes bypass reciprocal-link, degree, "
                "layer, connectivity, and entry-point maintenance."
            ),
        },
    }


def analyze_phase1_graph_reuse(
    transformation_dir: Path,
    output_dir: Path,
    k: int = 10,
) -> dict[str, Any]:
    """Measure kNN overlap, graph retention, and query-derived graph support."""
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments: dict[str, Any] = {}
    with (transformation_dir / "manifest.json").open(
        "r", encoding="utf-8"
    ) as file:
        transformation_manifest = json.load(file)
    experiment_names = tuple(transformation_manifest["experiments"].keys())
    subset_names = tuple(transformation_manifest.get("subset_names", SUBSET_NAMES))
    hnsw_parameters = transformation_manifest["hnsw"]

    for experiment_name in experiment_names:
        subset_results: dict[str, Any] = {}
        for subset_name in subset_names:
            subset_dir = transformation_dir / experiment_name / subset_name
            local_vectors = np.ascontiguousarray(
                np.load(subset_dir / "local_vectors.npy"), dtype=np.float32
            )
            shared_vectors = np.ascontiguousarray(
                np.load(subset_dir / "transformed_vectors.npy"), dtype=np.float32
            )
            with (subset_dir / "id_mapping.json").open(
                "r", encoding="utf-8"
            ) as file:
                id_mapping = json.load(file)
            global_ids = np.array(
                [record["global_id"] for record in id_mapping], dtype=np.int64
            )
            effective_k = min(k, len(local_vectors) - 1)

            local_index = faiss.read_index(str(subset_dir / "local_hnsw.faiss"))
            local_edges = _level_zero_edge_set(local_index)

            rebuilt_shared = faiss.IndexHNSWFlat(
                shared_vectors.shape[1],
                int(hnsw_parameters["m"]),
                faiss.METRIC_L2,
            )
            rebuilt_shared.hnsw.efConstruction = int(
                hnsw_parameters["ef_construction"]
            )
            rebuilt_shared.hnsw.efSearch = int(
                hnsw_parameters.get("ef_search", 32)
            )
            rebuilt_shared.add(shared_vectors)
            shared_edges = _level_zero_edge_set(rebuilt_shared)
            retained_edges = local_edges & shared_edges
            edge_retention = (
                len(retained_edges) / len(local_edges) if local_edges else 1.0
            )

            query_graph = _query_derived_graph(
                local_index, local_vectors, global_ids, effective_k
            )
            experiment_output_dir = output_dir / experiment_name / subset_name
            experiment_output_dir.mkdir(parents=True, exist_ok=True)
            query_graph_path = experiment_output_dir / "hnsw_derived_graph.npz"
            np.savez(
                query_graph_path,
                source_global_ids=global_ids,
                neighbor_global_ids=query_graph,
            )

            subset_results[subset_name] = {
                "size": int(len(local_vectors)),
                "k": effective_k,
                "exact_knn_overlap_local_vs_shared": _mean_knn_overlap(
                    local_vectors, shared_vectors, global_ids, effective_k
                ),
                "local_level0_directed_edges": len(local_edges),
                "rebuilt_shared_level0_directed_edges": len(shared_edges),
                "retained_level0_directed_edges": len(retained_edges),
                "level0_edge_retention": edge_retention,
                "hnsw_derived_graph": str(query_graph_path),
            }
        experiments[experiment_name] = {"subsets": subset_results}

    return {
        "source_transformations": str(transformation_dir),
        "k": k,
        "subset_names": list(subset_names),
        "experiment_names": list(experiment_names),
        "hnsw": hnsw_parameters,
        "edge_retention_definition": (
            "directed level-0 edge intersection between the original local "
            "HNSW and a same-parameter HNSW rebuilt on transformed vectors"
        ),
        "experiments": experiments,
    }


def run_backend_probe(
    transformation_dir: Path,
    output_dir: Path,
    k: int = 10,
) -> dict[str, Any]:
    """Run capability and Phase 1 graph-reuse probes and save one manifest."""
    started_at, start_counter = start_run()
    output_dir.mkdir(parents=True, exist_ok=True)
    capability = run_faiss_capability_probe(output_dir / "capability")
    graph_reuse = analyze_phase1_graph_reuse(
        transformation_dir, output_dir / "graph_reuse", k=k
    )
    manifest = add_run_metadata({
        "probe": "faiss_hnsw_backend",
        "scope": "minimal_backend_probe_not_merge_algorithm",
        "capability": capability,
        "graph_reuse": graph_reuse,
    }, started_at, start_counter)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the backend probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformation-dir",
        type=Path,
        default=Path("data/phase1_transformations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase1_backend_probe"),
    )
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Run the probe and print its main conclusions and measurements."""
    args = parse_args()
    manifest = run_backend_probe(
        transformation_dir=args.transformation_dir,
        output_dir=args.output_dir,
        k=args.k,
    )
    capability = manifest["capability"]
    mutation = capability["raw_mutation_probe"]
    print(f"FAISS {capability['faiss_version']} HNSW backend probe:")
    print("- Per-node, per-layer graph links are readable from Python.")
    print(
        "- Raw link overwrite/search/save/reload: "
        f"{mutation['mutation_visible_in_memory']}/"
        f"{mutation['search_completed_after_mutation']}/"
        f"{mutation['reload_preserved_mutation']}/"
        f"{mutation['search_completed_after_reload']}"
    )
    print("- Direct raw mutation is possible but not a safe graph-edit API.")
    for experiment_name, experiment in manifest["graph_reuse"][
        "experiments"
    ].items():
        values = []
        for subset_name, subset in experiment["subsets"].items():
            values.append(
                f"{subset_name}: kNN={subset['exact_knn_overlap_local_vs_shared']:.3f}, "
                f"edges={subset['level0_edge_retention']:.3f}"
            )
        print(f"- {experiment_name}: " + "; ".join(values))


if __name__ == "__main__":
    main()
