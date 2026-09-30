"""Build and evaluate full-rebuild HNSW baselines for Phase 1.

For each transformation experiment, this script rebuilds one global HNSW from
the unioned transformed vectors. It searches the same queries used by the
brute-force baseline and reports recall@k, build time, query latency, and index
size. This represents the cost and quality of discarding all local indices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.prepare_phase1_transformations import EXPERIMENT_NAMES
from data.phase1_common import add_run_metadata, start_run


def _load_json(path: Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def recall_at_k(
    predicted_neighbor_ids: np.ndarray,
    ground_truth_neighbor_ids: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return mean recall@k and one recall value per query."""
    if predicted_neighbor_ids.shape != ground_truth_neighbor_ids.shape:
        raise ValueError(
            "Predicted and ground-truth arrays must have the same shape, got "
            f"{predicted_neighbor_ids.shape} and {ground_truth_neighbor_ids.shape}."
        )
    if predicted_neighbor_ids.ndim != 2 or predicted_neighbor_ids.shape[1] == 0:
        raise ValueError("Neighbor arrays must be non-empty 2D matrices.")

    k = predicted_neighbor_ids.shape[1]
    per_query_recall = np.empty(len(predicted_neighbor_ids), dtype=np.float64)
    for query_index, (predicted, ground_truth) in enumerate(
        zip(predicted_neighbor_ids, ground_truth_neighbor_ids)
    ):
        per_query_recall[query_index] = (
            len(set(predicted.tolist()) & set(ground_truth.tolist())) / k
        )
    return float(np.mean(per_query_recall)), per_query_recall


def _build_hnsw(
    vectors: np.ndarray,
    global_ids: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> tuple[faiss.IndexIDMap2, float]:
    """Build a global squared-L2 HNSW and return its measured build time."""
    if m <= 0 or ef_construction <= 0 or ef_search <= 0:
        raise ValueError("HNSW parameters must all be positive.")

    start_time = perf_counter()
    hnsw = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_L2)
    hnsw.hnsw.efConstruction = ef_construction
    hnsw.hnsw.efSearch = ef_search
    index = faiss.IndexIDMap2(hnsw)
    index.add_with_ids(
        np.ascontiguousarray(vectors, dtype=np.float32),
        np.asarray(global_ids, dtype=np.int64),
    )
    build_seconds = perf_counter() - start_time
    return index, build_seconds


def _search_excluding_self(
    index: faiss.Index,
    query_vectors: np.ndarray,
    query_global_ids: np.ndarray,
    k: int,
    corpus_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Search HNSW and remove each query's own global ID from its results."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if k >= corpus_size:
        raise ValueError(f"k={k} must be smaller than corpus size {corpus_size}.")

    search_k = min(corpus_size, k + 1)
    start_time = perf_counter()
    raw_distances, raw_ids = index.search(
        np.ascontiguousarray(query_vectors, dtype=np.float32), search_k
    )
    search_seconds = perf_counter() - start_time

    neighbor_ids = np.empty((len(query_vectors), k), dtype=np.int64)
    neighbor_distances = np.empty((len(query_vectors), k), dtype=np.float32)
    for query_index, query_global_id in enumerate(query_global_ids):
        keep = raw_ids[query_index] != query_global_id
        filtered_ids = raw_ids[query_index][keep]
        filtered_distances = raw_distances[query_index][keep]
        if len(filtered_ids) < k or np.any(filtered_ids[:k] < 0):
            raise RuntimeError(
                f"HNSW returned fewer than {k} valid non-self neighbors for "
                f"query global ID {query_global_id}."
            )
        neighbor_ids[query_index] = filtered_ids[:k]
        neighbor_distances[query_index] = filtered_distances[:k]

    return neighbor_ids, neighbor_distances, search_seconds


def run_phase1_hnsw_rebuild_baseline(
    brute_force_dir: Path,
    output_dir: Path,
    m: int = 16,
    ef_construction: int = 40,
    ef_search: int = 32,
) -> dict[str, Any]:
    """Rebuild and evaluate one global HNSW for every Phase 1 experiment."""
    started_at, start_counter = start_run()
    brute_force_manifest = _load_json(brute_force_dir / "manifest.json")
    if not brute_force_manifest.get("exclude_self", False):
        raise ValueError("HNSW rebuild baseline requires self-excluded ground truth.")

    k = int(brute_force_manifest["k"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "baseline": "hnsw_full_rebuild",
        "source_brute_force": str(brute_force_dir),
        "distance_metric": "squared_l2",
        "k": k,
        "exclude_self": True,
        "hnsw": {
            "m": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
        },
        "experiments": {},
    }

    for experiment_name, ground_truth_experiment in brute_force_manifest[
        "experiments"
    ].items():
        vectors = np.ascontiguousarray(
            np.load(ground_truth_experiment["union_transformed_vectors"]),
            dtype=np.float32,
        )
        corpus_global_ids = np.asarray(
            np.load(ground_truth_experiment["union_global_ids"]), dtype=np.int64
        )
        query_global_ids = np.asarray(
            np.load(ground_truth_experiment["query_global_ids"]), dtype=np.int64
        )
        ground_truth_ids = np.asarray(
            np.load(ground_truth_experiment["neighbor_global_ids"]), dtype=np.int64
        )

        if len(vectors) != len(corpus_global_ids):
            raise ValueError(
                f"{experiment_name} vector and corpus ID counts do not match."
            )
        if not np.array_equal(query_global_ids, corpus_global_ids):
            raise ValueError(
                f"{experiment_name} queries must match union corpus row order."
            )
        if ground_truth_ids.shape != (len(query_global_ids), k):
            raise ValueError(
                f"{experiment_name} ground truth has shape "
                f"{ground_truth_ids.shape}; expected {(len(query_global_ids), k)}."
            )

        index, build_seconds = _build_hnsw(
            vectors=vectors,
            global_ids=corpus_global_ids,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        predicted_ids, predicted_distances, search_seconds = _search_excluding_self(
            index=index,
            query_vectors=vectors,
            query_global_ids=query_global_ids,
            k=k,
            corpus_size=len(vectors),
        )
        mean_recall, per_query_recall = recall_at_k(
            predicted_neighbor_ids=predicted_ids,
            ground_truth_neighbor_ids=ground_truth_ids,
        )

        experiment_output_dir = output_dir / experiment_name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        index_path = experiment_output_dir / "global_hnsw.faiss"
        neighbor_id_path = experiment_output_dir / "neighbor_global_ids.npy"
        distance_path = experiment_output_dir / "neighbor_distances.npy"
        recall_path = experiment_output_dir / "per_query_recall.npy"

        faiss.write_index(index, str(index_path))
        np.save(neighbor_id_path, predicted_ids)
        np.save(distance_path, predicted_distances)
        np.save(recall_path, per_query_recall)
        index_size_bytes = index_path.stat().st_size

        manifest["experiments"][experiment_name] = {
            "corpus_size": int(len(vectors)),
            "query_count": int(len(query_global_ids)),
            "dimension": int(vectors.shape[1]),
            "recall_at_k": mean_recall,
            "build_seconds": build_seconds,
            "total_search_seconds": search_seconds,
            "average_query_ms": search_seconds * 1000.0 / len(query_global_ids),
            "index_size_bytes": index_size_bytes,
            "index_path": str(index_path),
            "neighbor_global_ids": str(neighbor_id_path),
            "neighbor_distances": str(distance_path),
            "per_query_recall": str(recall_path),
        }

    add_run_metadata(manifest, started_at, start_counter)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the full-rebuild HNSW baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--brute-force-dir",
        type=Path,
        default=Path("data/phase1_bruteforce_baseline"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase1_hnsw_rebuild_baseline"),
    )
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=40)
    parser.add_argument("--ef-search", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Run all full-rebuild HNSW experiments and print their metrics."""
    args = parse_args()
    manifest = run_phase1_hnsw_rebuild_baseline(
        brute_force_dir=args.brute_force_dir,
        output_dir=args.output_dir,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    print(
        f"Prepared full-rebuild HNSW baseline "
        f"(k={manifest['k']}, M={manifest['hnsw']['m']}, "
        f"efConstruction={manifest['hnsw']['ef_construction']}, "
        f"efSearch={manifest['hnsw']['ef_search']}):"
    )
    for experiment_name, experiment in manifest["experiments"].items():
        print(
            f"- {experiment_name}: recall@{manifest['k']}="
            f"{experiment['recall_at_k']:.4f}, "
            f"build={experiment['build_seconds'] * 1000.0:.3f} ms, "
            f"query={experiment['average_query_ms']:.3f} ms/query, "
            f"index={experiment['index_size_bytes']} bytes"
        )


if __name__ == "__main__":
    main()
