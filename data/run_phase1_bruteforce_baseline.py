"""Run exact brute-force kNN baselines for Phase 1 transformed vectors.

Each transformation experiment has three subsets in recovered shared space.
This script unions those transformed vectors and computes exact squared-L2 kNN
results. The output is the ground truth used later for recall@k evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase1_common import add_run_metadata, start_run
from data.prepare_phase1_transformations import EXPERIMENT_NAMES, SUBSET_NAMES


def _load_json(path: Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_union_corpus(
    transformation_dir: Path,
    experiment_name: str,
    subset_names: tuple[str, ...] = SUBSET_NAMES,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load all transformed vectors and row metadata for one experiment."""
    vectors_by_subset = []
    global_ids_by_subset = []
    union_records: list[dict[str, Any]] = []
    expected_dimension: int | None = None
    union_row_id = 0

    for subset_name in subset_names:
        subset_dir = transformation_dir / experiment_name / subset_name
        vector_path = subset_dir / "transformed_vectors.npy"
        mapping_path = subset_dir / "id_mapping.json"
        vectors = np.load(vector_path)
        id_mapping = _load_json(mapping_path)

        if vectors.ndim != 2:
            raise ValueError(f"{vector_path} must be a 2D matrix, got {vectors.shape}.")
        if len(vectors) != len(id_mapping):
            raise ValueError(
                f"{subset_name} has {len(vectors)} vectors but "
                f"{len(id_mapping)} ID records."
            )
        if expected_dimension is None:
            expected_dimension = int(vectors.shape[1])
        elif vectors.shape[1] != expected_dimension:
            raise ValueError(
                f"{subset_name} dimension {vectors.shape[1]} does not match "
                f"{expected_dimension}."
            )

        global_ids = np.array(
            [record["global_id"] for record in id_mapping], dtype=np.int64
        )
        vectors_by_subset.append(np.ascontiguousarray(vectors, dtype=np.float32))
        global_ids_by_subset.append(global_ids)

        for local_row_id, record in enumerate(id_mapping):
            union_records.append(
                {
                    "union_row_id": union_row_id,
                    "experiment": experiment_name,
                    "subset_id": subset_name,
                    "local_id": int(record["local_id"]),
                    "global_id": int(record["global_id"]),
                    "item_id": record.get("item_id"),
                }
            )
            union_row_id += 1

    corpus_vectors = np.ascontiguousarray(np.vstack(vectors_by_subset), dtype=np.float32)
    corpus_global_ids = np.concatenate(global_ids_by_subset)
    unique_global_ids = np.unique(corpus_global_ids)
    if len(unique_global_ids) != len(corpus_global_ids):
        raise ValueError(f"{experiment_name} contains duplicate global IDs.")

    return corpus_vectors, corpus_global_ids, union_records


def exact_knn_squared_l2(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_global_ids: np.ndarray,
    k: int,
    exclude_query_global_ids: np.ndarray | None = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact squared-L2 nearest neighbors with deterministic tie breaks."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if query_vectors.ndim != 2 or corpus_vectors.ndim != 2:
        raise ValueError("query_vectors and corpus_vectors must both be 2D.")
    if query_vectors.shape[1] != corpus_vectors.shape[1]:
        raise ValueError(
            "query and corpus dimensions differ: "
            f"{query_vectors.shape[1]} versus {corpus_vectors.shape[1]}."
        )
    if len(corpus_vectors) != len(corpus_global_ids):
        raise ValueError("corpus_vectors and corpus_global_ids must have same length.")

    max_neighbors = len(corpus_vectors)
    if exclude_query_global_ids is not None:
        if len(exclude_query_global_ids) != len(query_vectors):
            raise ValueError("exclude_query_global_ids must match query count.")
        max_neighbors -= 1
    if k > max_neighbors:
        raise ValueError(f"k={k} exceeds available neighbors ({max_neighbors}).")

    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    corpus = np.ascontiguousarray(corpus_vectors, dtype=np.float32)
    corpus_ids = np.asarray(corpus_global_ids, dtype=np.int64)
    query_norms = np.sum(queries.astype(np.float64) ** 2, axis=1)
    corpus_norms = np.sum(corpus.astype(np.float64) ** 2, axis=1)

    neighbor_ids = np.empty((len(queries), k), dtype=np.int64)
    neighbor_distances = np.empty((len(queries), k), dtype=np.float32)

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch = queries[start:end]
        distances = (
            query_norms[start:end, None]
            + corpus_norms[None, :]
            - 2.0 * (batch.astype(np.float64) @ corpus.astype(np.float64).T)
        )
        distances = np.maximum(distances, 0.0)

        if exclude_query_global_ids is not None:
            excluded_ids = exclude_query_global_ids[start:end]
            distances[excluded_ids[:, None] == corpus_ids[None, :]] = np.inf

        candidate_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        for row_offset, candidates in enumerate(candidate_indices):
            candidate_distances = distances[row_offset, candidates]
            candidate_ids = corpus_ids[candidates]
            order = np.lexsort((candidate_ids, candidate_distances))
            selected = candidates[order[:k]]
            neighbor_ids[start + row_offset] = corpus_ids[selected]
            neighbor_distances[start + row_offset] = distances[row_offset, selected]

    return neighbor_ids, neighbor_distances


def run_phase1_bruteforce_baseline(
    transformation_dir: Path,
    output_dir: Path,
    k: int = 10,
    batch_size: int = 128,
    exclude_self: bool = True,
) -> dict[str, Any]:
    """Compute and persist exact union-level kNN for every transformation."""
    started_at, start_counter = start_run()
    transformation_manifest_path = transformation_dir / "manifest.json"
    transformation_manifest = _load_json(transformation_manifest_path)
    experiment_names = tuple(transformation_manifest["experiments"].keys())
    subset_names = tuple(transformation_manifest.get("subset_names", SUBSET_NAMES))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "baseline": "brute_force_exact_knn",
        "source_transformations": str(transformation_dir),
        "k": k,
        "batch_size": batch_size,
        "distance_metric": "squared_l2",
        "query_set": "union_transformed_vectors",
        "corpus": "union_transformed_vectors",
        "exclude_self": exclude_self,
        "subset_names": list(subset_names),
        "experiment_names": list(experiment_names),
        "experiments": {},
    }

    for experiment_name in experiment_names:
        experiment_output_dir = output_dir / experiment_name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        corpus_vectors, corpus_global_ids, union_records = _load_union_corpus(
            transformation_dir, experiment_name, subset_names=subset_names
        )
        excluded_ids = corpus_global_ids if exclude_self else None

        start_time = perf_counter()
        neighbor_ids, neighbor_distances = exact_knn_squared_l2(
            query_vectors=corpus_vectors,
            corpus_vectors=corpus_vectors,
            corpus_global_ids=corpus_global_ids,
            k=k,
            exclude_query_global_ids=excluded_ids,
            batch_size=batch_size,
        )
        elapsed_seconds = perf_counter() - start_time

        union_vector_path = experiment_output_dir / "union_transformed_vectors.npy"
        union_id_path = experiment_output_dir / "union_global_ids.npy"
        record_path = experiment_output_dir / "union_records.json"
        query_id_path = experiment_output_dir / "query_global_ids.npy"
        neighbor_id_path = experiment_output_dir / "neighbor_global_ids.npy"
        distance_path = experiment_output_dir / "neighbor_distances.npy"

        np.save(union_vector_path, corpus_vectors)
        np.save(union_id_path, corpus_global_ids)
        with record_path.open("w", encoding="utf-8") as file:
            json.dump(union_records, file, indent=2)
        np.save(query_id_path, corpus_global_ids)
        np.save(neighbor_id_path, neighbor_ids)
        np.save(distance_path, neighbor_distances)

        manifest["experiments"][experiment_name] = {
            "corpus_size": int(len(corpus_vectors)),
            "query_count": int(len(corpus_vectors)),
            "dimension": int(corpus_vectors.shape[1]),
            "elapsed_seconds": elapsed_seconds,
            "average_query_ms": elapsed_seconds * 1000.0 / len(corpus_vectors),
            "union_transformed_vectors": str(union_vector_path),
            "union_global_ids": str(union_id_path),
            "union_records": str(record_path),
            "query_global_ids": str(query_id_path),
            "neighbor_global_ids": str(neighbor_id_path),
            "neighbor_distances": str(distance_path),
        }

    add_run_metadata(manifest, started_at, start_counter)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line options for exact Phase 1 kNN baseline generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transformation-dir",
        type=Path,
        default=Path("data/phase1_transformations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase1_bruteforce_baseline"),
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Keep each query vector as a possible nearest neighbor.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the brute-force baseline and print a concise timing summary."""
    args = parse_args()
    manifest = run_phase1_bruteforce_baseline(
        transformation_dir=args.transformation_dir,
        output_dir=args.output_dir,
        k=args.k,
        batch_size=args.batch_size,
        exclude_self=not args.include_self,
    )

    print(
        f"Prepared brute-force exact kNN baseline "
        f"(k={manifest['k']}, exclude_self={manifest['exclude_self']}):"
    )
    for experiment_name, experiment in manifest["experiments"].items():
        print(
            f"- {experiment_name}: {experiment['query_count']} queries over "
            f"{experiment['corpus_size']} vectors, "
            f"{experiment['average_query_ms']:.3f} ms/query"
        )


if __name__ == "__main__":
    main()
