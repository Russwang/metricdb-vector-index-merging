"""Prepare frozen three-subset inputs for the Cycle 24 sequential evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.phase2_common import save_graph
from data.phase2_conditions import (
    add_controlled_noise,
    affine_isometry,
    apply_affine,
    build_hnsw,
    inverse_affine,
)
from data.phase2_graph import extract_raw_level0_graph, graph_with_updated_distances


def _derived_seed(base: int, *parts: str) -> int:
    payload = "/".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _load_source_subset(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.ascontiguousarray(np.load(directory / "shared_vectors.npy"), dtype=np.float32)
    mapping = json.loads((directory / "id_mapping.json").read_text(encoding="utf-8"))
    ids = np.asarray([row["global_id"] for row in mapping], dtype=np.int64)
    if len(vectors) != len(ids):
        raise ValueError(f"Source vectors and IDs are misaligned: {directory}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Source IDs are not unique: {directory}")
    return vectors, ids


def _condition_vectors(
    exact_corpus: np.ndarray,
    exact_queries: np.ndarray,
    condition: str,
    *,
    noise_seed: int,
    subset: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if condition == "rotation":
        return exact_corpus, exact_queries, {"family": "exact_isometry", "noise_level": 0.0}
    if condition != "gaussian_0.1":
        raise ValueError(f"Unsupported Cycle 24 condition: {condition}")
    corpus, corpus_metadata = add_controlled_noise(
        exact_corpus,
        "gaussian",
        0.1,
        np.random.default_rng(_derived_seed(noise_seed, subset, "corpus")),
    )
    queries, query_metadata = add_controlled_noise(
        exact_queries,
        "gaussian",
        0.1,
        np.random.default_rng(_derived_seed(noise_seed, subset, "queries")),
    )
    return corpus, queries, {
        "family": "imperfect_alignment",
        "noise_level": 0.1,
        "corpus": corpus_metadata,
        "queries": query_metadata,
    }


def prepare(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepared = config["prepared_input"]
    root = output_root or Path(config["output"]["prepared_root"])
    source_root = Path(prepared["source_root"])
    corpus_rows = int(prepared["corpus_rows_per_subset"])
    query_rows = int(prepared["held_out_queries_per_subset"])
    expected_dimension = int(prepared["dimension"])
    subsets = tuple(config["subset_sizes"])
    cases: list[dict[str, Any]] = []

    for seed in config["seeds"]:
        all_case_ids: list[np.ndarray] = []
        for subset in subsets:
            source = source_root / f"seed_{seed}" / prepared["source_subdir"] / subset
            shared, ids = _load_source_subset(source)
            if shared.shape[1] != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, got {shared.shape[1]} for {source}"
                )
            if len(shared) < corpus_rows + query_rows:
                raise ValueError(
                    f"{source} has {len(shared)} rows; {corpus_rows + query_rows} required"
                )
            corpus_shared = shared[:corpus_rows]
            corpus_ids = ids[:corpus_rows]
            query_shared = shared[corpus_rows : corpus_rows + query_rows]
            all_case_ids.append(corpus_ids)

            matrix, translation = affine_isometry(
                "rotation",
                expected_dimension,
                np.random.default_rng(
                    _derived_seed(int(prepared["transformation_seed"]), str(seed), subset)
                ),
            )
            local_corpus = inverse_affine(corpus_shared, matrix, translation)
            exact_corpus = apply_affine(local_corpus, matrix, translation)
            local_queries = inverse_affine(query_shared, matrix, translation)
            exact_queries = apply_affine(local_queries, matrix, translation)
            local_index = build_hnsw(
                local_corpus,
                corpus_ids,
                int(prepared["m"]),
                int(prepared["ef_construction"]),
                int(prepared["ef_search"]),
            )
            base_graph = extract_raw_level0_graph(
                local_index, exact_corpus, subset, "original_degree"
            )

            for condition in config["conditions"]:
                condition_name = "gaussian_0p100" if condition == "gaussian_0.1" else condition
                corpus, queries, condition_metadata = _condition_vectors(
                    exact_corpus,
                    exact_queries,
                    condition,
                    noise_seed=_derived_seed(int(prepared["noise_seed"]), str(seed)),
                    subset=subset,
                )
                graph = graph_with_updated_distances(base_graph, corpus, "no_repair")
                destination = root / f"seed_{seed}" / condition_name / subset
                destination.mkdir(parents=True, exist_ok=True)
                np.save(destination / "global_ids.npy", corpus_ids)
                np.save(destination / "local_vectors.npy", local_corpus)
                np.save(destination / "transformed_vectors.npy", corpus)
                np.save(destination / "held_out_queries.npy", queries)
                faiss.write_index(local_index, str(destination / "local_hnsw.faiss"))
                save_graph(destination / "graph.json", graph)
                metadata = {
                    "schema_version": "metricdb.cycle24-prepared-subset.v1",
                    "seed": int(seed),
                    "condition": condition_name,
                    "subset": subset,
                    "source": str(source),
                    "corpus_rows": corpus_rows,
                    "held_out_query_rows": query_rows,
                    "matrix": matrix.tolist(),
                    "translation": translation.tolist(),
                    **condition_metadata,
                }
                (destination / "condition.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                cases.append(
                    {
                        "seed": int(seed),
                        "condition": condition_name,
                        "subset": subset,
                        "directory": str(destination),
                        "corpus_rows": corpus_rows,
                        "held_out_query_rows": query_rows,
                    }
                )

        union_ids = np.concatenate(all_case_ids)
        if len(np.unique(union_ids)) != len(union_ids):
            raise ValueError(f"Cycle 24 seed {seed} contains duplicate global IDs across subsets")

    manifest = {
        "schema_version": "metricdb.cycle24-prepared.v1",
        "config": str(config_path),
        "source_root": str(source_root),
        "case_count": len(cases),
        "corpus_size_per_seed_condition": corpus_rows * len(subsets),
        "cases": cases,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cycle24_sequential_formal.json"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    result = prepare(args.config, args.output_root)
    print(json.dumps({"case_count": result["case_count"], "corpus_size": result["corpus_size_per_seed_condition"]}, indent=2))
