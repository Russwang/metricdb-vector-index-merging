"""Prepare frozen local-HNSW inputs for the Cycle 25 GloVe evaluation."""

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
from data.phase2_conditions import add_controlled_noise, affine_isometry, apply_affine, build_hnsw, inverse_affine
from data.phase2_graph import extract_raw_level0_graph, graph_with_updated_distances
from data.phase4_glove import sha256_file


def _derived_seed(base: int, *parts: str) -> int:
    payload = "/".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _condition_name(value: str) -> str:
    return "gaussian_0p100" if value == "gaussian_0.1" else value


def _load_and_validate_glove(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    root = Path(config["prepared_input"]["glove_root"])
    manifest = json.loads((root / "glove_ingest_manifest.json").read_text(encoding="utf-8"))
    vectors = np.ascontiguousarray(np.load(root / "vectors.npy"), dtype=np.float32)
    tokens = json.loads((root / "tokens.json").read_text(encoding="utf-8"))
    if vectors.shape != (int(config["dataset"]["valid_unique_rows"]), int(config["dataset"]["dimension"])):
        raise ValueError(f"Unexpected GloVe vector shape: {vectors.shape}")
    if len(tokens) != len(vectors) or len(set(tokens)) != len(tokens):
        raise ValueError("GloVe tokens are missing or non-unique.")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("GloVe vectors must be finite and non-zero.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("GloVe vectors are not row-wise L2 normalised.")
    if manifest["archive_sha256"] != config["dataset"]["archive_sha256"]:
        raise ValueError("Ingestion manifest archive SHA does not match Cycle 25 config.")

    arrays: dict[str, np.ndarray] = {}
    for name in ("D_A", "D_B", "query_A", "query_B"):
        path = root / f"{name}_indices.npy"
        values = np.asarray(np.load(path), dtype=np.int64)
        expected = int(config["split"][name] if name in {"D_A", "D_B"} else config["split"]["queries_per_side"])
        if len(values) != expected:
            raise ValueError(f"{name} has {len(values)} rows; expected {expected}.")
        recorded = manifest["split"]["artifacts"][f"{name}_indices"]["sha256"]
        if sha256_file(path) != recorded:
            raise ValueError(f"{name} checksum does not match ingestion manifest.")
        arrays[name] = values
    union = np.concatenate(list(arrays.values()))
    if len(np.unique(union)) != len(vectors) or set(union.tolist()) != set(range(len(vectors))):
        raise ValueError("Frozen GloVe split is not disjoint and exhaustive.")
    quality = {
        "vector_shape": list(vectors.shape),
        "finite": True,
        "zero_vector_count": 0,
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "split_total": int(len(union)),
        "split_unique": int(len(np.unique(union))),
        "split_exhaustive": True,
        "split_checksums_verified": True,
        "archive_sha256_verified": True,
    }
    return vectors, arrays, quality


def prepare(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vectors, split, data_quality = _load_and_validate_glove(config)
    settings = config["prepared_input"]
    search = config["search"]
    root = output_root or Path(settings["root"])
    cases: list[dict[str, Any]] = []

    for experiment_seed in config["experiment_seeds"]:
        for subset in ("D_A", "D_B"):
            suffix = subset[-1]
            corpus_ids = split[subset]
            query_ids = split[f"query_{suffix}"]
            shared_corpus = vectors[corpus_ids]
            shared_queries = vectors[query_ids]
            matrix, translation = affine_isometry(
                "rotation",
                int(config["dataset"]["dimension"]),
                np.random.default_rng(_derived_seed(int(settings["transformation_seed"]), str(experiment_seed), subset)),
            )
            local_corpus = inverse_affine(shared_corpus, matrix, translation)
            local_queries = inverse_affine(shared_queries, matrix, translation)
            exact_corpus = apply_affine(local_corpus, matrix, translation)
            exact_queries = apply_affine(local_queries, matrix, translation)
            local_index = build_hnsw(
                local_corpus,
                corpus_ids,
                int(search["m"]),
                int(search["ef_construction"]),
                int(search["ef_search"]),
            )
            base_graph = extract_raw_level0_graph(local_index, exact_corpus, subset, "original_degree")

            for condition in config["conditions"]:
                condition_name = _condition_name(str(condition))
                if condition == "rotation":
                    transformed = exact_corpus
                    queries = exact_queries
                    condition_metadata: dict[str, Any] = {"family": "exact_isometry", "noise_level": 0.0}
                elif condition == "gaussian_0.1":
                    transformed, corpus_noise = add_controlled_noise(
                        exact_corpus,
                        "gaussian",
                        0.1,
                        np.random.default_rng(_derived_seed(int(settings["noise_seed"]), str(experiment_seed), subset, "corpus")),
                    )
                    queries, query_noise = add_controlled_noise(
                        exact_queries,
                        "gaussian",
                        0.1,
                        np.random.default_rng(_derived_seed(int(settings["noise_seed"]), str(experiment_seed), subset, "queries")),
                    )
                    condition_metadata = {"family": "imperfect_alignment", "noise_level": 0.1, "corpus": corpus_noise, "queries": query_noise}
                else:
                    raise ValueError(f"Unsupported Cycle 25 condition: {condition}")
                graph = graph_with_updated_distances(base_graph, transformed, "no_repair")
                destination = root / f"seed_{experiment_seed}" / condition_name / subset
                destination.mkdir(parents=True, exist_ok=True)
                np.save(destination / "global_ids.npy", corpus_ids)
                np.save(destination / "query_global_ids.npy", query_ids)
                np.save(destination / "local_vectors.npy", local_corpus)
                np.save(destination / "local_queries.npy", local_queries)
                np.save(destination / "transformed_vectors.npy", transformed)
                np.save(destination / "held_out_queries.npy", queries)
                faiss.write_index(local_index, str(destination / "local_hnsw.faiss"))
                save_graph(destination / "graph.json", graph)
                metadata = {
                    "schema_version": "metricdb.cycle25-prepared-subset.v1",
                    "experiment_seed": int(experiment_seed),
                    "fixed_split_seed": int(config["split"]["seed"]),
                    "condition": condition_name,
                    "subset": subset,
                    "corpus_rows": int(len(corpus_ids)),
                    "held_out_query_rows": int(len(query_ids)),
                    "matrix": matrix.tolist(),
                    "translation": translation.tolist(),
                    **condition_metadata,
                }
                (destination / "condition.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                cases.append({"experiment_seed": int(experiment_seed), "condition": condition_name, "subset": subset, "directory": str(destination)})

    manifest = {
        "schema_version": "metricdb.cycle25-prepared.v1",
        "config": str(config_path),
        "fixed_split_seed": int(config["split"]["seed"]),
        "experiment_seeds": [int(value) for value in config["experiment_seeds"]],
        "case_count": len(cases),
        "corpus_size_per_seed_condition": int(config["split"]["corpus"]),
        "query_count_per_seed_condition": int(config["split"]["queries"]),
        "data_quality": data_quality,
        "cases": cases,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cycle25_glove_formal.json"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    result = prepare(args.config, args.output_root)
    print(json.dumps({"case_count": result["case_count"], "data_quality": result["data_quality"]}, indent=2))
