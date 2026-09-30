"""Run the frozen Cycle 25 GloVe method matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.phase2_common import load_graph, save_json
from data.phase3_baselines import (
    run_bruteforce_ground_truth,
    run_full_rebuild_hnsw,
    run_incremental_insertion_baseline,
    run_indexed_cross_knn_baseline,
    union_phase2_pair,
)
from data.phase3_io import Phase2SubsetInput, validate_phase2_graph


def _condition_name(value: str) -> str:
    return "gaussian_0p100" if value == "gaussian_0.1" else value


def _derived_seed(base: int, *parts: str) -> int:
    payload = "/".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _load_subset(root: Path, seed: int, condition: str, subset: str) -> Phase2SubsetInput:
    directory = root / f"seed_{seed}" / condition / subset
    graph = load_graph(directory / "graph.json")
    validate_phase2_graph(graph, subset, "raw_level0", "no_repair")
    metadata = json.loads((directory / "condition.json").read_text(encoding="utf-8"))
    vectors = np.ascontiguousarray(np.load(directory / "transformed_vectors.npy"), dtype=np.float32)
    local_vectors = np.ascontiguousarray(np.load(directory / "local_vectors.npy"), dtype=np.float32)
    ids = np.asarray(np.load(directory / "global_ids.npy"), dtype=np.int64)
    queries = np.ascontiguousarray(np.load(directory / "held_out_queries.npy"), dtype=np.float32)
    if len(vectors) != len(local_vectors) or len(vectors) != len(ids):
        raise ValueError(f"Misaligned Cycle 25 prepared arrays: {directory}")
    np.testing.assert_array_equal(ids, np.asarray(graph["node_global_ids"], dtype=np.int64))
    return Phase2SubsetInput(
        data_seed=int(seed),
        condition=condition,
        subset=subset,
        transformed_vectors=vectors,
        global_ids=ids,
        held_out_queries=queries,
        graph=graph,
        raw_no_repair_graph=graph,
        compiled_dir=directory,
        graph_path=directory / "graph.json",
        local_vectors=local_vectors,
        local_index_path=directory / "local_hnsw.faiss",
        condition_metadata=metadata,
    )


def _recall10(metrics: dict[str, Any]) -> float:
    return float(metrics["quality"]["overall"]["recall_at_10"])


def _cross_recall10(metrics: dict[str, Any]) -> float:
    return float(metrics["quality"]["source_target"]["recall_at_10"]["cross_subset"])


def _case(seed: int, condition: str, method: str, metrics: dict[str, Any], metrics_path: Path, *, direction: str | None = None) -> dict[str, Any]:
    result = {
        "experiment_seed": int(seed),
        "condition": condition,
        "pair_key": f"{seed}/{condition}",
        "method": method,
        "recall_at_10": _recall10(metrics),
        "cross_subset_recall_at_10": _cross_recall10(metrics),
        "integration_seconds_median": float(metrics["cost"]["integration_seconds_median"]),
        "average_query_ms": float(metrics["cost"]["average_query_ms"]),
        "metrics": str(metrics_path),
    }
    if direction is not None:
        result["direction"] = direction
    return result


def run(prepared_root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepared_manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    expected_prepared = len(config["experiment_seeds"]) * len(config["conditions"]) * 2
    if prepared_manifest.get("case_count") != expected_prepared:
        raise ValueError("Cycle 25 prepared manifest is incomplete.")
    cases: list[dict[str, Any]] = []

    for experiment_seed in config["experiment_seeds"]:
        for configured_condition in config["conditions"]:
            condition = _condition_name(str(configured_condition))
            pair_root = output_root / f"seed_{experiment_seed}" / condition
            first = _load_subset(prepared_root, int(experiment_seed), condition, "D_A")
            second = _load_subset(prepared_root, int(experiment_seed), condition, "D_B")
            union = union_phase2_pair(first, second)
            truth_dir = pair_root / "ground_truth"
            truth_metrics = run_bruteforce_ground_truth(
                union,
                truth_dir,
                [int(value) for value in config["search"]["evaluation_k"]],
                int(config["search"]["brute_force_batch_size"]),
            )
            truth_ids = np.load(truth_metrics["arrays"]["neighbor_global_ids"])

            full_dir = pair_root / "full_rebuild"
            full = run_full_rebuild_hnsw(
                union,
                truth_ids,
                full_dir,
                config["search"],
                int(config["timing_repetitions"]),
            )
            cases.append(_case(int(experiment_seed), condition, "full_rebuild", full, full_dir / "metrics.json"))

            incremental_dir = pair_root / "incremental"
            incremental = run_incremental_insertion_baseline(
                first,
                second,
                union,
                truth_ids,
                incremental_dir,
                config["search"],
                int(config["timing_repetitions"]),
            )
            for method, direction in (("incremental_A_to_B", "A<-B"), ("incremental_B_to_A", "B<-A")):
                direction_metrics = incremental["directions"][direction]
                cases.append(_case(int(experiment_seed), condition, method, direction_metrics, incremental_dir / "metrics.json", direction=direction))

            for method in ("random_direct", "bounded_n8_c2", "join1"):
                method_dir = pair_root / method
                cross_config = dict(config["cross_links"])
                cross_config.pop("variants", None)
                cross_config.update(config["cross_links"]["variants"][method])
                cross_config["random_seed"] = _derived_seed(int(config["cross_links"]["random_seed"]), str(experiment_seed), condition, method)
                metrics = run_indexed_cross_knn_baseline(
                    first,
                    second,
                    union,
                    truth_ids,
                    method_dir,
                    config["search"],
                    cross_config,
                    int(config["timing_repetitions"]),
                )
                cases.append(_case(int(experiment_seed), condition, method, metrics, method_dir / "metrics.json"))

            partial = {
                "schema_version": "metricdb.cycle25-results.v1",
                "config": str(config_path),
                "prepared_manifest": str(prepared_root / "manifest.json"),
                "fixed_split_seed": int(config["split"]["seed"]),
                "case_count": len(cases),
                "cases": cases,
            }
            output_root.mkdir(parents=True, exist_ok=True)
            save_json(output_root / "manifest.partial.json", partial)

    manifest = {
        "schema_version": "metricdb.cycle25-results.v1",
        "config": str(config_path),
        "prepared_manifest": str(prepared_root / "manifest.json"),
        "fixed_split_seed": int(config["split"]["seed"]),
        "experiment_seeds": [int(value) for value in config["experiment_seeds"]],
        "pairing_unit_note": config["statistics"]["fixed_split_caveat"],
        "case_count": len(cases),
        "cases": cases,
    }
    save_json(output_root / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cycle25_glove_formal.json"))
    parser.add_argument("--prepared-root", type=Path, default=Path("data/phase4_cycle25/prepared"))
    parser.add_argument("--output-root", type=Path, default=Path("data/phase4_cycle25/formal"))
    args = parser.parse_args()
    result = run(args.prepared_root, args.output_root, args.config)
    print(json.dumps({"case_count": result["case_count"]}, indent=2))
