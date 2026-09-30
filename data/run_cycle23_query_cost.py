"""Cycle 23 formal query-cost evaluation over frozen Stage 20 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter_ns
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from data.phase2_common import load_graph
from data.phase2_graph import external_graph_search
from data.phase4_loaded_base import measure_loaded_base_incremental


def _recall(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(row) & set(expected)) / len(expected) for row, expected in zip(predicted, truth)]))


def _native(index: faiss.Index, queries: np.ndarray, truth: np.ndarray, ef: int, warmup: int, repetitions: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = faiss.downcast_index(index.index) if hasattr(index, "index") else index
    raw.hnsw.efSearch = ef
    stats = getattr(faiss.cvar, "hnsw_stats", None)
    for query in queries[:warmup]:
        index.search(query[None, :], truth.shape[1])
    records, predictions = [], None
    for repetition in range(repetitions):
        rows = []
        for query_index, query in enumerate(queries):
            if stats is not None:
                stats.reset()
            started = perf_counter_ns()
            _, ids = index.search(query[None, :], truth.shape[1])
            latency = perf_counter_ns() - started
            rows.append(ids[0])
            records.append({"repetition": repetition, "query_index": query_index, "latency_ns": latency, "ndis": None if stats is None else int(stats.ndis), "nhops": None if stats is None else int(stats.nhops)})
        if predictions is None:
            predictions = np.asarray(rows)
    return _summary(records, _recall(predictions, truth)), records


def _external(graph: dict[str, Any], vectors: np.ndarray, queries: np.ndarray, truth: np.ndarray, ef: int, warmup: int, repetitions: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node_ids = np.asarray(graph["node_global_ids"], dtype=np.int64)
    row_lookup = {int(node): row for row, node in enumerate(node_ids)}
    corpus_float64 = vectors.astype(np.float64)
    for query in queries[:warmup]:
        external_graph_search(graph, vectors, query[None, :], truth.shape[1], ef, row_lookup=row_lookup, corpus_float64=corpus_float64)
    records, predictions = [], None
    for repetition in range(repetitions):
        rows = []
        for query_index, query in enumerate(queries):
            started = perf_counter_ns()
            ids, _, work = external_graph_search(graph, vectors, query[None, :], truth.shape[1], ef, return_diagnostics=True, row_lookup=row_lookup, corpus_float64=corpus_float64)
            latency = perf_counter_ns() - started
            rows.append(ids[0])
            records.append({"repetition": repetition, "query_index": query_index, "latency_ns": latency, "distance_evaluations": int(work["distance_evaluation_count"]), "visited_nodes": int(work["visited_node_count"])})
        if predictions is None:
            predictions = np.asarray(rows)
    return _summary(records, _recall(predictions, truth)), records


def _summary(records: list[dict[str, Any]], recall: float) -> dict[str, Any]:
    latency = np.asarray([row["latency_ns"] for row in records], dtype=np.float64)
    return {"recall_at_10": recall, "median_latency_ms": float(np.median(latency) / 1e6), "p95_latency_ms": float(np.percentile(latency, 95) / 1e6), "qps": float(len(latency) / (latency.sum() / 1e9)), "mean_ndis": _mean(records, "ndis"), "mean_nhops": _mean(records, "nhops"), "mean_distance_evaluations": _mean(records, "distance_evaluations"), "mean_visited_nodes": _mean(records, "visited_nodes")}


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in records if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def run(config_path: Path, stage20_root: Path, output_root: Path, *, only_seed: int | None = None, only_condition: str | None = None) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    faiss.omp_set_num_threads(1)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in config["seeds"]:
        if only_seed is not None and seed != only_seed:
            continue
        for configured_condition in config["conditions"]:
            if only_condition is not None and configured_condition != only_condition:
                continue
            condition = "gaussian_0p100" if configured_condition == "gaussian_0.1" else configured_condition
            case = stage20_root / "phase3"
            gt_dir = case / "phase3_ground_truth" / f"seed_{seed}" / condition
            queries = np.load(gt_dir / "query_vectors.npy").astype(np.float32)
            truth = np.load(gt_dir / "neighbor_global_ids.npy")[:, :10]
            vectors = np.load(gt_dir / "union_transformed_vectors.npy").astype(np.float32)
            methods: dict[str, tuple[str, Path]] = {
                "full_rebuild": ("native", case / "phase3_full_rebuild" / f"seed_{seed}" / condition / "full_rebuild_hnsw.faiss"),
                "incremental_A_to_B": ("native", case / "phase3_incremental_insertion" / f"seed_{seed}" / condition / "A_from_B" / "incremental_hnsw.faiss"),
                "incremental_B_to_A": ("native", case / "phase3_incremental_insertion" / f"seed_{seed}" / condition / "B_from_A" / "incremental_hnsw.faiss"),
            }
            for label in ("random_direct", "bounded_n4_c2", "bounded_n8_c2", "unlimited_join1"):
                methods[label] = ("external", case / "phase3_indexed_cross_knn" / f"seed_{seed}" / condition / f"indexed_cross_knn__{label}" / "graph.json")
            for method, (backend, artifact) in methods.items():
                for ef in config["query"]["ef_search"]:
                    if backend == "native":
                        summary, raw = _native(faiss.read_index(str(artifact)), queries, truth, ef, config["query"]["warmup_queries"], config["query"]["repetitions"])
                    else:
                        summary, raw = _external(load_graph(artifact), vectors, queries, truth, ef, config["query"]["warmup_queries"], config["query"]["repetitions"])
                    raw_path = output_root / "raw" / f"seed_{seed}" / condition / method / f"ef_{ef}.json"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_path.write_text(json.dumps({"schema_version": "metricdb.query-benchmark.v1", "warmup_queries": 20, "repetitions": 5, "rows": raw}) + "\n", encoding="utf-8")
                    summaries.append({"seed": seed, "condition": condition, "method": method, "backend": backend, "ef_search": ef, **summary, "raw_artifact": str(raw_path)})

            compiled = stage20_root / "phase2_compiled_graphs" / f"seed_{seed}" / condition
            a_vectors, b_vectors = np.load(compiled / "D_A" / "transformed_vectors.npy"), np.load(compiled / "D_B" / "transformed_vectors.npy")
            a_ids, b_ids = np.load(compiled / "D_A" / "global_ids.npy"), np.load(compiled / "D_B" / "global_ids.npy")
            for direction, base_vectors, base_ids, insert_vectors, insert_ids in (("A_to_B", b_vectors, b_ids, a_vectors, a_ids), ("B_to_A", a_vectors, a_ids, b_vectors, b_ids)):
                loaded = measure_loaded_base_incremental(base_vectors, base_ids, insert_vectors, insert_ids, repetitions=5)
                path = output_root / "loaded_base" / f"seed_{seed}" / condition / f"{direction}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(loaded, indent=2) + "\n", encoding="utf-8")
    suffix = "" if only_seed is None else f"_seed{only_seed}_{only_condition.replace('.', 'p')}"
    csv_path = output_root / f"query_cost_summary{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (output_root / f"manifest{suffix}.json").write_text(json.dumps({"schema_version": "metricdb.cycle23-results.v1", "config": str(config_path), "case_count": len(summaries), "summary": str(csv_path)}, indent=2) + "\n", encoding="utf-8")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cycle23_query_cost_formal.json"))
    parser.add_argument("--stage20-root", type=Path, default=Path("data/phase3_stage20_25k"))
    parser.add_argument("--output-root", type=Path, default=Path("data/phase4_cycle23"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition", choices=["rotation", "gaussian_0.1"])
    args = parser.parse_args()
    if (args.seed is None) != (args.condition is None):
        parser.error("--seed and --condition must be provided together")
    run(args.config, args.stage20_root, args.output_root, only_seed=args.seed, only_condition=args.condition)
