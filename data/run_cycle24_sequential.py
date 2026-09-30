"""Run the frozen three-subset Cycle 24 evaluation for every arrival order."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.phase2_common import load_graph, save_graph, save_json
from data.phase2_conditions import build_hnsw
from data.phase2_graph import external_graph_search
from data.phase3_baselines import run_full_rebuild_hnsw
from data.phase3_io import Phase2SubsetInput
from data.phase3_merge import add_indexed_cross_knn_links, no_cross_union_graph
from data.phase3_metrics import (
    graph_statistics,
    recall_at_k,
    source_target_recall_at_k,
    subset_lookup,
)
from data.run_phase1_bruteforce_baseline import exact_knn_squared_l2


def _condition_name(value: str) -> str:
    return "gaussian_0p100" if value == "gaussian_0.1" else value


def _load_input(prepared_root: Path, seed: int, condition: str, subset: str) -> Phase2SubsetInput:
    directory = prepared_root / f"seed_{seed}" / condition / subset
    graph = load_graph(directory / "graph.json")
    return Phase2SubsetInput(
        data_seed=int(seed),
        condition=condition,
        subset=subset,
        transformed_vectors=np.load(directory / "transformed_vectors.npy").astype(np.float32),
        local_vectors=np.load(directory / "local_vectors.npy").astype(np.float32),
        global_ids=np.load(directory / "global_ids.npy").astype(np.int64),
        held_out_queries=np.load(directory / "held_out_queries.npy").astype(np.float32),
        graph=graph,
        raw_no_repair_graph=graph,
        compiled_dir=directory,
        graph_path=directory / "graph.json",
        local_index_path=directory / "local_hnsw.faiss",
        condition_metadata=json.loads((directory / "condition.json").read_text(encoding="utf-8")),
    )


def _ordered_union(inputs: dict[str, Phase2SubsetInput], order: list[str] | tuple[str, ...]) -> dict[str, Any]:
    ids = np.concatenate([inputs[name].global_ids for name in order])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Cycle 24 union contains duplicate global IDs")
    return {
        "vectors": np.vstack([inputs[name].transformed_vectors for name in order]).astype(np.float32),
        "global_ids": ids,
        "queries": np.vstack([inputs[name].held_out_queries for name in sorted(inputs)]).astype(np.float32),
        "query_subsets": [name for name in sorted(inputs) for _ in range(len(inputs[name].held_out_queries))],
        "subset_to_ids": {name: value.global_ids for name, value in inputs.items()},
    }


def _concatenate_graphs(current: dict[str, Any], incoming: dict[str, Any], incoming_subset: str) -> dict[str, Any]:
    old_ids = np.asarray(current["node_global_ids"], dtype=np.int64)
    new_ids = np.asarray(incoming["node_global_ids"], dtype=np.int64)
    if set(old_ids.tolist()) & set(new_ids.tolist()):
        raise ValueError("Sequential graph inputs contain duplicate global IDs")
    neighbors = {int(node): [int(value) for value in values] for node, values in current["neighbors"].items()}
    neighbors.update({int(node): [int(value) for value in values] for node, values in incoming["neighbors"].items()})
    distances = {int(node): [float(value) for value in values] for node, values in current["edge_distances"].items()}
    distances.update({int(node): [float(value) for value in values] for node, values in incoming["edge_distances"].items()})
    subset_by_node = {int(node): str(value) for node, value in current.get("source_subset_by_node", {}).items()}
    if not subset_by_node:
        subset_by_node = {int(node): str(current.get("source_subset", "unknown")) for node in old_ids}
    subset_by_node.update({int(node): incoming_subset for node in new_ids})
    return {
        **current,
        "node_global_ids": np.concatenate([old_ids, new_ids]),
        "neighbors": neighbors,
        "edge_distances": distances,
        "source_subset": "merged",
        "source_subset_by_node": subset_by_node,
        "graph_kind": "sequential_indexed_cross_knn",
        "variant": "sequential",
        "node_levels": np.concatenate([np.asarray(current["node_levels"]), np.asarray(incoming["node_levels"])]),
        "original_degrees": np.concatenate([np.asarray(current["original_degrees"]), np.asarray(incoming["original_degrees"])]),
        "degree_budget": np.concatenate([np.asarray(current["degree_budget"]), np.asarray(incoming["degree_budget"])]),
    }


def _overlay_pair_cross_edges(target: dict[str, Any], pair: dict[str, Any]) -> int:
    pair_subset = {int(node): str(value) for node, value in pair["source_subset_by_node"].items()}
    added = 0
    for source, targets in pair["neighbors"].items():
        values = pair["edge_distances"][source]
        for index, destination in enumerate(targets):
            if pair_subset[int(source)] == pair_subset[int(destination)]:
                continue
            destination = int(destination)
            if destination in target["neighbors"][int(source)]:
                continue
            target["neighbors"][int(source)].append(destination)
            target["edge_distances"][int(source)].append(float(values[index]))
            added += 1
    return added


def _cross_config(method: str, config: dict[str, Any], random_seed: int) -> dict[str, Any]:
    settings = config["second_merge"]
    common = {
        "bridge_budget": {"mode": "fixed", "count": int(settings["portal_budget"])},
        "candidate_k": 64,
        "candidate_ef_search": 128,
        "cross_degree_cap": int(settings["cross_k"]),
        "random_seed": int(random_seed),
    }
    if method == "random_direct":
        return {**common, "bridge_selection": "random", "propagation_hops": 0}
    if method == "bounded_n8_c2":
        return {
            **common,
            "bridge_selection": "high_degree",
            "propagation_hops": 1,
            "propagation_neighbor_budget": 8,
            "propagation_candidate_cap": 2,
        }
    if method == "join1":
        return {**common, "bridge_selection": "high_degree", "propagation_hops": 1}
    raise ValueError(f"Unsupported external Cycle 24 method: {method}")


def _build_sequential_graph(
    inputs: dict[str, Phase2SubsetInput], order: list[str], method: str, config: dict[str, Any], random_seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = inputs[order[0]].graph
    integrated = [order[0]]
    steps: list[dict[str, Any]] = []
    for step_index, incoming_name in enumerate(order[1:], start=1):
        before_edges = {(int(s), int(t)) for s, values in current["neighbors"].items() for t in values}
        started = perf_counter()
        current = _concatenate_graphs(current, inputs[incoming_name].graph, incoming_name)
        candidate_count = 0
        pair_added = 0
        pair_details = []
        for prior_index, prior_name in enumerate(integrated):
            prior = replace(inputs[prior_name], graph=inputs[prior_name].graph)
            incoming = replace(inputs[incoming_name], graph=inputs[incoming_name].graph)
            pair, base = no_cross_union_graph(prior, incoming)
            variant = _cross_config(method, config, random_seed + step_index * 101 + prior_index)
            pair, diagnostics = add_indexed_cross_knn_links(
                pair,
                base,
                prior,
                incoming,
                cross_k=int(config["second_merge"]["cross_k"]),
                random_seed=int(variant["random_seed"]),
                cross_link_config=variant,
                bidirectional=True,
            )
            pair_added += _overlay_pair_cross_edges(current, pair)
            candidate_count += int(diagnostics["cross_candidate_distance_count"])
            pair_details.append(
                {
                    "prior_subset": prior_name,
                    "incoming_subset": incoming_name,
                    "candidate_distance_count": int(diagnostics["cross_candidate_distance_count"]),
                    "added_cross_link_count": int(diagnostics["added_cross_link_count"]),
                }
            )
        after_edges = {(int(s), int(t)) for s, values in current["neighbors"].items() for t in values}
        if not before_edges.issubset(after_edges):
            raise AssertionError("Sequential indexed integration removed a prior edge")
        statistics = graph_statistics(current)
        steps.append(
            {
                "step": step_index,
                "incoming_subset": incoming_name,
                "integration_seconds": float(perf_counter() - started),
                "candidate_distance_count": candidate_count,
                "added_cross_link_count": pair_added,
                "prior_edges_preserved": True,
                "node_count": statistics["node_count"],
                "weak_component_count": statistics["weak_connected_component_count"],
                "directed_reachable_count": statistics["directed_reachable_from_entry_count"],
                "pair_details": pair_details,
            }
        )
        integrated.append(incoming_name)
    return current, steps


def _evaluate_external(
    graph: dict[str, Any], union: dict[str, Any], truth: np.ndarray, evaluation_k: list[int], ef_search: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    maximum = min(max(evaluation_k), len(union["global_ids"]))
    started = perf_counter()
    predicted, distances, work = external_graph_search(
        graph,
        union["vectors"],
        union["queries"],
        maximum,
        ef_search,
        return_diagnostics=True,
    )
    search_seconds = perf_counter() - started
    overall, _ = recall_at_k(predicted, truth[:, :maximum], evaluation_k)
    split = source_target_recall_at_k(
        predicted,
        truth[:, :maximum],
        union["query_subsets"],
        subset_lookup(union["subset_to_ids"]),
        evaluation_k,
    )
    return {
        "overall": overall,
        "source_target": split,
        "search_seconds": search_seconds,
        "average_query_ms": search_seconds * 1000.0 / len(union["queries"]),
        "distance_evaluation_count": int(work["distance_evaluation_count"]),
        "visited_node_count": int(work["visited_node_count"]),
    }, predicted, distances


def _run_external_method(
    inputs: dict[str, Phase2SubsetInput],
    order: list[str],
    method: str,
    config: dict[str, Any],
    union: dict[str, Any],
    truth: np.ndarray,
    output_dir: Path,
    random_seed: int,
) -> dict[str, Any]:
    repetitions = int(config["integration_repetitions"])
    samples = []
    graph = None
    steps = None
    for repetition in range(repetitions):
        started = perf_counter()
        graph, steps = _build_sequential_graph(inputs, order, method, config, random_seed)
        samples.append(float(perf_counter() - started))
    assert graph is not None and steps is not None
    quality, predicted, distances = _evaluate_external(
        graph, union, truth, [10], int(config["prepared_input"]["ef_search"])
    )
    statistics = graph_statistics(graph)
    if statistics["node_count"] != int(config["corpus_size"]):
        raise ValueError(f"{method} produced {statistics['node_count']} nodes")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "graph.json"
    save_graph(graph_path, graph)
    np.save(output_dir / "neighbor_global_ids.npy", predicted)
    np.save(output_dir / "neighbor_distances.npy", distances)
    metrics = {
        "schema_version": "metricdb.cycle24-method.v1",
        "method": method,
        "backend": "external_level0_graph",
        "order": order,
        "quality": quality,
        "cost": {
            "integration_seconds_samples": samples,
            "integration_seconds_median": float(np.median(samples)),
            "integration_seconds_p95": float(np.percentile(samples, 95)),
            "first_merge": steps[0],
            "second_merge": steps[1],
        },
        "graph_statistics": statistics,
        "validity": {
            "unique_global_ids": len(np.unique(graph["node_global_ids"])) == len(graph["node_global_ids"]),
            "prior_edges_preserved": all(step["prior_edges_preserved"] for step in steps),
            "weakly_connected": statistics["weak_connected_component_count"] == 1,
            "directed_reachable_ratio": statistics["directed_reachable_from_entry_ratio"],
        },
        "graph_path": str(graph_path),
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def _run_incremental(
    inputs: dict[str, Phase2SubsetInput], order: list[str], config: dict[str, Any], union: dict[str, Any], truth: np.ndarray, output_dir: Path
) -> dict[str, Any]:
    settings = config["prepared_input"]
    samples = []
    step_samples: list[list[float]] = [[], []]
    index = None
    for _ in range(int(config["integration_repetitions"])):
        started = perf_counter()
        index = build_hnsw(
            inputs[order[0]].transformed_vectors,
            inputs[order[0]].global_ids,
            int(settings["m"]),
            int(settings["ef_construction"]),
            int(settings["ef_search"]),
        )
        for step, name in enumerate(order[1:]):
            step_started = perf_counter()
            index.add_with_ids(inputs[name].transformed_vectors, inputs[name].global_ids)
            step_samples[step].append(float(perf_counter() - step_started))
        samples.append(float(perf_counter() - started))
    assert index is not None
    maximum = 10
    distances, predicted = index.search(union["queries"], maximum)
    overall, _ = recall_at_k(predicted, truth[:, :maximum], [10])
    split = source_target_recall_at_k(
        predicted, truth[:, :maximum], union["query_subsets"], subset_lookup(union["subset_to_ids"]), [10]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "incremental_hnsw.faiss"
    faiss.write_index(index, str(path))
    np.save(output_dir / "neighbor_global_ids.npy", predicted)
    metrics = {
        "schema_version": "metricdb.cycle24-method.v1",
        "method": "sequential_incremental",
        "backend": "native_faiss_hnsw",
        "order": order,
        "quality": {"overall": overall, "source_target": split},
        "cost": {
            "integration_seconds_samples": samples,
            "integration_seconds_median": float(np.median(samples)),
            "first_add_seconds_samples": step_samples[0],
            "second_add_seconds_samples": step_samples[1],
            "second_add_seconds_median": float(np.median(step_samples[1])),
        },
        "validity": {"unique_global_ids": len(faiss.vector_to_array(index.id_map)) == len(np.unique(faiss.vector_to_array(index.id_map)))},
        "index_path": str(path),
    }
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def _order_effects(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault((case["seed"], case["condition"], case["method"]), []).append(case)
    result = []
    for (seed, condition, method), rows in sorted(grouped.items()):
        recalls = [float(row["recall_at_10"]) for row in rows]
        costs = [float(row["integration_seconds_median"]) for row in rows]
        result.append(
            {
                "seed": seed,
                "condition": condition,
                "method": method,
                "order_count": len(rows),
                "recall_min": min(recalls),
                "recall_max": max(recalls),
                "recall_range": max(recalls) - min(recalls),
                "integration_seconds_min": min(costs),
                "integration_seconds_max": max(costs),
            }
        )
    return result


def run(prepared_root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    faiss.omp_set_num_threads(1)
    cases: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        for configured_condition in config["conditions"]:
            condition = _condition_name(configured_condition)
            inputs = {name: _load_input(prepared_root, seed, condition, name) for name in config["subset_sizes"]}
            canonical = _ordered_union(inputs, sorted(inputs))
            truth, distances = exact_knn_squared_l2(
                canonical["queries"], canonical["vectors"], canonical["global_ids"], 10, None, 256
            )
            truth_dir = output_root / f"seed_{seed}" / condition / "ground_truth"
            truth_dir.mkdir(parents=True, exist_ok=True)
            np.save(truth_dir / "neighbor_global_ids.npy", truth)
            np.save(truth_dir / "neighbor_distances.npy", distances)
            for configured_order in config["orders"]:
                order = [str(value) for value in configured_order]
                union = _ordered_union(inputs, order)
                order_dir = output_root / f"seed_{seed}" / condition / "_".join(order)
                for method in config["methods"]:
                    destination = order_dir / method
                    if method == "full_rebuild":
                        metrics = run_full_rebuild_hnsw(
                            union,
                            truth,
                            destination,
                            {
                                "m": config["prepared_input"]["m"],
                                "ef_construction": config["prepared_input"]["ef_construction"],
                                "ef_search": config["prepared_input"]["ef_search"],
                                "evaluation_k": [10],
                            },
                            integration_repetitions=int(config["integration_repetitions"]),
                        )
                    elif method == "sequential_incremental":
                        metrics = _run_incremental(inputs, order, config, union, truth, destination)
                    else:
                        metrics = _run_external_method(
                            inputs, order, method, config, union, truth, destination, random_seed=442 + int(seed)
                        )
                    recall = float(metrics["quality"]["overall"]["recall_at_10"])
                    integration = float(metrics["cost"]["integration_seconds_median"])
                    cases.append(
                        {
                            "seed": int(seed),
                            "condition": condition,
                            "order": order,
                            "method": method,
                            "recall_at_10": recall,
                            "cross_subset_recall_at_10": float(metrics["quality"]["source_target"]["recall_at_10"]["cross_subset"]),
                            "integration_seconds_median": integration,
                            "metrics": str(destination / "metrics.json"),
                        }
                    )
    manifest = {
        "schema_version": "metricdb.cycle24-results.v2",
        "config": str(config_path),
        "prepared_manifest": str(prepared_root / "manifest.json"),
        "case_count": len(cases),
        "expected_case_count": len(config["seeds"]) * len(config["conditions"]) * len(config["orders"]) * len(config["methods"]),
        "cases": cases,
        "order_effects": _order_effects(cases),
    }
    if manifest["case_count"] != manifest["expected_case_count"]:
        raise AssertionError("Cycle 24 method matrix is incomplete")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, default=Path("data/phase4_cycle24_prepared"))
    parser.add_argument("--output-root", type=Path, default=Path("data/phase4_cycle24"))
    parser.add_argument("--config", type=Path, default=Path("configs/cycle24_sequential_formal.json"))
    args = parser.parse_args()
    result = run(args.prepared_root, args.output_root, args.config)
    print(json.dumps({"case_count": result["case_count"], "order_effect_count": len(result["order_effects"])}, indent=2))
