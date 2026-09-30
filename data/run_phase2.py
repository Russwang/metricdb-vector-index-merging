"""Run the complete Phase 2 transformation compilation and repair pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase1_common import add_run_metadata, start_run
from data.phase2_common import load_phase2_config, save_graph, save_json
from data.phase2_conditions import (
    add_controlled_noise,
    affine_isometry,
    apply_affine,
    build_hnsw,
    generate_held_out_shared_queries,
    inverse_affine,
    load_subset_inputs,
    transformation_metadata,
)
from data.phase2_graph import (
    extract_query_derived_graph,
    extract_raw_level0_graph,
    external_graph_search,
    graph_diagnostics,
    graph_with_updated_distances,
    repair_graph,
    structural_oracle_graph,
)
from data.run_phase1 import run_complete_phase1
from data.run_phase1_bruteforce_baseline import exact_knn_squared_l2
from data.run_phase1_hnsw_rebuild_baseline import recall_at_k


OUTPUT_DIRECTORIES = (
    "phase2_compiled_graphs",
    "phase2_repaired_graphs",
    "phase2_results",
    "phase2_phase1_seeds",
)


def _seed(base: int, *parts: str) -> int:
    value = int(base)
    for part in parts:
        for character in part:
            value = (value * 131 + ord(character)) % (2**32 - 1)
    return value


def _condition_name(family: str, level: float) -> str:
    return f"{family}_{level:.3f}".replace(".", "p")


def _exact_condition(
    name: str,
    shared: np.ndarray,
    query_shared: np.ndarray,
    global_ids: np.ndarray,
    subset_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    rng = np.random.default_rng(
        _seed(int(config["seeds"]["transformation"]), name, subset_name)
    )
    matrix, translation = affine_isometry(name, shared.shape[1], rng)
    local = inverse_affine(shared, matrix, translation)
    transformed = apply_affine(local, matrix, translation)
    query_local = inverse_affine(query_shared, matrix, translation)
    query_transformed = apply_affine(query_local, matrix, translation)
    search = config["search"]
    index = build_hnsw(
        local,
        global_ids,
        int(search["m"]),
        int(search["ef_construction"]),
        int(search["ef_search"]),
    )
    return {
        "name": name,
        "family": "exact_isometry",
        "local_vectors": local,
        "transformed_vectors": transformed,
        "held_out_queries": query_transformed,
        "local_index": index,
        "metadata": transformation_metadata(name, matrix, translation),
        "cost": {
            "vector_transformation_seconds": perf_counter() - started,
            "processed_node_ratio": 1.0,
        },
    }


def _general_linear_condition(
    phase1_dir: Path,
    subset_name: str,
    query_shared: np.ndarray,
) -> dict[str, Any]:
    started = perf_counter()
    directory = phase1_dir / "general_linear" / subset_name
    parameters = np.load(directory / "transformation.npz")
    matrix = np.asarray(parameters["matrix"], dtype=np.float64)
    local = np.ascontiguousarray(
        np.load(directory / "local_vectors.npy"), dtype=np.float32
    )
    transformed = np.ascontiguousarray(
        np.load(directory / "transformed_vectors.npy"), dtype=np.float32
    )
    return {
        "name": "general_linear",
        "family": "stress_test",
        "local_vectors": local,
        "transformed_vectors": transformed,
        "held_out_queries": query_shared.copy(),
        "local_index": faiss.read_index(str(directory / "local_hnsw.faiss")),
        "metadata": {
            "name": "general_linear",
            "condition_number": float(np.linalg.cond(matrix)),
            "stress_test_only": True,
        },
        "cost": {
            "vector_transformation_seconds": perf_counter() - started,
            "processed_node_ratio": 1.0,
        },
    }


def _noise_conditions(
    rotation: dict[str, Any],
    subset_name: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for family in config["noise"]["families"]:
        for level_value in config["noise"]["levels"]:
            started = perf_counter()
            level = float(level_value)
            name = _condition_name(family, level)
            corpus, metadata = add_controlled_noise(
                rotation["transformed_vectors"],
                family,
                level,
                np.random.default_rng(
                    _seed(int(config["seeds"]["noise"]), name, subset_name, "corpus")
                ),
            )
            queries, query_metadata = add_controlled_noise(
                rotation["held_out_queries"],
                family,
                level,
                np.random.default_rng(
                    _seed(int(config["seeds"]["noise"]), name, subset_name, "query")
                ),
            )
            result.append(
                {
                    "name": name,
                    "family": "imperfect_alignment",
                    "local_vectors": rotation["local_vectors"],
                    "transformed_vectors": corpus,
                    "held_out_queries": queries,
                    "local_index": rotation["local_index"],
                    "metadata": {
                        "base": "rotation",
                        "noise_family": family,
                        **metadata,
                        "query_measured_relative_perturbation": query_metadata[
                            "measured_relative_perturbation"
                        ],
                    },
                    "cost": {
                        "vector_transformation_seconds": perf_counter() - started,
                        "processed_node_ratio": 1.0,
                    },
                }
            )
    return result


def _recall_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    requested_k: list[int],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    means: dict[str, float] = {}
    per_query: dict[str, np.ndarray] = {}
    available = min(predicted.shape[1], truth.shape[1])
    for requested in requested_k:
        k = min(int(requested), available)
        mean, values = recall_at_k(predicted[:, :k], truth[:, :k])
        means[f"recall_at_{requested}"] = mean
        means[f"effective_k_{requested}"] = k
        per_query[str(requested)] = values
    return means, per_query


def _evaluate_external(
    graph: dict[str, Any],
    corpus: np.ndarray,
    global_ids: np.ndarray,
    held_out: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    requested = [int(k) for k in config["search"]["evaluation_k"]]
    corpus_query_count = min(
        int(config["queries"].get("corpus_point_count_per_subset", len(corpus))),
        len(corpus),
    )
    corpus_queries = corpus[:corpus_query_count]
    corpus_query_ids = global_ids[:corpus_query_count]
    max_corpus_k = min(max(requested), len(corpus) - 1)
    max_held_k = min(max(requested), len(corpus))
    batch_size = int(config["search"]["brute_force_batch_size"])
    corpus_truth, _ = exact_knn_squared_l2(
        corpus_queries,
        corpus,
        global_ids,
        max_corpus_k,
        corpus_query_ids,
        batch_size,
    )
    held_truth, _ = exact_knn_squared_l2(
        held_out, corpus, global_ids, max_held_k, None, batch_size
    )
    corpus_predicted, _ = external_graph_search(
        graph,
        corpus,
        corpus_queries,
        max_corpus_k,
        int(config["search"]["ef_search"]),
        corpus_query_ids,
        batch_size,
    )
    held_predicted, _ = external_graph_search(
        graph,
        corpus,
        held_out,
        max_held_k,
        int(config["search"]["ef_search"]),
        distance_batch_size=batch_size,
    )
    corpus_metrics, corpus_per_query = _recall_metrics(
        corpus_predicted, corpus_truth, requested
    )
    held_metrics, held_per_query = _recall_metrics(
        held_predicted, held_truth, requested
    )
    arrays = {
        "corpus_predicted": corpus_predicted,
        "corpus_truth": corpus_truth,
        "held_out_predicted": held_predicted,
        "held_out_truth": held_truth,
    }
    for k, values in corpus_per_query.items():
        arrays[f"corpus_recall_{k}"] = values
    for k, values in held_per_query.items():
        arrays[f"held_out_recall_{k}"] = values
    return {
        "corpus_point": corpus_metrics,
        "held_out": held_metrics,
    }, arrays


def _native_rebuild_metrics(
    index: faiss.Index,
    corpus: np.ndarray,
    global_ids: np.ndarray,
    held_out: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    requested = [int(k) for k in config["search"]["evaluation_k"]]
    max_k = min(max(requested), len(corpus))
    truth, _ = exact_knn_squared_l2(
        held_out,
        corpus,
        global_ids,
        max_k,
        None,
        int(config["search"]["brute_force_batch_size"]),
    )
    started = perf_counter()
    _, predicted = index.search(held_out, max_k)
    search_seconds = perf_counter() - started
    metrics, _ = _recall_metrics(predicted, truth, requested)
    metrics["total_search_seconds"] = search_seconds
    metrics["average_query_ms"] = search_seconds * 1000.0 / len(held_out)
    return metrics


def _save_variant(
    root: Path,
    data_seed: int,
    condition: str,
    subset: str,
    graph_kind: str | None,
    variant: str,
    graph: dict[str, Any],
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    directory = root / f"seed_{data_seed}" / condition / subset
    if graph_kind is not None:
        directory = directory / graph_kind
    directory = directory / variant
    directory.mkdir(parents=True, exist_ok=True)
    graph_path = directory / "graph.json"
    metrics_path = directory / "metrics.json"
    save_graph(graph_path, graph)
    save_json(metrics_path, metrics)
    for name, value in arrays.items():
        np.save(directory / f"{name}.npy", value)
    return {"graph": str(graph_path), "metrics": str(metrics_path)}


def _flatten_numeric(prefix: str, value: Any) -> dict[str, float]:
    """Flatten nested numeric metrics for aggregate summaries."""
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_numeric(child_prefix, child))
        return result
    return {}


def _aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate numeric metrics across data seeds."""
    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    for record in records:
        key = (
            record["condition"],
            record["subset"],
            record["graph_kind"],
            record["variant"],
        )
        metrics = grouped.setdefault(key, {})
        for metric_name, value in _flatten_numeric("", record["metrics"]).items():
            metrics.setdefault(metric_name, []).append(value)
    summary = []
    for (condition, subset, graph_kind, variant), metrics in sorted(grouped.items()):
        aggregate = {}
        for metric_name, values in sorted(metrics.items()):
            array = np.asarray(values, dtype=np.float64)
            aggregate[metric_name] = {
                "count": int(len(array)),
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
            }
        summary.append(
            {
                "condition": condition,
                "subset": subset,
                "graph_kind": graph_kind,
                "variant": variant,
                "metrics": aggregate,
            }
        )
    return summary


def _phase1_paths_for_seed(
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    data_seed: int,
    generate_phase1_inputs: bool,
) -> tuple[Path, Path, dict[str, Any] | None]:
    """Return Phase 1 subset/transformation paths for one data seed."""
    if not generate_phase1_inputs:
        return (
            Path(config["phase1"]["subset_dir"]),
            Path(config["phase1"]["transformation_dir"]),
            None,
        )
    phase1_config_path = Path(config["phase1"]["source_config"])
    if not phase1_config_path.is_absolute():
        phase1_config_path = (
            phase1_config_path
            if phase1_config_path.exists()
            else config_path.parent / phase1_config_path
        )
    with phase1_config_path.open("r", encoding="utf-8") as file:
        phase1_config = json.load(file)
    phase1_config["random_seed"] = int(data_seed)
    phase1_config.setdefault("stages", {})
    phase1_config["stages"]["run_tests"] = False
    seed_root = output_root / "phase2_phase1_seeds" / f"seed_{data_seed}"
    generated_config = seed_root / "phase1_config.json"
    seed_root.mkdir(parents=True, exist_ok=True)
    with generated_config.open("w", encoding="utf-8") as file:
        json.dump(phase1_config, file, indent=2)
    phase1_manifest = run_complete_phase1(generated_config, seed_root, clean=True)
    return (
        seed_root / "phase1_subsets",
        seed_root / "phase1_transformations",
        phase1_manifest,
    )


def _clean(output_root: Path) -> None:
    for name in OUTPUT_DIRECTORIES:
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
    manifest = output_root / "phase2_run_manifest.json"
    if manifest.exists():
        manifest.unlink()


def run_complete_phase2(
    config_path: Path,
    output_root: Path = Path("data"),
    clean: bool = True,
) -> dict[str, Any]:
    """Execute Phase 2 with explicit data-seed support and aggregate metrics."""
    started_at, start_counter = start_run()
    config = load_phase2_config(config_path)
    if clean:
        _clean(output_root)
    if config.get("stages", {}).get("run_tests", False):
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "tests"], check=True
        )

    compiled_root = output_root / "phase2_compiled_graphs"
    repaired_root = output_root / "phase2_repaired_graphs"
    results_root = output_root / "phase2_results"
    data_seeds = [int(seed) for seed in config["seeds"]["data"]]
    configured_subset_dir = Path(config["phase1"]["subset_dir"])
    configured_transformation_dir = Path(config["phase1"]["transformation_dir"])
    generate_phase1_inputs = (
        len(data_seeds) > 1
        or not configured_subset_dir.exists()
        or not configured_transformation_dir.exists()
    )
    aggregate_records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "pipeline": "phase2_complete",
        "profile": config["profile"],
        "config_path": str(config_path),
        "config": config,
        "result_scope": (
            "smoke_test_not_dissertation_formal_results"
            if config["profile"] == "smoke_test"
            else "formal_experiment"
        ),
        "data_seeds": data_seeds,
        "multi_seed_phase1_generation": generate_phase1_inputs,
        "generated_phase1_inputs": generate_phase1_inputs,
        "seeds": {},
        "conditions": {},
    }

    for data_seed in data_seeds:
        phase1_subset_dir, phase1_transformation_dir, phase1_manifest = (
            _phase1_paths_for_seed(
                config,
                config_path,
                output_root,
                data_seed,
                generate_phase1_inputs,
            )
        )
        seed_manifest: dict[str, Any] = {
            "phase1_subset_dir": str(phase1_subset_dir),
            "phase1_transformation_dir": str(phase1_transformation_dir),
            "generated_phase1_manifest": phase1_manifest,
            "conditions": {},
        }

        for subset_name in config["subsets"]:
            shared, global_ids = load_subset_inputs(phase1_subset_dir, subset_name)
            query_shared = generate_held_out_shared_queries(
                int(config["queries"]["held_out_count_per_subset"]),
                shared.shape[1],
                _seed(int(config["seeds"]["query"]), str(data_seed), subset_name),
            )
            exact = {
                name: _exact_condition(
                    name, shared, query_shared, global_ids, subset_name, config
                )
                for name in config["transformations"]["exact"]
            }
            conditions = list(exact.values())
            if "rotation" not in exact:
                raise ValueError("Noise experiments require the rotation condition.")
            conditions.extend(_noise_conditions(exact["rotation"], subset_name, config))
            if config["transformations"].get("include_general_linear_stress", False):
                conditions.append(
                    _general_linear_condition(
                        phase1_transformation_dir, subset_name, query_shared
                    )
                )

            for condition in conditions:
                condition_name = condition["name"]
                condition_key = f"{condition_name}/{subset_name}"
                manifest_key = (
                    f"seed_{data_seed}/{condition_key}"
                    if generate_phase1_inputs
                    else condition_key
                )
                condition_dir = (
                    compiled_root / f"seed_{data_seed}" / condition_name / subset_name
                )
                condition_dir.mkdir(parents=True, exist_ok=True)
                np.save(condition_dir / "local_vectors.npy", condition["local_vectors"])
                np.save(
                    condition_dir / "transformed_vectors.npy",
                    condition["transformed_vectors"],
                )
                np.save(
                    condition_dir / "held_out_queries.npy",
                    condition["held_out_queries"],
                )
                np.save(condition_dir / "global_ids.npy", global_ids)
                faiss.write_index(
                    condition["local_index"], str(condition_dir / "local_hnsw.faiss")
                )
                save_json(condition_dir / "condition.json", condition["metadata"])
                vector_seconds = condition.get("cost", {}).get(
                    "vector_transformation_seconds", 0.0
                )
                condition_manifest: dict[str, Any] = {
                    "family": condition["family"],
                    "metadata": condition["metadata"],
                    "cost": condition.get("cost", {}),
                    "graph_kinds": {},
                }

                rebuild_start = perf_counter()
                rebuild = build_hnsw(
                    condition["transformed_vectors"],
                    global_ids,
                    int(config["search"]["m"]),
                    int(config["search"]["ef_construction"]),
                    int(config["search"]["ef_search"]),
                )
                rebuild_build_seconds = perf_counter() - rebuild_start
                rebuild_extract_start = perf_counter()
                rebuilt_graph = extract_raw_level0_graph(
                    rebuild,
                    condition["transformed_vectors"],
                    subset_name,
                    config["graph"]["degree_budget_policy"],
                )
                rebuild_graph_extraction_seconds = (
                    perf_counter() - rebuild_extract_start
                )
                rebuilt_graph["graph_kind"] = "rebuilt"
                rebuilt_graph["variant"] = "local_rebuild"
                rebuild_quality, rebuild_arrays = _evaluate_external(
                    rebuilt_graph,
                    condition["transformed_vectors"],
                    global_ids,
                    condition["held_out_queries"],
                    config,
                )
                rebuild_metrics = {
                    "cost": {
                        "vector_transformation_seconds": vector_seconds,
                        "build_seconds": rebuild_build_seconds,
                        "graph_extraction_seconds": rebuild_graph_extraction_seconds,
                        "processed_node_ratio": 1.0,
                        "total_compile_and_rebuild_seconds": (
                            vector_seconds
                            + rebuild_build_seconds
                            + rebuild_graph_extraction_seconds
                        ),
                    },
                    "quality": rebuild_quality,
                    "native_faiss_held_out": _native_rebuild_metrics(
                        rebuild,
                        condition["transformed_vectors"],
                        global_ids,
                        condition["held_out_queries"],
                        config,
                    ),
                    "graph_diagnostics": graph_diagnostics(
                        rebuilt_graph, rebuilt_graph
                    ),
                }
                condition_manifest["local_rebuild_baseline"] = _save_variant(
                    results_root,
                    data_seed,
                    condition_name,
                    subset_name,
                    None,
                    "local_rebuild",
                    rebuilt_graph,
                    rebuild_metrics,
                    rebuild_arrays,
                )
                aggregate_records.append(
                    {
                        "data_seed": data_seed,
                        "condition": condition_name,
                        "subset": subset_name,
                        "graph_kind": "rebuilt",
                        "variant": "local_rebuild",
                        "metrics": rebuild_metrics,
                    }
                )

                for graph_kind in config["graph"]["kinds"]:
                    extraction_start = perf_counter()
                    if graph_kind == "raw_level0":
                        base = extract_raw_level0_graph(
                            condition["local_index"],
                            condition["transformed_vectors"],
                            subset_name,
                            config["graph"]["degree_budget_policy"],
                        )
                    else:
                        base = extract_query_derived_graph(
                            condition["local_index"],
                            condition["local_vectors"],
                            condition["transformed_vectors"],
                            subset_name,
                            int(config["graph"]["candidate_search_k"]),
                            config["graph"]["degree_budget_policy"],
                        )
                    graph_extraction_seconds = perf_counter() - extraction_start
                    no_repair = graph_with_updated_distances(
                        base, condition["transformed_vectors"], "no_repair"
                    )
                    requested_variants = set(config["graph"]["variants"])
                    variants = {}
                    if "no_repair" in requested_variants:
                        variants["no_repair"] = (
                            no_repair,
                            graph_diagnostics(base, no_repair),
                        )
                    if "practical_repair" in requested_variants:
                        repair_start = perf_counter()
                        practical, repair_diagnostics = repair_graph(
                            base,
                            condition["local_index"],
                            condition["local_vectors"],
                            condition["transformed_vectors"],
                            int(config["graph"]["candidate_search_k"]),
                            bool(config["graph"]["force_weak_connectivity"]),
                        )
                        repair_total_seconds = perf_counter() - repair_start
                        variants["practical_repair"] = (practical, repair_diagnostics)
                    if "structural_oracle" in requested_variants:
                        oracle_start = perf_counter()
                        oracle = structural_oracle_graph(
                            base,
                            condition["transformed_vectors"],
                            int(config["search"]["brute_force_batch_size"]),
                        )
                        oracle_seconds = perf_counter() - oracle_start
                        variants["structural_oracle"] = (
                            oracle,
                            graph_diagnostics(base, oracle),
                        )
                    graph_manifest: dict[str, Any] = {}
                    for variant_name, (variant_graph, diagnostics) in variants.items():
                        quality, arrays = _evaluate_external(
                            variant_graph,
                            condition["transformed_vectors"],
                            global_ids,
                            condition["held_out_queries"],
                            config,
                        )
                        if variant_name == "practical_repair":
                            repair_seconds = float(
                                diagnostics.get("repair_seconds", repair_total_seconds)
                            )
                            candidate_seconds = float(
                                diagnostics.get("candidate_generation_seconds", 0.0)
                            )
                            distance_count = int(
                                diagnostics.get("candidate_distance_computations", 0)
                            )
                            processed_ratio = float(
                                diagnostics.get("processed_node_ratio", 1.0)
                            )
                        elif variant_name == "structural_oracle":
                            repair_seconds = oracle_seconds
                            candidate_seconds = 0.0
                            distance_count = int(
                                len(condition["transformed_vectors"])
                                * (len(condition["transformed_vectors"]) - 1)
                            )
                            processed_ratio = 1.0
                        else:
                            repair_seconds = 0.0
                            candidate_seconds = 0.0
                            distance_count = 0
                            processed_ratio = 0.0
                        metrics = {
                            "cost": {
                                "vector_transformation_seconds": vector_seconds,
                                "graph_extraction_seconds": graph_extraction_seconds,
                                "candidate_generation_seconds": candidate_seconds,
                                "distance_computation_count": distance_count,
                                "repair_seconds": repair_seconds,
                                "processed_node_ratio": processed_ratio,
                                "total_compile_and_repair_seconds": (
                                    vector_seconds
                                    + graph_extraction_seconds
                                    + repair_seconds
                                ),
                            },
                            "quality": quality,
                            "graph_diagnostics": diagnostics,
                        }
                        graph_manifest[variant_name] = _save_variant(
                            repaired_root,
                            data_seed,
                            condition_name,
                            subset_name,
                            graph_kind,
                            variant_name,
                            variant_graph,
                            metrics,
                            arrays,
                        )
                        aggregate_records.append(
                            {
                                "data_seed": data_seed,
                                "condition": condition_name,
                                "subset": subset_name,
                                "graph_kind": graph_kind,
                                "variant": variant_name,
                                "metrics": metrics,
                            }
                        )
                    condition_manifest["graph_kinds"][graph_kind] = graph_manifest
                seed_manifest["conditions"][condition_key] = condition_manifest
                manifest["conditions"][manifest_key] = condition_manifest
        manifest["seeds"][str(data_seed)] = seed_manifest

    manifest["summary"] = {
        "data_seed_count": len(data_seeds),
        "aggregate_metrics": _aggregate_records(aggregate_records),
    }
    add_run_metadata(manifest, started_at, start_counter)
    save_json(output_root / "phase2_run_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase2_smoke.json")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_complete_phase2(
        args.config, args.output_root, clean=not args.no_clean
    )
    print(
        f"Completed Phase 2 profile={manifest['profile']} with "
        f"{len(manifest['conditions'])} condition/subset runs in "
        f"{manifest['runtime']['duration_seconds']:.3f} seconds."
    )


if __name__ == "__main__":
    main()
