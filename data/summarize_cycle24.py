"""Validate Cycle 24 completeness and write a compact checked-in artifact."""

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


def _condition_name(value: str) -> str:
    return "gaussian_0p100" if value == "gaussian_0.1" else value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_ids(path: Path) -> set[int]:
    index = faiss.read_index(str(path))
    if not hasattr(index, "id_map"):
        raise ValueError(f"Expected an ID-mapped FAISS index: {path}")
    return set(int(value) for value in faiss.vector_to_array(index.id_map))


def validate_and_summarize(
    manifest_path: Path,
    config_path: Path,
    prepared_root: Path,
    output_path: Path,
    *,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)
    prepared_manifest = _load_json(prepared_root / "manifest.json")
    methods = [str(value) for value in config["methods"]]
    conditions = [_condition_name(str(value)) for value in config["conditions"]]
    orders = [tuple(str(value) for value in order) for order in config["orders"]]
    expected = {
        (int(seed), condition, order, method)
        for seed in config["seeds"]
        for condition in conditions
        for order in orders
        for method in methods
    }
    actual: dict[tuple[int, str, tuple[str, ...], str], dict[str, Any]] = {}
    errors: list[str] = []
    checked_metrics = 0
    checked_indices = 0
    checked_graphs = 0
    validated_metrics: dict[tuple[int, str, tuple[str, ...], str], dict[str, Any]] = {}

    if manifest.get("schema_version") != "metricdb.cycle24-results.v2":
        errors.append("unexpected Cycle 24 result schema")
    if prepared_manifest.get("case_count") != len(config["seeds"]) * len(conditions) * len(config["subset_sizes"]):
        errors.append("prepared input case count does not match config")
    if prepared_manifest.get("corpus_size_per_seed_condition") != config["corpus_size"]:
        errors.append("prepared corpus size does not match config")

    expected_ids_by_case: dict[tuple[int, str], set[int]] = {}
    for seed in config["seeds"]:
        for condition in conditions:
            ids: set[int] = set()
            total = 0
            for subset in config["subset_sizes"]:
                path = prepared_root / f"seed_{seed}" / condition / subset / "global_ids.npy"
                if not path.is_file():
                    errors.append(f"missing prepared IDs: {path}")
                    continue
                values = np.load(path).astype(np.int64)
                total += len(values)
                before = len(ids)
                ids.update(int(value) for value in values)
                if len(ids) - before != len(values):
                    errors.append(f"duplicate prepared IDs for seed={seed} condition={condition}")
            if total != int(config["corpus_size"]) or len(ids) != total:
                errors.append(f"prepared union ID count mismatch for seed={seed} condition={condition}")
            expected_ids_by_case[(int(seed), condition)] = ids

    for row in manifest.get("cases", []):
        key = (int(row["seed"]), str(row["condition"]), tuple(row["order"]), str(row["method"]))
        if key in actual:
            errors.append(f"duplicate result case: {key}")
            continue
        actual[key] = row
        metrics_path = repo_root / row["metrics"]
        if not metrics_path.is_file():
            errors.append(f"missing metrics: {metrics_path}")
            continue
        metrics = _load_json(metrics_path)
        validated_metrics[key] = metrics
        checked_metrics += 1
        if str(metrics["method"]).replace("full_rebuild_hnsw", "full_rebuild") != key[3]:
            errors.append(f"method mismatch in {metrics_path}")
        if key[3] != "full_rebuild" and tuple(metrics.get("order", ())) != key[2]:
            errors.append(f"order mismatch in {metrics_path}")
        recall = float(metrics["quality"]["overall"]["recall_at_10"])
        cross_recall = float(metrics["quality"]["source_target"]["recall_at_10"]["cross_subset"])
        if not 0.0 <= recall <= 1.0 or not 0.0 <= cross_recall <= 1.0:
            errors.append(f"invalid recall in {metrics_path}")
        if not np.isclose(recall, float(row["recall_at_10"])):
            errors.append(f"manifest recall mismatch in {metrics_path}")
        samples = metrics["cost"].get("integration_seconds_samples", [])
        if len(samples) != int(config["integration_repetitions"]) or any(float(value) <= 0 for value in samples):
            errors.append(f"integration repetitions mismatch in {metrics_path}")
        if key[3] in {"full_rebuild", "sequential_incremental"}:
            index_key = "index_path"
            index_path = repo_root / metrics[index_key]
            if not index_path.is_file():
                errors.append(f"missing native index: {index_path}")
            elif _index_ids(index_path) != expected_ids_by_case[(key[0], key[1])]:
                errors.append(f"native index ID set mismatch: {index_path}")
            else:
                checked_indices += 1
        else:
            validity = metrics.get("validity", {})
            statistics = metrics.get("graph_statistics", {})
            if not validity.get("unique_global_ids") or not validity.get("prior_edges_preserved"):
                errors.append(f"external validity failure in {metrics_path}")
            if not validity.get("weakly_connected"):
                errors.append(f"external graph is not weakly connected in {metrics_path}")
            if int(statistics.get("node_count", -1)) != int(config["corpus_size"]):
                errors.append(f"external node count mismatch in {metrics_path}")
            second = metrics["cost"].get("second_merge", {})
            if not second.get("prior_edges_preserved") or int(second.get("candidate_distance_count", 0)) <= 0:
                errors.append(f"second-merge evidence missing in {metrics_path}")
            graph_path = repo_root / metrics["graph_path"]
            if not graph_path.is_file():
                errors.append(f"missing external graph: {graph_path}")
            else:
                checked_graphs += 1

    missing = expected - set(actual)
    extra = set(actual) - expected
    if missing:
        errors.append(f"missing result cases: {len(missing)}")
    if extra:
        errors.append(f"unexpected result cases: {len(extra)}")
    if int(manifest.get("case_count", -1)) != len(expected):
        errors.append("top-level case_count mismatch")
    effects = manifest.get("order_effects", [])
    expected_effects = len(config["seeds"]) * len(conditions) * len(methods)
    if len(effects) != expected_effects or any(int(row.get("order_count", 0)) != len(orders) for row in effects):
        errors.append("order-effect matrix is incomplete")

    aggregates = []
    for method in methods:
        rows = [row for key, row in actual.items() if key[3] == method]
        recalls = np.asarray([row["recall_at_10"] for row in rows], dtype=np.float64)
        cross = np.asarray([row["cross_subset_recall_at_10"] for row in rows], dtype=np.float64)
        costs = np.asarray([row["integration_seconds_median"] for row in rows], dtype=np.float64)
        aggregate = {
                "method": method,
                "case_count": len(rows),
                "mean_recall_at_10": float(np.mean(recalls)),
                "mean_cross_subset_recall_at_10": float(np.mean(cross)),
                "median_integration_seconds": float(np.median(costs)),
                "min_recall_at_10": float(np.min(recalls)),
                "max_recall_at_10": float(np.max(recalls)),
        }
        method_metrics = [metrics for key, metrics in validated_metrics.items() if key[3] == method]
        query_ms = [
            value["cost"]["average_query_ms"]
            if "average_query_ms" in value["cost"]
            else value["quality"]["average_query_ms"]
            for value in method_metrics
            if "average_query_ms" in value["cost"] or "average_query_ms" in value["quality"]
        ]
        if query_ms:
            aggregate["mean_query_ms"] = float(np.mean(query_ms))
        if method in {"random_direct", "bounded_n8_c2", "join1"}:
            first_work = np.asarray([value["cost"]["first_merge"]["candidate_distance_count"] for value in method_metrics], dtype=np.float64)
            second_work = np.asarray([value["cost"]["second_merge"]["candidate_distance_count"] for value in method_metrics], dtype=np.float64)
            second_seconds = np.asarray([value["cost"]["second_merge"]["integration_seconds"] for value in method_metrics], dtype=np.float64)
            aggregate.update(
                {
                    "mean_first_merge_candidate_distance_count": float(np.mean(first_work)),
                    "mean_second_merge_candidate_distance_count": float(np.mean(second_work)),
                    "second_to_first_candidate_work_ratio": float(np.mean(second_work / first_work)),
                    "median_second_merge_seconds": float(np.median(second_seconds)),
                }
            )
        elif method == "sequential_incremental":
            first_add = np.asarray([np.median(value["cost"]["first_add_seconds_samples"]) for value in method_metrics])
            second_add = np.asarray([value["cost"]["second_add_seconds_median"] for value in method_metrics])
            aggregate.update(
                {
                    "median_first_add_seconds": float(np.median(first_add)),
                    "median_second_add_seconds": float(np.median(second_add)),
                    "median_second_to_first_add_ratio": float(np.median(second_add / first_add)),
                }
            )
        aggregates.append(aggregate)

    result = {
        "schema_version": "metricdb.cycle24-compact.v1",
        "status": "complete" if not errors else "invalid",
        "config": str(config_path),
        "prepared_manifest": str(prepared_root / "manifest.json"),
        "formal_manifest": str(manifest_path),
        "source_checksums": {
            "config": _checksum(config_path),
            "prepared_manifest": _checksum(prepared_root / "manifest.json"),
            "formal_manifest": _checksum(manifest_path),
        },
        "validation": {
            "valid": not errors,
            "errors": errors,
            "expected_case_count": len(expected),
            "checked_metrics": checked_metrics,
            "checked_native_indices": checked_indices,
            "checked_external_graphs": checked_graphs,
            "order_effect_count": len(effects),
        },
        "aggregates": aggregates,
        "order_effects": effects,
        "cases": manifest.get("cases", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("Cycle 24 validation failed: " + "; ".join(errors))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/phase4_cycle24/manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/cycle24_sequential_formal.json"))
    parser.add_argument("--prepared-root", type=Path, default=Path("data/phase4_cycle24_prepared"))
    parser.add_argument("--output", type=Path, default=Path("results/cycle24_formal.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    summary = validate_and_summarize(args.manifest, args.config, args.prepared_root, args.output, repo_root=args.repo_root)
    print(json.dumps(summary["validation"], indent=2, sort_keys=True))
