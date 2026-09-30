"""Validate Cycle 25 raw artifacts and write the compact formal result."""

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

from data.phase2_common import load_graph
from data.phase3_metrics import graph_statistics, recall_at_k
from data.phase4_statistics import paired_bootstrap


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
        raise ValueError(f"Expected ID-mapped FAISS index: {path}")
    return set(int(value) for value in faiss.vector_to_array(index.id_map))


def _metric_payload(metrics: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if row["method"].startswith("incremental_"):
        return metrics["directions"][row["direction"]]
    return metrics


def validate_and_summarize(manifest_path: Path, config_path: Path, prepared_root: Path, output_path: Path, *, repo_root: Path = Path(".")) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)
    prepared = _load_json(prepared_root / "manifest.json")
    conditions = [_condition_name(str(value)) for value in config["conditions"]]
    expected = {(int(seed), condition, str(method)) for seed in config["experiment_seeds"] for condition in conditions for method in config["methods"]}
    actual: dict[tuple[int, str, str], dict[str, Any]] = {}
    errors: list[str] = []
    checked_metrics = checked_indices = checked_graphs = recalculated_recalls = 0

    if manifest.get("schema_version") != "metricdb.cycle25-results.v1":
        errors.append("unexpected Cycle 25 result schema")
    if prepared.get("case_count") != len(config["experiment_seeds"]) * len(conditions) * 2:
        errors.append("prepared input matrix is incomplete")
    if not prepared.get("data_quality", {}).get("split_exhaustive") or not prepared.get("data_quality", {}).get("split_checksums_verified"):
        errors.append("prepared input data-quality gate is incomplete")

    expected_ids: set[int] = set()
    for subset in ("D_A", "D_B"):
        values = np.load(prepared_root / f"seed_{config['experiment_seeds'][0]}" / conditions[0] / subset / "global_ids.npy")
        expected_ids.update(int(value) for value in values)
    if len(expected_ids) != int(config["split"]["corpus"]):
        errors.append("prepared union ID count does not match config")

    for row in manifest.get("cases", []):
        key = (int(row["experiment_seed"]), str(row["condition"]), str(row["method"]))
        if key in actual:
            errors.append(f"duplicate result case: {key}")
            continue
        actual[key] = row
        metrics_path = repo_root / row["metrics"]
        if not metrics_path.is_file():
            errors.append(f"missing metrics: {metrics_path}")
            continue
        metrics = _load_json(metrics_path)
        payload = _metric_payload(metrics, row)
        checked_metrics += 1
        samples = payload["cost"].get("integration_seconds_samples", [])
        if len(samples) != int(config["timing_repetitions"]) or any(float(value) <= 0 for value in samples):
            errors.append(f"integration repetitions mismatch: {metrics_path} {row['method']}")
        elif not np.isclose(float(np.median(samples)), float(row["integration_seconds_median"])):
            errors.append(f"integration median mismatch: {metrics_path} {row['method']}")
        recall = float(payload["quality"]["overall"]["recall_at_10"])
        if not 0.0 <= recall <= 1.0 or not np.isclose(recall, float(row["recall_at_10"])):
            errors.append(f"invalid or mismatched recall: {metrics_path} {row['method']}")
        predicted_path = repo_root / payload["arrays"]["neighbor_global_ids"]
        truth_path = manifest_path.parent / f"seed_{key[0]}" / key[1] / "ground_truth" / "neighbor_global_ids.npy"
        if predicted_path.is_file() and truth_path.is_file():
            recomputed, _ = recall_at_k(np.load(predicted_path), np.load(truth_path), [10])
            recalculated_recalls += 1
            if not np.isclose(float(recomputed["recall_at_10"]), recall):
                errors.append(f"independent recall recomputation mismatch: {metrics_path} {row['method']}")
        else:
            errors.append(f"missing prediction or truth arrays for {key}")

        if row["method"] == "full_rebuild" or row["method"].startswith("incremental_"):
            index_path = repo_root / payload["index_path"]
            if not index_path.is_file() or _index_ids(index_path) != expected_ids:
                errors.append(f"native index ID set mismatch: {index_path}")
            else:
                checked_indices += 1
        else:
            graph_path = repo_root / payload["graph_path"]
            if not graph_path.is_file():
                errors.append(f"missing external graph: {graph_path}")
            else:
                graph = load_graph(graph_path)
                statistics = graph_statistics(graph)
                ids = set(int(value) for value in graph["node_global_ids"])
                if ids != expected_ids or statistics["weak_connected_component_count"] != 1:
                    errors.append(f"external graph validity failure: {graph_path}")
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

    aggregate_rows = []
    statistic_rows = []
    for row in actual.values():
        statistic_rows.append({"pair_key": row["pair_key"], "method": row["method"], "recall": row["recall_at_10"], "integration": row["integration_seconds_median"]})
    for method in config["methods"]:
        rows = [row for row in actual.values() if row["method"] == method]
        aggregate_rows.append({
            "method": method,
            "pair_count": len(rows),
            "mean_recall_at_10": float(np.mean([row["recall_at_10"] for row in rows])) if rows else float("nan"),
            "mean_cross_subset_recall_at_10": float(np.mean([row["cross_subset_recall_at_10"] for row in rows])) if rows else float("nan"),
            "median_integration_seconds": float(np.median([row["integration_seconds_median"] for row in rows])) if rows else float("nan"),
            "mean_query_ms": float(np.mean([row["average_query_ms"] for row in rows])) if rows else float("nan"),
        })
    confidence_intervals = []
    for method in config["methods"]:
        if method == "full_rebuild":
            continue
        confidence_intervals.append({
            "method": method,
            "recall_delta_vs_full_rebuild": paired_bootstrap(statistic_rows, baseline="full_rebuild", method=method, value_key="recall", statistic="difference", samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"])),
            "integration_ratio_vs_full_rebuild": paired_bootstrap(statistic_rows, baseline="full_rebuild", method=method, value_key="integration", statistic="log_ratio", samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"])),
        })

    result = {
        "schema_version": "metricdb.cycle25-compact.v1",
        "status": "complete" if not errors else "invalid",
        "config": str(config_path),
        "prepared_manifest": str(prepared_root / "manifest.json"),
        "formal_manifest": str(manifest_path),
        "source_checksums": {
            "config": _checksum(config_path),
            "prepared_manifest": _checksum(prepared_root / "manifest.json"),
            "formal_manifest": _checksum(manifest_path),
        },
        "dataset": {"name": config["dataset"]["name"], "rows": int(config["dataset"]["valid_unique_rows"]), "dimension": int(config["dataset"]["dimension"]), "data_quality": prepared.get("data_quality", {})},
        "design": {
            "fixed_split_seed": int(config["split"]["seed"]),
            "experiment_seeds": [int(value) for value in config["experiment_seeds"]],
            "conditions": conditions,
            "pairing_unit": config["statistics"]["pairing_unit"],
            "pair_count": len(config["experiment_seeds"]) * len(conditions),
            "caveat": config["statistics"]["fixed_split_caveat"],
        },
        "validation": {"valid": not errors, "errors": errors, "expected_case_count": len(expected), "checked_metrics": checked_metrics, "checked_native_indices": checked_indices, "checked_external_graphs": checked_graphs, "independently_recalculated_recalls": recalculated_recalls},
        "aggregates": aggregate_rows,
        "confidence_intervals": confidence_intervals,
        "cases": manifest.get("cases", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("Cycle 25 validation failed: " + "; ".join(errors))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/phase4_cycle25/formal/manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/cycle25_glove_formal.json"))
    parser.add_argument("--prepared-root", type=Path, default=Path("data/phase4_cycle25/prepared"))
    parser.add_argument("--output", type=Path, default=Path("results/cycle25_formal.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    summary = validate_and_summarize(args.manifest, args.config, args.prepared_root, args.output, repo_root=args.repo_root)
    print(json.dumps(summary["validation"], indent=2, sort_keys=True))
