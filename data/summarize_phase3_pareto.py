"""Aggregate Phase 3 manifests into timing/quality/work and Pareto tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase2_common import save_json


def _integration(cost: dict[str, Any]) -> float | None:
    for key in (
        "integration_seconds_median",
        "integration_seconds",
        "total_integration_seconds",
        "build_seconds",
    ):
        if key in cost:
            return float(cost[key])
    return None


def _row(case: str, method: str, result: dict[str, Any]) -> dict[str, Any]:
    cost = result.get("cost", {})
    quality = result.get("quality", {}).get("overall", {})
    return {
        "case": case,
        "method": method,
        "corpus_size": result.get("corpus_size"),
        "recall_at_10": quality.get("recall_at_10"),
        "integration_seconds": _integration(cost),
        "integration_repetitions": cost.get("integration_repetitions", 1),
        "integration_p25": cost.get("integration_seconds_p25"),
        "integration_p75": cost.get("integration_seconds_p75"),
        "candidate_distance_work": cost.get("cross_candidate_distance_count", 0),
        "relative_full_rebuild": None,
        "recall_delta_random_direct": None,
        "integration_ratio_random_direct": None,
        "work_ratio_join1": None,
    }


def manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, methods in manifest.get("method_results", {}).items():
        for method, result in methods.items():
            if "directions" in result:
                for direction, direction_result in result["directions"].items():
                    rows.append(_row(case, f"{method}/{direction}", direction_result))
            elif result.get("quality"):
                rows.append(_row(case, method, result))
    for case in sorted({row["case"] for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        rebuild = next(
            (row["integration_seconds"] for row in case_rows if row["method"] == "full_rebuild_hnsw"),
            None,
        )
        if rebuild:
            for row in case_rows:
                if row["integration_seconds"] is not None:
                    row["relative_full_rebuild"] = row["integration_seconds"] / rebuild
        direct = next((row for row in case_rows if row["method"] == "indexed_cross_knn/random_direct"), None)
        join1 = next(
            (row for row in case_rows if row["method"] in {
                "indexed_cross_knn/high_degree_join1", "indexed_cross_knn/unlimited_join1"
            }),
            None,
        )
        for row in case_rows:
            if direct and row["recall_at_10"] is not None and direct["recall_at_10"] is not None:
                row["recall_delta_random_direct"] = row["recall_at_10"] - direct["recall_at_10"]
            if direct and row["integration_seconds"] is not None and direct["integration_seconds"]:
                row["integration_ratio_random_direct"] = row["integration_seconds"] / direct["integration_seconds"]
            if join1 and join1["candidate_distance_work"]:
                row["work_ratio_join1"] = row["candidate_distance_work"] / join1["candidate_distance_work"]
    return rows


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if any(left[key] is None or right[key] is None for key in ("recall_at_10", "integration_seconds")):
        return False
    return (
        left["recall_at_10"] >= right["recall_at_10"]
        and left["integration_seconds"] <= right["integration_seconds"]
        and (
            left["recall_at_10"] > right["recall_at_10"]
            or left["integration_seconds"] < right["integration_seconds"]
        )
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    methods: list[dict[str, Any]] = []
    for method, values in sorted(by_method.items()):
        recalls = [float(value["recall_at_10"]) for value in values if value["recall_at_10"] is not None]
        timings = [float(value["integration_seconds"]) for value in values if value["integration_seconds"] is not None]
        work = [float(value["candidate_distance_work"]) for value in values]
        ratios = [float(value["relative_full_rebuild"]) for value in values if value["relative_full_rebuild"] is not None]
        methods.append({
            "method": method,
            "case_count": len(values),
            "mean_recall_at_10": statistics.fmean(recalls) if recalls else None,
            "median_integration_seconds": statistics.median(timings) if timings else None,
            "mean_candidate_distance_work": statistics.fmean(work) if work else 0.0,
            "median_relative_full_rebuild": statistics.median(ratios) if ratios else None,
        })
    topology = [item for item in methods if item["method"].startswith("indexed_cross_knn")]
    for item in topology:
        item["pareto_nondominated"] = not any(
            other is not item and _dominates(
                {"recall_at_10": other["mean_recall_at_10"], "integration_seconds": other["median_integration_seconds"]},
                {"recall_at_10": item["mean_recall_at_10"], "integration_seconds": item["median_integration_seconds"]},
            )
            for other in topology
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "case_count": len({row["case"] for row in rows}),
        "methods": methods,
    }
    rebuild = next((item for item in methods if item["method"] == "full_rebuild_hnsw"), None)
    incremental = [item for item in methods if item["method"].startswith("incremental_insertion/")]
    if rebuild and incremental:
        best_incremental_recall = max(item["mean_recall_at_10"] for item in incremental)
        best_incremental_timing = min(item["median_integration_seconds"] for item in incremental)
        gates = []
        for item in methods:
            if not item["method"].startswith("indexed_cross_knn/bounded_"):
                continue
            candidate_rows = [row for row in rows if row["method"] == item["method"]]
            relative_values = [
                float(row["relative_full_rebuild"])
                for row in candidate_rows
                if row["relative_full_rebuild"] is not None
            ]
            if not relative_values:
                continue
            recall_gap = rebuild["mean_recall_at_10"] - item["mean_recall_at_10"]
            integration_ratio = statistics.median(relative_values)
            faster_cases = sum(
                float(row["relative_full_rebuild"]) < 1.0
                for row in candidate_rows
                if row["relative_full_rebuild"] is not None
            )
            incremental_advantage = (
                item["mean_recall_at_10"] > best_incremental_recall
                or item["median_integration_seconds"] < best_incremental_timing
            )
            checks = {
                "recall_gap_le_0p10": recall_gap <= 0.10,
                "integration_ratio_le_0p95": integration_ratio <= 0.95,
                "faster_cases_ge_4_of_6": faster_cases >= 4,
                "has_advantage_over_best_incremental": incremental_advantage,
            }
            gates.append({
                "method": item["method"],
                "mean_recall_gap_to_full_rebuild": recall_gap,
                "median_integration_ratio_to_full_rebuild": integration_ratio,
                "paired_cases_faster_than_full_rebuild": faster_cases,
                "paired_case_count": len(candidate_rows),
                "best_incremental_mean_recall_at_10": best_incremental_recall,
                "best_incremental_median_integration_seconds": best_incremental_timing,
                "checks": checks,
                "advance_to_100k": all(checks.values()),
            })
        if gates:
            result["stage21_gate"] = {
                "candidates": gates,
                "any_advance_to_100k": any(item["advance_to_100k"] for item in gates),
            }
    return result


def summarize(manifest_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in manifest_paths:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        source_rows = manifest_rows(manifest)
        for row in source_rows:
            if len(manifest_paths) > 1:
                row["case"] = f"{path.parent.name}:{row['case']}"
            row["manifest"] = str(path)
        rows.extend(source_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "manifest", "case", "method", "corpus_size", "recall_at_10",
        "integration_seconds", "integration_repetitions", "integration_p25",
        "integration_p75", "candidate_distance_work", "relative_full_rebuild",
        "recall_delta_random_direct", "integration_ratio_random_direct", "work_ratio_join1",
    ]
    with (output_dir / "paired_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate_rows(rows)
    summary["manifests"] = [str(path) for path in manifest_paths]
    summary["paired_cases_csv"] = str(output_dir / "paired_cases.csv")
    save_json(output_dir / "pareto_summary.json", summary)
    method_fields = sorted({key for item in summary["methods"] for key in item})
    with (output_dir / "method_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=method_fields)
        writer.writeheader()
        writer.writerows(summary["methods"])
    summary["method_summary_csv"] = str(output_dir / "method_summary.csv")
    save_json(output_dir / "pareto_summary.json", summary)
    if len(manifest_paths) > 1:
        save_json(output_dir / "unified_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.manifests, args.output_dir)
    print(f"Aggregated {summary['case_count']} cases into {args.output_dir}.")


if __name__ == "__main__":
    main()
