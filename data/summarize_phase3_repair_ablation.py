"""Aggregate paired practical-repair versus no-repair Phase 3 results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BATCHES = ("exact", "gaussian", "laplacian")
PAIRED_METHODS = ("dual_index", "no_cross_merged", "random_cross_links", "sampled_cross_knn")
PRACTICAL_METHODS = {
    "bruteforce_ground_truth", "full_rebuild_hnsw", "dual_index",
    "no_cross_merged", "random_cross_links", "sampled_cross_knn", "incremental_insertion",
}
NO_REPAIR_METHODS = {
    "bruteforce_ground_truth", "dual_index", "no_cross_merged",
    "random_cross_links", "sampled_cross_knn",
}
TIE_TOLERANCE = 1e-12


class RepairAblationError(ValueError):
    """Raised when paired repair inputs are incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RepairAblationError(f"Input file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RepairAblationError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RepairAblationError(f"JSON root must be an object: {path}")
    return value


def _number(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RepairAblationError(f"Missing numeric field {context}.{key}")
    return float(value)


def _optional(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("Cannot summarize empty values")
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _optional_stats(values: Iterable[float | None]) -> dict[str, Any] | None:
    present = [value for value in values if value is not None]
    return _stats(present) if present else None


def _case_parts(key: str) -> tuple[int, str]:
    try:
        seed_text, condition = key.split("/", 1)
        return int(seed_text.removeprefix("seed_")), condition
    except ValueError as error:
        raise RepairAblationError(f"Invalid case key: {key}") from error


def _quality(item: dict[str, Any], context: str) -> dict[str, float]:
    overall = item.get("quality", {}).get("overall", {})
    cross = item.get("quality", {}).get("source_target", {}).get("recall_at_10", {})
    return {
        "recall_at_1": _number(overall, "recall_at_1", context),
        "recall_at_10": _number(overall, "recall_at_10", context),
        "recall_at_50": _number(overall, "recall_at_50", context),
        "cross_subset_recall_at_10": _number(cross, "cross_subset", context),
    }


def _validate_manifest(
    batch: str, manifest: dict[str, Any], variant: str, expected_methods: set[str]
) -> dict[str, dict[str, Any]]:
    readiness = manifest.get("readiness", {})
    if readiness.get("ready") is not True or readiness.get("missing_count") != 0:
        raise RepairAblationError(f"{batch}/{variant} manifest is not ready or has missing inputs")
    if readiness.get("graph_kind") != "raw_level0":
        raise RepairAblationError(f"{batch}/{variant} graph_kind must be raw_level0")
    if readiness.get("within_subset_variant") != variant:
        raise RepairAblationError(
            f"{batch} expected variant {variant}, got {readiness.get('within_subset_variant')}"
        )
    results = manifest.get("method_results")
    if not isinstance(results, dict) or not results:
        raise RepairAblationError(f"{batch}/{variant} has no method_results")
    for key, case in results.items():
        if set(case) != expected_methods:
            raise RepairAblationError(
                f"{batch}/{variant}/{key} method completeness failure; "
                f"missing={sorted(expected_methods - set(case))}, extra={sorted(set(case) - expected_methods)}"
            )
        _case_parts(key)
        for method in PAIRED_METHODS:
            item = case[method]
            input_data = item.get("input", {})
            if input_data.get("graph_kind") != "raw_level0" or input_data.get("within_subset_variant") != variant:
                raise RepairAblationError(f"Input metadata mismatch for {batch}/{variant}/{key}/{method}")
            if item.get("diagnostics", {}).get("global_id_output") is not True:
                raise RepairAblationError(f"Global-ID output not confirmed for {batch}/{variant}/{key}/{method}")
            _quality(item, f"{batch}/{variant}/{key}/{method}")
        for method in ("random_cross_links", "sampled_cross_knn"):
            graph = case[method].get("graph", {})
            if _number(graph, "sample_ratio", f"{key}/{method}.graph") != 0.05:
                raise RepairAblationError(f"Unexpected sample_ratio for {batch}/{variant}/{key}/{method}")
            if _number(graph, "cross_k", f"{key}/{method}.graph") != 4:
                raise RepairAblationError(f"Unexpected cross_k for {batch}/{variant}/{key}/{method}")
        if _number(case["no_cross_merged"].get("graph", {}), "cross_subset_edge_count", key) != 0:
            raise RepairAblationError(f"no-cross contains cross edges for {batch}/{variant}/{key}")
        sampled_cost = case["sampled_cross_knn"].get("cost", {})
        _number(sampled_cost, "cross_candidate_distance_count", f"{key}/sampled.cost")
        _number(sampled_cost, "cross_scan_work_ratio", f"{key}/sampled.cost")
    return results


def load_paired_manifests(
    practical_paths: dict[str, Path], no_repair_paths: dict[str, Path]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for batch in BATCHES:
        practical = _validate_manifest(
            batch, _read_json(practical_paths[batch]), "practical_repair", PRACTICAL_METHODS
        )
        no_repair = _validate_manifest(
            batch, _read_json(no_repair_paths[batch]), "no_repair", NO_REPAIR_METHODS
        )
        if set(practical) != set(no_repair):
            raise RepairAblationError(
                f"{batch} practical/no-repair case keys differ; "
                f"practical_only={sorted(set(practical)-set(no_repair))}, "
                f"no_repair_only={sorted(set(no_repair)-set(practical))}"
            )
        for key in sorted(practical):
            seed, condition = _case_parts(key)
            if (seed, condition) in seen:
                raise RepairAblationError(f"Case appears in multiple batches: {key}")
            seen.add((seed, condition))
            for method in PAIRED_METHODS:
                p_item = practical[key][method]
                n_item = no_repair[key][method]
                p_quality = _quality(p_item, f"{batch}/practical/{key}/{method}")
                n_quality = _quality(n_item, f"{batch}/no_repair/{key}/{method}")
                record: dict[str, Any] = {
                    "batch": batch, "seed": seed, "condition": condition, "method": method,
                }
                for metric in p_quality:
                    record[f"practical_{metric}"] = p_quality[metric]
                    record[f"no_repair_{metric}"] = n_quality[metric]
                    record[f"delta_{metric}"] = p_quality[metric] - n_quality[metric]
                for prefix, item in (("practical", p_item), ("no_repair", n_item)):
                    cost = item.get("cost", {})
                    stats = item.get("graph_statistics", {})
                    record[f"{prefix}_integration_seconds"] = _optional(cost, "integration_seconds")
                    record[f"{prefix}_total_search_seconds"] = _optional(cost, "total_search_seconds")
                    record[f"{prefix}_average_query_ms"] = _optional(cost, "average_query_ms")
                    record[f"{prefix}_weak_components"] = _optional(stats, "weak_connected_component_count")
                    record[f"{prefix}_reachable_ratio"] = _optional(stats, "directed_reachable_from_entry_ratio")
                for metric in ("integration_seconds", "total_search_seconds", "average_query_ms", "weak_components", "reachable_ratio"):
                    p_value, n_value = record[f"practical_{metric}"], record[f"no_repair_{metric}"]
                    record[f"delta_{metric}"] = p_value - n_value if p_value is not None and n_value is not None else None
                records.append(record)
    if len(seen) != 55:
        raise RepairAblationError(f"Expected 55 paired cases, found {len(seen)}")
    return records


def _group_labels(record: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("all", "all"),
        ("batch", record["batch"]),
        ("condition", record["condition"]),
    ]


def summarize_paired(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for level, group in _group_labels(record):
            grouped[(level, group, record["method"])].append(record)
    rows: list[dict[str, Any]] = []
    for (level, group, method), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "group_level": level, "group": group, "method": method,
            "paired_case_count": len(values),
        }
        for metric in ("recall_at_1", "recall_at_10", "recall_at_50", "cross_subset_recall_at_10"):
            practical = [value[f"practical_{metric}"] for value in values]
            no_repair = [value[f"no_repair_{metric}"] for value in values]
            deltas = [value[f"delta_{metric}"] for value in values]
            delta_stats = _stats(deltas)
            row.update({
                f"practical_{metric}_mean": statistics.fmean(practical),
                f"no_repair_{metric}_mean": statistics.fmean(no_repair),
                f"delta_{metric}_mean": delta_stats["mean"],
                f"delta_{metric}_std": delta_stats["std"],
                f"delta_{metric}_median": delta_stats["median"],
                f"delta_{metric}_min": delta_stats["min"],
                f"delta_{metric}_max": delta_stats["max"],
                f"{metric}_practical_wins": sum(delta > TIE_TOLERANCE for delta in deltas),
                f"{metric}_ties": sum(abs(delta) <= TIE_TOLERANCE for delta in deltas),
                f"{metric}_no_repair_wins": sum(delta < -TIE_TOLERANCE for delta in deltas),
            })
        for metric in ("integration_seconds", "total_search_seconds", "average_query_ms", "weak_components", "reachable_ratio"):
            for variant in ("practical", "no_repair", "delta"):
                summary = _optional_stats(value[f"{variant}_{metric}"] for value in values)
                row[f"{variant}_{metric}_mean"] = summary["mean"] if summary else None
                row[f"{variant}_{metric}_std"] = summary["std"] if summary else None
        rows.append(row)
    return rows


def load_phase2_snapshots(paths: dict[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch in BATCHES:
        snapshot = _read_json(paths[batch])
        if snapshot.get("batch") != batch or snapshot.get("graph_kind") != "raw_level0":
            raise RepairAblationError(f"Invalid Phase 2 snapshot metadata: {paths[batch]}")
        batch_records = snapshot.get("records")
        if not isinstance(batch_records, list) or len(batch_records) != snapshot.get("record_count"):
            raise RepairAblationError(f"Invalid Phase 2 snapshot records: {paths[batch]}")
        expected = {
            (int(seed), condition, subset, variant)
            for seed in snapshot["seeds"] for condition in snapshot["conditions"]
            for subset in snapshot["subsets"] for variant in ("practical_repair", "no_repair")
        }
        actual = {(r["seed"], r["condition"], r["subset"], r["variant"]) for r in batch_records}
        if actual != expected:
            raise RepairAblationError(f"Incomplete Phase 2 snapshot records: {paths[batch]}")
        records.extend(batch_records)
    return records


def build_phase2_cost_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, int, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_case[(record["batch"], record["seed"], record["condition"])][
            (record["subset"], record["variant"])
        ] = record
    rows: list[dict[str, Any]] = []
    sum_fields = (
        "repair_seconds", "candidate_generation_seconds", "distance_computation_count",
        "total_compile_and_repair_seconds", "vector_transformation_seconds", "graph_extraction_seconds",
    )
    mean_cost_fields = ("processed_node_ratio",)
    mean_diag_fields = (
        "retained_edge_ratio", "removed_edge_ratio", "added_edge_ratio", "reciprocal_edge_ratio",
        "connected_component_count", "directed_reachable_from_entry_ratio",
    )
    mean_recall_fields = ("recall_at_1", "recall_at_10", "recall_at_50")
    for (batch, seed, condition), case in sorted(by_case.items()):
        for scope, subsets in (("D_A", ("D_A",)), ("D_B", ("D_B",)), ("pairwise_sum", ("D_A", "D_B"))):
            row: dict[str, Any] = {"batch": batch, "seed": seed, "condition": condition, "scope": scope}
            for field in sum_fields:
                for variant in ("practical_repair", "no_repair"):
                    row[f"{variant}_{field}"] = sum(case[(s, variant)]["cost"][field] for s in subsets)
                row[f"delta_{field}"] = row[f"practical_repair_{field}"] - row[f"no_repair_{field}"]
            for field in mean_cost_fields:
                for variant in ("practical_repair", "no_repair"):
                    row[f"{variant}_{field}"] = statistics.fmean(case[(s, variant)]["cost"][field] for s in subsets)
                row[f"delta_{field}"] = row[f"practical_repair_{field}"] - row[f"no_repair_{field}"]
            for field in mean_diag_fields:
                for variant in ("practical_repair", "no_repair"):
                    row[f"{variant}_{field}"] = statistics.fmean(case[(s, variant)]["graph_diagnostics"][field] for s in subsets)
                row[f"delta_{field}"] = row[f"practical_repair_{field}"] - row[f"no_repair_{field}"]
            for field in mean_recall_fields:
                for variant in ("practical_repair", "no_repair"):
                    row[f"{variant}_local_held_out_{field}"] = statistics.fmean(case[(s, variant)]["local_held_out_recall"][field] for s in subsets)
                row[f"delta_local_held_out_{field}"] = row[f"practical_repair_local_held_out_{field}"] - row[f"no_repair_local_held_out_{field}"]
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _markdown(summary_rows: list[dict[str, Any]]) -> str:
    all_rows = {row["method"]: row for row in summary_rows if row["group_level"] == "all"}
    lines = [
        "# Phase 3 Repair Ablation Tables", "",
        "Deltas are paired practical_repair minus no_repair; std is sample std (ddof=1).", "",
        "| Method | Practical R@10 | No-repair R@10 | Delta mean +/- std | Wins/ties/losses |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in PAIRED_METHODS:
        row = all_rows[method]
        lines.append(
            f"| {method} | {row['practical_recall_at_10_mean']:.5f} | {row['no_repair_recall_at_10_mean']:.5f} | "
            f"{row['delta_recall_at_10_mean']:.5f} +/- {row['delta_recall_at_10_std']:.5f} | "
            f"{row['recall_at_10_practical_wins']}/{row['recall_at_10_ties']}/{row['recall_at_10_no_repair_wins']} |"
        )
    return "\n".join(lines) + "\n"


def aggregate_repair_ablation(
    practical_paths: dict[str, Path], no_repair_paths: dict[str, Path],
    phase2_paths: dict[str, Path], output_dir: Path,
) -> dict[str, Any]:
    records = load_paired_manifests(practical_paths, no_repair_paths)
    summary_rows = summarize_paired(records)
    phase2_records = load_phase2_snapshots(phase2_paths)
    phase2_rows = build_phase2_cost_rows(phase2_records)
    recall_rows = [{k: row[k] for k in row if "recall_at_10" in k or k in ("group_level", "group", "method", "paired_case_count")} for row in summary_rows]
    cross_rows = [{k: row[k] for k in row if "cross_subset_recall_at_10" in k or k in ("group_level", "group", "method", "paired_case_count")} for row in summary_rows]
    connectivity_rows = [{k: row[k] for k in row if any(token in k for token in ("weak_components", "reachable_ratio")) or k in ("group_level", "group", "method", "paired_case_count")} for row in summary_rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "repair_ablation_recall10.csv", recall_rows)
    _write_csv(output_dir / "repair_ablation_paired_deltas.csv", records)
    _write_csv(output_dir / "repair_ablation_cross_recall10.csv", cross_rows)
    _write_csv(output_dir / "repair_ablation_connectivity.csv", connectivity_rows)
    _write_csv(output_dir / "repair_ablation_phase2_cost.csv", phase2_rows)
    (output_dir / "repair_ablation_tables.md").write_text(_markdown(summary_rows), encoding="utf-8")
    summary = {
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_manifests": {
            "practical_repair": {b: str(practical_paths[b]) for b in BATCHES},
            "no_repair": {b: str(no_repair_paths[b]) for b in BATCHES},
            "phase2_metric_snapshots": {b: str(phase2_paths[b]) for b in BATCHES},
        },
        "scope": {"graph_kind": "raw_level0", "subset_pair": ["D_A", "D_B"], "case_count": 55, "paired_methods": list(PAIRED_METHODS), "sample_ratio": 0.05, "cross_k": 4},
        "statistics": {"unit": "seed x condition paired case", "delta": "practical_repair - no_repair", "sample_std_ddof": 1, "single_observation_std": 0, "tie_tolerance": TIE_TOLERANCE},
        "paired_summaries": summary_rows,
        "phase2_cost_row_count": len(phase2_rows),
        "schema_warnings": [
            "External graph timings are prototype measurements and are not backend-equivalent to native FAISS timings.",
            "Phase 2 preparation can be cached; combined Phase 2 plus Phase 3 time is not a universal deployment cost.",
            "practical_repair is the current Python prototype implementation.",
            "Missing optional Phase 3 cost/connectivity values remain null or blank, not zero.",
        ],
    }
    (output_dir / "phase3_repair_ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _defaults(prefix: str) -> dict[str, Path]:
    if prefix == "practical":
        return {b: Path(f"data/phase3_formal_batch_{b}_manifest.json") for b in BATCHES}
    if prefix == "no_repair":
        return {b: Path(f"data/phase3_repair_ablation_no_repair_{b}_manifest.json") for b in BATCHES}
    return {b: Path(f"data/phase3_repair_ablation_phase2_metrics/{b}.json") for b in BATCHES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for batch in BATCHES:
        parser.add_argument(f"--practical-{batch}-manifest", type=Path, default=_defaults("practical")[batch])
        parser.add_argument(f"--no-repair-{batch}-manifest", type=Path, default=_defaults("no_repair")[batch])
        parser.add_argument(f"--phase2-{batch}-metrics", type=Path, default=_defaults("phase2")[batch])
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase3_repair_ablation_summary"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    practical = {b: getattr(args, f"practical_{b}_manifest") for b in BATCHES}
    no_repair = {b: getattr(args, f"no_repair_{b}_manifest") for b in BATCHES}
    phase2 = {b: getattr(args, f"phase2_{b}_metrics") for b in BATCHES}
    try:
        summary = aggregate_repair_ablation(practical, no_repair, phase2, args.output_dir)
    except (RepairAblationError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Completed Phase 3 repair ablation aggregation cases={summary['scope']['case_count']} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
