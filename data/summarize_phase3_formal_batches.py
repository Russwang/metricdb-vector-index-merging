"""Validate and aggregate Phase 3 first-round formal batch manifests."""

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


BATCH_ORDER = ("exact", "gaussian", "laplacian")
REQUIRED_METHODS = {
    "bruteforce_ground_truth",
    "full_rebuild_hnsw",
    "dual_index",
    "no_cross_merged",
    "random_cross_links",
    "sampled_cross_knn",
    "incremental_insertion",
}
ANN_METHODS = (
    "full_rebuild_hnsw",
    "dual_index",
    "no_cross_merged",
    "random_cross_links",
    "sampled_cross_knn",
    "incremental_insertion/A<-B",
    "incremental_insertion/B<-A",
)
CONNECTIVITY_METHODS = (
    "full_rebuild_hnsw",
    "dual_index",
    "no_cross_merged",
    "random_cross_links",
    "sampled_cross_knn",
    "incremental_insertion/A<-B",
    "incremental_insertion/B<-A",
)
COST_METHODS = ("bruteforce_ground_truth",) + ANN_METHODS


class ManifestValidationError(ValueError):
    """Raised when a formal manifest cannot be safely aggregated."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestValidationError(f"Manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestValidationError(f"Cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestValidationError(f"Manifest root must be an object: {path}")
    return value


def _case_parts(key: str) -> tuple[int, str]:
    try:
        seed_text, condition = key.split("/", 1)
        if not seed_text.startswith("seed_") or not condition:
            raise ValueError
        return int(seed_text.removeprefix("seed_")), condition
    except ValueError as error:
        raise ManifestValidationError(f"Invalid method result key: {key!r}") from error


def _method_items(result: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for method in sorted(REQUIRED_METHODS - {"incremental_insertion"}):
        yield method, "", result[method]
    directions = result["incremental_insertion"].get("directions")
    if not isinstance(directions, dict) or set(directions) != {"A<-B", "B<-A"}:
        raise ManifestValidationError(
            "incremental_insertion must contain separate A<-B and B<-A directions"
        )
    for direction in ("A<-B", "B<-A"):
        yield "incremental_insertion", direction, directions[direction]


def _display_method(method: str, direction: str) -> str:
    return f"{method}/{direction}" if direction else method


def _required_number(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ManifestValidationError(f"Missing numeric {context}.{key}")
    return float(value)


def _optional_number(mapping: dict[str, Any], key: str) -> float | int | None:
    value = mapping.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _quality_values(item: dict[str, Any], context: str) -> dict[str, float]:
    quality = item.get("quality")
    if not isinstance(quality, dict):
        raise ManifestValidationError(f"Missing quality object for {context}")
    overall = quality.get("overall", {})
    source_target = quality.get("source_target", {}).get("recall_at_10", {})
    return {
        "recall_at_1": _required_number(overall, "recall_at_1", context),
        "recall_at_10": _required_number(overall, "recall_at_10", context),
        "recall_at_50": _required_number(overall, "recall_at_50", context),
        "cross_subset_recall_at_10": _required_number(
            source_target, "cross_subset", context
        ),
    }


def _statistics(values: Iterable[float | int]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def _optional_statistics(values: Iterable[float | int | None]) -> dict[str, Any] | None:
    present = [value for value in values if value is not None]
    return _statistics(present) if present else None


def _validate_manifest(
    batch: str,
    path: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    readiness = manifest.get("readiness")
    if not isinstance(readiness, dict) or readiness.get("ready") is not True:
        raise ManifestValidationError(f"{batch} manifest readiness.ready is not true: {path}")
    if readiness.get("missing_count") != 0 or readiness.get("missing"):
        raise ManifestValidationError(f"{batch} manifest reports missing inputs: {path}")
    method_results = manifest.get("method_results")
    if not isinstance(method_results, dict) or not method_results:
        raise ManifestValidationError(f"{batch} manifest has no method_results: {path}")

    records: list[dict[str, Any]] = []
    for key, result in sorted(method_results.items()):
        if not isinstance(result, dict) or set(result) != REQUIRED_METHODS:
            missing = sorted(REQUIRED_METHODS - set(result or {}))
            extra = sorted(set(result or {}) - REQUIRED_METHODS)
            raise ManifestValidationError(
                f"Incomplete methods for {batch}/{key}; missing={missing}, extra={extra}"
            )
        seed, condition = _case_parts(key)
        for method, direction, item in _method_items(result):
            if not isinstance(item, dict):
                raise ManifestValidationError(
                    f"Method result must be an object: {batch}/{key}/{method}/{direction}"
                )
            records.append(
                {
                    "batch": batch,
                    "seed": seed,
                    "condition": condition,
                    "method": method,
                    "direction": direction,
                    "display_method": _display_method(method, direction),
                    "item": item,
                    "case_key": key,
                    "source_manifest": str(path),
                }
            )
    return records


def load_and_validate_manifests(
    manifest_paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(manifest_paths) != set(BATCH_ORDER):
        raise ManifestValidationError(
            f"Expected manifests for {list(BATCH_ORDER)}, got {sorted(manifest_paths)}"
        )
    records: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    seen_cases: set[tuple[int, str]] = set()
    case_counts: dict[str, int] = {}
    for batch in BATCH_ORDER:
        path = Path(manifest_paths[batch])
        manifest = _read_json(path)
        batch_records = _validate_manifest(batch, path, manifest)
        cases = {(record["seed"], record["condition"]) for record in batch_records}
        overlap = seen_cases & cases
        if overlap:
            raise ManifestValidationError(f"Cases occur in multiple batches: {sorted(overlap)}")
        seen_cases.update(cases)
        case_counts[batch] = len(cases)
        records.extend(batch_records)
        manifests[batch] = manifest

    graph_sources: set[str] = {
        str(manifest["readiness"].get("graph_kind")) for manifest in manifests.values()
    }
    variants: set[str] = {
        str(manifest["readiness"].get("within_subset_variant"))
        for manifest in manifests.values()
    }
    subset_pairs: set[tuple[str, ...]] = {
        tuple(sorted(manifest["readiness"].get("subsets", [])))
        for manifest in manifests.values()
    }
    parameter_pairs: set[tuple[float, int]] = set()
    for record in records:
        item = record["item"]
        input_data = item.get("input", {})
        if record["method"] == "bruteforce_ground_truth":
            continue
        if input_data.get("graph_kind") is not None:
            graph_sources.add(str(input_data["graph_kind"]))
        if input_data.get("within_subset_variant") is not None:
            variants.add(str(input_data["within_subset_variant"]))
        if input_data.get("subsets") is not None:
            subset_pairs.add(tuple(sorted(input_data["subsets"])))
        if record["method"] in {"random_cross_links", "sampled_cross_knn"}:
            graph = item.get("graph", {})
            sample_ratio = _required_number(
                graph, "sample_ratio", f"{record['case_key']}/{record['method']}.graph"
            )
            cross_k = int(
                _required_number(
                    graph, "cross_k", f"{record['case_key']}/{record['method']}.graph"
                )
            )
            parameter_pairs.add((sample_ratio, cross_k))
    if graph_sources != {"raw_level0"}:
        raise ManifestValidationError(f"Graph sources are inconsistent: {sorted(graph_sources)}")
    if variants != {"practical_repair"}:
        raise ManifestValidationError(f"Within-subset variants are inconsistent: {sorted(variants)}")
    if subset_pairs != {("D_A", "D_B")}:
        raise ManifestValidationError(f"Subset pairs are inconsistent: {sorted(subset_pairs)}")
    if len(parameter_pairs) != 1:
        raise ManifestValidationError(
            f"sampled/random parameters are inconsistent: {sorted(parameter_pairs)}"
        )
    sample_ratio, cross_k = next(iter(parameter_pairs))
    metadata = {
        "manifests": manifests,
        "case_counts_by_batch": case_counts,
        "case_count": len(seen_cases),
        "seed_list": sorted({seed for seed, _ in seen_cases}),
        "condition_groups": {
            batch: sorted(
                {record["condition"] for record in records if record["batch"] == batch}
            )
            for batch in BATCH_ORDER
        },
        "graph_source": "raw_level0",
        "within_subset_variant": "practical_repair",
        "subset_pair": ["D_A", "D_B"],
        "fixed_parameters": {"sample_ratio": sample_ratio, "cross_k": cross_k},
    }
    return records, metadata


def _group_records(
    records: list[dict[str, Any]], methods: Iterable[str]
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    selected = set(methods)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["display_method"] in selected:
            grouped[(record["batch"], record["condition"], record["display_method"])].append(
                record
            )
    return grouped


def build_recall_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for (batch, condition, display_method), group in sorted(
        _group_records(records, ANN_METHODS).items()
    ):
        qualities = [
            _quality_values(record["item"], f"{record['case_key']}/{display_method}")
            for record in group
        ]
        metrics = {
            key: _statistics([quality[key] for quality in qualities])
            for key in (
                "recall_at_1",
                "recall_at_10",
                "recall_at_50",
                "cross_subset_recall_at_10",
            )
        }
        method, _, direction = display_method.partition("/")
        row = {
            "batch": batch,
            "condition": condition,
            "method": method,
            "direction": direction,
            "seed_count": len(group),
            "seed_list": sorted(record["seed"] for record in group),
            "recall10_mean": metrics["recall_at_10"]["mean"],
            "recall10_std": metrics["recall_at_10"]["std"],
            "recall10_min": metrics["recall_at_10"]["min"],
            "recall10_max": metrics["recall_at_10"]["max"],
            "cross_subset_recall10_mean": metrics["cross_subset_recall_at_10"]["mean"],
            "cross_subset_recall10_std": metrics["cross_subset_recall_at_10"]["std"],
        }
        rows.append(row)
        nested[batch][condition][display_method] = {
            "case_count": len(group),
            "seed_list": row["seed_list"],
            **metrics,
        }
    return rows, {batch: dict(conditions) for batch, conditions in nested.items()}


def _records_by_case(records: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(record["batch"], record["seed"], record["condition"])][
            record["display_method"]
        ] = record
    return grouped


def build_improvement_rows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {
        "case_count": 0,
        "sampled_beats_random": 0,
        "sampled_beats_no_cross": 0,
        "sampled_beats_full_rebuild": 0,
        "sampled_connected_and_reachable": 0,
        "no_cross_has_two_weak_components": 0,
    }
    for (batch, seed, condition), case in sorted(_records_by_case(records).items()):
        sampled = case["sampled_cross_knn"]["item"]
        random = case["random_cross_links"]["item"]
        no_cross = case["no_cross_merged"]["item"]
        full = case["full_rebuild_hnsw"]["item"]
        sampled_r10 = _quality_values(sampled, "sampled_cross_knn")["recall_at_10"]
        random_r10 = _quality_values(random, "random_cross_links")["recall_at_10"]
        no_cross_r10 = _quality_values(no_cross, "no_cross_merged")["recall_at_10"]
        full_r10 = _quality_values(full, "full_rebuild_hnsw")["recall_at_10"]
        sampled_stats = sampled.get("graph_statistics", {})
        no_cross_stats = no_cross.get("graph_statistics", {})
        sampled_cost = sampled.get("cost", {})
        sampled_graph = sampled.get("graph", {})
        row = {
            "batch": batch,
            "seed": seed,
            "condition": condition,
            "sampled_recall10": sampled_r10,
            "random_recall10": random_r10,
            "no_cross_recall10": no_cross_r10,
            "full_rebuild_recall10": full_r10,
            "sampled_minus_random": sampled_r10 - random_r10,
            "sampled_minus_no_cross": sampled_r10 - no_cross_r10,
            "sampled_minus_full_rebuild": sampled_r10 - full_r10,
            "sampled_cross_subset_recall10": _quality_values(
                sampled, "sampled_cross_knn"
            )["cross_subset_recall_at_10"],
            "sampled_integration_seconds": _required_number(
                sampled_cost, "integration_seconds", "sampled_cross_knn.cost"
            ),
            "sampled_candidate_distance_count": int(
                _required_number(
                    sampled_cost,
                    "cross_candidate_distance_count",
                    "sampled_cross_knn.cost",
                )
            ),
            "sampled_scan_work_ratio": _required_number(
                sampled_cost, "cross_scan_work_ratio", "sampled_cross_knn.cost"
            ),
            "sampled_added_cross_link_count": int(
                _required_number(
                    sampled_graph, "added_cross_link_count", "sampled_cross_knn.graph"
                )
            ),
        }
        rows.append(row)
        counts["case_count"] += 1
        counts["sampled_beats_random"] += sampled_r10 > random_r10
        counts["sampled_beats_no_cross"] += sampled_r10 > no_cross_r10
        counts["sampled_beats_full_rebuild"] += sampled_r10 > full_r10
        counts["sampled_connected_and_reachable"] += (
            sampled_stats.get("weak_connected_component_count") == 1
            and sampled_stats.get("directed_reachable_from_entry_ratio") == 1.0
        )
        counts["no_cross_has_two_weak_components"] += (
            no_cross_stats.get("weak_connected_component_count") == 2
        )
    return rows, counts


def build_connectivity_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (batch, condition, display_method), group in sorted(
        _group_records(records, CONNECTIVITY_METHODS).items()
    ):
        stats = [record["item"].get("graph_statistics") for record in group]
        present = [value for value in stats if isinstance(value, dict)]
        method, _, direction = display_method.partition("/")
        if not present:
            rows.append(
                {
                    "batch": batch,
                    "condition": condition,
                    "method": method,
                    "direction": direction,
                    "case_count": len(group),
                    "weak_component_count_mean": None,
                    "weak_component_count_min": None,
                    "weak_component_count_max": None,
                    "fully_connected_case_count": None,
                    "directed_reachable_ratio_mean": None,
                    "directed_reachable_ratio_min": None,
                    "directed_reachable_ratio_max": None,
                    "fully_reachable_case_count": None,
                    "cross_subset_edge_count_mean": None,
                    "added_cross_link_count_mean": None,
                    "availability_note": "not_reported_by_backend",
                }
            )
            continue
        components = [value["weak_connected_component_count"] for value in present]
        reachable = [value["directed_reachable_from_entry_ratio"] for value in present]
        cross_edges = [
            _optional_number(record["item"].get("graph", {}), "cross_subset_edge_count")
            for record in group
        ]
        added_edges = [
            _optional_number(record["item"].get("graph", {}), "added_cross_link_count")
            for record in group
        ]
        cross_summary = _optional_statistics(cross_edges)
        added_summary = _optional_statistics(added_edges)
        rows.append(
            {
                "batch": batch,
                "condition": condition,
                "method": method,
                "direction": direction,
                "case_count": len(group),
                "weak_component_count_mean": statistics.fmean(components),
                "weak_component_count_min": min(components),
                "weak_component_count_max": max(components),
                "fully_connected_case_count": sum(value == 1 for value in components),
                "directed_reachable_ratio_mean": statistics.fmean(reachable),
                "directed_reachable_ratio_min": min(reachable),
                "directed_reachable_ratio_max": max(reachable),
                "fully_reachable_case_count": sum(value == 1.0 for value in reachable),
                "cross_subset_edge_count_mean": cross_summary["mean"] if cross_summary else None,
                "added_cross_link_count_mean": added_summary["mean"] if added_summary else None,
                "availability_note": "",
            }
        )
    return rows


COST_FIELDS = (
    "integration_or_build_seconds",
    "graph_extraction_seconds",
    "auxiliary_total_integration_seconds",
    "base_build_seconds",
    "inserted_add_seconds",
    "total_search_seconds",
    "average_query_ms",
    "p95_query_ms",
    "processed_vector_count",
    "processed_vector_ratio",
    "candidate_distance_count",
    "scan_work_ratio",
    "rerank_distance_count",
    "endpoint_selection_distance_count",
    "edge_weight_distance_count",
    "edge_count",
    "graph_or_index_size_bytes",
)


def _cost_values(display_method: str, item: dict[str, Any]) -> tuple[dict[str, Any], str]:
    cost = item.get("cost", {})
    graph_stats = item.get("graph_statistics", {})
    values: dict[str, Any] = {field: None for field in COST_FIELDS}
    if display_method == "bruteforce_ground_truth":
        values["integration_or_build_seconds"] = _optional_number(cost, "total_seconds")
        values["candidate_distance_count"] = _optional_number(cost, "exact_distance_count")
        semantics = "exact reference distance evaluation; not an ANN integration method"
    elif display_method == "full_rebuild_hnsw":
        values["integration_or_build_seconds"] = _optional_number(cost, "build_seconds")
        values["graph_extraction_seconds"] = _optional_number(
            cost, "graph_extraction_seconds"
        )
        values["auxiliary_total_integration_seconds"] = _optional_number(
            cost, "total_integration_seconds"
        )
        values["processed_vector_count"] = _optional_number(item, "corpus_size")
        values["processed_vector_ratio"] = _optional_number(cost, "processed_vector_ratio")
        semantics = "native FAISS HNSW build cost; graph extraction excluded from primary build field"
    elif display_method == "dual_index":
        values["integration_or_build_seconds"] = _optional_number(cost, "integration_seconds")
        values["processed_vector_count"] = 0
        values["processed_vector_ratio"] = _optional_number(cost, "processed_vector_ratio")
        values["rerank_distance_count"] = _optional_number(cost, "rerank_distance_count")
        semantics = "no-build dual local-index search; integration is input selection, query/rerank is primary cost"
    elif display_method == "no_cross_merged":
        values["integration_or_build_seconds"] = _optional_number(cost, "integration_seconds")
        values["processed_vector_count"] = 0
        values["processed_vector_ratio"] = _optional_number(cost, "processed_vector_ratio")
        semantics = "external graph concatenation cost without cross-subset links"
    elif display_method == "random_cross_links":
        values["integration_or_build_seconds"] = _optional_number(cost, "integration_seconds")
        values["processed_vector_count"] = _optional_number(
            item.get("graph", {}), "sampled_bridge_node_count"
        )
        values["processed_vector_ratio"] = _optional_number(cost, "processed_vector_ratio")
        values["candidate_distance_count"] = _optional_number(cost, "endpoint_selection_distance_count")
        values["endpoint_selection_distance_count"] = _optional_number(
            cost, "endpoint_selection_distance_count"
        )
        values["edge_weight_distance_count"] = _optional_number(
            cost, "edge_weight_distance_count"
        )
        semantics = "external graph integration; candidate field is endpoint-selection distance work, edge-weight work is separate in manifests"
    elif display_method == "sampled_cross_knn":
        values["integration_or_build_seconds"] = _optional_number(cost, "integration_seconds")
        values["processed_vector_count"] = _optional_number(
            item.get("graph", {}), "sampled_bridge_node_count"
        )
        values["processed_vector_ratio"] = _optional_number(cost, "processed_vector_ratio")
        values["candidate_distance_count"] = _optional_number(cost, "cross_candidate_distance_count")
        values["scan_work_ratio"] = _optional_number(cost, "cross_scan_work_ratio")
        values["edge_weight_distance_count"] = _optional_number(
            cost, "edge_weight_distance_count"
        )
        semantics = "external graph integration plus sampled exact cross-candidate scans"
    else:
        values["integration_or_build_seconds"] = _optional_number(cost, "total_integration_seconds")
        values["base_build_seconds"] = _optional_number(cost, "base_index_build_seconds")
        values["inserted_add_seconds"] = _optional_number(cost, "inserted_add_seconds")
        values["processed_vector_count"] = _optional_number(cost, "processed_inserted_vector_count")
        values["processed_vector_ratio"] = _optional_number(cost, "processed_inserted_vector_ratio")
        semantics = "native FAISS base rebuild plus inserted-subset add; not physical loaded-base reuse"
    values["total_search_seconds"] = _optional_number(cost, "total_search_seconds")
    values["average_query_ms"] = _optional_number(cost, "average_query_ms")
    values["p95_query_ms"] = _optional_number(cost, "p95_query_ms")
    values["edge_count"] = _optional_number(graph_stats, "edge_count")
    values["graph_or_index_size_bytes"] = _optional_number(item, "index_size_bytes")
    return values, semantics


def build_cost_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (batch, condition, display_method), group in sorted(
        _group_records(records, COST_METHODS).items()
    ):
        extracted = [_cost_values(display_method, record["item"]) for record in group]
        method, _, direction = display_method.partition("/")
        row: dict[str, Any] = {
            "batch": batch,
            "condition": condition,
            "method": method,
            "direction": direction,
            "case_count": len(group),
            "measurement_semantics": extracted[0][1],
        }
        missing: list[str] = []
        for field in COST_FIELDS:
            summary = _optional_statistics(values[field] for values, _ in extracted)
            row[f"{field}_mean"] = summary["mean"] if summary else None
            row[f"{field}_std"] = summary["std"] if summary else None
            if summary is None:
                missing.append(field)
        row["not_reported_fields"] = ";".join(missing)
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


def _format_stat(summary: dict[str, Any]) -> str:
    return f"{summary['mean']:.5f} +/- {summary['std']:.5f}"


def _markdown_tables(
    recall_summary: dict[str, Any],
    win_counts: dict[str, int],
    metadata: dict[str, Any],
) -> str:
    labels = (
        ("no_cross_merged", "no-cross"),
        ("random_cross_links", "random"),
        ("sampled_cross_knn", "sampled"),
        ("full_rebuild_hnsw", "full rebuild"),
        ("dual_index", "dual-index"),
        ("incremental_insertion/A<-B", "incremental A<-B"),
        ("incremental_insertion/B<-A", "incremental B<-A"),
    )
    lines = [
        "# Phase 3 First-Round Formal Tables",
        "",
        "All values are mean +/- sample standard deviation across seeds (ddof=1).",
        "",
        "| Batch | Condition | " + " | ".join(label for _, label in labels) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for batch in BATCH_ORDER:
        for condition, methods in sorted(recall_summary[batch].items()):
            values = [_format_stat(methods[method]["recall_at_10"]) for method, _ in labels]
            lines.append(f"| {batch} | {condition} | " + " | ".join(values) + " |")
    total = win_counts["case_count"]
    lines.extend(
        [
            "",
            "## Consistency Counts",
            "",
            f"- sampled > random: {win_counts['sampled_beats_random']}/{total}",
            f"- sampled > no-cross: {win_counts['sampled_beats_no_cross']}/{total}",
            f"- sampled > full rebuild: {win_counts['sampled_beats_full_rebuild']}/{total}",
            f"- sampled connected and fully reachable: {win_counts['sampled_connected_and_reachable']}/{total}",
            f"- no-cross has two weak components: {win_counts['no_cross_has_two_weak_components']}/{total}",
            "",
            "## Fixed Parameters",
            "",
            f"- graph source: `{metadata['graph_source']}`",
            f"- within-subset variant: `{metadata['within_subset_variant']}`",
            f"- subset pair: `{' + '.join(metadata['subset_pair'])}`",
            f"- sample ratio: `{metadata['fixed_parameters']['sample_ratio']}`",
            f"- cross-k: `{metadata['fixed_parameters']['cross_k']}`",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_formal_batches(
    manifest_paths: dict[str, Path], output_dir: Path
) -> dict[str, Any]:
    records, metadata = load_and_validate_manifests(manifest_paths)
    recall_rows, recall_summary = build_recall_rows(records)
    improvement_rows, win_counts = build_improvement_rows(records)
    connectivity_rows = build_connectivity_rows(records)
    cost_rows = build_cost_rows(records)

    expected_counts = {
        "case_count": 55,
        "sampled_beats_random": 55,
        "sampled_beats_no_cross": 55,
        "sampled_beats_full_rebuild": 0,
        "sampled_connected_and_reachable": 55,
        "no_cross_has_two_weak_components": 55,
    }
    consistency_warnings = [
        f"Consistency check differs for {key}: expected {expected}, observed {win_counts.get(key, metadata.get(key))}"
        for key, expected in expected_counts.items()
        if (win_counts.get(key, metadata.get(key))) != expected
    ]
    warnings = [
        "Sample standard deviation uses ddof=1; a single observation is assigned std=0.",
        "External graph and native FAISS latency are not backend-equivalent.",
        "Incremental insertion rebuilds the native base HNSW; physical base-index loading is not measured.",
        "Missing optional cost/connectivity fields remain null or blank rather than being imputed as zero.",
        "P95 query latency is absent from the current manifests and is therefore left blank.",
        *consistency_warnings,
    ]
    summary = {
        "source_manifests": {batch: str(Path(manifest_paths[batch])) for batch in BATCH_ORDER},
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "graph_source": metadata["graph_source"],
        "within_subset_variant": metadata["within_subset_variant"],
        "subset_pair": metadata["subset_pair"],
        "fixed_sampled_random_parameters": metadata["fixed_parameters"],
        "seed_list": metadata["seed_list"],
        "condition_groups": metadata["condition_groups"],
        "aggregation_definition": {
            "unit": "seed x condition case",
            "mean": "arithmetic mean",
            "standard_deviation": "sample standard deviation (ddof=1)",
            "single_observation_standard_deviation": 0,
            "incremental_directions": "reported separately; never pooled or best-of-selected",
            "bruteforce_role": "exact quality/cost reference; excluded from sampled ANN win counts",
        },
        "per_condition_per_method": recall_summary,
        "global_validation_counts": {
            "case_count": metadata["case_count"],
            "case_counts_by_batch": metadata["case_counts_by_batch"],
            **win_counts,
        },
        "consistency_expectations": expected_counts,
        "schema_warnings": warnings,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase3_first_round_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "phase3_recall10_by_condition.csv", recall_rows)
    _write_csv(output_dir / "phase3_sampled_improvements.csv", improvement_rows)
    _write_csv(output_dir / "phase3_connectivity_summary.csv", connectivity_rows)
    _write_csv(output_dir / "phase3_cost_summary.csv", cost_rows)
    (output_dir / "phase3_first_round_tables.md").write_text(
        _markdown_tables(recall_summary, win_counts, metadata), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact-manifest",
        type=Path,
        default=Path("data/phase3_formal_batch_exact_manifest.json"),
    )
    parser.add_argument(
        "--gaussian-manifest",
        type=Path,
        default=Path("data/phase3_formal_batch_gaussian_manifest.json"),
    )
    parser.add_argument(
        "--laplacian-manifest",
        type=Path,
        default=Path("data/phase3_formal_batch_laplacian_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/phase3_formal_summary")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {
        "exact": args.exact_manifest,
        "gaussian": args.gaussian_manifest,
        "laplacian": args.laplacian_manifest,
    }
    try:
        summary = aggregate_formal_batches(paths, args.output_dir)
    except (ManifestValidationError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    counts = summary["global_validation_counts"]
    print(
        "Completed Phase 3 formal aggregation "
        f"cases={counts['case_count']} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
