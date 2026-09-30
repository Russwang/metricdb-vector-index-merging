"""Export the release-sized, self-contained summary of the formal 50k run.

The full Stage 21 output contains indices, vectors, and per-query arrays that
are intentionally not versioned.  This exporter preserves the six paired
observations needed for the dissertation tables and bootstrap intervals,
together with checksums of the immutable full-run sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


COMPACT_SCHEMA = "metricdb.stage21-50k-compact.v1"
REQUIRED_METHODS = {
    "full_rebuild_hnsw",
    "indexed_cross_knn/random_direct",
    "indexed_cross_knn/bounded_n4_c2",
    "indexed_cross_knn/bounded_n8_c2",
    "indexed_cross_knn/unlimited_join1",
}


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_metrics(result: dict[str, Any]) -> dict[str, float]:
    return {
        "recall_at_10": float(result["quality"]["overall"]["recall_at_10"]),
        "integration_seconds_median": float(result["cost"]["integration_seconds_median"]),
        "average_query_ms": float(result["cost"]["average_query_ms"]),
    }


def build(
    manifest_path: Path,
    summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build a minimal, independent representation of Stage 21 measurements."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    compact_cases: dict[str, dict[str, dict[str, float]]] = {}
    for pair_key, values in sorted(manifest["method_results"].items()):
        case = {
            method: _compact_metrics(values[method])
            for method in REQUIRED_METHODS
        }
        for direction, result in values["incremental_insertion"]["directions"].items():
            case[f"incremental_insertion/{direction}"] = _compact_metrics(result)
        compact_cases[pair_key] = case

    result = {
        "schema_version": COMPACT_SCHEMA,
        "source": {
            "full_manifest": str(manifest_path),
            "full_manifest_sha256": _checksum(manifest_path),
            "aggregate_summary": str(summary_path),
            "aggregate_summary_sha256": _checksum(summary_path),
        },
        "profile": manifest["profile"],
        "result_scope": manifest["result_scope"],
        "evaluated_pair_size": int(next(iter(manifest["method_results"].values()))["full_rebuild_hnsw"]["corpus_size"]),
        "method_results": compact_cases,
        "stage21_gate": summary["stage21_gate"],
    }
    validation = validate(result)
    if not validation["valid"]:
        raise ValueError("Invalid Stage 21 compact result: " + "; ".join(validation["errors"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema_version") != COMPACT_SCHEMA:
        errors.append(f"schema_version must be {COMPACT_SCHEMA!r}")
    source = payload.get("source", {})
    for field in ("full_manifest", "full_manifest_sha256", "aggregate_summary", "aggregate_summary_sha256"):
        if not isinstance(source.get(field), str) or not source[field]:
            errors.append(f"source.{field} must be a non-empty string")
    cases = payload.get("method_results")
    if not isinstance(cases, dict) or len(cases) != 6:
        errors.append("method_results must contain exactly six seed×condition pairs")
        return {"valid": not errors, "errors": errors, "pair_count": 0}
    for pair_key, methods in cases.items():
        missing = REQUIRED_METHODS - set(methods)
        if missing:
            errors.append(f"{pair_key} missing methods: {sorted(missing)}")
        for method, metrics in methods.items():
            for field in ("recall_at_10", "integration_seconds_median", "average_query_ms"):
                value = metrics.get(field) if isinstance(metrics, dict) else None
                if not isinstance(value, (int, float)):
                    errors.append(f"{pair_key} {method} has invalid {field}")
    return {"valid": not errors, "errors": errors, "pair_count": len(cases)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cycle21_50k_compact.json"))
    args = parser.parse_args()
    payload = build(args.manifest, args.summary, args.output)
    print(json.dumps(validate(payload), indent=2, sort_keys=True))
