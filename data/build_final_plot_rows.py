"""Build cross-cycle dissertation rows from validated frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = {
    "full_rebuild_hnsw": "Full rebuild",
    "indexed_cross_knn/bounded_n8_c2": "Bounded N8/C2",
    "indexed_cross_knn/unlimited_join1": "Unlimited JOIN1",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage21_rows(root: Path) -> list[dict[str, Any]]:
    compact = _load(root / "results/cycle21_50k_compact.json")
    cases = compact["method_results"]
    rows = []
    for method, display in METHODS.items():
        measurements = [case[method] for case in cases.values()]
        rows.append({"label": "Synthetic 50k", "method": display, "recall": float(np.mean([row["recall_at_10"] for row in measurements])), "latency_ms": float(np.mean([row["average_query_ms"] for row in measurements])), "integration_seconds": float(np.median([row["integration_seconds_median"] for row in measurements]))})
    return rows


def _cycle24_rows(root: Path) -> list[dict[str, Any]]:
    compact = _load(root / "results/cycle24_formal.json")
    if compact.get("status") != "complete" or not compact["validation"]["valid"]:
        raise ValueError("Cycle 24 artifact is not validated and complete.")
    method_names = {"full_rebuild": "Full rebuild", "bounded_n8_c2": "Bounded N8/C2", "join1": "Unlimited JOIN1"}
    aggregates = {row["method"]: row for row in compact["aggregates"]}
    rows = []
    for method, display in method_names.items():
        summary = aggregates[method]
        rows.append({"label": "Sequential 15k", "method": display, "recall": summary["mean_recall_at_10"], "latency_ms": summary["mean_query_ms"], "integration_seconds": summary["median_integration_seconds"]})
    return rows


def _cycle25_rows(root: Path) -> list[dict[str, Any]]:
    compact = _load(root / "results/cycle25_formal.json")
    if compact.get("status") != "complete" or not compact["validation"]["valid"]:
        raise ValueError("Cycle 25 artifact is not validated and complete.")
    method_names = {"full_rebuild": "Full rebuild", "bounded_n8_c2": "Bounded N8/C2", "join1": "Unlimited JOIN1"}
    aggregates = {row["method"]: row for row in compact["aggregates"]}
    return [{"label": "GloVe 25k", "method": display, "recall": aggregates[method]["mean_recall_at_10"], "latency_ms": aggregates[method]["mean_query_ms"], "integration_seconds": aggregates[method]["median_integration_seconds"]} for method, display in method_names.items()]


def build(repo_root: Path, output: Path) -> list[dict[str, Any]]:
    rows = [*_stage21_rows(repo_root), *_cycle24_rows(repo_root), *_cycle25_rows(repo_root)]
    payload = {"schema_version": "metricdb.final-plot-rows.v1", "sources": ["results/cycle21_50k_compact.json", "results/cycle24_formal.json", "results/cycle25_formal.json"], "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/final_plot_rows.json"))
    args = parser.parse_args()
    rows = build(args.repo_root, args.output)
    print(json.dumps({"row_count": len(rows)}, indent=2))
