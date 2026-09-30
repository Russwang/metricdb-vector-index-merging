"""Snapshot Phase 2 repair metrics before shared batch outputs are replaced."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase3_common import load_phase3_config, save_json


VARIANTS = ("practical_repair", "no_repair")
COST_FIELDS = (
    "repair_seconds",
    "candidate_generation_seconds",
    "distance_computation_count",
    "processed_node_ratio",
    "total_compile_and_repair_seconds",
    "vector_transformation_seconds",
    "graph_extraction_seconds",
)
GRAPH_FIELDS = (
    "retained_edge_ratio",
    "removed_edge_ratio",
    "added_edge_ratio",
    "reciprocal_edge_ratio",
    "connected_component_count",
    "directed_reachable_from_entry_ratio",
)
RECALL_FIELDS = ("recall_at_1", "recall_at_10", "recall_at_50")


class MetricsSnapshotError(ValueError):
    """Raised when expected Phase 2 metrics cannot be safely snapshotted."""


def _numeric(mapping: dict[str, Any], field: str, context: str) -> int | float:
    value = mapping.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MetricsSnapshotError(f"Missing numeric field {context}.{field}")
    return value


def collect_phase2_metrics(batch: str, phase3_config: Path) -> dict[str, Any]:
    config = load_phase3_config(phase3_config)
    repaired_root = Path(config["phase2"]["repaired_graph_root"])
    graph_kind = config["input"]["graph_kind"]
    records: list[dict[str, Any]] = []
    for seed in config["input"]["data_seeds"]:
        for condition in config["input"]["conditions"]:
            for subset in config["subsets"]["pair"]:
                for variant in VARIANTS:
                    path = (
                        repaired_root
                        / f"seed_{int(seed)}"
                        / condition
                        / subset
                        / graph_kind
                        / variant
                        / "metrics.json"
                    )
                    if not path.is_file():
                        raise MetricsSnapshotError(f"Missing metrics file: {path}")
                    try:
                        metrics = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as error:
                        raise MetricsSnapshotError(f"Cannot read {path}: {error}") from error
                    cost = metrics.get("cost", {})
                    diagnostics = metrics.get("graph_diagnostics", {})
                    held_out = metrics.get("quality", {}).get("held_out", {})
                    record = {
                        "batch": batch,
                        "seed": int(seed),
                        "condition": condition,
                        "subset": subset,
                        "variant": variant,
                        "source_metrics": str(path),
                        "cost": {
                            field: _numeric(cost, field, f"{path}.cost")
                            for field in COST_FIELDS
                        },
                        "graph_diagnostics": {
                            field: _numeric(diagnostics, field, f"{path}.graph_diagnostics")
                            for field in GRAPH_FIELDS
                        },
                        "local_held_out_recall": {
                            field: _numeric(held_out, field, f"{path}.quality.held_out")
                            for field in RECALL_FIELDS
                        },
                    }
                    records.append(record)
    expected = (
        len(config["input"]["data_seeds"])
        * len(config["input"]["conditions"])
        * len(config["subsets"]["pair"])
        * len(VARIANTS)
    )
    if len(records) != expected:
        raise MetricsSnapshotError(f"Expected {expected} records, collected {len(records)}")
    return {
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "batch": batch,
        "phase3_config": str(phase3_config),
        "graph_kind": graph_kind,
        "subsets": list(config["subsets"]["pair"]),
        "seeds": [int(seed) for seed in config["input"]["data_seeds"]],
        "conditions": list(config["input"]["conditions"]),
        "variants": list(VARIANTS),
        "record_count": len(records),
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, choices=("exact", "gaussian", "laplacian"))
    parser.add_argument("--phase3-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot = collect_phase2_metrics(args.batch, args.phase3_config)
        save_json(args.output, snapshot)
    except (MetricsSnapshotError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Collected Phase 2 repair metrics batch={args.batch} records={snapshot['record_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
