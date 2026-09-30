"""Deterministic acceptance run for all three sequential arrival orders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.phase4_sequential import sequential_bounded_merge


def _ring(ids: np.ndarray, subset: str) -> dict:
    neighbors = {int(node): [int(ids[(row + 1) % len(ids)])] for row, node in enumerate(ids)}
    return {"node_global_ids": ids, "neighbors": neighbors, "edge_distances": {node: [1.0] for node in neighbors}, "source_subset": subset, "original_entry_point": int(ids[0]), "node_levels": np.ones(len(ids), dtype=np.int64), "original_degrees": np.ones(len(ids), dtype=np.int64), "degree_budget": np.ones(len(ids), dtype=np.int64)}


def run(output: Path) -> dict:
    rng = np.random.default_rng(24)
    inputs = {}
    for offset, subset in enumerate(("D_A", "D_B", "D_C")):
        ids = np.arange(offset * 100, offset * 100 + 20, dtype=np.int64)
        inputs[subset] = (_ring(ids, subset), rng.normal(loc=offset * 0.1, size=(20, 8)).astype(np.float32))
    cases = []
    for order in (("D_A", "D_B", "D_C"), ("D_A", "D_C", "D_B"), ("D_B", "D_C", "D_A")):
        current, vectors = inputs[order[0]]
        first, first_cost = sequential_bounded_merge(current, inputs[order[1]][0], vectors, inputs[order[1]][1], new_subset=order[1])
        union_vectors = np.vstack([vectors, inputs[order[1]][1]])
        final, second_cost = sequential_bounded_merge(first, inputs[order[2]][0], union_vectors, inputs[order[2]][1], new_subset=order[2])
        cases.append({"order": order, "first": first_cost, "second": second_cost, "unique_ids": len(np.unique(final["node_global_ids"])) == len(final["node_global_ids"])})
    result = {"schema_version": "metricdb.cycle24-smoke.v1", "seed": 24, "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/cycle24_smoke.json"))
    args = parser.parse_args(); run(args.output)
