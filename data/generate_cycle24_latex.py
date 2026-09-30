"""Generate the dissertation Cycle 24 table from the validated compact artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "full_rebuild": "Full rebuild",
    "sequential_incremental": "Sequential incremental",
    "random_direct": "Random direct",
    "bounded_n8_c2": "Bounded \\method{N8/C2}",
    "join1": "Unlimited \\method{JOIN1}",
}


def generate(source: Path, output: Path) -> None:
    artifact = json.loads(source.read_text(encoding="utf-8"))
    if artifact.get("status") != "complete" or not artifact.get("validation", {}).get("valid"):
        raise ValueError("Cycle 24 LaTeX generation requires a complete validated artifact")
    rows = {row["method"]: row for row in artifact["aggregates"]}
    if set(rows) != set(LABELS):
        raise ValueError("Cycle 24 aggregate method set is incomplete")
    lines = [
        "% Generated from metricdb/results/cycle24_formal.json; do not edit.",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Recall@10 & Cross-subset Recall@10 & Integration (s) \\\\",
        "\\midrule",
    ]
    for method in LABELS:
        row = rows[method]
        lines.append(
            f"{LABELS[method]} & {row['mean_recall_at_10']:.4f} & "
            f"{row['mean_cross_subset_recall_at_10']:.4f} & "
            f"{row['median_integration_seconds']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("results/cycle24_formal.json"))
    parser.add_argument("--output", type=Path, default=Path("../overleaf_metricdb/generated/cycle24_results_table.tex"))
    args = parser.parse_args()
    generate(args.source, args.output)
