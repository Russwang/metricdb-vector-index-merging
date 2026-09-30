"""Generate the dissertation Cycle 25 table from the validated compact artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "full_rebuild": "Full rebuild",
    "incremental_A_to_B": "Incremental A then B",
    "incremental_B_to_A": "Incremental B then A",
    "random_direct": "Random direct",
    "bounded_n8_c2": "Bounded \\method{N8/C2}",
    "join1": "Unlimited \\method{JOIN1}",
}


def generate(compact_path: Path, output_path: Path) -> None:
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    if compact.get("status") != "complete" or not compact.get("validation", {}).get("valid"):
        raise ValueError("Cycle 25 compact artifact is not validated and complete.")
    intervals = {row["method"]: row for row in compact["confidence_intervals"]}
    lines = [
        "% Generated from results/cycle25_formal.json; do not edit.",
        "\\begin{tabular}{lrrrr}",
        "Method & Recall@10 & Cross-subset Recall@10 & $\\Delta$ Recall@10 [95\\% CI] & Integration (s) \\\\",
        "\\hline",
    ]
    for row in compact["aggregates"]:
        method = row["method"]
        if method == "full_rebuild":
            delta = "reference"
        else:
            ci = intervals[method]["recall_delta_vs_full_rebuild"]
            delta = f"{ci['estimate']:+.4f} [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]"
        lines.append(
            f"{LABELS[method]} & {row['mean_recall_at_10']:.4f} & "
            f"{row['mean_cross_subset_recall_at_10']:.4f} & {delta} & "
            f"{row['median_integration_seconds']:.3f} \\\\"
        )
    lines.extend(["\\end{tabular}", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", type=Path, default=Path("results/cycle25_formal.json"))
    parser.add_argument("--output", type=Path, default=Path("../overleaf_metricdb/generated/cycle25_results_table.tex"))
    args = parser.parse_args()
    generate(args.compact, args.output)
