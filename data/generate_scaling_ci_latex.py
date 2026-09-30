"""Generate the dissertation scaling-CI table from its checked artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate(source: Path, output: Path) -> None:
    artifact = json.loads(source.read_text(encoding="utf-8"))
    if artifact.get("status") != "complete" or not artifact.get("validation", {}).get("valid"):
        raise ValueError("Scaling confidence-interval artifact is not valid and complete.")
    lines = [
        "% Generated from results/scaling_confidence_intervals.json; do not edit.",
        "\\begin{tabular}{llrr}",
        "Scale & Candidate & $\\Delta$ Recall@10 vs random [95\\% CI] & Integration ratio [95\\% CI] \\\\",
        "\\hline",
    ]
    labels = {
        "indexed_cross_knn/bounded_n8_c2": "Bounded \\method{N8/C2}",
        "indexed_cross_knn/unlimited_join1": "Unlimited \\method{JOIN1}",
    }
    for scale in artifact["scales"]:
        for row in scale["comparisons"]:
            if row["baseline"] != "indexed_cross_knn/random_direct":
                continue
            recall = row["recall_delta"]
            ratio = row["integration_ratio"]
            lines.append(
                f"{scale['label']} & {labels[row['candidate']]} & "
                f"{recall['estimate']:+.4f} [{recall['ci95'][0]:+.4f}, {recall['ci95'][1]:+.4f}] & "
                f"{ratio['estimate']:.3f} [{ratio['ci95'][0]:.3f}, {ratio['ci95'][1]:.3f}] \\\\"
            )
    lines.extend(["\\end{tabular}", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("results/scaling_confidence_intervals.json"))
    parser.add_argument("--output", type=Path, default=Path("../overleaf_metricdb/generated/scaling_ci_table.tex"))
    args = parser.parse_args()
    generate(args.source, args.output)
