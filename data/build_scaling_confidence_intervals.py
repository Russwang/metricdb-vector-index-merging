"""Build paired bootstrap intervals for the frozen 10k/25k/50k evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.phase4_glove import sha256_file
from data.phase4_statistics import paired_bootstrap


def _method_values(method: dict[str, Any]) -> tuple[float, float]:
    if "recall_at_10" in method:
        return float(method["recall_at_10"]), float(method["integration_seconds_median"])
    return float(method["quality"]["overall"]["recall_at_10"]), float(method["cost"]["integration_seconds_median"])


def build(config_path: Path, output_path: Path, *, repo_root: Path = Path(".")) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scales = []
    errors: list[str] = []
    for scale in config["scales"]:
        manifest_path = repo_root / scale["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = manifest.get("method_results", {})
        expected_methods = {config["baseline"], *config["candidates"], *scale["comparisons"]}
        rows = []
        for pair_key, methods in sorted(cases.items()):
            missing = expected_methods - set(methods)
            if missing:
                errors.append(f"{scale['label']} {pair_key} missing methods: {sorted(missing)}")
                continue
            for method in expected_methods:
                recall, integration = _method_values(methods[method])
                rows.append({"pair_key": pair_key, "method": method, "recall": recall, "integration": integration})
        pair_keys = sorted(cases)
        if len(pair_keys) != 6:
            errors.append(f"{scale['label']} expected 6 seed×condition pairs, found {len(pair_keys)}")
        comparisons = []
        for candidate in config["candidates"]:
            for baseline in scale["comparisons"]:
                comparisons.append({
                    "candidate": candidate,
                    "baseline": baseline,
                    "recall_delta": paired_bootstrap(rows, baseline=baseline, method=candidate, value_key="recall", statistic="difference", samples=int(config["bootstrap_samples"]), seed=int(config["bootstrap_seed"])),
                    "integration_ratio": paired_bootstrap(rows, baseline=baseline, method=candidate, value_key="integration", statistic="log_ratio", samples=int(config["bootstrap_samples"]), seed=int(config["bootstrap_seed"])),
                })
        scales.append({
            "label": scale["label"],
            "source_scale": int(scale["source_scale"]),
            "evaluated_pair_size": int(scale["evaluated_pair_size"]),
            "repair_input": scale["repair_input"],
            "source_manifest": str(scale["manifest"]),
            "source_manifest_sha256": sha256_file(manifest_path),
            "pair_keys": pair_keys,
            "pair_count": len(pair_keys),
            "comparisons": comparisons,
        })
    result = {
        "schema_version": "metricdb.scaling-confidence-intervals.v1",
        "status": "complete" if not errors else "invalid",
        "config": str(config_path),
        "pairing_unit": config["pairing_unit"],
        "bootstrap_samples": int(config["bootstrap_samples"]),
        "bootstrap_seed": int(config["bootstrap_seed"]),
        "scope_caveat": config["scope_caveat"],
        "validation": {"valid": not errors, "errors": errors, "scale_count": len(scales), "pair_count_per_scale": [row["pair_count"] for row in scales]},
        "scales": scales,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("Scaling confidence interval validation failed: " + "; ".join(errors))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_scaling_statistics.json"))
    parser.add_argument("--output", type=Path, default=Path("results/scaling_confidence_intervals.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = build(args.config, args.output, repo_root=args.repo_root)
    print(json.dumps(result["validation"], indent=2, sort_keys=True))
