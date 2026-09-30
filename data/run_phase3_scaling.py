"""Run isolated Stage 17 topology-reuse scaling experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase2_common import save_json
from data.run_phase2 import run_complete_phase2
from data.run_phase3 import run_complete_phase3


def _write(path: Path, value: dict[str, Any]) -> None:
    save_json(path, value)


def _phase1_config(template: Path, size: int, seed: int) -> dict[str, Any]:
    with template.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["data"]["num_samples"] = int(size)
    config["random_seed"] = int(seed)
    config.setdefault("stages", {})
    config["stages"]["run_tests"] = False
    config["stages"]["backend_probe"] = False
    config["stages"]["brute_force_baseline"] = False
    config["stages"]["hnsw_rebuild_baseline"] = False
    config["description"] = f"Stage 17 isolated scaling input, N={size}, seed={seed}."
    return config


def _phase2_config(phase1_config: Path, seed: int) -> dict[str, Any]:
    return {
        "profile": "formal_experiment",
        "description": "Stage 17 no-repair pairwise input generation.",
        "phase1": {
            "transformation_dir": "unused_in_multiseed",
            "subset_dir": "unused_in_multiseed",
            "source_config": str(phase1_config),
        },
        "subsets": ["D_A", "D_B"],
        "transformations": {"exact": ["rotation"], "include_general_linear_stress": False},
        "noise": {"families": ["gaussian"], "levels": [0.1]},
        "graph": {
            "kinds": ["raw_level0"],
            "variants": ["no_repair"],
            "degree_budget_policy": "original_degree",
            "candidate_search_k": 64,
            "force_weak_connectivity": False,
        },
        "search": {"m": 16, "ef_construction": 200, "ef_search": 64, "evaluation_k": [1, 10, 50], "brute_force_batch_size": 256},
        "queries": {"corpus_point_count_per_subset": 256, "held_out_count_per_subset": 256},
        "seeds": {"data": [int(seed)], "transformation": 142, "noise": 242, "query": 342},
        "stages": {"run_tests": False},
    }


def _phase3_config(
    scale_root: Path,
    seed: int,
    condition: str,
    include_exact_sampled: bool,
) -> dict[str, Any]:
    methods = [
        "bruteforce_ground_truth",
        "full_rebuild_hnsw",
        "incremental_insertion",
        "indexed_cross_knn",
    ]
    if include_exact_sampled:
        methods.insert(3, "sampled_cross_knn")
    return {
        "profile": "formal_experiment",
        "description": "Stage 17 isolated scaling comparison.",
        "phase2": {
            "compiled_graph_root": str(scale_root / "phase2_compiled_graphs"),
            "repaired_graph_root": str(scale_root / "phase2_repaired_graphs"),
            "manifest": str(scale_root / "phase2_run_manifest.json"),
            "recommended_pairwise_config": "generated_in_scale_root",
        },
        "subsets": {"pair": ["D_A", "D_B"]},
        "input": {
            "data_seeds": [int(seed)],
            "conditions": [condition],
            "graph_kind": "raw_level0",
            "within_subset_variant": "no_repair",
            "require_raw_no_repair_graph": True,
        },
        "methods": {"enabled": methods, "first_round": methods, "second_round_planned": []},
        "search": {"m": 16, "ef_construction": 200, "ef_search": 64, "evaluation_k": [1, 10, 50], "brute_force_batch_size": 256},
        "cross_links": {
            "sample_ratios": [0.05], "cross_k": [4], "smoke_default": {"sample_ratio": 0.05, "cross_k": 4},
            "bridge_budget": {"mode": "sqrt", "sqrt_multiplier": 2.0},
            "candidate_k": 64, "candidate_ef_search": 128, "cross_degree_cap": 4, "random_seed": 442,
        },
        "indexed_sweep": {
            "enabled": True,
            "variants": [
                {"name": "random_direct", "bridge_selection": "random", "propagation_hops": 0, "cross_degree_cap": 4},
                {"name": "high_degree_join1", "bridge_selection": "high_degree", "propagation_hops": 1, "cross_degree_cap": 4},
            ],
        },
        "metrics": {"latency_warmup_queries": 20, "latency_repetitions": 3},
        "output": {
            "root": str(scale_root / "phase3" / condition),
            "manifest": str(scale_root / "phase3" / condition / "manifest.json"),
        },
        "stages": {"run_tests": False, "readiness_check_only": False, "fail_on_missing_inputs": True},
    }


def run_scale(
    plan: dict[str, Any],
    size: int,
    conditions: list[str],
    skip_phase2: bool,
) -> dict[str, Any]:
    root = Path(plan["output_root"]) / f"scale_{int(size)}"
    root.mkdir(parents=True, exist_ok=True)
    seed = int(plan["seed"])
    phase1_path = root / "configs" / "phase1.json"
    phase2_path = root / "configs" / "phase2.json"
    _write(phase1_path, _phase1_config(Path(plan["phase1_template"]), size, seed))
    _write(phase2_path, _phase2_config(phase1_path, seed))
    phase2 = (
        run_complete_phase2(phase2_path, root, clean=True)
        if not skip_phase2
        else None
    )
    phase3_manifests = {}
    for condition in conditions:
        phase3_path = root / "configs" / f"phase3_{condition}.json"
        _write(
            phase3_path,
            _phase3_config(
                root,
                seed,
                condition,
                int(size) <= int(plan["exact_sampled_max_size"]),
            ),
        )
        phase3_manifests[condition] = run_complete_phase3(phase3_path)
    summary = {
        "stage": "phase3_stage17_scaling",
        "size": int(size),
        "seed": seed,
        "exact_sampled_included": int(size) <= int(plan["exact_sampled_max_size"]),
        "phase2_manifest": str(root / "phase2_run_manifest.json"),
        "phase3_manifests": {
            condition: str(root / "phase3" / condition / "manifest.json")
            for condition in conditions
        },
        "phase2_runtime_seconds": (
            phase2["runtime"]["duration_seconds"] if phase2 is not None else None
        ),
        "phase3_runtime_seconds": {
            condition: manifest["runtime"]["duration_seconds"]
            for condition, manifest in phase3_manifests.items()
        },
        "method_results": {
            condition: manifest["method_results"]
            for condition, manifest in phase3_manifests.items()
        },
    }
    _write(root / "scaling_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/phase3_scaling_plan.json"))
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--skip-phase2", action="store_true")
    args = parser.parse_args()
    with args.plan.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    sizes = args.sizes or [int(value) for value in plan["sizes"]]
    conditions = args.conditions or [str(value) for value in plan["conditions"]]
    for size in sizes:
        summary = run_scale(plan, int(size), conditions, args.skip_phase2)
        print(
            f"Completed Stage 17 scale={size} phase2={summary['phase2_runtime_seconds']} "
            f"conditions={','.join(conditions)}"
        )


if __name__ == "__main__":
    main()
