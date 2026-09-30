"""Run Phase 3 common readiness checks and shared manifest generation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase3_common import add_run_metadata, load_phase3_config, save_json, start_run
from data.phase3_baselines import (
    run_dual_index_baseline,
    run_bruteforce_ground_truth,
    run_full_rebuild_hnsw,
    run_incremental_insertion_baseline,
    run_indexed_cross_knn_baseline,
    run_no_cross_merged_baseline,
    run_random_cross_links_baseline,
    run_sampled_cross_knn_baseline,
    union_phase2_pair,
)
from data.phase3_io import check_pairwise_phase2_inputs, load_pairwise_phase2_inputs


IMPLEMENTED_METHODS = {
    "bruteforce_ground_truth",
    "dual_index",
    "full_rebuild_hnsw",
    "incremental_insertion",
    "indexed_cross_knn",
    "no_cross_merged",
    "random_cross_links",
    "sampled_cross_knn",
}


def _parameter_slug(value: float | int) -> str:
    text = f"{float(value):g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def _cross_link_parameter_grid(
    config: dict[str, Any],
    method: str,
) -> list[tuple[float, int]]:
    """Return one default pair or the configured validation sweep grid."""
    sweep = config.get("sweep", {})
    if not sweep.get("enabled", False) or method not in sweep.get("methods", []):
        default = config["cross_links"].get("smoke_default", {})
        sample_ratio = float(
            default.get("sample_ratio", config["cross_links"]["sample_ratios"][0])
        )
        cross_k = int(default.get("cross_k", config["cross_links"]["cross_k"][0]))
        return [(sample_ratio, cross_k)]
    return [
        (float(sample_ratio), int(cross_k))
        for sample_ratio in sweep["sample_ratios"]
        for cross_k in sweep["cross_k"]
    ]


def _cross_link_config_for(
    config: dict[str, Any],
    sample_ratio: float,
    cross_k: int,
) -> dict[str, Any]:
    """Inject one selected pair while preserving the configured allowed grid."""
    selected = dict(config["cross_links"])
    selected["smoke_default"] = {
        "sample_ratio": float(sample_ratio),
        "cross_k": int(cross_k),
    }
    return selected


def _sweep_result_key(method: str, sample_ratio: float, cross_k: int) -> str:
    return (
        f"{method}/sample_ratio_{_parameter_slug(sample_ratio)}"
        f"/cross_k_{int(cross_k)}"
    )


def _sweep_output_dir(
    base_dir: Path,
    sample_ratio: float,
    cross_k: int,
    include_parameters: bool,
) -> Path:
    if not include_parameters:
        return base_dir
    return (
        base_dir
        / f"sample_ratio_{_parameter_slug(sample_ratio)}"
        / f"cross_k_{int(cross_k)}"
    )


def _indexed_variants(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return named indexed-cross-kNN variants for the Stage 16 validation matrix."""
    sweep = config.get("indexed_sweep", {})
    base = dict(config["cross_links"])
    if not sweep.get("enabled", False):
        return [("indexed_cross_knn", base)]
    variants: list[tuple[str, dict[str, Any]]] = []
    for item in sweep["variants"]:
        selected = dict(base)
        selected.update({key: value for key, value in item.items() if key != "name"})
        variants.append((f"indexed_cross_knn/{item['name']}", selected))
    return variants


def _run_implemented_methods(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Execute currently implemented Phase 3 methods for configured inputs."""
    enabled = [method for method in config["methods"]["enabled"] if method in IMPLEMENTED_METHODS]
    methods: dict[str, Any] = {}
    if not enabled:
        return methods
    for data_seed in config["input"]["data_seeds"]:
        for condition in config["input"]["conditions"]:
            first, second = load_pairwise_phase2_inputs(config, int(data_seed), condition)
            union = union_phase2_pair(first, second)
            key = f"seed_{int(data_seed)}/{condition}"
            methods[key] = {}
            ground_truth = None
            methods_requiring_ground_truth = {
                "dual_index",
                "full_rebuild_hnsw",
                "incremental_insertion",
                "no_cross_merged",
                "random_cross_links",
                "sampled_cross_knn",
                "indexed_cross_knn",
            }
            if "bruteforce_ground_truth" in enabled or (
                set(enabled) & methods_requiring_ground_truth
            ):
                ground_truth_dir = (
                    output_root
                    / "phase3_ground_truth"
                    / f"seed_{int(data_seed)}"
                    / condition
                )
                ground_truth = run_bruteforce_ground_truth(
                    union,
                    ground_truth_dir,
                    [int(k) for k in config["search"]["evaluation_k"]],
                    int(config["search"]["brute_force_batch_size"]),
                )
                methods[key]["bruteforce_ground_truth"] = ground_truth
            if "full_rebuild_hnsw" in enabled:
                if ground_truth is None:
                    raise RuntimeError("Full rebuild requires brute-force ground truth.")
                truth_ids = np.load(ground_truth["arrays"]["neighbor_global_ids"])
                full_rebuild_dir = (
                    output_root
                    / "phase3_full_rebuild"
                    / f"seed_{int(data_seed)}"
                    / condition
                )
                methods[key]["full_rebuild_hnsw"] = run_full_rebuild_hnsw(
                    union,
                    truth_ids,
                    full_rebuild_dir,
                    config["search"],
                    int(config["metrics"].get("integration_repetitions", 1)),
                )
            if "dual_index" in enabled:
                if ground_truth is None:
                    raise RuntimeError("Dual-index requires brute-force ground truth.")
                truth_ids = np.load(ground_truth["arrays"]["neighbor_global_ids"])
                dual_index_dir = (
                    output_root
                    / "phase3_dual_index"
                    / f"seed_{int(data_seed)}"
                    / condition
                )
                methods[key]["dual_index"] = run_dual_index_baseline(
                    first,
                    second,
                    union,
                    truth_ids,
                    dual_index_dir,
                    config["search"],
                )
            if "incremental_insertion" in enabled:
                if ground_truth is None:
                    raise RuntimeError(
                        "Incremental insertion requires brute-force ground truth."
                    )
                truth_ids = np.load(ground_truth["arrays"]["neighbor_global_ids"])
                incremental_dir = (
                    output_root
                    / "phase3_incremental_insertion"
                    / f"seed_{int(data_seed)}"
                    / condition
                )
                methods[key]["incremental_insertion"] = (
                    run_incremental_insertion_baseline(
                        first,
                        second,
                        union,
                        truth_ids,
                        incremental_dir,
                        config["search"],
                        int(config["metrics"].get("integration_repetitions", 1)),
                    )
                )
            if "no_cross_merged" in enabled:
                if ground_truth is None:
                    raise RuntimeError("No-cross merged requires brute-force ground truth.")
                truth_ids = np.load(ground_truth["arrays"]["neighbor_global_ids"])
                no_cross_dir = (
                    output_root
                    / "phase3_no_cross_merged"
                    / f"seed_{int(data_seed)}"
                    / condition
                )
                methods[key]["no_cross_merged"] = run_no_cross_merged_baseline(
                    first,
                    second,
                    union,
                    truth_ids,
                    no_cross_dir,
                    config["search"],
                )
            if "random_cross_links" in enabled:
                if ground_truth is None:
                    raise RuntimeError("Random cross-links requires brute-force ground truth.")
                truth_ids = np.load(ground_truth["arrays"]["neighbor_global_ids"])
                base_random_dir = (
                    output_root
                    / "phase3_random_cross_links"
                    / f"seed_{int(data_seed)}"
                    / condition
                )
                random_grid = _cross_link_parameter_grid(
                    config, "random_cross_links"
                )
                random_is_sweep = (
                    config.get("sweep", {}).get("enabled", False)
                    and "random_cross_links" in config.get("sweep", {}).get("methods", [])
                )
                for sample_ratio, cross_k in random_grid:
                    result_key = (
                        _sweep_result_key(
                            "random_cross_links", sample_ratio, cross_k
                        )
                        if random_is_sweep
                        else "random_cross_links"
                    )
                    random_dir = _sweep_output_dir(
                        base_random_dir, sample_ratio, cross_k, random_is_sweep
                    )
                    methods[key][result_key] = run_random_cross_links_baseline(
                        first,
                        second,
                        union,
                        truth_ids,
                        random_dir,
                        config["search"],
                        _cross_link_config_for(config, sample_ratio, cross_k),
                    )
            if "sampled_cross_knn" in enabled:
                if ground_truth is None:
                    raise RuntimeError("Sampled cross-kNN requires brute-force ground truth.")
                truth_ids = np.load(ground_truth["arrays"]["neighbor_global_ids"])
                base_sampled_dir = (
                    output_root
                    / "phase3_sampled_cross_knn"
                    / f"seed_{int(data_seed)}"
                    / condition
                )
                sampled_grid = _cross_link_parameter_grid(
                    config, "sampled_cross_knn"
                )
                sampled_is_sweep = (
                    config.get("sweep", {}).get("enabled", False)
                    and "sampled_cross_knn" in config.get("sweep", {}).get("methods", [])
                )
                for sample_ratio, cross_k in sampled_grid:
                    result_key = (
                        _sweep_result_key(
                            "sampled_cross_knn", sample_ratio, cross_k
                        )
                        if sampled_is_sweep
                        else "sampled_cross_knn"
                    )
                    sampled_dir = _sweep_output_dir(
                        base_sampled_dir, sample_ratio, cross_k, sampled_is_sweep
                    )
                    methods[key][result_key] = run_sampled_cross_knn_baseline(
                        first,
                        second,
                        union,
                        truth_ids,
                        sampled_dir,
                        config["search"],
                        _cross_link_config_for(config, sample_ratio, cross_k),
                    )
            if "indexed_cross_knn" in enabled:
                if ground_truth is None:
                    raise RuntimeError("Indexed cross-kNN requires brute-force ground truth.")
                truth_ids = np.load(ground_truth["arrays"]["neighbor_global_ids"])
                default_ratio, default_cross_k = _cross_link_parameter_grid(
                    config, "indexed_cross_knn"
                )[0]
                for result_key, variant_config in _indexed_variants(config):
                    indexed_dir = (
                        output_root
                        / "phase3_indexed_cross_knn"
                        / f"seed_{int(data_seed)}"
                        / condition
                        / result_key.replace("/", "__")
                    )
                    selected_config = _cross_link_config_for(
                        {**config, "cross_links": variant_config},
                        default_ratio,
                        default_cross_k,
                    )
                    methods[key][result_key] = run_indexed_cross_knn_baseline(
                        first,
                        second,
                        union,
                        truth_ids,
                        indexed_dir,
                        config["search"],
                        selected_config,
                        int(config["metrics"].get("integration_repetitions", 1)),
                    )
    return methods


def run_complete_phase3(
    config_path: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Run Phase 3 common-infrastructure readiness checks.

    Stage 2 intentionally does not execute any Phase 3 method. It validates the
    config, checks that pairwise Phase 2 inputs exist, and writes a manifest.
    """
    started_at, start_counter = start_run()
    config = load_phase3_config(config_path)
    if output_root is None:
        output_root = Path(config["output"].get("root", "data"))
    if config.get("stages", {}).get("run_tests", False):
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "tests"], check=True
        )
    readiness = check_pairwise_phase2_inputs(config)
    method_results: dict[str, Any] = {}
    if readiness["ready"] and not config.get("stages", {}).get(
        "readiness_check_only", True
    ):
        method_results = _run_implemented_methods(config, output_root)
    manifest: dict[str, Any] = {
        "pipeline": "phase3_common_infrastructure",
        "stage": (
            "method_execution"
            if method_results
            else "common_readiness_check"
        ),
        "profile": config["profile"],
        "config_path": str(config_path),
        "config": config,
        "readiness": readiness,
        "methods": config["methods"],
        "method_results": method_results,
        "result_scope": {
            "smoke_test": "smoke_test_not_dissertation_formal_results",
            "validation_sweep": "first_round_validation_not_formal_results",
            "formal_experiment": "formal_experiment",
        }[config["profile"]],
    }
    add_run_metadata(manifest, started_at, start_counter)
    manifest_path = Path(config["output"].get("manifest", output_root / "phase3_run_manifest.json"))
    if not manifest_path.is_absolute():
        manifest_path = Path(manifest_path)
    save_json(manifest_path, manifest)
    if (
        not readiness["ready"]
        and config.get("stages", {}).get("fail_on_missing_inputs", True)
    ):
        first = readiness["missing"][0]
        raise FileNotFoundError(
            "Phase 3 pairwise inputs are not ready. First missing record: "
            f"seed={first['data_seed']} condition={first['condition']} "
            f"subset={first['subset']} missing={first['missing_paths']}"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase3_smoke.json")
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_complete_phase3(args.config, args.output_root)
    readiness = manifest["readiness"]
    print(
        f"Completed Phase 3 {manifest['stage']} profile={manifest['profile']} "
        f"ready={readiness['ready']} checked={readiness['checked_count']} "
        f"missing={readiness['missing_count']}."
    )


if __name__ == "__main__":
    main()
