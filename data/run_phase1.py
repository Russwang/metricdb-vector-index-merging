"""Run the complete reproducible Phase 1 pipeline from one JSON config."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.data_generator import generate_synthetic_data
from data.phase1_common import (
    add_run_metadata,
    load_phase1_config,
    start_run,
    subset_names,
)
from data.prepare_phase1_subsets import prepare_phase1_subsets
from data.prepare_phase1_transformations import prepare_phase1_transformations
from data.probe_phase1_hnsw_backend import run_backend_probe
from data.run_phase1_bruteforce_baseline import run_phase1_bruteforce_baseline
from data.run_phase1_hnsw_rebuild_baseline import (
    run_phase1_hnsw_rebuild_baseline,
)


GENERATED_DIRECTORIES = (
    "phase1_subsets",
    "phase1_transformations",
    "phase1_bruteforce_baseline",
    "phase1_hnsw_rebuild_baseline",
    "phase1_backend_probe",
)


def _run_stage(
    name: str,
    function: Callable[..., dict[str, Any]],
    stage_results: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one stage and record its elapsed time and output manifest."""
    start = perf_counter()
    result = function(**kwargs)
    stage_results[name] = {
        "duration_seconds": perf_counter() - start,
        "manifest": result,
    }
    return result


def _clean_generated_outputs(output_root: Path) -> None:
    """Remove only known generated Phase 1 outputs before a full rerun."""
    for directory_name in GENERATED_DIRECTORIES:
        directory = output_root / directory_name
        if directory.exists():
            shutil.rmtree(directory)
    for filename in (
        "vectors.npy",
        "metadata.json",
        "data_generation_manifest.json",
        "phase1_run_manifest.json",
    ):
        path = output_root / filename
        if path.exists():
            path.unlink()


def _run_test_suite() -> dict[str, Any]:
    """Run the complete unittest suite and fail the pipeline on any test error."""
    started_at, start_counter = start_run()
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "tests"],
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Phase 1 test suite failed with exit code {completed.returncode}."
        )
    return add_run_metadata(
        {
            "stage": "full_test_suite",
            "command": [sys.executable, "-m", "unittest", "discover", "tests"],
            "return_code": completed.returncode,
            "status": "passed",
        },
        started_at,
        start_counter,
    )


def run_complete_phase1(
    config_path: Path,
    output_root: Path = Path("data"),
    clean: bool = True,
) -> dict[str, Any]:
    """Execute data generation through backend probe from one configuration."""
    started_at, start_counter = start_run()
    config = load_phase1_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean_generated_outputs(output_root)

    names = subset_names(int(config["subsets"]["count"]))
    transformations = tuple(config["transformations"]["types"])
    seed = int(config["random_seed"])
    search = config["search"]
    stage_results: dict[str, Any] = {}

    if config.get("stages", {}).get("run_tests", False):
        _run_stage(
            "test_suite",
            _run_test_suite,
            stage_results,
        )
        # Integration tests generate temporary default data in data/. Clean it
        # before creating the configured final artifacts.
        if clean:
            _clean_generated_outputs(output_root)

    _run_stage(
        "data_generation",
        generate_synthetic_data,
        stage_results,
        num_samples=int(config["data"]["num_samples"]),
        dimension=int(config["data"]["dimension"]),
        seed=seed,
        output_dir=output_root,
    )
    _run_stage(
        "subset_preparation",
        prepare_phase1_subsets,
        stage_results,
        vector_path=output_root / "vectors.npy",
        metadata_path=output_root / "metadata.json",
        output_dir=output_root / "phase1_subsets",
        seed=seed,
        subset_names=names,
    )
    _run_stage(
        "transformation_preparation",
        prepare_phase1_transformations,
        stage_results,
        subset_dir=output_root / "phase1_subsets",
        output_dir=output_root / "phase1_transformations",
        seed=seed,
        condition_number=float(
            config["transformations"]["general_linear_condition_number"]
        ),
        m=int(search["m"]),
        ef_construction=int(search["ef_construction"]),
        ef_search=int(search["ef_search"]),
        experiment_names=transformations,
    )
    if config.get("stages", {}).get("brute_force_baseline", True):
        _run_stage(
            "brute_force_baseline",
            run_phase1_bruteforce_baseline,
            stage_results,
            transformation_dir=output_root / "phase1_transformations",
            output_dir=output_root / "phase1_bruteforce_baseline",
            k=int(search["k"]),
            batch_size=int(search["brute_force_batch_size"]),
            exclude_self=True,
        )
    if config.get("stages", {}).get("hnsw_rebuild_baseline", True):
        if not config.get("stages", {}).get("brute_force_baseline", True):
            raise ValueError("hnsw_rebuild_baseline requires brute_force_baseline.")
        _run_stage(
            "hnsw_rebuild_baseline",
            run_phase1_hnsw_rebuild_baseline,
            stage_results,
            brute_force_dir=output_root / "phase1_bruteforce_baseline",
            output_dir=output_root / "phase1_hnsw_rebuild_baseline",
            m=int(search["m"]),
            ef_construction=int(search["ef_construction"]),
            ef_search=int(search["ef_search"]),
        )
    if config.get("stages", {}).get("backend_probe", True):
        _run_stage(
            "backend_probe",
            run_backend_probe,
            stage_results,
            transformation_dir=output_root / "phase1_transformations",
            output_dir=output_root / "phase1_backend_probe",
            k=int(search["k"]),
        )

    manifest = add_run_metadata(
        {
            "pipeline": "phase1_complete",
            "profile": config["profile"],
            "config_path": str(config_path),
            "config": config,
            "output_root": str(output_root),
            "clean_start": clean,
            "result_scope": (
                "smoke_test_not_dissertation_formal_results"
                if config["profile"] == "smoke_test"
                else "formal_experiment"
            ),
            "phase2_backend_decision": {
                "implementation": "python",
                "graph_representation": "external_hnsw_derived_adjacency",
                "modify_faiss_raw_neighbor_arrays": False,
                "faiss_roles": [
                    "local_hnsw",
                    "full_rebuild_baseline",
                    "candidate_neighbor_queries",
                ],
                "rationale": (
                    "FAISS raw neighbor arrays are readable and mechanically "
                    "writable, but direct writes do not maintain HNSW graph "
                    "invariants."
                ),
            },
            "stages": stage_results,
        },
        started_at,
        start_counter,
    )
    manifest_path = output_root / "phase1_run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse complete-pipeline options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1_smoke.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing generated files instead of starting from a clean output root.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured pipeline and print its stage timing summary."""
    args = parse_args()
    manifest = run_complete_phase1(
        config_path=args.config,
        output_root=args.output_root,
        clean=not args.no_clean,
    )
    print(
        f"Completed Phase 1 profile={manifest['profile']} in "
        f"{manifest['runtime']['duration_seconds']:.3f} seconds:"
    )
    for stage_name, stage in manifest["stages"].items():
        print(f"- {stage_name}: {stage['duration_seconds']:.3f} seconds")


if __name__ == "__main__":
    main()
