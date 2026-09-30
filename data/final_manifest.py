"""Compact evidence manifest and generated dissertation source artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import faiss
import numpy as np

FINAL_MANIFEST_SCHEMA = "metricdb.final-evidence.v1"
REQUIRED_EXPERIMENT_FIELDS = {"cycle", "config", "source", "status", "artifacts"}
EXPERIMENT_STATUSES = {"complete", "pending"}


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_final_manifest(experiments: list[dict[str, Any]], output_path: str | Path, *, repo_root: str | Path = ".") -> dict[str, Any]:
    for index, experiment in enumerate(experiments):
        missing = REQUIRED_EXPERIMENT_FIELDS - set(experiment)
        if missing:
            raise ValueError(f"Experiment {index} missing fields: {sorted(missing)}")
        if experiment["status"] not in EXPERIMENT_STATUSES:
            raise ValueError(
                f"Experiment {index} has invalid status {experiment['status']!r}; "
                f"expected one of {sorted(EXPERIMENT_STATUSES)}"
            )
    root = Path(repo_root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    normalized = []
    for experiment in experiments:
        artifacts = []
        for artifact in experiment["artifacts"]:
            path = Path(artifact)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise ValueError(f"Manifest artifact does not exist: {path}")
            artifacts.append({"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path), "bytes": path.stat().st_size, "sha256": _checksum(path)})
        normalized.append({**experiment, "artifacts": artifacts})
    manifest = {
        "schema_version": FINAL_MANIFEST_SCHEMA,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "faiss_threads": int(faiss.omp_get_max_threads()),
            "git_commit": commit,
        },
        "experiments": normalized,
        "scope_notes": {
            "topology": "stored level-0 topology; not a complete multi-layer HNSW physical merge",
            "incremental_legacy": "rebuild-base-plus-add",
            "loaded_base": "prebuilt shared-space base only",
            "stage17_100k": "exploratory single-seed",
            "stage21_100k": "no gated multi-seed 100k run",
            "repair_wins": {"local_or_dual": "0/55", "sampled_union": "55/55"},
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_final_manifest(
    manifest: dict[str, Any], *, repo_root: str | Path = ".", require_complete: bool = False
) -> dict[str, Any]:
    """Verify configs, artifact metadata, checksums, and completion state.

    A valid manifest may intentionally contain pending experiments.  Set
    ``require_complete`` for the final dissertation release gate.
    """
    root = Path(repo_root).resolve()
    errors: list[str] = []
    pending_cycles: list[int] = []
    verified_artifacts = 0

    if manifest.get("schema_version") != FINAL_MANIFEST_SCHEMA:
        errors.append(
            f"schema_version must be {FINAL_MANIFEST_SCHEMA!r}, got {manifest.get('schema_version')!r}"
        )
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list):
        return {
            "valid": False,
            "release_ready": False,
            "errors": errors + ["experiments must be a list"],
            "pending_cycles": [],
            "verified_artifacts": 0,
        }

    seen_cycles: set[int] = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            errors.append(f"experiment {index} must be an object")
            continue
        missing = REQUIRED_EXPERIMENT_FIELDS - set(experiment)
        if missing:
            errors.append(f"experiment {index} missing fields: {sorted(missing)}")
            continue
        cycle = experiment["cycle"]
        if not isinstance(cycle, int):
            errors.append(f"experiment {index} cycle must be an integer")
        elif cycle in seen_cycles:
            errors.append(f"duplicate cycle: {cycle}")
        else:
            seen_cycles.add(cycle)
        status = experiment["status"]
        if status not in EXPERIMENT_STATUSES:
            errors.append(f"cycle {cycle} has invalid status: {status!r}")
        elif status == "pending":
            pending_cycles.append(cycle)

        config_path = root / str(experiment["config"])
        if not config_path.is_file():
            errors.append(f"cycle {cycle} config missing: {experiment['config']}")
        artifacts = experiment["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"cycle {cycle} artifacts must be a non-empty list")
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not {"path", "bytes", "sha256"}.issubset(artifact):
                errors.append(f"cycle {cycle} has malformed artifact metadata")
                continue
            path = Path(str(artifact["path"]))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file():
                errors.append(f"cycle {cycle} artifact missing: {path}")
                continue
            actual_bytes = resolved.stat().st_size
            if actual_bytes != artifact["bytes"]:
                errors.append(
                    f"cycle {cycle} artifact size mismatch for {path}: "
                    f"expected {artifact['bytes']}, got {actual_bytes}"
                )
                continue
            actual_sha = _checksum(resolved)
            if actual_sha != artifact["sha256"]:
                errors.append(f"cycle {cycle} artifact checksum mismatch: {path}")
                continue
            verified_artifacts += 1

    if require_complete and pending_cycles:
        errors.append(f"pending cycles remain: {sorted(pending_cycles)}")
    valid = not errors
    return {
        "valid": valid,
        "release_ready": valid and not pending_cycles,
        "errors": errors,
        "pending_cycles": sorted(pending_cycles),
        "verified_artifacts": verified_artifacts,
    }


def generate_latex_sources(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    """Generate one traceable table fragment and pgfplots CSV from compact rows."""
    required = {"label", "method", "recall", "latency_ms", "integration_seconds"}
    for index, row in enumerate(rows):
        if missing := required - set(row):
            raise ValueError(f"Result row {index} missing fields: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "final_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(required), lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in sorted(required)} for row in rows])
    tex_path = output / "final_results_table.tex"
    lines = ["% Generated from final manifest source data; do not edit.", "\\begin{tabular}{llrrr}", "Label & Method & Recall & Latency (ms) & Integration (s) \\\\", "\\hline"]
    for row in rows:
        label = str(row["label"]).replace("_", "\\_")
        method = str(row["method"]).replace("_", "\\_")
        lines.append(f"{label} & {method} & {float(row['recall']):.3f} & {float(row['latency_ms']):.3f} & {float(row['integration_seconds']):.3f} \\\\")
    lines.extend(["\\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "latex_table": str(tex_path)}
