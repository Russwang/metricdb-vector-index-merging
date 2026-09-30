"""Shared configuration, naming, and runtime metadata helpers for Phase 1."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np


SUPPORTED_TRANSFORMATIONS = ("identity", "orthogonal", "general_linear")


def utc_now_iso() -> str:
    """Return the current UTC time in an ISO 8601 representation."""
    return datetime.now(timezone.utc).isoformat()


def runtime_environment() -> dict[str, str]:
    """Return software versions needed to reproduce a Phase 1 run."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def start_run() -> tuple[str, float]:
    """Return wall-clock and monotonic start markers for a manifest."""
    return utc_now_iso(), perf_counter()


def add_run_metadata(
    manifest: dict[str, Any], started_at: str, start_counter: float
) -> dict[str, Any]:
    """Attach completion time, duration, and software versions to a manifest."""
    manifest["runtime"] = {
        "started_at_utc": started_at,
        "completed_at_utc": utc_now_iso(),
        "duration_seconds": perf_counter() - start_counter,
        "environment": runtime_environment(),
    }
    return manifest


def subset_names(count: int) -> tuple[str, ...]:
    """Generate stable subset names D_A, D_B, ..., D_Z."""
    if count < 2:
        raise ValueError("subset_count must be at least 2.")
    if count > 26:
        raise ValueError("subset_count greater than 26 is not supported.")
    return tuple(f"D_{chr(ord('A') + index)}" for index in range(count))


def load_phase1_config(path: Path) -> dict[str, Any]:
    """Load and validate a unified Phase 1 JSON configuration."""
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_sections = {"profile", "data", "subsets", "transformations", "search"}
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Phase 1 config is missing sections: {sorted(missing)}")

    if config["profile"] not in {"smoke_test", "formal_experiment"}:
        raise ValueError("profile must be smoke_test or formal_experiment.")
    if int(config["data"]["num_samples"]) <= 0:
        raise ValueError("data.num_samples must be positive.")
    if int(config["data"]["dimension"]) <= 0:
        raise ValueError("data.dimension must be positive.")
    if int(config["subsets"]["count"]) > int(config["data"]["num_samples"]):
        raise ValueError("subset count cannot exceed the number of samples.")

    names = subset_names(int(config["subsets"]["count"]))
    transformations = tuple(config["transformations"]["types"])
    if not transformations:
        raise ValueError("At least one transformation type is required.")
    unsupported = set(transformations) - set(SUPPORTED_TRANSFORMATIONS)
    if unsupported:
        raise ValueError(f"Unsupported transformations: {sorted(unsupported)}")
    if "identity" not in transformations:
        raise ValueError("The transformation list must include identity.")

    k = int(config["search"]["k"])
    if k <= 0:
        raise ValueError("search.k must be positive.")
    if k >= int(config["data"]["num_samples"]):
        raise ValueError("search.k must be smaller than data.num_samples.")
    if int(config["search"]["m"]) <= 0:
        raise ValueError("search.m must be positive.")
    if int(config["search"]["ef_construction"]) <= 0:
        raise ValueError("search.ef_construction must be positive.")
    if int(config["search"]["ef_search"]) <= 0:
        raise ValueError("search.ef_search must be positive.")

    normalized = json.loads(json.dumps(config))
    normalized["subsets"]["names"] = list(names)
    normalized["transformations"]["types"] = list(transformations)
    return normalized
