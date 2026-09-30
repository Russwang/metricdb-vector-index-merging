"""Shared configuration and serialization helpers for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data.phase1_common import add_run_metadata, runtime_environment, start_run


EXACT_TRANSFORMATIONS = (
    "identity",
    "rotation",
    "reflection",
    "translation",
    "orthogonal_translation",
)
NOISE_FAMILIES = ("gaussian", "laplacian")
GRAPH_KINDS = ("raw_level0", "query_derived")


def load_phase2_config(path: Path) -> dict[str, Any]:
    """Load and validate one Phase 2 JSON configuration."""
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = {
        "profile",
        "phase1",
        "subsets",
        "transformations",
        "noise",
        "graph",
        "search",
        "queries",
        "seeds",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Phase 2 config is missing sections: {sorted(missing)}")
    if config["profile"] not in {"smoke_test", "formal_experiment"}:
        raise ValueError("profile must be smoke_test or formal_experiment.")
    exact = tuple(config["transformations"]["exact"])
    unsupported = set(exact) - set(EXACT_TRANSFORMATIONS)
    if unsupported:
        raise ValueError(f"Unsupported exact transformations: {sorted(unsupported)}")
    families = tuple(config["noise"]["families"])
    unsupported = set(families) - set(NOISE_FAMILIES)
    if unsupported:
        raise ValueError(f"Unsupported noise families: {sorted(unsupported)}")
    graph_kinds = tuple(config["graph"]["kinds"])
    unsupported = set(graph_kinds) - set(GRAPH_KINDS)
    if unsupported:
        raise ValueError(f"Unsupported graph kinds: {sorted(unsupported)}")
    if config["graph"]["degree_budget_policy"] not in {
        "original_degree",
        "maxM0",
    }:
        raise ValueError("degree_budget_policy must be original_degree or maxM0.")
    if any(float(level) < 0 for level in config["noise"]["levels"]):
        raise ValueError("Noise levels must be non-negative.")
    if any(int(k) <= 0 for k in config["search"]["evaluation_k"]):
        raise ValueError("All evaluation k values must be positive.")
    if int(config["graph"]["candidate_search_k"]) <= 0:
        raise ValueError("candidate_search_k must be positive.")
    variants = config["graph"].get(
        "variants", ["no_repair", "practical_repair", "structural_oracle"]
    )
    allowed_variants = {"no_repair", "practical_repair", "structural_oracle"}
    if not variants or set(variants) - allowed_variants:
        raise ValueError("graph.variants must be a non-empty supported variant list.")
    if int(config["queries"]["held_out_count_per_subset"]) <= 0:
        raise ValueError("held_out_count_per_subset must be positive.")
    if int(config["queries"].get("corpus_point_count_per_subset", 1)) <= 0:
        raise ValueError("corpus_point_count_per_subset must be positive.")
    normalized = json.loads(json.dumps(config))
    normalized["graph"]["variants"] = list(variants)
    return normalized


def save_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)


def save_graph(path: Path, graph: dict[str, Any]) -> None:
    """Persist an external graph without relying on pickle/object arrays."""
    serializable = dict(graph)
    serializable["node_global_ids"] = np.asarray(
        graph["node_global_ids"], dtype=np.int64
    ).tolist()
    serializable["node_levels"] = np.asarray(
        graph["node_levels"], dtype=np.int64
    ).tolist()
    serializable["original_degrees"] = np.asarray(
        graph["original_degrees"], dtype=np.int64
    ).tolist()
    serializable["degree_budget"] = np.asarray(
        graph["degree_budget"], dtype=np.int64
    ).tolist()
    serializable["neighbors"] = {
        str(int(node)): [int(value) for value in neighbors]
        for node, neighbors in graph["neighbors"].items()
    }
    serializable["edge_distances"] = {
        str(int(node)): [float(value) for value in distances]
        for node, distances in graph["edge_distances"].items()
    }
    save_json(path, serializable)


def load_graph(path: Path) -> dict[str, Any]:
    """Load a graph saved by :func:`save_graph`."""
    with path.open("r", encoding="utf-8") as file:
        graph = json.load(file)
    for key in ("node_global_ids", "node_levels", "original_degrees", "degree_budget"):
        graph[key] = np.asarray(graph[key], dtype=np.int64)
    graph["neighbors"] = {
        int(node): [int(value) for value in neighbors]
        for node, neighbors in graph["neighbors"].items()
    }
    graph["edge_distances"] = {
        int(node): [float(value) for value in distances]
        for node, distances in graph["edge_distances"].items()
    }
    return graph


__all__ = [
    "EXACT_TRANSFORMATIONS",
    "GRAPH_KINDS",
    "NOISE_FAMILIES",
    "add_run_metadata",
    "load_graph",
    "load_phase2_config",
    "runtime_environment",
    "save_graph",
    "save_json",
    "start_run",
]
