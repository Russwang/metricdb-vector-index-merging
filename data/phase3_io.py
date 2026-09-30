"""Phase 3 input loading and readiness checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data.phase2_common import load_graph


@dataclass(frozen=True)
class Phase2SubsetInput:
    """Loaded Phase 2 artifacts for one seed/condition/subset."""

    data_seed: int
    condition: str
    subset: str
    transformed_vectors: np.ndarray
    global_ids: np.ndarray
    held_out_queries: np.ndarray
    graph: dict[str, Any]
    raw_no_repair_graph: dict[str, Any] | None
    compiled_dir: Path
    graph_path: Path
    local_vectors: np.ndarray | None = None
    local_index_path: Path | None = None
    condition_metadata: dict[str, Any] = field(default_factory=dict)


def phase2_compiled_subset_dir(
    compiled_root: Path,
    data_seed: int,
    condition: str,
    subset: str,
) -> Path:
    """Return the Phase 2 compiled artifact directory for one subset."""
    return compiled_root / f"seed_{int(data_seed)}" / condition / subset


def phase2_variant_graph_path(
    repaired_root: Path,
    data_seed: int,
    condition: str,
    subset: str,
    graph_kind: str,
    variant: str,
) -> Path:
    """Return the saved graph path for one Phase 2 graph variant."""
    return (
        repaired_root
        / f"seed_{int(data_seed)}"
        / condition
        / subset
        / graph_kind
        / variant
        / "graph.json"
    )


def _missing_required(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def _input_paths_for_subset(
    compiled_root: Path,
    repaired_root: Path,
    data_seed: int,
    condition: str,
    subset: str,
    graph_kind: str,
    variant: str,
    require_raw_no_repair_graph: bool,
) -> dict[str, Path]:
    compiled_dir = phase2_compiled_subset_dir(
        compiled_root, data_seed, condition, subset
    )
    paths = {
        "compiled_dir": compiled_dir,
        "transformed_vectors": compiled_dir / "transformed_vectors.npy",
        "local_vectors": compiled_dir / "local_vectors.npy",
        "global_ids": compiled_dir / "global_ids.npy",
        "held_out_queries": compiled_dir / "held_out_queries.npy",
        "local_index": compiled_dir / "local_hnsw.faiss",
        "condition_metadata": compiled_dir / "condition.json",
        "selected_graph": phase2_variant_graph_path(
            repaired_root, data_seed, condition, subset, graph_kind, variant
        ),
    }
    if require_raw_no_repair_graph:
        paths["raw_no_repair_graph"] = phase2_variant_graph_path(
            repaired_root, data_seed, condition, subset, graph_kind, "no_repair"
        )
    return paths


def check_pairwise_phase2_inputs(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Check whether Phase 2 outputs can support Phase 3 pairwise loading."""
    compiled_root = Path(config["phase2"]["compiled_graph_root"])
    repaired_root = Path(config["phase2"]["repaired_graph_root"])
    graph_kind = str(config["input"]["graph_kind"])
    variant = str(config["input"]["within_subset_variant"])
    require_raw = bool(config["input"].get("require_raw_no_repair_graph", True))
    require_indexed = "indexed_cross_knn" in config["methods"].get("enabled", [])
    subsets = [str(value) for value in config["subsets"]["pair"]]
    missing: list[dict[str, Any]] = []
    available: list[dict[str, Any]] = []
    for data_seed in config["input"]["data_seeds"]:
        for condition in config["input"]["conditions"]:
            for subset in subsets:
                paths = _input_paths_for_subset(
                    compiled_root,
                    repaired_root,
                    int(data_seed),
                    str(condition),
                    subset,
                    graph_kind,
                    variant,
                    require_raw,
                )
                required_paths = [
                    paths["transformed_vectors"],
                    paths["global_ids"],
                    paths["held_out_queries"],
                    paths["selected_graph"],
                ]
                if require_raw:
                    required_paths.append(paths["raw_no_repair_graph"])
                if require_indexed:
                    required_paths.extend(
                        [
                            paths["local_vectors"],
                            paths["local_index"],
                            paths["condition_metadata"],
                        ]
                    )
                missing_paths = _missing_required(required_paths)
                record = {
                    "data_seed": int(data_seed),
                    "condition": str(condition),
                    "subset": subset,
                    "missing_paths": missing_paths,
                }
                if missing_paths:
                    missing.append(record)
                else:
                    available.append(record)
    return {
        "ready": not missing,
        "graph_kind": graph_kind,
        "within_subset_variant": variant,
        "subsets": subsets,
        "checked_count": len(available) + len(missing),
        "available_count": len(available),
        "missing_count": len(missing),
        "missing": missing,
    }


def validate_phase2_graph(
    graph: dict[str, Any],
    subset: str,
    graph_kind: str,
    variant: str,
) -> None:
    """Validate graph identity and basic global-ID integrity."""
    if graph.get("source_subset") != subset:
        raise ValueError(
            f"Expected graph source_subset={subset}, got {graph.get('source_subset')}"
        )
    if graph.get("graph_kind") != graph_kind:
        raise ValueError(
            f"Expected graph_kind={graph_kind}, got {graph.get('graph_kind')}"
        )
    if graph.get("variant") != variant:
        raise ValueError(f"Expected variant={variant}, got {graph.get('variant')}")
    node_ids = np.asarray(graph["node_global_ids"], dtype=np.int64)
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("Phase 2 graph contains duplicate global IDs.")
    valid = set(int(value) for value in node_ids)
    if set(int(node) for node in graph["neighbors"]) != valid:
        raise ValueError("Graph neighbor keys do not match node_global_ids.")
    for node, neighbors in graph["neighbors"].items():
        if int(node) not in valid:
            raise ValueError(f"Unknown graph node: {node}")
        unknown = set(int(value) for value in neighbors) - valid
        if unknown:
            raise ValueError(f"Graph contains unknown neighbor IDs: {sorted(unknown)}")


def load_phase2_subset_input(
    config: dict[str, Any],
    data_seed: int,
    condition: str,
    subset: str,
) -> Phase2SubsetInput:
    """Load one Phase 2 subset input selected by the Phase 3 config."""
    compiled_root = Path(config["phase2"]["compiled_graph_root"])
    repaired_root = Path(config["phase2"]["repaired_graph_root"])
    graph_kind = str(config["input"]["graph_kind"])
    variant = str(config["input"]["within_subset_variant"])
    require_raw = bool(config["input"].get("require_raw_no_repair_graph", True))
    require_indexed = "indexed_cross_knn" in config["methods"].get("enabled", [])
    paths = _input_paths_for_subset(
        compiled_root,
        repaired_root,
        int(data_seed),
        condition,
        subset,
        graph_kind,
        variant,
        require_raw,
    )
    missing = _missing_required(
        [
            paths["transformed_vectors"],
            paths["global_ids"],
            paths["held_out_queries"],
            paths["selected_graph"],
        ]
        + ([paths["raw_no_repair_graph"]] if require_raw else [])
        + (
            [
                paths["local_vectors"],
                paths["local_index"],
                paths["condition_metadata"],
            ]
            if require_indexed
            else []
        )
    )
    if missing:
        raise FileNotFoundError(
            "Missing Phase 2 inputs for Phase 3: " + ", ".join(missing)
        )
    vectors = np.load(paths["transformed_vectors"])
    local_vectors = np.load(paths["local_vectors"]) if require_indexed else None
    global_ids = np.load(paths["global_ids"])
    held_out = np.load(paths["held_out_queries"])
    condition_metadata: dict[str, Any] = {}
    if require_indexed:
        import json

        with paths["condition_metadata"].open("r", encoding="utf-8") as handle:
            condition_metadata = json.load(handle)
    graph = load_graph(paths["selected_graph"])
    validate_phase2_graph(graph, subset, graph_kind, variant)
    raw_graph = None
    if require_raw:
        raw_graph = load_graph(paths["raw_no_repair_graph"])
        validate_phase2_graph(raw_graph, subset, graph_kind, "no_repair")
    if (
        len(vectors) != len(global_ids)
        or (local_vectors is not None and len(vectors) != len(local_vectors))
        or len(vectors) != len(graph["node_global_ids"])
    ):
        raise ValueError("Vector, global-ID, and graph node counts differ.")
    np.testing.assert_array_equal(global_ids.astype(np.int64), graph["node_global_ids"])
    return Phase2SubsetInput(
        data_seed=int(data_seed),
        condition=condition,
        subset=subset,
        transformed_vectors=vectors,
        local_vectors=local_vectors,
        global_ids=global_ids.astype(np.int64),
        held_out_queries=held_out,
        graph=graph,
        raw_no_repair_graph=raw_graph,
        compiled_dir=paths["compiled_dir"],
        graph_path=paths["selected_graph"],
        local_index_path=paths["local_index"] if require_indexed else None,
        condition_metadata=condition_metadata,
    )


def load_pairwise_phase2_inputs(
    config: dict[str, Any],
    data_seed: int,
    condition: str,
) -> tuple[Phase2SubsetInput, Phase2SubsetInput]:
    """Load and validate the configured pair of Phase 2 subset inputs."""
    first, second = [
        load_phase2_subset_input(config, data_seed, condition, subset)
        for subset in config["subsets"]["pair"]
    ]
    overlap = set(first.global_ids.tolist()) & set(second.global_ids.tolist())
    if overlap:
        raise ValueError(f"Pairwise subsets have overlapping global IDs: {sorted(overlap)[:5]}")
    return first, second


__all__ = [
    "Phase2SubsetInput",
    "check_pairwise_phase2_inputs",
    "load_pairwise_phase2_inputs",
    "load_phase2_subset_input",
    "phase2_compiled_subset_dir",
    "phase2_variant_graph_path",
    "validate_phase2_graph",
]
