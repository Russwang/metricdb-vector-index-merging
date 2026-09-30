"""Shared configuration and metadata helpers for Phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.phase1_common import add_run_metadata, runtime_environment, start_run
from data.phase2_common import save_json


GRAPH_KINDS = ("raw_level0", "query_derived")
PRIMARY_GRAPH_KIND = "raw_level0"
WITHIN_SUBSET_VARIANTS = ("practical_repair", "no_repair")
PRIMARY_WITHIN_SUBSET_VARIANT = "practical_repair"

FIRST_ROUND_METHODS = (
    "bruteforce_ground_truth",
    "full_rebuild_hnsw",
    "dual_index",
    "no_cross_merged",
    "random_cross_links",
    "sampled_cross_knn",
    "incremental_insertion",
)

SECOND_ROUND_METHODS = (
    "indexed_cross_knn",
    "anchor_pseudo_anchor_cross_links",
    "join_set_propagation_merge",
    "raw_level0_vs_query_derived_ablation",
    "no_repair_vs_practical_repair_ablation",
)


def load_phase3_config(path: Path) -> dict[str, Any]:
    """Load and validate one Phase 3 JSON configuration."""
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = {
        "profile",
        "phase2",
        "subsets",
        "input",
        "methods",
        "search",
        "cross_links",
        "metrics",
        "output",
        "stages",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Phase 3 config is missing sections: {sorted(missing)}")
    if config["profile"] not in {
        "smoke_test",
        "validation_sweep",
        "formal_experiment",
    }:
        raise ValueError(
            "profile must be smoke_test, validation_sweep, or formal_experiment."
        )

    pair = tuple(config["subsets"].get("pair", ()))
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("subsets.pair must contain two distinct subset names.")

    graph_kind = config["input"].get("graph_kind")
    if isinstance(graph_kind, list):
        raise ValueError(
            "Phase 3 first-round input.graph_kind must be a single graph source; "
            "raw_level0/query_derived ablation belongs to a separate run."
        )
    if graph_kind not in GRAPH_KINDS:
        raise ValueError(f"Unsupported graph_kind: {graph_kind!r}")

    variant = config["input"].get("within_subset_variant")
    if variant not in WITHIN_SUBSET_VARIANTS:
        raise ValueError(f"Unsupported within_subset_variant: {variant!r}")

    seeds = [int(seed) for seed in config["input"].get("data_seeds", [])]
    if not seeds:
        raise ValueError("input.data_seeds must be a non-empty list.")
    conditions = [str(condition) for condition in config["input"].get("conditions", [])]
    if not conditions:
        raise ValueError("input.conditions must be a non-empty list.")

    enabled = tuple(config["methods"].get("enabled", ()))
    supported = set(FIRST_ROUND_METHODS) | set(SECOND_ROUND_METHODS)
    unsupported = set(enabled) - supported
    if unsupported:
        raise ValueError(f"Unsupported Phase 3 methods: {sorted(unsupported)}")

    evaluation_k = [int(k) for k in config["search"].get("evaluation_k", [])]
    if not evaluation_k or any(k <= 0 for k in evaluation_k):
        raise ValueError("search.evaluation_k must contain positive integers.")
    if int(config["search"].get("ef_search", 0)) <= 0:
        raise ValueError("search.ef_search must be positive.")
    if int(config["search"].get("brute_force_batch_size", 0)) <= 0:
        raise ValueError("search.brute_force_batch_size must be positive.")

    sample_ratios = [float(value) for value in config["cross_links"].get("sample_ratios", [])]
    if not sample_ratios or any(value <= 0.0 or value > 1.0 for value in sample_ratios):
        raise ValueError("cross_links.sample_ratios must be in (0, 1].")
    cross_k = [int(value) for value in config["cross_links"].get("cross_k", [])]
    if not cross_k or any(value <= 0 for value in cross_k):
        raise ValueError("cross_links.cross_k must contain positive integers.")
    bridge_budget = config["cross_links"].get("bridge_budget", {})
    if bridge_budget:
        mode = bridge_budget.get("mode", "ratio")
        if mode not in {"ratio", "sqrt", "fixed"}:
            raise ValueError("bridge_budget.mode must be ratio, sqrt, or fixed.")
        if mode == "sqrt" and float(bridge_budget.get("sqrt_multiplier", 0.0)) <= 0.0:
            raise ValueError("bridge_budget.sqrt_multiplier must be positive.")
        if mode == "fixed" and int(bridge_budget.get("count", 0)) <= 0:
            raise ValueError("bridge_budget.count must be positive.")
    bridge_selection = config["cross_links"].get("bridge_selection", "random")
    if bridge_selection not in {"random", "high_degree", "upper_layer", "hybrid"}:
        raise ValueError("cross_links.bridge_selection is unsupported.")
    if "candidate_k" in config["cross_links"]:
        if int(config["cross_links"]["candidate_k"]) <= 0:
            raise ValueError("cross_links.candidate_k must be positive.")
        if int(config["cross_links"]["candidate_k"]) < min(cross_k):
            raise ValueError("cross_links.candidate_k must be >= every cross_k.")
    if "candidate_ef_search" in config["cross_links"] and int(config["cross_links"]["candidate_ef_search"]) <= 0:
        raise ValueError("cross_links.candidate_ef_search must be positive.")
    if int(config["cross_links"].get("cross_degree_cap", min(cross_k))) <= 0:
        raise ValueError("cross_links.cross_degree_cap must be positive.")
    if int(config["cross_links"].get("propagation_hops", 0)) not in {0, 1}:
        raise ValueError("cross_links.propagation_hops must be 0 or 1.")
    for field in ("propagation_neighbor_budget", "propagation_candidate_cap"):
        if field in config["cross_links"] and int(config["cross_links"][field]) <= 0:
            raise ValueError(f"cross_links.{field} must be positive.")
    repair_policy = config["cross_links"].get("repair_policy", "input_graph")
    if repair_policy not in {"input_graph", "no_repair", "bridge_local"}:
        raise ValueError("cross_links.repair_policy is unsupported.")
    if repair_policy == "bridge_local":
        if int(config["cross_links"].get("repair_radius", 1)) < 0:
            raise ValueError("cross_links.repair_radius must be non-negative.")
        if int(config["cross_links"].get("repair_candidate_k", 64)) <= 0:
            raise ValueError("cross_links.repair_candidate_k must be positive.")
    indexed_sweep = config.get("indexed_sweep", {})
    if indexed_sweep.get("enabled", False):
        variants = indexed_sweep.get("variants", [])
        names = [str(item.get("name", "")) for item in variants]
        if not variants or not all(names) or len(set(names)) != len(names):
            raise ValueError("indexed_sweep.variants must have unique names.")
        for item in variants:
            if str(item.get("bridge_selection", "random")) not in {"random", "high_degree", "upper_layer", "hybrid"}:
                raise ValueError("indexed_sweep bridge_selection is unsupported.")
            if int(item.get("propagation_hops", 0)) not in {0, 1}:
                raise ValueError("indexed_sweep propagation_hops must be 0 or 1.")
            if int(item.get("cross_degree_cap", min(cross_k))) <= 0:
                raise ValueError("indexed_sweep cross_degree_cap must be positive.")
            item_budget = item.get("bridge_budget")
            if item_budget is not None:
                item_mode = item_budget.get("mode", "ratio")
                if item_mode not in {"ratio", "sqrt", "fixed"}:
                    raise ValueError("indexed_sweep bridge_budget.mode is unsupported.")
                if item_mode == "sqrt" and float(item_budget.get("sqrt_multiplier", 0.0)) <= 0.0:
                    raise ValueError("indexed_sweep bridge_budget.sqrt_multiplier must be positive.")
                if item_mode == "fixed" and int(item_budget.get("count", 0)) <= 0:
                    raise ValueError("indexed_sweep bridge_budget.count must be positive.")
            for field in ("propagation_neighbor_budget", "propagation_candidate_cap"):
                if field in item and int(item[field]) <= 0:
                    raise ValueError(f"indexed_sweep {field} must be positive.")

    integration_repetitions = int(config["metrics"].get("integration_repetitions", 1))
    if integration_repetitions <= 0:
        raise ValueError("metrics.integration_repetitions must be positive.")

    sweep = config.get("sweep", {})
    if sweep.get("enabled", False):
        sweep_ratios = [
            float(value)
            for value in sweep.get("sample_ratios", config["cross_links"]["sample_ratios"])
        ]
        if not sweep_ratios or any(value <= 0.0 or value > 1.0 for value in sweep_ratios):
            raise ValueError("sweep.sample_ratios must be in (0, 1].")
        unsupported_ratios = set(sweep_ratios) - set(sample_ratios)
        if unsupported_ratios:
            raise ValueError(
                "sweep.sample_ratios must be listed in cross_links.sample_ratios."
            )
        sweep_cross_k = [
            int(value)
            for value in sweep.get("cross_k", config["cross_links"]["cross_k"])
        ]
        if not sweep_cross_k or any(value <= 0 for value in sweep_cross_k):
            raise ValueError("sweep.cross_k must contain positive integers.")
        unsupported_cross_k = set(sweep_cross_k) - set(cross_k)
        if unsupported_cross_k:
            raise ValueError("sweep.cross_k must be listed in cross_links.cross_k.")
        sweep_methods = [
            str(value)
            for value in sweep.get(
                "methods", ["random_cross_links", "sampled_cross_knn"]
            )
        ]
        unsupported_sweep_methods = set(sweep_methods) - {
            "random_cross_links",
            "sampled_cross_knn",
        }
        if unsupported_sweep_methods:
            raise ValueError(
                "sweep.methods may only contain random_cross_links and sampled_cross_knn."
            )

    normalized = json.loads(json.dumps(config))
    normalized["subsets"]["pair"] = list(pair)
    normalized["input"]["data_seeds"] = seeds
    normalized["input"]["conditions"] = conditions
    normalized["methods"]["enabled"] = list(enabled)
    normalized["search"]["evaluation_k"] = evaluation_k
    normalized["cross_links"]["sample_ratios"] = sample_ratios
    normalized["cross_links"]["cross_k"] = cross_k
    normalized["metrics"]["integration_repetitions"] = integration_repetitions
    if sweep.get("enabled", False):
        normalized.setdefault("sweep", {})
        normalized["sweep"]["enabled"] = True
        normalized["sweep"]["sample_ratios"] = sweep_ratios
        normalized["sweep"]["cross_k"] = sweep_cross_k
        normalized["sweep"]["methods"] = sweep_methods
    return normalized


__all__ = [
    "FIRST_ROUND_METHODS",
    "GRAPH_KINDS",
    "PRIMARY_GRAPH_KIND",
    "PRIMARY_WITHIN_SUBSET_VARIANT",
    "SECOND_ROUND_METHODS",
    "WITHIN_SUBSET_VARIANTS",
    "add_run_metadata",
    "load_phase3_config",
    "runtime_environment",
    "save_json",
    "start_run",
]
