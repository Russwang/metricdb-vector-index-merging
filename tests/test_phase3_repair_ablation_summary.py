import csv
import json
import tempfile
import unittest
from pathlib import Path

from data.collect_phase2_repair_ablation_metrics import (
    MetricsSnapshotError,
    collect_phase2_metrics,
)
from data.summarize_phase3_repair_ablation import (
    RepairAblationError,
    aggregate_repair_ablation,
    load_paired_manifests,
    summarize_paired,
)


CONDITIONS = {
    "exact": ["identity", "rotation", "reflection", "translation", "orthogonal_translation"],
    "gaussian": ["gaussian_0p010", "gaussian_0p050", "gaussian_0p100"],
    "laplacian": ["laplacian_0p010", "laplacian_0p050", "laplacian_0p100"],
}


def _quality(value):
    return {
        "overall": {"recall_at_1": value + 0.05, "recall_at_10": value, "recall_at_50": value - 0.05},
        "source_target": {"recall_at_10": {"cross_subset": value - 0.02}},
    }


def _method(method, variant, value):
    item = {
        "input": {"graph_kind": "raw_level0", "within_subset_variant": variant, "subsets": ["D_A", "D_B"]},
        "diagnostics": {"global_id_output": True},
        "quality": _quality(value),
        "cost": {"integration_seconds": 0.3, "total_search_seconds": 0.2, "average_query_ms": 1.0},
    }
    if method in ("no_cross_merged", "random_cross_links", "sampled_cross_knn"):
        item["graph_statistics"] = {"weak_connected_component_count": 2 if method == "no_cross_merged" else 1, "directed_reachable_from_entry_ratio": 0.5 if method == "no_cross_merged" else 1.0}
        item["graph"] = {"cross_subset_edge_count": 0 if method == "no_cross_merged" else 8, "added_cross_link_count": 0 if method == "no_cross_merged" else 8}
    if method in ("random_cross_links", "sampled_cross_knn"):
        item["graph"].update({"sample_ratio": 0.05, "cross_k": 4})
    if method == "sampled_cross_knn":
        item["cost"].update({"cross_candidate_distance_count": 40, "cross_scan_work_ratio": 0.05})
    return item


def _case(variant, delta):
    base = 0.7 + delta
    methods = {
        "bruteforce_ground_truth": {"cost": {"total_seconds": 0.1}},
        "dual_index": _method("dual_index", variant, base),
        "no_cross_merged": _method("no_cross_merged", variant, base - 0.2),
        "random_cross_links": _method("random_cross_links", variant, base - 0.1),
        "sampled_cross_knn": _method("sampled_cross_knn", variant, base + 0.05),
    }
    if variant == "practical_repair":
        methods["full_rebuild_hnsw"] = {"quality": _quality(0.9)}
        methods["incremental_insertion"] = {"directions": {"A<-B": {}, "B<-A": {}}}
    return methods


def _manifest(batch, variant):
    results = {}
    for seed in range(42, 47):
        for condition in CONDITIONS[batch]:
            # Practical is +0.01 for even seeds, tied for seed 43, and -0.01 for seed 45.
            delta = 0.01 if seed in (42, 44, 46) else (0.0 if seed == 43 else -0.01)
            results[f"seed_{seed}/{condition}"] = _case(variant, delta if variant == "practical_repair" else 0.0)
    return {
        "readiness": {"ready": True, "missing_count": 0, "graph_kind": "raw_level0", "within_subset_variant": variant},
        "method_results": results,
    }


def _metric_record(batch, seed, condition, subset, variant):
    practical = variant == "practical_repair"
    return {
        "batch": batch, "seed": seed, "condition": condition, "subset": subset, "variant": variant,
        "cost": {
            "repair_seconds": 2.0 if practical else 0.0,
            "candidate_generation_seconds": 1.0 if practical else 0.0,
            "distance_computation_count": 100 if practical else 0,
            "processed_node_ratio": 1.0 if practical else 0.0,
            "total_compile_and_repair_seconds": 3.0 if practical else 1.0,
            "vector_transformation_seconds": 0.4,
            "graph_extraction_seconds": 0.6,
        },
        "graph_diagnostics": {
            "retained_edge_ratio": 0.8 if practical else 1.0,
            "removed_edge_ratio": 0.2 if practical else 0.0,
            "added_edge_ratio": 0.2 if practical else 0.0,
            "reciprocal_edge_ratio": 0.9,
            "connected_component_count": 1,
            "directed_reachable_from_entry_ratio": 1.0,
        },
        "local_held_out_recall": {"recall_at_1": 0.8, "recall_at_10": 0.75 if practical else 0.74, "recall_at_50": 0.7},
    }


def _snapshot(batch):
    records = [
        _metric_record(batch, seed, condition, subset, variant)
        for seed in range(42, 47) for condition in CONDITIONS[batch]
        for subset in ("D_A", "D_B") for variant in ("practical_repair", "no_repair")
    ]
    return {
        "batch": batch, "graph_kind": "raw_level0", "seeds": list(range(42, 47)),
        "conditions": CONDITIONS[batch], "subsets": ["D_A", "D_B"],
        "record_count": len(records), "records": records,
    }


class TestRepairAblationSummary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.practical, self.no_repair, self.phase2 = {}, {}, {}
        for batch in CONDITIONS:
            self.practical[batch] = self.root / f"p_{batch}.json"
            self.no_repair[batch] = self.root / f"n_{batch}.json"
            self.phase2[batch] = self.root / f"m_{batch}.json"
            self.practical[batch].write_text(json.dumps(_manifest(batch, "practical_repair")))
            self.no_repair[batch].write_text(json.dumps(_manifest(batch, "no_repair")))
            self.phase2[batch].write_text(json.dumps(_snapshot(batch)))

    def tearDown(self):
        self.temp.cleanup()

    def test_loads_55_paired_cases_and_excludes_nonpaired_methods(self):
        records = load_paired_manifests(self.practical, self.no_repair)
        self.assertEqual(len(records), 55 * 4)
        self.assertEqual({r["method"] for r in records}, {"dual_index", "no_cross_merged", "random_cross_links", "sampled_cross_knn"})

    def test_case_keys_must_match(self):
        payload = json.loads(self.no_repair["exact"].read_text())
        del payload["method_results"]["seed_42/identity"]
        self.no_repair["exact"].write_text(json.dumps(payload))
        with self.assertRaisesRegex(RepairAblationError, "case keys differ"):
            load_paired_manifests(self.practical, self.no_repair)

    def test_variants_and_method_completeness_are_validated(self):
        payload = json.loads(self.no_repair["exact"].read_text())
        payload["readiness"]["within_subset_variant"] = "practical_repair"
        self.no_repair["exact"].write_text(json.dumps(payload))
        with self.assertRaisesRegex(RepairAblationError, "expected variant"):
            load_paired_manifests(self.practical, self.no_repair)

    def test_parameters_must_match_fixed_values(self):
        payload = json.loads(self.no_repair["gaussian"].read_text())
        payload["method_results"]["seed_42/gaussian_0p010"]["sampled_cross_knn"]["graph"]["cross_k"] = 8
        self.no_repair["gaussian"].write_text(json.dumps(payload))
        with self.assertRaisesRegex(RepairAblationError, "cross_k"):
            load_paired_manifests(self.practical, self.no_repair)

    def test_paired_delta_sign_wins_ties_losses_and_ddof(self):
        rows = summarize_paired(load_paired_manifests(self.practical, self.no_repair))
        sampled = next(r for r in rows if r["group_level"] == "all" and r["method"] == "sampled_cross_knn")
        self.assertGreater(sampled["delta_recall_at_10_mean"], 0)
        self.assertEqual(sampled["recall_at_10_practical_wins"], 33)
        self.assertEqual(sampled["recall_at_10_ties"], 11)
        self.assertEqual(sampled["recall_at_10_no_repair_wins"], 11)
        self.assertGreater(sampled["delta_recall_at_10_std"], 0)
        condition = next(r for r in rows if r["group_level"] == "condition" and r["group"] == "identity" and r["method"] == "sampled_cross_knn")
        self.assertEqual(condition["paired_case_count"], 5)

    def test_single_observation_std_is_zero(self):
        records = load_paired_manifests(self.practical, self.no_repair)
        one = summarize_paired([records[0]])
        all_row = next(r for r in one if r["group_level"] == "all")
        self.assertEqual(all_row["delta_recall_at_10_std"], 0.0)

    def test_generated_outputs_parse_and_csv_headers(self):
        output = self.root / "output"
        summary = aggregate_repair_ablation(self.practical, self.no_repair, self.phase2, output)
        parsed = json.loads((output / "phase3_repair_ablation_summary.json").read_text())
        self.assertEqual(parsed["scope"]["case_count"], 55)
        self.assertEqual(summary["statistics"]["sample_std_ddof"], 1)
        with (output / "repair_ablation_paired_deltas.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 55 * 4)
        self.assertIn("delta_recall_at_10", rows[0])
        with (output / "repair_ablation_phase2_cost.csv").open(newline="") as handle:
            cost_rows = list(csv.DictReader(handle))
        self.assertEqual(len(cost_rows), 55 * 3)

    def test_missing_optional_connectivity_remains_blank(self):
        output = self.root / "output"
        aggregate_repair_ablation(self.practical, self.no_repair, self.phase2, output)
        with (output / "repair_ablation_connectivity.csv").open(newline="") as handle:
            dual = next(r for r in csv.DictReader(handle) if r["method"] == "dual_index")
        self.assertEqual(dual["practical_weak_components_mean"], "")


class TestPhase2RepairMetricsCollector(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "phase3.json"
        config = {
            "profile": "formal_experiment",
            "phase2": {"compiled_graph_root": str(self.root / "compiled"), "repaired_graph_root": str(self.root / "repaired"), "manifest": str(self.root / "manifest.json"), "recommended_pairwise_config": "unused.json"},
            "subsets": {"pair": ["D_A", "D_B"]},
            "input": {"data_seeds": [42], "conditions": ["identity"], "graph_kind": "raw_level0", "within_subset_variant": "no_repair", "require_raw_no_repair_graph": True},
            "methods": {"enabled": ["bruteforce_ground_truth"], "first_round": ["bruteforce_ground_truth"], "second_round_planned": []},
            "search": {"m": 16, "ef_construction": 200, "ef_search": 64, "evaluation_k": [1, 10, 50], "brute_force_batch_size": 256},
            "cross_links": {"sample_ratios": [0.05], "cross_k": [4], "smoke_default": {"sample_ratio": 0.05, "cross_k": 4}, "random_seed": 442},
            "sweep": {"enabled": False}, "metrics": {"latency_warmup_queries": 1, "latency_repetitions": 1},
            "output": {"root": str(self.root / "output"), "manifest": str(self.root / "phase3_manifest.json")},
            "stages": {"run_tests": False, "readiness_check_only": False, "fail_on_missing_inputs": True},
        }
        self.config.write_text(json.dumps(config))
        for subset in ("D_A", "D_B"):
            for variant in ("practical_repair", "no_repair"):
                path = self.root / "repaired" / "seed_42" / "identity" / subset / "raw_level0" / variant / "metrics.json"
                path.parent.mkdir(parents=True)
                record = _metric_record("exact", 42, "identity", subset, variant)
                path.write_text(json.dumps({"cost": record["cost"], "graph_diagnostics": record["graph_diagnostics"], "quality": {"held_out": record["local_held_out_recall"]}}))

    def tearDown(self):
        self.temp.cleanup()

    def test_collector_expected_schema_and_numeric_fields(self):
        result = collect_phase2_metrics("exact", self.config)
        self.assertEqual(result["record_count"], 4)
        self.assertEqual(result["seeds"], [42])
        self.assertEqual(result["conditions"], ["identity"])
        self.assertEqual(set(result["variants"]), {"practical_repair", "no_repair"})
        self.assertIsInstance(result["records"][0]["cost"]["repair_seconds"], (int, float))

    def test_collector_rejects_missing_metrics_file(self):
        missing = self.root / "repaired" / "seed_42" / "identity" / "D_B" / "raw_level0" / "no_repair" / "metrics.json"
        missing.unlink()
        with self.assertRaisesRegex(MetricsSnapshotError, "Missing metrics file"):
            collect_phase2_metrics("exact", self.config)

    def test_collector_does_not_impute_missing_field(self):
        path = self.root / "repaired" / "seed_42" / "identity" / "D_A" / "raw_level0" / "practical_repair" / "metrics.json"
        payload = json.loads(path.read_text())
        del payload["cost"]["repair_seconds"]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(MetricsSnapshotError, "repair_seconds"):
            collect_phase2_metrics("exact", self.config)


if __name__ == "__main__":
    unittest.main()
