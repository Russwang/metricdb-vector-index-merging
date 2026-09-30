import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from data.summarize_phase3_formal_batches import (
    ManifestValidationError,
    aggregate_formal_batches,
    build_improvement_rows,
    build_recall_rows,
    load_and_validate_manifests,
    main,
)


def _quality(recall10: float) -> dict:
    return {
        "overall": {
            "recall_at_1": recall10 + 0.05,
            "recall_at_10": recall10,
            "recall_at_50": recall10 - 0.05,
        },
        "source_target": {
            "recall_at_10": {
                "source_side": recall10,
                "target_side": recall10 - 0.01,
                "cross_subset": recall10 - 0.01,
            }
        },
    }


def _ann_item(method: str, recall10: float) -> dict:
    item = {
        "method": method,
        "input": {
            "graph_kind": "raw_level0",
            "within_subset_variant": "practical_repair",
            "subsets": ["D_A", "D_B"],
        },
        "quality": _quality(recall10),
        "cost": {"total_search_seconds": 1.0, "average_query_ms": 2.0},
    }
    if method == "full_rebuild_hnsw":
        item["cost"].update({"build_seconds": 3.0, "processed_vector_ratio": 1.0})
        item["graph_statistics"] = {
            "weak_connected_component_count": 1,
            "directed_reachable_from_entry_ratio": 1.0,
            "edge_count": 20,
        }
    elif method == "dual_index":
        item["cost"].update(
            {"integration_seconds": 0.001, "rerank_distance_count": 100}
        )
    elif method == "no_cross_merged":
        item["cost"].update({"integration_seconds": 0.2, "processed_vector_ratio": 0.0})
        item["graph"] = {"cross_subset_edge_count": 0, "added_cross_link_count": 0}
        item["graph_statistics"] = {
            "weak_connected_component_count": 2,
            "directed_reachable_from_entry_ratio": 0.5,
            "edge_count": 10,
        }
    elif method == "random_cross_links":
        item["cost"].update(
            {
                "integration_seconds": 0.3,
                "processed_vector_ratio": 0.05,
                "endpoint_selection_distance_count": 0,
                "edge_weight_distance_count": 8,
            }
        )
        item["graph"] = {
            "sample_ratio": 0.05,
            "cross_k": 4,
            "cross_subset_edge_count": 8,
            "added_cross_link_count": 8,
        }
        item["graph_statistics"] = {
            "weak_connected_component_count": 1,
            "directed_reachable_from_entry_ratio": 1.0,
            "edge_count": 18,
        }
    elif method == "sampled_cross_knn":
        item["cost"].update(
            {
                "integration_seconds": 0.5,
                "processed_vector_ratio": 0.05,
                "cross_candidate_distance_count": 40,
                "cross_scan_work_ratio": 0.05,
            }
        )
        item["graph"] = {
            "sample_ratio": 0.05,
            "cross_k": 4,
            "cross_subset_edge_count": 8,
            "added_cross_link_count": 8,
        }
        item["graph_statistics"] = {
            "weak_connected_component_count": 1,
            "directed_reachable_from_entry_ratio": 1.0,
            "edge_count": 18,
        }
    return item


def _incremental_direction(direction: str, recall10: float) -> dict:
    base, inserted = ("D_A", "D_B") if direction == "A<-B" else ("D_B", "D_A")
    return {
        **_ann_item("incremental_insertion", recall10),
        "direction": direction,
        "input": {
            "graph_kind": "raw_level0",
            "within_subset_variant": "practical_repair",
            "subsets": [base, inserted],
        },
        "cost": {
            "base_index_build_seconds": 0.2,
            "inserted_add_seconds": 0.3,
            "total_integration_seconds": 0.5,
            "total_search_seconds": 0.01,
            "average_query_ms": 0.1,
            "processed_inserted_vector_count": 5,
            "processed_inserted_vector_ratio": 0.5,
        },
        "index_size_bytes": 1234,
    }


def _case(seed_offset: float = 0.0) -> dict:
    return {
        "bruteforce_ground_truth": {
            "method": "bruteforce_ground_truth",
            "cost": {"exact_distance_count": 1000, "total_seconds": 0.1},
        },
        "full_rebuild_hnsw": _ann_item("full_rebuild_hnsw", 0.90 + seed_offset),
        "dual_index": _ann_item("dual_index", 0.95 + seed_offset),
        "no_cross_merged": _ann_item("no_cross_merged", 0.40 + seed_offset),
        "random_cross_links": _ann_item("random_cross_links", 0.55 + seed_offset),
        "sampled_cross_knn": _ann_item("sampled_cross_knn", 0.75 + seed_offset),
        "incremental_insertion": {
            "directions": {
                "A<-B": _incremental_direction("A<-B", 0.88 + seed_offset),
                "B<-A": _incremental_direction("B<-A", 0.89 + seed_offset),
            }
        },
    }


def _manifest(condition: str, two_seeds: bool = False) -> dict:
    results = {f"seed_42/{condition}": _case()}
    if two_seeds:
        results[f"seed_43/{condition}"] = _case(0.02)
    return {
        "readiness": {
            "ready": True,
            "missing_count": 0,
            "missing": [],
            "graph_kind": "raw_level0",
            "within_subset_variant": "practical_repair",
            "subsets": ["D_A", "D_B"],
        },
        "method_results": results,
    }


class TestPhase3FormalSummary(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = {
            "exact": self.root / "exact.json",
            "gaussian": self.root / "gaussian.json",
            "laplacian": self.root / "laplacian.json",
        }
        payloads = {
            "exact": _manifest("identity", two_seeds=True),
            "gaussian": _manifest("gaussian_0p100"),
            "laplacian": _manifest("laplacian_0p100"),
        }
        for batch, path in self.paths.items():
            path.write_text(json.dumps(payloads[batch]), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_three_manifests_and_condition_groups_are_loaded(self):
        records, metadata = load_and_validate_manifests(self.paths)
        self.assertEqual(metadata["case_count"], 4)
        self.assertEqual(metadata["condition_groups"]["exact"], ["identity"])
        self.assertEqual(metadata["condition_groups"]["gaussian"], ["gaussian_0p100"])
        self.assertEqual(metadata["condition_groups"]["laplacian"], ["laplacian_0p100"])
        self.assertEqual(len(records), 4 * 8)

    def test_method_completeness_is_validated(self):
        payload = json.loads(self.paths["exact"].read_text())
        del payload["method_results"]["seed_42/identity"]["dual_index"]
        self.paths["exact"].write_text(json.dumps(payload))
        with self.assertRaisesRegex(ManifestValidationError, "Incomplete methods"):
            load_and_validate_manifests(self.paths)

    def test_missing_manifest_is_rejected_and_cli_returns_nonzero(self):
        self.paths["laplacian"].unlink()
        with self.assertRaisesRegex(ManifestValidationError, "does not exist"):
            load_and_validate_manifests(self.paths)
        with contextlib.redirect_stderr(io.StringIO()):
            result = main(
                [
                    "--exact-manifest", str(self.paths["exact"]),
                    "--gaussian-manifest", str(self.paths["gaussian"]),
                    "--laplacian-manifest", str(self.paths["laplacian"]),
                    "--output-dir", str(self.root / "output"),
                ]
            )
        self.assertEqual(result, 1)

    def test_not_ready_is_rejected(self):
        payload = json.loads(self.paths["exact"].read_text())
        payload["readiness"]["ready"] = False
        self.paths["exact"].write_text(json.dumps(payload))
        with self.assertRaisesRegex(ManifestValidationError, "readiness.ready"):
            load_and_validate_manifests(self.paths)

    def test_parameter_mismatch_is_rejected(self):
        payload = json.loads(self.paths["gaussian"].read_text())
        payload["method_results"]["seed_42/gaussian_0p100"]["random_cross_links"]["graph"]["cross_k"] = 8
        self.paths["gaussian"].write_text(json.dumps(payload))
        with self.assertRaisesRegex(ManifestValidationError, "parameters are inconsistent"):
            load_and_validate_manifests(self.paths)

    def test_incremental_directions_remain_separate(self):
        records, _ = load_and_validate_manifests(self.paths)
        rows, _ = build_recall_rows(records)
        directions = {
            row["direction"] for row in rows if row["method"] == "incremental_insertion"
        }
        self.assertEqual(directions, {"A<-B", "B<-A"})

    def test_mean_sample_std_min_max_are_correct(self):
        records, _ = load_and_validate_manifests(self.paths)
        _, summary = build_recall_rows(records)
        values = summary["exact"]["identity"]["sampled_cross_knn"]["recall_at_10"]
        self.assertAlmostEqual(values["mean"], 0.76)
        self.assertAlmostEqual(values["std"], (0.0002) ** 0.5)
        self.assertAlmostEqual(values["min"], 0.75)
        self.assertAlmostEqual(values["max"], 0.77)
        single = summary["gaussian"]["gaussian_0p100"]["sampled_cross_knn"]["recall_at_10"]
        self.assertEqual(single["std"], 0.0)

    def test_win_counts_exclude_bruteforce(self):
        records, _ = load_and_validate_manifests(self.paths)
        rows, counts = build_improvement_rows(records)
        self.assertEqual(len(rows), 4)
        self.assertEqual(counts["sampled_beats_random"], 4)
        self.assertEqual(counts["sampled_beats_no_cross"], 4)
        self.assertEqual(counts["sampled_beats_full_rebuild"], 0)
        self.assertNotIn("sampled_beats_bruteforce", counts)

    def test_outputs_have_expected_csv_headers_and_rows(self):
        output = self.root / "output"
        aggregate_formal_batches(self.paths, output)
        with (output / "phase3_recall10_by_condition.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3 * 7)
        self.assertIn("cross_subset_recall10_std", rows[0])
        with (output / "phase3_sampled_improvements.csv").open(newline="") as handle:
            improvement_rows = list(csv.DictReader(handle))
        self.assertEqual(len(improvement_rows), 4)

    def test_missing_optional_cost_fields_are_blank_not_zero(self):
        output = self.root / "output"
        aggregate_formal_batches(self.paths, output)
        with (output / "phase3_cost_summary.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        dual = next(row for row in rows if row["method"] == "dual_index")
        self.assertEqual(dual["p95_query_ms_mean"], "")
        self.assertIn("p95_query_ms", dual["not_reported_fields"])

    def test_generated_json_can_be_parsed(self):
        output = self.root / "output"
        summary = aggregate_formal_batches(self.paths, output)
        parsed = json.loads((output / "phase3_first_round_summary.json").read_text())
        self.assertEqual(parsed["global_validation_counts"]["case_count"], 4)
        self.assertEqual(parsed["seed_list"], [42, 43])
        self.assertEqual(summary["aggregation_definition"]["single_observation_standard_deviation"], 0)


if __name__ == "__main__":
    unittest.main()
