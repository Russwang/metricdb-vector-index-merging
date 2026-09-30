import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.final_manifest import (
    FINAL_MANIFEST_SCHEMA,
    build_final_manifest,
    generate_latex_sources,
    validate_final_manifest,
)
from data.build_scaling_confidence_intervals import build as build_scaling_intervals
from data.export_stage21_compact import build as build_stage21_compact, validate as validate_stage21_compact
from data.phase4_glove import fixed_glove_split, parse_glove_text
from data.phase4_loaded_base import measure_loaded_base_incremental
from data.phase4_sequential import sequential_bounded_merge
from data.phase4_statistics import paired_bootstrap
from data.prepare_cycle24_inputs import prepare as prepare_cycle24_inputs
from data.run_cycle24_sequential import run as run_cycle24_sequential
from data.summarize_cycle24 import validate_and_summarize as summarize_cycle24


def graph(ids, subset):
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "node_global_ids": ids,
        "neighbors": {int(node): [int(ids[(i + 1) % len(ids)])] for i, node in enumerate(ids)},
        "source_subset": subset,
        "original_entry_point": int(ids[0]),
    }


class TestPhase4Supplemental(unittest.TestCase):
    def test_cycle24_preparer_writes_three_unique_subsets_and_both_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            rng = np.random.default_rng(24)
            for offset, subset in enumerate(("D_A", "D_B", "D_C")):
                source = source_root / "seed_7" / "phase1_subsets" / subset
                source.mkdir(parents=True)
                np.save(source / "shared_vectors.npy", rng.normal(size=(9, 4)).astype(np.float32))
                records = [{"global_id": offset * 100 + row} for row in range(9)]
                (source / "id_mapping.json").write_text(json.dumps(records), encoding="utf-8")
            output = root / "prepared"
            config = {
                "corpus_size": 15,
                "subset_sizes": {"D_A": 5, "D_B": 5, "D_C": 5},
                "seeds": [7],
                "conditions": ["rotation", "gaussian_0.1"],
                "orders": [["D_A", "D_B", "D_C"], ["D_A", "D_C", "D_B"], ["D_B", "D_C", "D_A"]],
                "methods": ["full_rebuild", "sequential_incremental", "random_direct", "bounded_n8_c2", "join1"],
                "integration_repetitions": 1,
                "prepared_input": {
                    "source_root": str(source_root),
                    "source_subdir": "phase1_subsets",
                    "corpus_rows_per_subset": 5,
                    "held_out_queries_per_subset": 2,
                    "dimension": 4,
                    "m": 2,
                    "ef_construction": 20,
                    "ef_search": 8,
                    "transformation_seed": 142,
                    "noise_seed": 242,
                },
                "second_merge": {"portal_budget": 2, "cross_k": 1},
                "output": {"prepared_root": str(output), "root": str(root / "results")},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = prepare_cycle24_inputs(config_path)
            self.assertEqual(manifest["case_count"], 6)
            self.assertEqual(manifest["corpus_size_per_seed_condition"], 15)
            all_ids = []
            for subset in ("D_A", "D_B", "D_C"):
                case = output / "seed_7" / "rotation" / subset
                all_ids.extend(np.load(case / "global_ids.npy").tolist())
                self.assertTrue((case / "graph.json").is_file())
                self.assertEqual(np.load(case / "held_out_queries.npy").shape, (2, 4))
            self.assertEqual(len(all_ids), len(set(all_ids)))
            result = run_cycle24_sequential(output, root / "results", config_path)
            self.assertEqual(result["case_count"], 30)
            self.assertEqual(len(result["order_effects"]), 10)
            self.assertEqual({row["method"] for row in result["cases"]}, set(config["methods"]))
            compact = summarize_cycle24(
                root / "results" / "manifest.json",
                config_path,
                output,
                root / "compact.json",
                repo_root=Path("."),
            )
            self.assertTrue(compact["validation"]["valid"])
            self.assertEqual(compact["validation"]["checked_metrics"], 30)

    def test_loaded_base_reload_add_has_complete_ids(self):
        rng = np.random.default_rng(1)
        result = measure_loaded_base_incremental(
            rng.normal(size=(12, 4)), np.arange(12), rng.normal(size=(5, 4)), np.arange(100, 105), repetitions=2, m=4
        )
        self.assertIsNone(result["backend_limitation"])
        self.assertTrue(result["final_ids_complete"])
        self.assertEqual(len(result["add_only"]["samples_seconds"]), 2)

    def test_second_merge_preserves_edges_and_budget(self):
        first = graph([0, 1, 2], "A")
        second = graph([10, 11, 12], "B")
        rng = np.random.default_rng(2)
        merged, first_cost = sequential_bounded_merge(first, second, rng.normal(size=(3, 3)), rng.normal(size=(3, 3)), new_subset="B", portal_budget=2, cross_k=1)
        prior_edges = {(s, t) for s, values in merged["neighbors"].items() for t in values}
        third = graph([20, 21, 22], "C")
        final, second_cost = sequential_bounded_merge(merged, third, rng.normal(size=(6, 3)), rng.normal(size=(3, 3)), new_subset="C", portal_budget=2, cross_k=1)
        final_edges = {(s, t) for s, values in final["neighbors"].items() for t in values}
        self.assertTrue(prior_edges.issubset(final_edges))
        self.assertEqual(len(np.unique(final["node_global_ids"])), 9)
        self.assertLessEqual(second_cost["candidate_distance_count"], 4)
        self.assertTrue(second_cost["prior_edges_preserved"])
        self.assertEqual(first_cost["weak_component_count"], 1)

    def test_glove_parser_rejects_bad_duplicate_nonfinite_and_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.txt"
            path.write_text("bad 1\ndup 3 4\ndup 1 2\nnan nan 2\nzero 0 0\ngood 5 12\n", encoding="utf-8")
            tokens, vectors, manifest = parse_glove_text(path, dimension=2, limit=2)
        self.assertEqual(tokens, ["dup", "good"])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
        self.assertEqual(manifest["rejected"]["duplicate_token"], 1)
        self.assertEqual(manifest["rejected"]["non_numeric_or_non_finite"], 1)

    def test_frozen_glove_split_is_deterministic_disjoint_and_exhaustive(self):
        tokens = [f"token-{index}" for index in range(25_512)]
        vectors = np.ones((25_512, 2), dtype=np.float32)
        first = fixed_glove_split(tokens, vectors, seed=42)
        second = fixed_glove_split(tokens, vectors, seed=42)
        names = ("D_A_indices", "D_B_indices", "query_A_indices", "query_B_indices")
        union = np.concatenate([first[name] for name in names])
        self.assertEqual(len(union), 25_512)
        self.assertEqual(len(np.unique(union)), 25_512)
        np.testing.assert_array_equal(np.sort(union), np.arange(25_512))
        for name in names:
            np.testing.assert_array_equal(first[name], second[name])

    def test_bootstrap_is_deterministic_and_requires_pairing(self):
        rows = [
            {"pair_key": "42/rotation", "method": "base", "value": 0.5},
            {"pair_key": "43/gaussian", "method": "base", "value": 0.7},
            {"pair_key": "42/rotation", "method": "new", "value": 0.6},
            {"pair_key": "43/gaussian", "method": "new", "value": 0.8},
        ]
        first = paired_bootstrap(rows, baseline="base", method="new", value_key="value", samples=100)
        second = paired_bootstrap(rows, baseline="base", method="new", value_key="value", samples=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 0.1)
        with self.assertRaisesRegex(ValueError, "same"):
            paired_bootstrap(rows[:-1], baseline="base", method="new", value_key="value")

    def test_scaling_interval_builder_requires_and_preserves_six_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scales = []
            for label in ("10k", "25k", "50k"):
                methods = {}
                for index in range(6):
                    pair = f"seed_{index // 2}/{index % 2}"
                    methods[pair] = {
                        "random": {"quality": {"overall": {"recall_at_10": 0.5}}, "cost": {"integration_seconds_median": 2.0}},
                        "bounded": {"quality": {"overall": {"recall_at_10": 0.6}}, "cost": {"integration_seconds_median": 1.5}},
                    }
                path = root / f"{label}.json"
                path.write_text(json.dumps({"method_results": methods}), encoding="utf-8")
                scales.append({"label": label, "source_scale": 1, "evaluated_pair_size": 1, "repair_input": "x", "manifest": path.name, "comparisons": ["random"]})
            config = {"schema_version": "x", "bootstrap_samples": 100, "bootstrap_seed": 7, "pairing_unit": "seed_x_condition", "baseline": "random", "candidates": ["bounded"], "scales": scales, "scope_caveat": "test"}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = build_scaling_intervals(config_path, root / "output.json", repo_root=root)
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(result["validation"]["pair_count_per_scale"], [6, 6, 6])
            for scale in result["scales"]:
                self.assertAlmostEqual(scale["comparisons"][0]["recall_delta"]["estimate"], 0.1)

    def test_stage21_compact_export_keeps_only_paired_release_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            methods = {
                "full_rebuild_hnsw": {"quality": {"overall": {"recall_at_10": 0.5}}, "cost": {"integration_seconds_median": 2.0, "average_query_ms": 0.1}, "corpus_size": 10},
                "indexed_cross_knn/random_direct": {"quality": {"overall": {"recall_at_10": 0.4}}, "cost": {"integration_seconds_median": 1.9, "average_query_ms": 2.0}},
                "indexed_cross_knn/bounded_n4_c2": {"quality": {"overall": {"recall_at_10": 0.45}}, "cost": {"integration_seconds_median": 2.1, "average_query_ms": 2.1}},
                "indexed_cross_knn/bounded_n8_c2": {"quality": {"overall": {"recall_at_10": 0.46}}, "cost": {"integration_seconds_median": 2.2, "average_query_ms": 2.2}},
                "indexed_cross_knn/unlimited_join1": {"quality": {"overall": {"recall_at_10": 0.47}}, "cost": {"integration_seconds_median": 2.3, "average_query_ms": 2.3}},
                "incremental_insertion": {"directions": {"A<-B": {"quality": {"overall": {"recall_at_10": 0.5}}, "cost": {"integration_seconds_median": 2.0, "average_query_ms": 0.1}}}},
            }
            full_manifest = {"profile": "formal", "result_scope": "test", "methods": methods, "method_results": {f"seed_{index}/x": methods for index in range(6)}}
            manifest_path = root / "full.json"
            summary_path = root / "summary.json"
            manifest_path.write_text(json.dumps(full_manifest), encoding="utf-8")
            summary_path.write_text(json.dumps({"stage21_gate": {"any_advance_to_100k": False}}), encoding="utf-8")
            compact = build_stage21_compact(manifest_path, summary_path, root / "compact.json")
            self.assertTrue(validate_stage21_compact(compact)["valid"])
            self.assertEqual(len(compact["method_results"]), 6)
            self.assertNotIn("quality", compact["method_results"]["seed_0/x"]["full_rebuild_hnsw"])

    def test_manifest_and_table_generation_fail_on_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "result.json"
            artifact.write_text("{}", encoding="utf-8")
            (root / "c.json").write_text("{}\n", encoding="utf-8")
            manifest = build_final_manifest([{"cycle": 25, "config": "c.json", "source": "smoke", "status": "pending", "artifacts": [artifact]}], root / "manifest.json", repo_root=root)
            self.assertEqual(manifest["schema_version"], FINAL_MANIFEST_SCHEMA)
            self.assertEqual(len(manifest["experiments"][0]["artifacts"][0]["sha256"]), 64)
            validation = validate_final_manifest(manifest, repo_root=root)
            self.assertTrue(validation["valid"])
            self.assertFalse(validation["release_ready"])
            self.assertEqual(validation["pending_cycles"], [25])
            strict = validate_final_manifest(manifest, repo_root=root, require_complete=True)
            self.assertFalse(strict["valid"])
            paths = generate_latex_sources([{"label": "x", "method": "m", "recall": 0.8, "latency_ms": 1.0, "integration_seconds": 2.0}], root / "generated")
            self.assertTrue(Path(paths["csv"]).exists())
            with self.assertRaisesRegex(ValueError, "missing fields"):
                generate_latex_sources([{"label": "x"}], root)
            with self.assertRaisesRegex(ValueError, "missing fields"):
                build_final_manifest([{"cycle": 1}], root / "bad.json", repo_root=root)

    def test_manifest_validation_detects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            artifact = root / "result.json"
            artifact.write_text("{}\n", encoding="utf-8")
            manifest = build_final_manifest(
                [{"cycle": 1, "config": "config.json", "source": "formal", "status": "complete", "artifacts": [artifact]}],
                root / "manifest.json",
                repo_root=root,
            )
            artifact.write_text('{"changed": true}\n', encoding="utf-8")
            validation = validate_final_manifest(manifest, repo_root=root)
            self.assertFalse(validation["valid"])
            self.assertIn("size mismatch", validation["errors"][0])


if __name__ == "__main__":
    unittest.main()
