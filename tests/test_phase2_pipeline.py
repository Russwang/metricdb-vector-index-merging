import json
import os
import tempfile
import unittest
from pathlib import Path

from data.run_phase1 import run_complete_phase1
from data.run_phase2 import run_complete_phase2


class TestPhase2Pipeline(unittest.TestCase):
    def test_smoke_pipeline_writes_manifest_and_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1_config = root / "phase1.json"
            phase2_config = root / "phase2.json"
            output = root / "data"
            phase1_config.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "description": "phase2 prerequisite",
                        "data": {"num_samples": 30, "dimension": 8},
                        "subsets": {"count": 2},
                        "random_seed": 11,
                        "transformations": {
                            "types": [
                                "identity",
                                "orthogonal",
                                "general_linear",
                            ],
                            "general_linear_condition_number": 3.0,
                        },
                        "search": {
                            "k": 3,
                            "m": 4,
                            "ef_construction": 20,
                            "ef_search": 16,
                            "brute_force_batch_size": 8,
                        },
                        "stages": {"backend_probe": False, "run_tests": False},
                    }
                ),
                encoding="utf-8",
            )
            run_complete_phase1(phase1_config, output)
            phase2_config.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "phase1": {
                            "transformation_dir": str(
                                output / "phase1_transformations"
                            ),
                            "subset_dir": str(output / "phase1_subsets"),
                            "source_config": str(phase1_config),
                        },
                        "subsets": ["D_A"],
                        "transformations": {
                            "exact": ["identity", "rotation"],
                            "include_general_linear_stress": False,
                        },
                        "noise": {"families": ["gaussian"], "levels": [0.0]},
                        "graph": {
                            "kinds": ["raw_level0"],
                            "degree_budget_policy": "original_degree",
                            "candidate_search_k": 5,
                            "force_weak_connectivity": True,
                        },
                        "search": {
                            "m": 4,
                            "ef_construction": 20,
                            "ef_search": 16,
                            "evaluation_k": [1, 2],
                            "brute_force_batch_size": 8,
                        },
                        "queries": {"held_out_count_per_subset": 4},
                        "seeds": {
                            "data": [11],
                            "transformation": 12,
                            "noise": 13,
                            "query": 14,
                        },
                        "stages": {"run_tests": False},
                    }
                ),
                encoding="utf-8",
            )
            manifest = run_complete_phase2(phase2_config, output)

            self.assertEqual(manifest["profile"], "smoke_test")
            self.assertEqual(len(manifest["conditions"]), 3)
            self.assertEqual(manifest["summary"]["data_seed_count"], 1)
            self.assertTrue((output / "phase2_run_manifest.json").exists())
            identity = manifest["conditions"]["identity/D_A"]
            variants = identity["graph_kinds"]["raw_level0"]
            self.assertEqual(
                set(variants),
                {
                    "no_repair",
                    "practical_repair",
                    "structural_oracle",
                },
            )
            self.assertIn("local_rebuild_baseline", identity)
            for variant in variants.values():
                self.assertTrue(Path(variant["graph"]).exists())
                self.assertTrue(Path(variant["metrics"]).exists())
            self.assertTrue(
                Path(identity["local_rebuild_baseline"]["graph"]).exists()
            )

    def test_multi_seed_pipeline_generates_phase1_inputs_and_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1_config = root / "phase1.json"
            phase2_config = root / "phase2.json"
            output = root / "data"
            phase1_config.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "description": "phase2 multi-seed prerequisite",
                        "data": {"num_samples": 24, "dimension": 6},
                        "subsets": {"count": 2},
                        "random_seed": 0,
                        "transformations": {
                            "types": [
                                "identity",
                                "orthogonal",
                                "general_linear",
                            ],
                            "general_linear_condition_number": 3.0,
                        },
                        "search": {
                            "k": 3,
                            "m": 4,
                            "ef_construction": 16,
                            "ef_search": 12,
                            "brute_force_batch_size": 8,
                        },
                        "stages": {"backend_probe": False, "run_tests": False},
                    }
                ),
                encoding="utf-8",
            )
            phase2_config.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "phase1": {
                            "transformation_dir": "unused_in_multiseed",
                            "subset_dir": "unused_in_multiseed",
                            "source_config": str(phase1_config),
                        },
                        "subsets": ["D_A"],
                        "transformations": {
                            "exact": ["identity", "rotation"],
                            "include_general_linear_stress": False,
                        },
                        "noise": {"families": ["gaussian"], "levels": [0.0]},
                        "graph": {
                            "kinds": ["raw_level0"],
                            "degree_budget_policy": "original_degree",
                            "candidate_search_k": 4,
                            "force_weak_connectivity": True,
                        },
                        "search": {
                            "m": 4,
                            "ef_construction": 16,
                            "ef_search": 12,
                            "evaluation_k": [1, 2],
                            "brute_force_batch_size": 8,
                        },
                        "queries": {"held_out_count_per_subset": 3},
                        "seeds": {
                            "data": [21, 22],
                            "transformation": 12,
                            "noise": 13,
                            "query": 14,
                        },
                        "stages": {"run_tests": False},
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_complete_phase2(phase2_config, output)

            self.assertTrue(manifest["multi_seed_phase1_generation"])
            self.assertEqual(set(manifest["seeds"]), {"21", "22"})
            self.assertIn("seed_21/identity/D_A", manifest["conditions"])
            self.assertIn("seed_22/identity/D_A", manifest["conditions"])
            self.assertEqual(manifest["summary"]["data_seed_count"], 2)
            aggregate = manifest["summary"]["aggregate_metrics"]
            self.assertTrue(aggregate)
            sample = next(
                item
                for item in aggregate
                if item["condition"] == "identity"
                and item["graph_kind"] == "raw_level0"
                and item["variant"] == "no_repair"
            )
            self.assertEqual(
                sample["metrics"]["quality.held_out.recall_at_1"]["count"], 2
            )

    def test_multi_seed_source_config_can_be_repo_root_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            phase1_config = config_dir / "phase1.json"
            phase2_config = config_dir / "phase2.json"
            output = root / "data"
            phase1_config.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "description": "relative source config prerequisite",
                        "data": {"num_samples": 20, "dimension": 5},
                        "subsets": {"count": 2},
                        "random_seed": 0,
                        "transformations": {
                            "types": [
                                "identity",
                                "orthogonal",
                                "general_linear",
                            ],
                            "general_linear_condition_number": 3.0,
                        },
                        "search": {
                            "k": 3,
                            "m": 4,
                            "ef_construction": 12,
                            "ef_search": 10,
                            "brute_force_batch_size": 8,
                        },
                        "stages": {"backend_probe": False, "run_tests": False},
                    }
                ),
                encoding="utf-8",
            )
            phase2_config.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "phase1": {
                            "transformation_dir": "unused_in_multiseed",
                            "subset_dir": "unused_in_multiseed",
                            "source_config": "configs/phase1.json",
                        },
                        "subsets": ["D_A"],
                        "transformations": {
                            "exact": ["identity", "rotation"],
                            "include_general_linear_stress": False,
                        },
                        "noise": {"families": ["gaussian"], "levels": [0.0]},
                        "graph": {
                            "kinds": ["raw_level0"],
                            "degree_budget_policy": "original_degree",
                            "candidate_search_k": 4,
                            "force_weak_connectivity": True,
                        },
                        "search": {
                            "m": 4,
                            "ef_construction": 12,
                            "ef_search": 10,
                            "evaluation_k": [1, 2],
                            "brute_force_batch_size": 8,
                        },
                        "queries": {"held_out_count_per_subset": 3},
                        "seeds": {
                            "data": [31, 32],
                            "transformation": 12,
                            "noise": 13,
                            "query": 14,
                        },
                        "stages": {"run_tests": False},
                    }
                ),
                encoding="utf-8",
            )

            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                manifest = run_complete_phase2(phase2_config, output)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(set(manifest["seeds"]), {"31", "32"})
            self.assertTrue(
                (output / "phase2_phase1_seeds" / "seed_31").exists()
            )

    def test_single_seed_missing_phase1_dirs_generates_from_source_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1_config = root / "phase1.json"
            phase2_config = root / "phase2.json"
            output = root / "data"
            phase1_config.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "description": "single-seed generated prerequisite",
                        "data": {"num_samples": 20, "dimension": 5},
                        "subsets": {"count": 2},
                        "random_seed": 0,
                        "transformations": {
                            "types": [
                                "identity",
                                "orthogonal",
                                "general_linear",
                            ],
                            "general_linear_condition_number": 3.0,
                        },
                        "search": {
                            "k": 3,
                            "m": 4,
                            "ef_construction": 12,
                            "ef_search": 10,
                            "brute_force_batch_size": 8,
                        },
                        "stages": {"backend_probe": False, "run_tests": False},
                    }
                ),
                encoding="utf-8",
            )
            phase2_config.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "phase1": {
                            "transformation_dir": "missing_transformations",
                            "subset_dir": "missing_subsets",
                            "source_config": str(phase1_config),
                        },
                        "subsets": ["D_A", "D_B"],
                        "transformations": {
                            "exact": ["identity", "rotation"],
                            "include_general_linear_stress": False,
                        },
                        "noise": {"families": ["gaussian"], "levels": [0.0]},
                        "graph": {
                            "kinds": ["raw_level0"],
                            "degree_budget_policy": "original_degree",
                            "candidate_search_k": 4,
                            "force_weak_connectivity": True,
                        },
                        "search": {
                            "m": 4,
                            "ef_construction": 12,
                            "ef_search": 10,
                            "evaluation_k": [1, 2],
                            "brute_force_batch_size": 8,
                        },
                        "queries": {"held_out_count_per_subset": 3},
                        "seeds": {
                            "data": [41],
                            "transformation": 12,
                            "noise": 13,
                            "query": 14,
                        },
                        "stages": {"run_tests": False},
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_complete_phase2(phase2_config, output)

            self.assertTrue(manifest["generated_phase1_inputs"])
            self.assertIn("seed_41/identity/D_A", manifest["conditions"])
            self.assertIn("seed_41/identity/D_B", manifest["conditions"])
            self.assertTrue(
                (output / "phase2_phase1_seeds" / "seed_41").exists()
            )


if __name__ == "__main__":
    unittest.main()
