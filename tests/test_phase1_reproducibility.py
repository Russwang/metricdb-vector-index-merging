import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.data_generator import generate_synthetic_data
from data.run_phase1 import run_complete_phase1


class TestPhase1Reproducibility(unittest.TestCase):
    def test_data_generator_repeats_exactly_for_same_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"

            generate_synthetic_data(20, 7, first, seed=123)
            generate_synthetic_data(20, 7, second, seed=123)

            np.testing.assert_array_equal(
                np.load(first / "vectors.npy"),
                np.load(second / "vectors.npy"),
            )
            self.assertEqual(
                (first / "metadata.json").read_text(encoding="utf-8"),
                (second / "metadata.json").read_text(encoding="utf-8"),
            )

    def test_complete_pipeline_uses_one_config_and_records_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "phase1_test.json"
            output_root = root / "output"
            config_path.write_text(
                json.dumps(
                    {
                        "profile": "smoke_test",
                        "description": "unit test",
                        "data": {"num_samples": 24, "dimension": 8},
                        "subsets": {"count": 4},
                        "random_seed": 19,
                        "transformations": {
                            "types": [
                                "identity",
                                "orthogonal",
                                "general_linear",
                            ],
                            "general_linear_condition_number": 5.0,
                        },
                        "search": {
                            "k": 3,
                            "m": 4,
                            "ef_construction": 20,
                            "ef_search": 16,
                            "brute_force_batch_size": 8,
                        },
                        "stages": {
                            "backend_probe": True,
                            "run_tests": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_complete_phase1(
                config_path=config_path,
                output_root=output_root,
            )

            self.assertEqual(manifest["profile"], "smoke_test")
            self.assertEqual(
                manifest["result_scope"],
                "smoke_test_not_dissertation_formal_results",
            )
            self.assertFalse(
                manifest["phase2_backend_decision"][
                    "modify_faiss_raw_neighbor_arrays"
                ]
            )
            self.assertEqual(
                manifest["config"]["subsets"]["names"],
                ["D_A", "D_B", "D_C", "D_D"],
            )
            self.assertEqual(
                set(manifest["stages"]),
                {
                    "data_generation",
                    "subset_preparation",
                    "transformation_preparation",
                    "brute_force_baseline",
                    "hnsw_rebuild_baseline",
                    "backend_probe",
                },
            )
            self.assertTrue((output_root / "phase1_run_manifest.json").exists())

            for stage in manifest["stages"].values():
                runtime = stage["manifest"]["runtime"]
                self.assertGreaterEqual(runtime["duration_seconds"], 0.0)
                self.assertIn("python", runtime["environment"])
                self.assertIn("numpy", runtime["environment"])
                self.assertIn("faiss", runtime["environment"])

            transformation_manifest = manifest["stages"][
                "transformation_preparation"
            ]["manifest"]
            self.assertEqual(transformation_manifest["hnsw"]["m"], 4)
            self.assertEqual(transformation_manifest["hnsw"]["ef_search"], 16)
            self.assertEqual(
                transformation_manifest["subset_names"],
                ["D_A", "D_B", "D_C", "D_D"],
            )


if __name__ == "__main__":
    unittest.main()
