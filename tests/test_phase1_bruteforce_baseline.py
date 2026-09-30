import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.prepare_phase1_transformations import EXPERIMENT_NAMES, SUBSET_NAMES
from data.run_phase1_bruteforce_baseline import (
    exact_knn_squared_l2,
    run_phase1_bruteforce_baseline,
)


class TestPhase1BruteforceBaseline(unittest.TestCase):
    def test_exact_knn_excludes_self_and_breaks_ties_by_global_id(self):
        vectors = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )
        global_ids = np.array([30, 20, 10, 40], dtype=np.int64)

        neighbor_ids, neighbor_distances = exact_knn_squared_l2(
            query_vectors=vectors[:1],
            corpus_vectors=vectors,
            corpus_global_ids=global_ids,
            k=2,
            exclude_query_global_ids=np.array([30], dtype=np.int64),
        )

        np.testing.assert_array_equal(neighbor_ids[0], np.array([10, 20]))
        np.testing.assert_allclose(neighbor_distances[0], np.array([1.0, 1.0]))

    def test_run_baseline_persists_ground_truth_for_each_experiment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transformation_dir = root / "transformations"
            output_dir = root / "bruteforce"
            transformation_dir.mkdir()

            for experiment_name in EXPERIMENT_NAMES:
                for subset_name in SUBSET_NAMES:
                    subset_dir = transformation_dir / experiment_name / subset_name
                    subset_dir.mkdir(parents=True)

                    if subset_name == "D_A":
                        vectors = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
                        ids = [0, 2]
                    elif subset_name == "D_B":
                        vectors = np.array([[0.0, 3.0], [10.0, 0.0]], dtype=np.float32)
                        ids = [3, 10]
                    else:
                        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
                        ids = [1]

                    np.save(subset_dir / "transformed_vectors.npy", vectors)
                    id_mapping = [
                        {
                            "subset_id": subset_name,
                            "local_id": local_id,
                            "global_id": global_id,
                            "item_id": f"item_{global_id}",
                        }
                        for local_id, global_id in enumerate(ids)
                    ]
                    (subset_dir / "id_mapping.json").write_text(
                        json.dumps(id_mapping), encoding="utf-8"
                    )

            (transformation_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "experiments": {
                            experiment_name: {"subsets": {}}
                            for experiment_name in EXPERIMENT_NAMES
                        }
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_phase1_bruteforce_baseline(
                transformation_dir=transformation_dir,
                output_dir=output_dir,
                k=2,
                batch_size=2,
            )

            self.assertEqual(set(manifest["experiments"]), set(EXPERIMENT_NAMES))
            identity_dir = output_dir / "identity"
            query_ids = np.load(identity_dir / "query_global_ids.npy")
            neighbor_ids = np.load(identity_dir / "neighbor_global_ids.npy")
            neighbor_distances = np.load(identity_dir / "neighbor_distances.npy")

            self.assertEqual(neighbor_ids.shape, (5, 2))
            row_for_global_zero = int(np.where(query_ids == 0)[0][0])
            np.testing.assert_array_equal(
                neighbor_ids[row_for_global_zero], np.array([1, 2])
            )
            np.testing.assert_allclose(
                neighbor_distances[row_for_global_zero], np.array([1.0, 4.0])
            )

            records = json.loads(
                (identity_dir / "union_records.json").read_text(encoding="utf-8")
            )
            self.assertEqual([record["union_row_id"] for record in records], list(range(5)))


if __name__ == "__main__":
    unittest.main()
