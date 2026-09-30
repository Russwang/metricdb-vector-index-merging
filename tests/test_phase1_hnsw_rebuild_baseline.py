import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from data.prepare_phase1_transformations import EXPERIMENT_NAMES
from data.run_phase1_bruteforce_baseline import exact_knn_squared_l2
from data.run_phase1_hnsw_rebuild_baseline import (
    recall_at_k,
    run_phase1_hnsw_rebuild_baseline,
)


class TestPhase1HnswRebuildBaseline(unittest.TestCase):
    def test_recall_at_k_uses_neighbor_set_overlap(self):
        predicted = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        ground_truth = np.array([[3, 2, 8], [4, 9, 10]], dtype=np.int64)

        mean_recall, per_query_recall = recall_at_k(predicted, ground_truth)

        np.testing.assert_allclose(per_query_recall, np.array([2 / 3, 1 / 3]))
        self.assertAlmostEqual(mean_recall, 0.5)

    def test_full_rebuild_persists_index_results_and_metrics(self):
        rng = np.random.default_rng(17)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            brute_force_dir = root / "bruteforce"
            output_dir = root / "hnsw"
            brute_force_dir.mkdir()
            experiment_manifest = {}

            for experiment_name in EXPERIMENT_NAMES:
                experiment_dir = brute_force_dir / experiment_name
                experiment_dir.mkdir()
                vectors = rng.normal(size=(30, 5)).astype(np.float32)
                global_ids = rng.permutation(np.arange(100, 130)).astype(np.int64)
                ground_truth_ids, _ = exact_knn_squared_l2(
                    query_vectors=vectors,
                    corpus_vectors=vectors,
                    corpus_global_ids=global_ids,
                    k=4,
                    exclude_query_global_ids=global_ids,
                )

                vector_path = experiment_dir / "union_transformed_vectors.npy"
                union_id_path = experiment_dir / "union_global_ids.npy"
                query_id_path = experiment_dir / "query_global_ids.npy"
                ground_truth_path = experiment_dir / "neighbor_global_ids.npy"
                np.save(vector_path, vectors)
                np.save(union_id_path, global_ids)
                np.save(query_id_path, global_ids)
                np.save(ground_truth_path, ground_truth_ids)

                experiment_manifest[experiment_name] = {
                    "union_transformed_vectors": str(vector_path),
                    "union_global_ids": str(union_id_path),
                    "query_global_ids": str(query_id_path),
                    "neighbor_global_ids": str(ground_truth_path),
                }

            (brute_force_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "k": 4,
                        "exclude_self": True,
                        "experiments": experiment_manifest,
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_phase1_hnsw_rebuild_baseline(
                brute_force_dir=brute_force_dir,
                output_dir=output_dir,
                m=8,
                ef_construction=40,
                ef_search=32,
            )

            self.assertEqual(set(manifest["experiments"]), set(EXPERIMENT_NAMES))
            for experiment_name, experiment in manifest["experiments"].items():
                self.assertGreaterEqual(experiment["recall_at_k"], 0.0)
                self.assertLessEqual(experiment["recall_at_k"], 1.0)
                self.assertGreater(experiment["index_size_bytes"], 0)

                index = faiss.read_index(experiment["index_path"])
                self.assertEqual(index.ntotal, 30)
                neighbor_ids = np.load(experiment["neighbor_global_ids"])
                per_query_recall = np.load(experiment["per_query_recall"])
                self.assertEqual(neighbor_ids.shape, (30, 4))
                self.assertEqual(per_query_recall.shape, (30,))


if __name__ == "__main__":
    unittest.main()
