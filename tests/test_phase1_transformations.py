import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from data.prepare_phase1_transformations import (
    EXPERIMENT_NAMES,
    SUBSET_NAMES,
    prepare_phase1_transformations,
)


class TestPhase1Transformations(unittest.TestCase):
    def test_controlled_transformations_have_expected_geometry(self):
        rng = np.random.default_rng(9)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subset_dir = root / "subsets"
            output_dir = root / "transformations"
            subset_dir.mkdir()

            source_vectors = {}
            for subset_name in SUBSET_NAMES:
                vectors = rng.normal(size=(12, 6)).astype(np.float32)
                source_vectors[subset_name] = vectors
                current_subset_dir = subset_dir / subset_name
                current_subset_dir.mkdir()
                np.save(current_subset_dir / "shared_vectors.npy", vectors)
                id_mapping = [
                    {
                        "subset_id": subset_name,
                        "local_id": local_id,
                        "global_id": global_id,
                        "item_id": f"item_{global_id}",
                    }
                    for local_id, global_id in enumerate(range(len(vectors)))
                ]
                (current_subset_dir / "id_mapping.json").write_text(
                    json.dumps(id_mapping), encoding="utf-8"
                )

            (subset_dir / "manifest.json").write_text(
                json.dumps({"dimension": 6}), encoding="utf-8"
            )
            manifest = prepare_phase1_transformations(
                subset_dir=subset_dir,
                output_dir=output_dir,
                seed=42,
                condition_number=8.0,
            )

            self.assertEqual(set(manifest["experiments"]), set(EXPERIMENT_NAMES))
            for experiment_name in EXPERIMENT_NAMES:
                shared_a = np.load(
                    output_dir / experiment_name / "D_A" / "shared_vectors.npy"
                )
                local_a = np.load(
                    output_dir / experiment_name / "D_A" / "local_vectors.npy"
                )
                transformed_a = np.load(
                    output_dir
                    / experiment_name
                    / "D_A"
                    / "transformed_vectors.npy"
                )
                np.testing.assert_array_equal(shared_a, source_vectors["D_A"])
                np.testing.assert_array_equal(local_a, source_vectors["D_A"])
                np.testing.assert_array_equal(transformed_a, source_vectors["D_A"])

            for subset_name in ("D_B", "D_C"):
                shared = source_vectors[subset_name]
                shared_distances = self._pairwise_squared_l2(shared)

                identity_local = np.load(
                    output_dir
                    / "identity"
                    / subset_name
                    / "local_vectors.npy"
                )
                identity_transformed = np.load(
                    output_dir / "identity" / subset_name / "transformed_vectors.npy"
                )
                np.testing.assert_array_equal(identity_local, shared)
                np.testing.assert_array_equal(identity_transformed, shared)

                orthogonal_dir = output_dir / "orthogonal" / subset_name
                orthogonal_local = np.load(orthogonal_dir / "local_vectors.npy")
                orthogonal_transformed = np.load(
                    orthogonal_dir / "transformed_vectors.npy"
                )
                orthogonal_parameters = np.load(
                    orthogonal_dir / "transformation.npz"
                )
                q = orthogonal_parameters["matrix"]
                np.testing.assert_allclose(q.T @ q, np.eye(6), atol=1e-10)
                np.testing.assert_allclose(
                    self._pairwise_squared_l2(orthogonal_local),
                    shared_distances,
                    rtol=1e-5,
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    orthogonal_transformed, shared, rtol=1e-6, atol=1e-6
                )
                orthogonal_index = faiss.read_index(
                    str(orthogonal_dir / "local_hnsw.faiss")
                )
                _, result_ids = orthogonal_index.search(orthogonal_local[:1], 1)
                self.assertEqual(result_ids[0, 0], 0)

                general_dir = output_dir / "general_linear" / subset_name
                general_local = np.load(general_dir / "local_vectors.npy")
                general_transformed = np.load(
                    general_dir / "transformed_vectors.npy"
                )
                general_parameters = np.load(
                    general_dir / "transformation.npz"
                )
                matrix = general_parameters["matrix"]
                self.assertAlmostEqual(np.linalg.cond(matrix), 8.0, places=8)
                self.assertFalse(
                    np.allclose(
                        self._pairwise_squared_l2(general_local),
                        shared_distances,
                        rtol=1e-3,
                        atol=1e-3,
                    )
                )
                np.testing.assert_allclose(
                    general_transformed, shared, rtol=1e-5, atol=1e-5
                )
                general_index = faiss.read_index(
                    str(general_dir / "local_hnsw.faiss")
                )
                self.assertEqual(general_index.ntotal, len(general_local))

            matrix_b = np.load(
                output_dir / "orthogonal" / "D_B" / "transformation.npz"
            )["matrix"]
            matrix_c = np.load(
                output_dir / "orthogonal" / "D_C" / "transformation.npz"
            )["matrix"]
            self.assertFalse(np.allclose(matrix_b, matrix_c))

    @staticmethod
    def _pairwise_squared_l2(vectors):
        differences = vectors[:, None, :] - vectors[None, :, :]
        return np.sum(differences * differences, axis=-1)


if __name__ == "__main__":
    unittest.main()
