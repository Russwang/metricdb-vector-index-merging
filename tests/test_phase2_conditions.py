import unittest

import numpy as np

from data.phase2_conditions import (
    add_controlled_noise,
    affine_isometry,
    apply_affine,
    inverse_affine,
)


class TestPhase2Conditions(unittest.TestCase):
    def test_exact_conditions_preserve_pairwise_l2_and_determinants(self):
        vectors = np.random.default_rng(1).normal(size=(20, 7)).astype(np.float32)
        original = np.sum(
            (vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=2
        )
        for offset, name in enumerate(
            (
                "identity",
                "rotation",
                "reflection",
                "translation",
                "orthogonal_translation",
            )
        ):
            matrix, translation = affine_isometry(
                name, vectors.shape[1], np.random.default_rng(10 + offset)
            )
            transformed = apply_affine(vectors, matrix, translation)
            observed = np.sum(
                (transformed[:, None, :] - transformed[None, :, :]) ** 2,
                axis=2,
            )
            np.testing.assert_allclose(observed, original, atol=1e-5)
            if name == "rotation":
                self.assertAlmostEqual(np.linalg.det(matrix), 1.0, places=8)
            if name == "reflection":
                self.assertAlmostEqual(np.linalg.det(matrix), -1.0, places=8)

    def test_inverse_affine_recovers_shared_vectors(self):
        shared = np.random.default_rng(4).normal(size=(15, 6)).astype(np.float32)
        matrix, translation = affine_isometry(
            "orthogonal_translation", 6, np.random.default_rng(5)
        )
        local = inverse_affine(shared, matrix, translation)
        recovered = apply_affine(local, matrix, translation)
        np.testing.assert_allclose(recovered, shared, atol=1e-5)

    def test_noise_is_reproducible_and_matches_target_scale(self):
        vectors = np.random.default_rng(6).normal(size=(2000, 16)).astype(np.float32)
        for family in ("gaussian", "laplacian"):
            first, first_meta = add_controlled_noise(
                vectors, family, 0.1, np.random.default_rng(7)
            )
            second, second_meta = add_controlled_noise(
                vectors, family, 0.1, np.random.default_rng(7)
            )
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first_meta, second_meta)
            self.assertAlmostEqual(
                first_meta["measured_relative_perturbation"], 0.1, delta=0.01
            )


if __name__ == "__main__":
    unittest.main()
