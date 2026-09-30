import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.prepare_phase1_subsets import SUBSET_NAMES, prepare_phase1_subsets


class TestPhase1Subsets(unittest.TestCase):
    def test_split_persists_shared_vectors_and_id_mappings(self):
        rng = np.random.default_rng(7)
        vectors = rng.random((10, 8), dtype=np.float32)
        metadata = [{"id": f"item_{i}"} for i in range(len(vectors))]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vector_path = root / "vectors.npy"
            metadata_path = root / "metadata.json"
            output_dir = root / "phase1"
            np.save(vector_path, vectors)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            manifest = prepare_phase1_subsets(
                vector_path=vector_path,
                metadata_path=metadata_path,
                output_dir=output_dir,
                seed=42,
            )

            self.assertEqual(set(manifest["subsets"]), set(SUBSET_NAMES))
            self.assertEqual(
                sorted(subset["size"] for subset in manifest["subsets"].values()),
                [3, 3, 4],
            )

            observed_global_ids = []
            for subset_name in SUBSET_NAMES:
                subset_dir = output_dir / subset_name
                subset_vectors = np.load(subset_dir / "shared_vectors.npy")
                id_mapping = json.loads(
                    (subset_dir / "id_mapping.json").read_text(encoding="utf-8")
                )

                self.assertEqual(
                    [record["local_id"] for record in id_mapping],
                    list(range(len(subset_vectors))),
                )

                observed_global_ids.extend(
                    record["global_id"] for record in id_mapping
                )
                for local_id, record in enumerate(id_mapping):
                    np.testing.assert_array_equal(
                        subset_vectors[local_id],
                        vectors[record["global_id"]],
                    )

            self.assertEqual(sorted(observed_global_ids), list(range(len(vectors))))


if __name__ == "__main__":
    unittest.main()
