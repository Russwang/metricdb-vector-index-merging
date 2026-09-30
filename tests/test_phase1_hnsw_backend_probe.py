import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from data.probe_phase1_hnsw_backend import (
    extract_hnsw_layers,
    run_faiss_capability_probe,
)


class TestPhase1HnswBackendProbe(unittest.TestCase):
    def test_extract_hnsw_layers_reads_all_nodes_and_valid_external_ids(self):
        rng = np.random.default_rng(4)
        vectors = rng.normal(size=(20, 6)).astype(np.float32)
        global_ids = np.arange(100, 120, dtype=np.int64)
        raw = faiss.IndexHNSWFlat(6, 4, faiss.METRIC_L2)
        index = faiss.IndexIDMap2(raw)
        index.add_with_ids(vectors, global_ids)

        graph = extract_hnsw_layers(index)

        self.assertEqual(set(graph), set(global_ids.tolist()))
        for layers in graph.values():
            self.assertIn(0, layers)
            for neighbors in layers.values():
                self.assertTrue(set(neighbors).issubset(set(global_ids.tolist())))

    def test_capability_probe_mutates_searches_and_reloads_disposable_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_faiss_capability_probe(Path(temporary_directory))

            mutation = result["raw_mutation_probe"]
            self.assertTrue(mutation["mutation_visible_in_memory"])
            self.assertTrue(mutation["search_completed_after_mutation"])
            self.assertTrue(mutation["reload_preserved_mutation"])
            self.assertTrue(mutation["search_completed_after_reload"])
            self.assertFalse(
                result["safety_conclusion"]["safe_high_level_link_edit_api"]
            )
            self.assertTrue(Path(mutation["modified_index"]).exists())


if __name__ == "__main__":
    unittest.main()
