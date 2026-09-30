import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from data.phase2_common import load_graph, save_graph
from data.phase2_graph import (
    directed_reachable_from_entry,
    extract_query_derived_graph,
    extract_raw_level0_graph,
    external_graph_search,
    repair_graph,
    structural_oracle_graph,
    weak_components,
)


class TestPhase2Graph(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.vectors = rng.normal(size=(30, 8)).astype(np.float32)
        self.global_ids = np.arange(100, 130, dtype=np.int64)
        raw = faiss.IndexHNSWFlat(8, 4, faiss.METRIC_L2)
        raw.hnsw.efConstruction = 30
        raw.hnsw.efSearch = 20
        self.index = faiss.IndexIDMap2(raw)
        self.index.add_with_ids(self.vectors, self.global_ids)

    def test_raw_extraction_is_read_only_and_uses_global_ids(self):
        raw = faiss.downcast_index(self.index.index)
        before = np.asarray(
            faiss.vector_to_array(raw.hnsw.neighbors)
        ).copy()
        graph = extract_raw_level0_graph(
            self.index, self.vectors, "D_A", "original_degree"
        )
        after = np.asarray(faiss.vector_to_array(raw.hnsw.neighbors))
        np.testing.assert_array_equal(before, after)
        self.assertEqual(graph["graph_kind"], "raw_level0")
        self.assertEqual(graph["maxM0"], raw.hnsw.nb_neighbors(0))
        valid = set(self.global_ids.tolist())
        for node, neighbors in graph["neighbors"].items():
            self.assertIn(node, valid)
            self.assertTrue(set(neighbors).issubset(valid))

    def test_query_graph_is_distinct_and_serializable(self):
        graph = extract_query_derived_graph(
            self.index,
            self.vectors,
            self.vectors,
            "D_A",
            5,
        )
        self.assertEqual(graph["graph_kind"], "query_derived")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            save_graph(path, graph)
            loaded = load_graph(path)
        np.testing.assert_array_equal(
            loaded["node_global_ids"], graph["node_global_ids"]
        )
        self.assertEqual(loaded["neighbors"], graph["neighbors"])

    def test_repair_and_oracle_respect_valid_ids(self):
        graph = extract_raw_level0_graph(self.index, self.vectors, "D_A")
        noisy = self.vectors.copy()
        noisy[:, 0] *= 3
        repaired, diagnostics = repair_graph(
            graph,
            self.index,
            self.vectors,
            noisy,
            candidate_search_k=10,
        )
        self.assertEqual(len(weak_components(repaired)), 1)
        self.assertEqual(diagnostics["unresolved_component_count"], 0)
        oracle = structural_oracle_graph(graph, noisy)
        valid = set(self.global_ids.tolist())
        for current in (repaired, oracle):
            for node, neighbors in current["neighbors"].items():
                self.assertNotIn(node, neighbors)
                self.assertEqual(len(neighbors), len(set(neighbors)))
                self.assertTrue(set(neighbors).issubset(valid))

    def test_external_search_returns_valid_neighbors(self):
        graph = extract_raw_level0_graph(self.index, self.vectors, "D_A")
        ids, distances = external_graph_search(
            graph,
            self.vectors,
            self.vectors[:3],
            k=5,
            ef_search=20,
            exclude_global_ids=self.global_ids[:3],
        )
        self.assertEqual(ids.shape, (3, 5))
        self.assertEqual(distances.shape, (3, 5))
        self.assertFalse(np.any(ids == self.global_ids[:3, None]))
        self.assertTrue(set(ids.ravel()).issubset(set(self.global_ids.tolist())))

    def test_directed_reachability_is_reported_after_repair(self):
        graph = extract_raw_level0_graph(self.index, self.vectors, "D_A")
        nodes = [int(value) for value in graph["node_global_ids"]]
        entry = int(graph["original_entry_point"])
        graph["neighbors"] = {node: [] for node in nodes}
        graph["neighbors"][entry] = [nodes[0]] if nodes[0] != entry else [nodes[1]]
        self.assertLess(len(directed_reachable_from_entry(graph)), len(nodes))

        repaired, diagnostics = repair_graph(
            graph,
            self.index,
            self.vectors,
            self.vectors,
            candidate_search_k=8,
        )

        self.assertEqual(len(directed_reachable_from_entry(repaired)), len(nodes))
        self.assertEqual(diagnostics["directed_unreachable_from_entry_count"], 0)


if __name__ == "__main__":
    unittest.main()
