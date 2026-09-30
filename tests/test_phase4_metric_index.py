import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from data.phase2_graph import extract_raw_level0_graph
from data.phase4_benchmark import QUERY_BENCHMARK_SCHEMA, benchmark_queries
from main import SimpleMetricRelation
from modules.metric_indexing.indexing import MetricIndex


class TestPhase4MetricIndex(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.vectors = rng.normal(size=(40, 6)).astype(np.float32)
        self.relation = SimpleMetricRelation(
            "test", self.vectors, [{"id": f"id-{i}"} for i in range(40)]
        )
        raw = faiss.IndexHNSWFlat(6, 8, faiss.METRIC_L2)
        raw.hnsw.efSearch = 40
        wrapped = faiss.IndexIDMap2(raw)
        self.global_ids = np.arange(200, 240, dtype=np.int64)
        wrapped.add_with_ids(self.vectors, self.global_ids)
        self.graph = extract_raw_level0_graph(wrapped, self.vectors, "union")

    def test_both_backends_return_legal_string_ids(self):
        for index in (
            MetricIndex(ef_search=64),
            MetricIndex("external_graph", graph=self.graph, item_ids=self.relation.ids, ef_search=64),
        ):
            index.build(self.relation)
            result = index.search(self.vectors[0], 5)
            self.assertEqual(len(result), 5)
            self.assertTrue(set(result).issubset(set(self.relation.ids)))
            self.assertTrue(all(isinstance(value, str) for value in result))

    def test_validation(self):
        with self.assertRaisesRegex(RuntimeError, "before build"):
            MetricIndex().search(self.vectors[0], 2)
        duplicate = SimpleMetricRelation("bad", self.vectors[:2], [{"id": "x"}, {"id": "x"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MetricIndex().build(duplicate)
        index = MetricIndex()
        index.build(self.relation)
        for bad_k in (0, 41):
            with self.assertRaises(ValueError):
                index.search(self.vectors[0], bad_k)
        with self.assertRaisesRegex(ValueError, "shape"):
            index.search(np.zeros(7, dtype=np.float32), 2)

    def test_external_alignment_validation(self):
        bad = dict(self.graph)
        bad["node_global_ids"] = bad["node_global_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "row-aligned"):
            MetricIndex("external_graph", graph=bad).build(self.relation)

    def test_benchmark_raw_schema_and_capacity(self):
        index = MetricIndex(m=8, ef_search=32)
        index.build(self.relation)
        result = benchmark_queries(index, self.vectors[:4], k=3, warmup_queries=2, repetitions=2)
        self.assertEqual(result["schema_version"], QUERY_BENCHMARK_SCHEMA)
        self.assertEqual(len(result["raw_queries"]), 8)
        self.assertEqual(result["warmup_query_count"], 2)
        self.assertGreater(result["summary"]["qps"], 0)
        self.assertIn("ndis", result["raw_queries"][0])
        with tempfile.TemporaryDirectory() as directory:
            size = index.serialized_size_bytes(Path(directory) / "index.faiss")
            capacity = index.capacity_metrics(size)
        self.assertGreater(capacity["directed_edge_count"], 0)
        self.assertEqual(capacity["serialized_artifact_bytes"], size)

    def test_external_work_counters(self):
        index = MetricIndex("external_graph", graph=self.graph, item_ids=self.relation.ids)
        index.build(self.relation)
        index.search(self.vectors[0], 3)
        self.assertGreater(index.last_search_diagnostics["distance_evaluations"], 0)
        self.assertGreater(index.last_search_diagnostics["visited_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
