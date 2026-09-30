"""Focused tests for Stage 13/14 external-search and indexed cross-kNN changes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from data.phase2_conditions import build_hnsw
from data.phase2_graph import external_graph_search
from data.phase3_io import Phase2SubsetInput
from data.phase3_merge import add_indexed_cross_knn_links, no_cross_union_graph, select_bridge_nodes


def _graph(ids: np.ndarray, subset: str) -> dict:
    neighbours = {int(node): [] for node in ids}
    for left, right in zip(ids[:-1], ids[1:]):
        neighbours[int(left)].append(int(right))
        neighbours[int(right)].append(int(left))
    return {
        "node_global_ids": ids,
        "neighbors": neighbours,
        "edge_distances": {int(node): [1.0] * len(values) for node, values in neighbours.items()},
        "source_subset": subset,
        "graph_kind": "raw_level0",
        "variant": "practical_repair",
        "node_levels": np.ones(len(ids), dtype=np.int64),
        "upper_layer_nodes_by_level": [],
        "original_entry_point": int(ids[0]),
        "hnsw_M": 4,
        "maxM0": 8,
        "original_degrees": np.asarray([len(neighbours[int(node)]) for node in ids]),
        "degree_budget": np.asarray([len(neighbours[int(node)]) for node in ids]),
    }


class TestStage13And14(unittest.TestCase):
    def test_external_search_counts_only_lazy_visited_distances(self):
        ids = np.arange(10, dtype=np.int64)
        graph = _graph(ids, "D_A")
        vectors = np.arange(10, dtype=np.float32).reshape(-1, 1)
        result_ids, _, diagnostics = external_graph_search(
            graph,
            vectors,
            np.asarray([[0.0]], dtype=np.float32),
            k=1,
            ef_search=2,
            return_diagnostics=True,
        )
        self.assertEqual(int(result_ids[0, 0]), 0)
        self.assertEqual(diagnostics["distance_mode"], "lazy_visited_nodes")
        self.assertLess(diagnostics["distance_evaluation_count"], len(ids))
        self.assertEqual(diagnostics["visited_node_count"], 2)

    def test_indexed_cross_knn_queries_existing_target_index_and_reranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_vectors = np.asarray([[0.0], [10.0], [20.0]], dtype=np.float32)
            second_vectors = np.asarray([[1.0], [11.0], [30.0]], dtype=np.float32)
            first_ids = np.asarray([100, 101, 102], dtype=np.int64)
            second_ids = np.asarray([200, 201, 202], dtype=np.int64)
            first_path = root / "first.faiss"
            second_path = root / "second.faiss"
            faiss.write_index(build_hnsw(first_vectors, first_ids, 4, 12, 32), str(first_path))
            faiss.write_index(build_hnsw(second_vectors, second_ids, 4, 12, 32), str(second_path))
            identity = {"matrix": [[1.0]], "translation": [0.0]}
            first = Phase2SubsetInput(
                7, "identity", "D_A", first_vectors, first_ids,
                np.zeros((1, 1), dtype=np.float32), _graph(first_ids, "D_A"), None,
                root, root / "first.graph", first_vectors, first_path, identity,
            )
            second = Phase2SubsetInput(
                7, "identity", "D_B", second_vectors, second_ids,
                np.zeros((1, 1), dtype=np.float32), _graph(second_ids, "D_B"), None,
                root, root / "second.graph", second_vectors, second_path, identity,
            )
            union, base = no_cross_union_graph(first, second)
            merged, diagnostics = add_indexed_cross_knn_links(
                union, base, first, second, cross_k=1, random_seed=3,
                cross_link_config={
                    "bridge_budget": {"mode": "fixed", "count": 2},
                    "candidate_k": 3,
                    "candidate_ef_search": 32,
                },
            )
            self.assertEqual(diagnostics["candidate_hnsw_call_count"], 4)
            self.assertEqual(diagnostics["cross_candidate_distance_count"], 12)
            self.assertGreater(diagnostics["added_cross_link_count"], 0)
            self.assertEqual(merged["cross_link_strategy"], "indexed_local_hnsw_rerank")
            self.assertTrue(all(
                target in set(second_ids) or target in set(first_ids)
                for targets in merged["neighbors"].values() for target in targets
            ))

            propagated_union, propagated_base = no_cross_union_graph(first, second)
            propagated, propagated_diagnostics = add_indexed_cross_knn_links(
                propagated_union, propagated_base, first, second, cross_k=1, random_seed=3,
                cross_link_config={
                    "bridge_budget": {"mode": "fixed", "count": 2},
                    "candidate_k": 3,
                    "candidate_ef_search": 32,
                    "cross_degree_cap": 1,
                    "propagation_hops": 1,
                },
            )
            subset = propagated["source_subset_by_node"]
            self.assertGreater(propagated_diagnostics["propagation_distance_count"], 0)
            self.assertTrue(all(
                sum(subset[target] != subset[source] for target in targets) <= 1
                for source, targets in propagated["neighbors"].items()
            ))

    def test_high_degree_bridge_selection_is_deterministic(self):
        ids = np.asarray([10, 11, 12], dtype=np.int64)
        graph = _graph(ids, "D_A")
        graph["neighbors"][11].append(10)
        first, second = select_bridge_nodes(
            ids,
            ids + 100,
            {"bridge_budget": {"mode": "fixed", "count": 1}, "bridge_selection": "high_degree"},
            9,
            graph,
            {**graph, "node_global_ids": ids + 100, "neighbors": {110: [], 111: [110, 112], 112: []}},
        )
        self.assertEqual(first.tolist(), [11])
        self.assertEqual(second.tolist(), [111])

    def test_bounded_propagation_respects_neighbor_and_candidate_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_vectors = np.arange(5, dtype=np.float32).reshape(-1, 1)
            second_vectors = (np.arange(5, dtype=np.float32) + 0.25).reshape(-1, 1)
            first_ids = np.arange(100, 105, dtype=np.int64)
            second_ids = np.arange(200, 205, dtype=np.int64)
            first_path, second_path = root / "first.faiss", root / "second.faiss"
            faiss.write_index(build_hnsw(first_vectors, first_ids, 4, 12, 32), str(first_path))
            faiss.write_index(build_hnsw(second_vectors, second_ids, 4, 12, 32), str(second_path))
            identity = {"matrix": [[1.0]], "translation": [0.0]}
            first = Phase2SubsetInput(
                7, "identity", "D_A", first_vectors, first_ids,
                np.zeros((1, 1), dtype=np.float32), _graph(first_ids, "D_A"), None,
                root, root / "first.graph", first_vectors, first_path, identity,
            )
            second = Phase2SubsetInput(
                7, "identity", "D_B", second_vectors, second_ids,
                np.zeros((1, 1), dtype=np.float32), _graph(second_ids, "D_B"), None,
                root, root / "second.graph", second_vectors, second_path, identity,
            )
            first.graph["edge_distances"][101] = [10.0, 1.0]
            second.graph["edge_distances"][201] = [10.0, 1.0]
            union, base = no_cross_union_graph(first, second)
            _, diagnostics = add_indexed_cross_knn_links(
                union, base, first, second, cross_k=2, random_seed=1,
                cross_link_config={
                    "bridge_budget": {"mode": "fixed", "count": 1},
                    "bridge_selection": "high_degree",
                    "candidate_k": 5,
                    "candidate_ef_search": 32,
                    "cross_degree_cap": 2,
                    "propagation_hops": 1,
                    "propagation_neighbor_budget": 1,
                    "propagation_candidate_cap": 1,
                },
            )
            self.assertLessEqual(diagnostics["propagated_neighbor_count"], 2)
            self.assertEqual(diagnostics["propagated_neighbor_ids"], [102, 202])
            self.assertLessEqual(diagnostics["max_propagated_candidates_per_neighbor"], 1)
            self.assertLessEqual(diagnostics["propagation_distance_count"], 4)


if __name__ == "__main__":
    unittest.main()
