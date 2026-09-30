import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.phase2_common import save_graph
from data.phase3_common import load_phase3_config
from data.phase3_io import (
    check_pairwise_phase2_inputs,
    load_pairwise_phase2_inputs,
)
from data.phase3_metrics import (
    graph_statistics,
    random_cross_link_cost,
    recall_at_k,
    sampled_cross_knn_cost,
    source_target_recall_at_k,
    subset_lookup,
)
from data.phase3_baselines import dual_index_search_external
from data.phase3_merge import (
    count_cross_subset_edges,
    no_cross_union_graph,
    random_cross_link_union_graph,
    sampled_cross_knn_union_graph,
)
from data.run_phase3 import run_complete_phase3


def _base_config(root: Path) -> dict:
    return {
        "profile": "smoke_test",
        "phase2": {
            "compiled_graph_root": str(root / "phase2_compiled_graphs"),
            "repaired_graph_root": str(root / "phase2_repaired_graphs"),
            "manifest": str(root / "phase2_run_manifest.json"),
            "recommended_pairwise_config": "configs/phase2_pairwise_validation.json",
        },
        "subsets": {"pair": ["D_A", "D_B"]},
        "input": {
            "data_seeds": [7],
            "conditions": ["identity"],
            "graph_kind": "raw_level0",
            "within_subset_variant": "practical_repair",
            "require_raw_no_repair_graph": True,
        },
        "methods": {
            "enabled": ["bruteforce_ground_truth"],
            "first_round": ["bruteforce_ground_truth"],
            "second_round_planned": [],
        },
        "search": {
            "m": 4,
            "ef_construction": 12,
            "ef_search": 10,
            "evaluation_k": [1, 2],
            "brute_force_batch_size": 8,
        },
        "cross_links": {
            "sample_ratios": [0.5, 1.0],
            "cross_k": [1, 2],
            "random_seed": 99,
        },
        "metrics": {"latency_warmup_queries": 1, "latency_repetitions": 1},
        "output": {
            "root": str(root),
            "manifest": str(root / "phase3_run_manifest.json"),
        },
        "stages": {
            "run_tests": False,
            "readiness_check_only": True,
            "fail_on_missing_inputs": False,
        },
    }


def _write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config), encoding="utf-8")


def _graph(ids: np.ndarray, subset: str, variant: str) -> dict:
    neighbors = {}
    edge_distances = {}
    for index, node in enumerate(ids):
        target = int(ids[(index + 1) % len(ids)]) if len(ids) > 1 else int(node)
        values = [] if target == int(node) else [target]
        neighbors[int(node)] = values
        edge_distances[int(node)] = [1.0 for _ in values]
    return {
        "node_global_ids": ids.astype(np.int64),
        "neighbors": neighbors,
        "edge_distances": edge_distances,
        "source_subset": subset,
        "graph_kind": "raw_level0",
        "directed": True,
        "node_levels": np.ones(len(ids), dtype=np.int64),
        "upper_layer_nodes_by_level": [],
        "original_entry_point": int(ids[0]),
        "hnsw_M": 4,
        "maxM0": 8,
        "original_degrees": np.array([len(neighbors[int(node)]) for node in ids]),
        "degree_budget": np.array([len(neighbors[int(node)]) for node in ids]),
        "variant": variant,
    }


def _complete_graph(ids: np.ndarray, subset: str, variant: str) -> dict:
    neighbors = {}
    edge_distances = {}
    for node in ids:
        values = [int(target) for target in ids if int(target) != int(node)]
        neighbors[int(node)] = values
        edge_distances[int(node)] = [1.0 for _ in values]
    return {
        "node_global_ids": ids.astype(np.int64),
        "neighbors": neighbors,
        "edge_distances": edge_distances,
        "source_subset": subset,
        "graph_kind": "raw_level0",
        "directed": True,
        "node_levels": np.ones(len(ids), dtype=np.int64),
        "upper_layer_nodes_by_level": [],
        "original_entry_point": int(ids[0]),
        "hnsw_M": 4,
        "maxM0": 8,
        "original_degrees": np.array([len(neighbors[int(node)]) for node in ids]),
        "degree_budget": np.array([len(neighbors[int(node)]) for node in ids]),
        "variant": variant,
    }


def _write_subset(root: Path, seed: int, condition: str, subset: str, ids: np.ndarray):
    compiled = root / "phase2_compiled_graphs" / f"seed_{seed}" / condition / subset
    compiled.mkdir(parents=True)
    np.save(compiled / "transformed_vectors.npy", np.arange(len(ids) * 3).reshape(len(ids), 3).astype(np.float32))
    np.save(compiled / "global_ids.npy", ids.astype(np.int64))
    np.save(compiled / "held_out_queries.npy", np.zeros((2, 3), dtype=np.float32))
    for variant in ("no_repair", "practical_repair"):
        path = (
            root
            / "phase2_repaired_graphs"
            / f"seed_{seed}"
            / condition
            / subset
            / "raw_level0"
            / variant
            / "graph.json"
        )
        save_graph(path, _graph(ids, subset, variant))


class TestPhase3Common(unittest.TestCase):
    def test_config_loader_rejects_mixed_graph_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base_config(root)
            config["input"]["graph_kind"] = ["raw_level0", "query_derived"]
            path = root / "phase3.json"
            _write_config(path, config)
            with self.assertRaisesRegex(ValueError, "single graph source"):
                load_phase3_config(path)

    def test_config_defaults_and_validates_integration_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base_config(root)
            path = root / "phase3.json"
            _write_config(path, config)
            self.assertEqual(load_phase3_config(path)["metrics"]["integration_repetitions"], 1)
            config["metrics"]["integration_repetitions"] = 0
            _write_config(path, config)
            with self.assertRaisesRegex(ValueError, "integration_repetitions"):
                load_phase3_config(path)

    def test_config_rejects_nonpositive_bounded_propagation_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _base_config(root)
            config["cross_links"]["propagation_neighbor_budget"] = 0
            path = root / "phase3.json"
            _write_config(path, config)
            with self.assertRaisesRegex(ValueError, "propagation_neighbor_budget"):
                load_phase3_config(path)

    def test_pairwise_readiness_detects_missing_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11]))
            config = _base_config(root)
            result = check_pairwise_phase2_inputs(config)
            self.assertFalse(result["ready"])
            self.assertEqual(result["checked_count"], 2)
            self.assertEqual(result["available_count"], 1)
            self.assertEqual(result["missing"][0]["subset"], "D_B")

    def test_load_pairwise_inputs_checks_ids_and_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21]))
            first, second = load_pairwise_phase2_inputs(
                _base_config(root), 7, "identity"
            )
            self.assertEqual(first.subset, "D_A")
            self.assertEqual(second.subset, "D_B")
            self.assertEqual(first.graph["graph_kind"], "raw_level0")
            self.assertEqual(first.graph["variant"], "practical_repair")

    def test_runner_writes_readiness_manifest_without_running_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11]))
            config = _base_config(root)
            path = root / "phase3.json"
            _write_config(path, config)
            manifest = run_complete_phase3(path, root)
            self.assertEqual(manifest["stage"], "common_readiness_check")
            self.assertFalse(manifest["readiness"]["ready"])
            self.assertTrue((root / "phase3_run_manifest.json").exists())

    def test_runner_executes_ground_truth_and_full_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11, 12]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21, 22]))
            config = _base_config(root)
            config["methods"]["enabled"] = [
                "bruteforce_ground_truth",
                "full_rebuild_hnsw",
            ]
            config["stages"]["readiness_check_only"] = False
            path = root / "phase3.json"
            _write_config(path, config)

            manifest = run_complete_phase3(path, root)

            self.assertEqual(manifest["stage"], "method_execution")
            self.assertTrue(manifest["readiness"]["ready"])
            result = manifest["method_results"]["seed_7/identity"]
            self.assertIn("bruteforce_ground_truth", result)
            self.assertIn("full_rebuild_hnsw", result)
            timing = result["full_rebuild_hnsw"]["cost"]
            self.assertEqual(timing["integration_repetitions"], 1)
            self.assertEqual(len(timing["integration_seconds_samples"]), 1)
            self.assertEqual(timing["integration_seconds"], timing["integration_seconds_median"])
            self.assertTrue(
                (
                    root
                    / "phase3_ground_truth"
                    / "seed_7"
                    / "identity"
                    / "neighbor_global_ids.npy"
                ).exists()
            )
            self.assertTrue(
                (
                    root
                    / "phase3_full_rebuild"
                    / "seed_7"
                    / "identity"
                    / "full_rebuild_hnsw.faiss"
                ).exists()
            )

    def test_dual_index_external_search_reranks_global_candidates(self):
        from data.phase3_io import Phase2SubsetInput

        first = Phase2SubsetInput(
            data_seed=7,
            condition="identity",
            subset="D_A",
            transformed_vectors=np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
            global_ids=np.array([100, 101], dtype=np.int64),
            held_out_queries=np.zeros((1, 2), dtype=np.float32),
            graph=_complete_graph(np.array([100, 101]), "D_A", "practical_repair"),
            raw_no_repair_graph=None,
            compiled_dir=Path("."),
            graph_path=Path("."),
        )
        second = Phase2SubsetInput(
            data_seed=7,
            condition="identity",
            subset="D_B",
            transformed_vectors=np.array([[0.2, 0.0], [8.0, 0.0]], dtype=np.float32),
            global_ids=np.array([200, 201], dtype=np.int64),
            held_out_queries=np.zeros((1, 2), dtype=np.float32),
            graph=_complete_graph(np.array([200, 201]), "D_B", "practical_repair"),
            raw_no_repair_graph=None,
            compiled_dir=Path("."),
            graph_path=Path("."),
        )
        ids, distances, diagnostics = dual_index_search_external(
            first,
            second,
            np.array([[0.1, 0.0]], dtype=np.float32),
            k=3,
            candidate_count_per_subset=2,
            ef_search=4,
            distance_batch_size=4,
        )

        self.assertEqual(ids.shape, (1, 3))
        self.assertEqual(ids[0, 0], 100)
        self.assertEqual(ids[0, 1], 200)
        self.assertTrue(set(ids.ravel()).issubset({100, 101, 200, 201}))
        self.assertLessEqual(distances[0, 0], distances[0, 1])
        self.assertEqual(diagnostics["backend"], "external_graph_search")
        self.assertFalse(diagnostics["constructs_single_union_graph"])
        self.assertEqual(diagnostics["rerank_distance_count"], 4)

    def test_runner_executes_dual_index_and_records_cost_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11, 12]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21, 22]))
            config = _base_config(root)
            config["methods"]["enabled"] = [
                "bruteforce_ground_truth",
                "dual_index",
            ]
            config["stages"]["readiness_check_only"] = False
            path = root / "phase3.json"
            _write_config(path, config)

            manifest = run_complete_phase3(path, root)

            result = manifest["method_results"]["seed_7/identity"]["dual_index"]
            self.assertEqual(result["method"], "dual_index")
            self.assertEqual(result["backend"], "external_graph_search")
            self.assertFalse(result["constructs_single_union_graph"])
            self.assertIn("integration_seconds", result["cost"])
            self.assertIn("total_search_seconds", result["cost"])
            self.assertIn("rerank_distance_count", result["cost"])
            self.assertTrue(
                (
                    root
                    / "phase3_dual_index"
                    / "seed_7"
                    / "identity"
                    / "neighbor_global_ids.npy"
                ).exists()
            )

    def test_no_cross_union_graph_preserves_local_nodes_edges_only(self):
        from data.phase2_graph import directed_reachable_from_entry, weak_components
        from data.phase3_io import Phase2SubsetInput

        first = Phase2SubsetInput(
            data_seed=7,
            condition="identity",
            subset="D_A",
            transformed_vectors=np.zeros((3, 2), dtype=np.float32),
            global_ids=np.array([10, 11, 12], dtype=np.int64),
            held_out_queries=np.zeros((1, 2), dtype=np.float32),
            graph=_graph(np.array([10, 11, 12]), "D_A", "practical_repair"),
            raw_no_repair_graph=None,
            compiled_dir=Path("."),
            graph_path=Path("."),
        )
        second = Phase2SubsetInput(
            data_seed=7,
            condition="identity",
            subset="D_B",
            transformed_vectors=np.zeros((2, 2), dtype=np.float32),
            global_ids=np.array([20, 21], dtype=np.int64),
            held_out_queries=np.zeros((1, 2), dtype=np.float32),
            graph=_graph(np.array([20, 21]), "D_B", "practical_repair"),
            raw_no_repair_graph=None,
            compiled_dir=Path("."),
            graph_path=Path("."),
        )

        graph, diagnostics = no_cross_union_graph(first, second)

        self.assertEqual(set(graph["node_global_ids"].tolist()), {10, 11, 12, 20, 21})
        self.assertEqual(graph["neighbors"][10], [11])
        self.assertEqual(graph["neighbors"][20], [21])
        self.assertEqual(count_cross_subset_edges(graph), 0)
        self.assertEqual(diagnostics["added_cross_link_count"], 0)
        self.assertEqual(diagnostics["reused_within_subset_edge_ratio"], 1.0)
        self.assertEqual(len(weak_components(graph)), 2)
        self.assertEqual(directed_reachable_from_entry(graph), {10, 11, 12})
        stats = graph_statistics(graph)
        self.assertEqual(stats["weak_connected_component_count"], 2)
        self.assertAlmostEqual(stats["directed_reachable_from_entry_ratio"], 3 / 5)

    def test_runner_executes_no_cross_merged_and_records_graph_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11, 12]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21, 22]))
            config = _base_config(root)
            config["methods"]["enabled"] = [
                "bruteforce_ground_truth",
                "no_cross_merged",
            ]
            config["stages"]["readiness_check_only"] = False
            path = root / "phase3.json"
            _write_config(path, config)

            manifest = run_complete_phase3(path, root)

            result = manifest["method_results"]["seed_7/identity"]["no_cross_merged"]
            self.assertEqual(result["method"], "no_cross_merged")
            self.assertTrue(result["constructs_single_union_graph"])
            self.assertEqual(result["backend"], "external_graph_search")
            self.assertEqual(result["graph"]["added_cross_link_count"], 0)
            self.assertEqual(result["graph"]["cross_link_strategy"], "none")
            self.assertEqual(result["graph"]["cross_subset_edge_count"], 0)
            self.assertEqual(result["graph_statistics"]["weak_connected_component_count"], 2)
            self.assertIn("graph_merge_seconds", result["cost"])
            self.assertIn("total_search_seconds", result["cost"])
            self.assertTrue(result["diagnostics"]["global_id_output"])
            self.assertTrue(
                (
                    root
                    / "phase3_no_cross_merged"
                    / "seed_7"
                    / "identity"
                    / "neighbor_global_ids.npy"
                ).exists()
            )

    def test_random_cross_links_add_only_weighted_cross_edges_reproducibly(self):
        from data.phase3_io import Phase2SubsetInput

        first = Phase2SubsetInput(
            data_seed=7,
            condition="identity",
            subset="D_A",
            transformed_vectors=np.array(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32
            ),
            global_ids=np.array([10, 11, 12], dtype=np.int64),
            held_out_queries=np.zeros((1, 2), dtype=np.float32),
            graph=_graph(np.array([10, 11, 12]), "D_A", "practical_repair"),
            raw_no_repair_graph=None,
            compiled_dir=Path("."),
            graph_path=Path("."),
        )
        second = Phase2SubsetInput(
            data_seed=7,
            condition="identity",
            subset="D_B",
            transformed_vectors=np.array(
                [[10.0, 0.0], [11.0, 0.0]], dtype=np.float32
            ),
            global_ids=np.array([20, 21], dtype=np.int64),
            held_out_queries=np.zeros((1, 2), dtype=np.float32),
            graph=_graph(np.array([20, 21]), "D_B", "practical_repair"),
            raw_no_repair_graph=None,
            compiled_dir=Path("."),
            graph_path=Path("."),
        )

        graph, diagnostics = random_cross_link_union_graph(
            first,
            second,
            sample_ratio=1.0,
            cross_k=2,
            random_seed=123,
            bidirectional=True,
        )
        repeat_graph, repeat_diagnostics = random_cross_link_union_graph(
            first,
            second,
            sample_ratio=1.0,
            cross_k=2,
            random_seed=123,
            bidirectional=True,
        )

        subset_by_node = graph["source_subset_by_node"]
        original_edges = {
            (source, target)
            for source, targets in first.graph["neighbors"].items()
            for target in targets
        } | {
            (source, target)
            for source, targets in second.graph["neighbors"].items()
            for target in targets
        }
        current_edges = {
            (source, target)
            for source, targets in graph["neighbors"].items()
            for target in targets
        }
        self.assertTrue(original_edges.issubset(current_edges))
        self.assertFalse(any(source == target for source, target in current_edges))
        cross_edges = [
            (source, target)
            for source, target in current_edges
            if subset_by_node[source] != subset_by_node[target]
        ]
        self.assertEqual(len(cross_edges), diagnostics["added_cross_link_count"])
        self.assertEqual(len(cross_edges), 12)
        self.assertEqual(diagnostics["endpoint_selection_distance_count"], 0)
        self.assertEqual(
            diagnostics["edge_weight_distance_count"],
            diagnostics["added_cross_link_count"],
        )
        self.assertEqual(count_cross_subset_edges(graph), 12)
        self.assertEqual(
            {
                (source, target)
                for source, targets in repeat_graph["neighbors"].items()
                for target in targets
            },
            current_edges,
        )
        self.assertEqual(
            repeat_diagnostics["added_cross_link_count"],
            diagnostics["added_cross_link_count"],
        )

        vectors = {
            10: np.array([0.0, 0.0]),
            11: np.array([1.0, 0.0]),
            12: np.array([2.0, 0.0]),
            20: np.array([10.0, 0.0]),
            21: np.array([11.0, 0.0]),
        }
        for source, targets in graph["neighbors"].items():
            for offset, target in enumerate(targets):
                if subset_by_node[source] == subset_by_node[target]:
                    continue
                expected = float(np.sum((vectors[source] - vectors[target]) ** 2))
                self.assertAlmostEqual(graph["edge_distances"][source][offset], expected)

    def test_runner_executes_random_cross_links_and_records_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11, 12]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21, 22]))
            config = _base_config(root)
            config["methods"]["enabled"] = [
                "bruteforce_ground_truth",
                "random_cross_links",
            ]
            config["stages"]["readiness_check_only"] = False
            path = root / "phase3.json"
            _write_config(path, config)

            manifest = run_complete_phase3(path, root)

            result = manifest["method_results"]["seed_7/identity"]["random_cross_links"]
            self.assertEqual(result["method"], "random_cross_links")
            self.assertEqual(result["baseline_type"], "weak_random_cross_link_baseline")
            self.assertEqual(result["graph"]["cross_link_strategy"], "random")
            self.assertTrue(result["constructs_single_union_graph"])
            self.assertGreater(result["graph"]["cross_subset_edge_count"], 0)
            self.assertEqual(
                result["cost"]["endpoint_selection_distance_count"],
                0,
            )
            self.assertEqual(
                result["cost"]["edge_weight_distance_count"],
                result["graph"]["added_cross_link_count"],
            )
            self.assertIn("graph_merge_seconds", result["cost"])
            self.assertIn("cross_link_construction_seconds", result["cost"])
            self.assertIn("total_search_seconds", result["cost"])
            self.assertGreater(result["graph_statistics"]["edge_count"], 0)
            self.assertTrue(
                (
                    root
                    / "phase3_random_cross_links"
                    / "seed_7"
                    / "identity"
                    / "neighbor_global_ids.npy"
                ).exists()
            )

    def test_sampled_cross_knn_selects_exact_opposite_subset_topk(self):
        from data.phase3_io import Phase2SubsetInput

        first = Phase2SubsetInput(
            data_seed=7,
            condition="identity",
            subset="D_A",
            transformed_vectors=np.array(
                [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32
            ),
            global_ids=np.array([100, 101, 102], dtype=np.int64),
            held_out_queries=np.zeros((1, 2), dtype=np.float32),
            graph=_graph(np.array([100, 101, 102]), "D_A", "practical_repair"),
            raw_no_repair_graph=None,
            compiled_dir=Path("."),
            graph_path=Path("."),
        )
        second = Phase2SubsetInput(
            data_seed=7,
            condition="identity",
            subset="D_B",
            transformed_vectors=np.array(
                [[1.0, 0.0], [11.0, 0.0], [30.0, 0.0]], dtype=np.float32
            ),
            global_ids=np.array([200, 201, 202], dtype=np.int64),
            held_out_queries=np.zeros((1, 2), dtype=np.float32),
            graph=_graph(np.array([200, 201, 202]), "D_B", "practical_repair"),
            raw_no_repair_graph=None,
            compiled_dir=Path("."),
            graph_path=Path("."),
        )

        graph, diagnostics = sampled_cross_knn_union_graph(
            first,
            second,
            sample_ratio=1.0,
            cross_k=1,
            random_seed=123,
            bidirectional=True,
        )
        repeat_graph, repeat_diagnostics = sampled_cross_knn_union_graph(
            first,
            second,
            sample_ratio=1.0,
            cross_k=1,
            random_seed=123,
            bidirectional=True,
        )

        subset_by_node = graph["source_subset_by_node"]
        original_edges = {
            (source, target)
            for source, targets in first.graph["neighbors"].items()
            for target in targets
        } | {
            (source, target)
            for source, targets in second.graph["neighbors"].items()
            for target in targets
        }
        current_edges = {
            (source, target)
            for source, targets in graph["neighbors"].items()
            for target in targets
        }
        self.assertTrue(original_edges.issubset(current_edges))
        self.assertFalse(any(source == target for source, target in current_edges))
        cross_edges = [
            (source, target)
            for source, target in current_edges
            if subset_by_node[source] != subset_by_node[target]
        ]
        self.assertEqual(len(cross_edges), diagnostics["added_cross_link_count"])
        self.assertEqual(count_cross_subset_edges(graph), len(cross_edges))
        self.assertEqual(diagnostics["cross_candidate_distance_count"], 18)
        self.assertEqual(diagnostics["edge_weight_distance_count"], 0)
        self.assertAlmostEqual(diagnostics["cross_scan_work_ratio"], 1.0)
        self.assertEqual(
            {
                (source, target)
                for source, targets in repeat_graph["neighbors"].items()
                for target in targets
            },
            current_edges,
        )
        self.assertEqual(
            repeat_diagnostics["sampled_node_ids_A"],
            diagnostics["sampled_node_ids_A"],
        )
        self.assertEqual(
            repeat_diagnostics["sampled_node_ids_B"],
            diagnostics["sampled_node_ids_B"],
        )

        expected_top = {
            100: 200,
            101: 201,
            102: 201,
            200: 100,
            201: 101,
            202: 102,
        }
        for source, target in expected_top.items():
            self.assertIn(target, graph["neighbors"][source])
            offset = graph["neighbors"][source].index(target)
            source_vector = (
                first.transformed_vectors[list(first.global_ids).index(source)]
                if source in set(first.global_ids.tolist())
                else second.transformed_vectors[list(second.global_ids).index(source)]
            )
            target_vector = (
                first.transformed_vectors[list(first.global_ids).index(target)]
                if target in set(first.global_ids.tolist())
                else second.transformed_vectors[list(second.global_ids).index(target)]
            )
            expected_distance = float(np.sum((source_vector - target_vector) ** 2))
            self.assertAlmostEqual(
                graph["edge_distances"][source][offset],
                expected_distance,
            )

    def test_runner_executes_sampled_cross_knn_and_records_proposed_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11, 12]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21, 22]))
            config = _base_config(root)
            config["methods"]["enabled"] = [
                "bruteforce_ground_truth",
                "sampled_cross_knn",
            ]
            config["stages"]["readiness_check_only"] = False
            path = root / "phase3.json"
            _write_config(path, config)

            manifest = run_complete_phase3(path, root)

            result = manifest["method_results"]["seed_7/identity"]["sampled_cross_knn"]
            self.assertEqual(result["method"], "sampled_cross_knn")
            self.assertEqual(result["method_type"], "proposed_topology_reuse")
            self.assertEqual(result["graph"]["cross_link_strategy"], "sampled_cross_knn")
            self.assertTrue(result["constructs_single_union_graph"])
            self.assertGreater(result["graph"]["cross_subset_edge_count"], 0)
            self.assertEqual(result["graph"]["degree_pruning"], "none")
            self.assertEqual(result["graph"]["connectivity_repair"], "none")
            self.assertEqual(
                result["cost"]["cross_candidate_distance_count"],
                result["graph"]["sampled_node_count_A"] * 3
                + result["graph"]["sampled_node_count_B"] * 3,
            )
            self.assertIn("cross_candidate_search_seconds", result["cost"])
            self.assertEqual(result["cost"]["edge_weight_distance_count"], 0)
            self.assertEqual(
                result["cost"]["edge_weight_distance_source"],
                "reused_from_cross_candidate_search",
            )
            self.assertGreater(result["graph_statistics"]["edge_count"], 0)
            self.assertTrue(
                (
                    root
                    / "phase3_sampled_cross_knn"
                    / "seed_7"
                    / "identity"
                    / "neighbor_global_ids.npy"
                ).exists()
            )

    def test_validation_sweep_expands_cross_link_parameter_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11, 12]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21, 22]))
            config = _base_config(root)
            config["profile"] = "validation_sweep"
            config["methods"]["enabled"] = [
                "bruteforce_ground_truth",
                "random_cross_links",
                "sampled_cross_knn",
            ]
            config["sweep"] = {
                "enabled": True,
                "methods": ["random_cross_links", "sampled_cross_knn"],
                "sample_ratios": [0.5, 1.0],
                "cross_k": [1, 2],
            }
            config["stages"]["readiness_check_only"] = False
            path = root / "phase3_validation_sweep.json"
            _write_config(path, config)

            manifest = run_complete_phase3(path, root)

            self.assertEqual(manifest["profile"], "validation_sweep")
            self.assertEqual(
                manifest["result_scope"],
                "first_round_validation_not_formal_results",
            )
            result = manifest["method_results"]["seed_7/identity"]
            random_keys = [
                key for key in result if key.startswith("random_cross_links/")
            ]
            sampled_keys = [
                key for key in result if key.startswith("sampled_cross_knn/")
            ]
            self.assertEqual(len(random_keys), 4)
            self.assertEqual(len(sampled_keys), 4)
            self.assertNotIn("random_cross_links", result)
            self.assertNotIn("sampled_cross_knn", result)
            expected_suffixes = {
                "sample_ratio_0p5/cross_k_1",
                "sample_ratio_0p5/cross_k_2",
                "sample_ratio_1/cross_k_1",
                "sample_ratio_1/cross_k_2",
            }
            self.assertEqual(
                {key.removeprefix("random_cross_links/") for key in random_keys},
                expected_suffixes,
            )
            self.assertEqual(
                {key.removeprefix("sampled_cross_knn/") for key in sampled_keys},
                expected_suffixes,
            )
            random_result = result["random_cross_links/sample_ratio_0p5/cross_k_2"]
            sampled_result = result["sampled_cross_knn/sample_ratio_1/cross_k_1"]
            self.assertEqual(random_result["graph"]["sample_ratio"], 0.5)
            self.assertEqual(random_result["graph"]["cross_k"], 2)
            self.assertEqual(sampled_result["graph"]["sample_ratio"], 1.0)
            self.assertEqual(sampled_result["graph"]["cross_k"], 1)
            self.assertTrue(
                (
                    root
                    / "phase3_random_cross_links"
                    / "seed_7"
                    / "identity"
                    / "sample_ratio_0p5"
                    / "cross_k_2"
                    / "neighbor_global_ids.npy"
                ).exists()
            )
            self.assertTrue(
                (
                    root
                    / "phase3_sampled_cross_knn"
                    / "seed_7"
                    / "identity"
                    / "sample_ratio_1"
                    / "cross_k_1"
                    / "neighbor_global_ids.npy"
                ).exists()
            )

    def test_runner_uses_config_output_root_when_not_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11, 12]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21, 22]))
            config = _base_config(root)
            isolated_output = root / "phase3_validation_sweep"
            config["profile"] = "validation_sweep"
            config["methods"]["enabled"] = [
                "bruteforce_ground_truth",
                "random_cross_links",
            ]
            config["sweep"] = {
                "enabled": True,
                "methods": ["random_cross_links"],
                "sample_ratios": [0.5],
                "cross_k": [1],
            }
            config["output"]["root"] = str(isolated_output)
            config["output"]["manifest"] = str(root / "validation_manifest.json")
            config["stages"]["readiness_check_only"] = False
            path = root / "phase3_validation_sweep.json"
            _write_config(path, config)

            run_complete_phase3(path)

            self.assertTrue(
                (
                    isolated_output
                    / "phase3_random_cross_links"
                    / "seed_7"
                    / "identity"
                    / "sample_ratio_0p5"
                    / "cross_k_1"
                    / "neighbor_global_ids.npy"
                ).exists()
            )
            self.assertFalse(
                (
                    Path("data")
                    / "phase3_random_cross_links"
                    / "seed_7"
                    / "identity"
                    / "sample_ratio_0p5"
                    / "cross_k_1"
                    / "neighbor_global_ids.npy"
                ).exists()
            )

    def test_runner_executes_incremental_insertion_in_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_subset(root, 7, "identity", "D_A", np.array([10, 11, 12]))
            _write_subset(root, 7, "identity", "D_B", np.array([20, 21, 22]))
            config = _base_config(root)
            config["methods"]["enabled"] = [
                "bruteforce_ground_truth",
                "incremental_insertion",
            ]
            config["stages"]["readiness_check_only"] = False
            path = root / "phase3.json"
            _write_config(path, config)

            manifest = run_complete_phase3(path, root)

            result = manifest["method_results"]["seed_7/identity"][
                "incremental_insertion"
            ]
            self.assertEqual(result["method"], "incremental_insertion")
            self.assertEqual(
                result["baseline_type"],
                "strong_incremental_insertion_baseline",
            )
            self.assertEqual(result["method_type"], "baseline")
            self.assertEqual(result["backend"], "native_faiss_hnsw")
            self.assertTrue(result["constructs_single_union_graph"])
            self.assertEqual(result["direction_count"], 2)
            self.assertEqual(set(result["direction_labels"]), {"A<-B", "B<-A"})
            self.assertFalse(result["reuse_summary"]["physical_base_loaded"])
            self.assertFalse(result["reuse_summary"]["reuses_inserted_topology"])
            self.assertTrue(
                result["reuse_summary"]["not_proposed_topology_reuse_method"]
            )

            for direction, expected_base, expected_inserted in (
                ("A<-B", "D_A", "D_B"),
                ("B<-A", "D_B", "D_A"),
            ):
                direction_result = result["directions"][direction]
                self.assertEqual(direction_result["direction"], direction)
                self.assertEqual(direction_result["base_subset"], expected_base)
                self.assertEqual(
                    direction_result["inserted_subset"], expected_inserted
                )
                self.assertEqual(
                    direction_result["cost"]["processed_inserted_vector_count"],
                    3,
                )
                self.assertAlmostEqual(
                    direction_result["cost"]["processed_inserted_vector_ratio"],
                    0.5,
                )
                self.assertIn(
                    "base_index_prepare_seconds",
                    direction_result["cost"],
                )
                self.assertIn("base_index_build_seconds", direction_result["cost"])
                self.assertIn("inserted_add_seconds", direction_result["cost"])
                self.assertIn("total_integration_seconds", direction_result["cost"])
                self.assertFalse(
                    direction_result["reuse"]["physical_base_loaded"]
                )
                self.assertFalse(
                    direction_result["reuse"]["reuses_base_topology"]
                )
                self.assertFalse(
                    direction_result["reuse"]["reuses_inserted_topology"]
                )
                self.assertTrue(
                    direction_result["diagnostics"][
                        "final_index_contains_all_union_ids"
                    ]
                )
                self.assertEqual(
                    direction_result["diagnostics"]["final_index_id_count"],
                    6,
                )
                predicted = np.load(
                    direction_result["arrays"]["neighbor_global_ids"]
                )
                self.assertTrue(set(predicted.ravel()).issubset({10, 11, 12, 20, 21, 22}))
                self.assertTrue(Path(direction_result["index_path"]).exists())

            self.assertTrue(
                (
                    root
                    / "phase3_incremental_insertion"
                    / "seed_7"
                    / "identity"
                    / "metrics.json"
                ).exists()
            )

    def test_recall_and_source_target_metrics(self):
        predicted = np.array([[1, 9, 3], [4, 5, 6]])
        truth = np.array([[1, 3, 9], [5, 7, 8]])
        means, per_query = recall_at_k(predicted, truth, [1, 3])
        self.assertEqual(means["recall_at_1"], 0.5)
        np.testing.assert_allclose(per_query["3"], np.array([1.0, 1 / 3]))

        lookup = subset_lookup(
            {"D_A": np.array([1, 2, 3]), "D_B": np.array([4, 5, 6, 7, 8, 9])}
        )
        split = source_target_recall_at_k(
            predicted, truth, ["D_A", "D_B"], lookup, [3]
        )
        self.assertAlmostEqual(split["recall_at_3"]["source_side"], 2 / 3)
        self.assertAlmostEqual(split["recall_at_3"]["target_side"], 1.0)

    def test_cost_metrics_keep_sample_and_random_work_separate(self):
        sampled = sampled_cross_knn_cost(
            size_a=100, size_b=200, sampled_a=10, sampled_b=20
        )
        self.assertEqual(sampled["sample_ratio"], 0.1)
        self.assertEqual(sampled["sample_ratio_A"], 0.1)
        self.assertEqual(sampled["sample_ratio_B"], 0.1)
        self.assertEqual(sampled["cross_candidate_distance_count"], 4000)
        self.assertAlmostEqual(sampled["cross_scan_work_ratio"], 0.1)

        random = random_cross_link_cost(
            edge_count=40,
            edge_weight_distance_count=40,
            endpoint_selection_distance_count=0,
        )
        self.assertEqual(random["endpoint_selection_distance_count"], 0)
        self.assertEqual(random["edge_weight_distance_count"], 40)


if __name__ == "__main__":
    unittest.main()
