"""Run a compact, deterministic two-backend interface acceptance case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.phase2_graph import extract_raw_level0_graph
from data.phase4_benchmark import benchmark_queries
from main import SimpleMetricRelation
from modules.metric_indexing.indexing import MetricIndex


def run(output: Path) -> dict:
    rng = np.random.default_rng(22)
    vectors = rng.normal(size=(128, 16)).astype(np.float32)
    relation = SimpleMetricRelation("cycle22", vectors, [{"id": f"item-{i}"} for i in range(len(vectors))])
    native_raw = faiss.IndexHNSWFlat(16, 16, faiss.METRIC_L2)
    native_wrapped = faiss.IndexIDMap2(native_raw)
    graph_ids = np.arange(1000, 1128, dtype=np.int64)
    native_wrapped.add_with_ids(vectors, graph_ids)
    graph = extract_raw_level0_graph(native_wrapped, vectors, "merged")
    indices = {
        "faiss_hnsw": MetricIndex(),
        "external_graph": MetricIndex("external_graph", graph=graph, item_ids=relation.ids),
    }
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, index in indices.items():
            index.build(relation)
            benchmark = benchmark_queries(index, vectors[:24], warmup_queries=20, repetitions=5)
            suffix = "faiss" if name == "faiss_hnsw" else "json"
            size = index.serialized_size_bytes(Path(directory) / f"{name}.{suffix}")
            results[name] = {"search_sample": index.search(vectors[0], 10), "benchmark": benchmark, "capacity": index.capacity_metrics(size)}
    manifest = {"schema_version": "metricdb.cycle22-results.v1", "seed": 22, "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/cycle22_smoke.json"))
    args = parser.parse_args()
    run(args.output)
