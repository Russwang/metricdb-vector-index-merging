"""Shared, raw-preserving query benchmark and capacity reporting."""

from __future__ import annotations

from time import perf_counter_ns
from typing import Any

import faiss
import numpy as np

from modules.metric_indexing.indexing import MetricIndex

QUERY_BENCHMARK_SCHEMA = "metricdb.query-benchmark.v1"


def benchmark_queries(
    index: MetricIndex,
    queries: np.ndarray,
    *,
    k: int = 10,
    warmup_queries: int = 20,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Benchmark one-query calls, retaining every measured latency and work value."""
    values = np.asarray(queries, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("queries must be a non-empty 2-D array.")
    if warmup_queries < 0 or repetitions <= 0:
        raise ValueError("warmup_queries must be non-negative and repetitions positive.")
    if index.backend == "faiss_hnsw":
        faiss.omp_set_num_threads(1)
    for query in values[: min(warmup_queries, len(values))]:
        index.search(query, k)

    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for query_index, query in enumerate(values):
            started = perf_counter_ns()
            result = index.search(query, k)
            latency = perf_counter_ns() - started
            rows.append({
                "repetition": repetition,
                "query_index": query_index,
                "latency_ns": int(latency),
                "result_count": len(result),
                **index.last_search_diagnostics,
            })
    latencies = np.asarray([row["latency_ns"] for row in rows], dtype=np.float64)
    total_seconds = float(latencies.sum() / 1e9)
    return {
        "schema_version": QUERY_BENCHMARK_SCHEMA,
        "backend": index.backend,
        "k": int(k),
        "query_count": int(len(values)),
        "warmup_query_count": int(min(warmup_queries, len(values))),
        "repetitions": int(repetitions),
        "faiss_threads": 1 if index.backend == "faiss_hnsw" else None,
        "raw_queries": rows,
        "summary": {
            "median_latency_ms": float(np.median(latencies) / 1e6),
            "p95_latency_ms": float(np.percentile(latencies, 95) / 1e6),
            "qps": float(len(rows) / total_seconds) if total_seconds else float("inf"),
        },
    }
