"""Loaded shared-space base measurements for Cycle 23."""

from __future__ import annotations

import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np


def _build(vectors: np.ndarray, ids: np.ndarray, m: int, ef_construction: int, ef_search: int) -> faiss.Index:
    raw = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_L2)
    raw.hnsw.efConstruction = ef_construction
    raw.hnsw.efSearch = ef_search
    index = faiss.IndexIDMap2(raw)
    index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), np.asarray(ids, dtype=np.int64))
    return index


def measure_loaded_base_incremental(
    base_vectors: np.ndarray,
    base_ids: np.ndarray,
    inserted_vectors: np.ndarray,
    inserted_ids: np.ndarray,
    *,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 64,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Separate offline build, disk load, add-only, load+add and rebuild+add.

    Both arrays must already be represented in the final shared coordinate
    space. This routine never describes a local-space physical index as reused.
    """
    base = np.ascontiguousarray(base_vectors, dtype=np.float32)
    inserted = np.ascontiguousarray(inserted_vectors, dtype=np.float32)
    base_ids = np.asarray(base_ids, dtype=np.int64)
    inserted_ids = np.asarray(inserted_ids, dtype=np.int64)
    if base.ndim != 2 or inserted.ndim != 2 or base.shape[1] != inserted.shape[1]:
        raise ValueError("Base and inserted vectors must be 2-D with matching dimensions.")
    if len(base) != len(base_ids) or len(inserted) != len(inserted_ids):
        raise ValueError("Vector and ID arrays must be row-aligned.")
    if len(np.unique(np.concatenate([base_ids, inserted_ids]))) != len(base_ids) + len(inserted_ids):
        raise ValueError("Base and inserted IDs must be globally unique.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    faiss.omp_set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shared_space_base.faiss"
        started = perf_counter()
        pristine = _build(base, base_ids, m, ef_construction, ef_search)
        offline_build = perf_counter() - started
        faiss.write_index(pristine, str(path))
        artifact_bytes = path.stat().st_size

        load_samples: list[float] = []
        add_samples: list[float] = []
        load_add_samples: list[float] = []
        rebuild_add_samples: list[float] = []
        limitation: dict[str, str] | None = None
        final_ids: np.ndarray | None = None
        for _ in range(repetitions):
            load_started = perf_counter()
            loaded = faiss.read_index(str(path))
            load_seconds = perf_counter() - load_started
            add_started = perf_counter()
            try:
                loaded.add_with_ids(inserted, inserted_ids)
                add_seconds = perf_counter() - add_started
                final_ids = np.asarray(faiss.vector_to_array(loaded.id_map), dtype=np.int64)
            except Exception as error:  # reproducible backend limitation, no fallback
                limitation = {"type": type(error).__name__, "message": str(error)}
                break
            load_samples.append(load_seconds)
            add_samples.append(add_seconds)
            load_add_samples.append(load_seconds + add_seconds)

            rebuild_started = perf_counter()
            rebuilt = _build(base, base_ids, m, ef_construction, ef_search)
            rebuilt.add_with_ids(inserted, inserted_ids)
            rebuild_add_samples.append(perf_counter() - rebuild_started)

    def summary(samples: list[float]) -> dict[str, Any]:
        return {"samples_seconds": samples, "median_seconds": float(np.median(samples))} if samples else {"samples_seconds": [], "median_seconds": None}

    return {
        "schema_version": "metricdb.loaded-base.v1",
        "semantic_scope": "prebuilt shared-space base; not a reused local-space physical index",
        "repetitions_requested": int(repetitions),
        "offline_base_build_seconds": float(offline_build),
        "serialized_base_bytes": int(artifact_bytes),
        "load": summary(load_samples),
        "add_only": summary(add_samples),
        "load_plus_add": summary(load_add_samples),
        "rebuild_base_plus_add": summary(rebuild_add_samples),
        "backend_limitation": limitation,
        "final_id_count": None if final_ids is None else int(len(final_ids)),
        "final_ids_complete": None if final_ids is None else bool(set(final_ids) == set(np.concatenate([base_ids, inserted_ids]))),
    }
