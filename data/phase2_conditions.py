"""Generate exact-isometry, noisy, and stress-test conditions for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def _rotation(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix, remainder = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    matrix *= np.where(np.diag(remainder) < 0, -1.0, 1.0)
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def affine_isometry(
    name: str, dimension: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return A, b for T(x)=x@A+b under one exact condition."""
    identity = np.eye(dimension, dtype=np.float64)
    zero = np.zeros(dimension, dtype=np.float64)
    if name == "identity":
        return identity, zero
    if name == "rotation":
        return _rotation(dimension, rng), zero
    if name == "reflection":
        matrix = _rotation(dimension, rng)
        matrix[:, 0] *= -1
        return matrix, zero
    if name == "translation":
        return identity, rng.normal(0.0, 0.25, size=dimension)
    if name == "orthogonal_translation":
        return _rotation(dimension, rng), rng.normal(0.0, 0.25, size=dimension)
    raise ValueError(f"Unknown exact transformation: {name}")


def apply_affine(vectors: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a row-vector affine transformation."""
    return np.ascontiguousarray(
        vectors.astype(np.float64) @ matrix + translation, dtype=np.float32
    )


def inverse_affine(
    shared_vectors: np.ndarray, matrix: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Construct local vectors whose affine transform recovers shared vectors."""
    return np.ascontiguousarray(
        (shared_vectors.astype(np.float64) - translation) @ matrix.T,
        dtype=np.float32,
    )


def add_controlled_noise(
    exact_vectors: np.ndarray,
    family: str,
    level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Add noise with a target RMS-L2 relative perturbation."""
    dimension = exact_vectors.shape[1]
    mean_norm = float(
        np.mean(np.linalg.norm(exact_vectors.astype(np.float64), axis=1))
    )
    if family == "gaussian":
        scale = level * mean_norm / np.sqrt(dimension)
        epsilon = rng.normal(0.0, scale, size=exact_vectors.shape)
        scale_name = "sigma"
    elif family == "laplacian":
        scale = level * mean_norm / np.sqrt(2.0 * dimension)
        epsilon = rng.laplace(0.0, scale, size=exact_vectors.shape)
        scale_name = "b"
    else:
        raise ValueError(f"Unknown noise family: {family}")
    noisy = np.ascontiguousarray(
        exact_vectors.astype(np.float64) + epsilon, dtype=np.float32
    )
    measured = float(
        np.sqrt(np.mean(np.sum(epsilon**2, axis=1))) / mean_norm
        if mean_norm > 0
        else 0.0
    )
    return noisy, {
        "noise_level": float(level),
        scale_name: float(scale),
        "mean_norm": mean_norm,
        "measured_relative_perturbation": measured,
    }


def build_hnsw(
    vectors: np.ndarray,
    global_ids: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> faiss.IndexIDMap2:
    """Build one ID-mapped local HNSW."""
    raw = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_L2)
    raw.hnsw.efConstruction = ef_construction
    raw.hnsw.efSearch = ef_search
    index = faiss.IndexIDMap2(raw)
    index.add_with_ids(
        np.ascontiguousarray(vectors, dtype=np.float32),
        np.asarray(global_ids, dtype=np.int64),
    )
    return index


def load_subset_inputs(
    phase1_subset_dir: Path, subset_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load shared vectors and row-aligned global IDs from Phase 1."""
    subset_dir = phase1_subset_dir / subset_name
    vectors = np.ascontiguousarray(
        np.load(subset_dir / "shared_vectors.npy"), dtype=np.float32
    )
    with (subset_dir / "id_mapping.json").open("r", encoding="utf-8") as file:
        mapping = json.load(file)
    global_ids = np.asarray(
        [record["global_id"] for record in mapping], dtype=np.int64
    )
    if len(vectors) != len(global_ids):
        raise ValueError(f"{subset_name} vector and ID counts differ.")
    return vectors, global_ids


def generate_held_out_shared_queries(
    count: int, dimension: int, seed: int
) -> np.ndarray:
    """Generate independent queries from Phase 1's uniform source distribution."""
    return np.random.default_rng(seed).random(
        (count, dimension), dtype=np.float32
    )


def transformation_metadata(
    name: str, matrix: np.ndarray, translation: np.ndarray
) -> dict[str, Any]:
    """Return geometric metadata required by the Phase 2 plan."""
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    return {
        "name": name,
        "matrix": matrix.tolist(),
        "translation": translation.tolist(),
        "determinant": float(np.linalg.det(matrix)),
        "isometry_error": float(np.linalg.norm(matrix.T @ matrix - identity)),
    }
