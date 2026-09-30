"""Strict GloVe ingestion used by the frozen real-embedding evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_glove_text(path: str | Path, *, dimension: int = 50, limit: int = 25_512) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Read the first unique, finite, non-zero vectors and L2-normalise them."""
    if dimension <= 0 or limit <= 0:
        raise ValueError("dimension and limit must be positive.")
    tokens: list[str] = []
    rows: list[np.ndarray] = []
    seen: set[str] = set()
    rejected = {"bad_field_count": 0, "duplicate_token": 0, "non_numeric_or_non_finite": 0, "zero_norm": 0}
    with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            parts = line.rstrip("\n").split()
            if len(parts) != dimension + 1:
                rejected["bad_field_count"] += 1
                continue
            token = parts[0]
            if token in seen:
                rejected["duplicate_token"] += 1
                continue
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                rejected["non_numeric_or_non_finite"] += 1
                continue
            if not np.isfinite(vector).all():
                rejected["non_numeric_or_non_finite"] += 1
                continue
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm == 0.0:
                rejected["zero_norm"] += 1
                continue
            seen.add(token)
            tokens.append(token)
            rows.append(vector / norm)
            if len(rows) == limit:
                break
    if len(rows) < limit:
        raise ValueError(f"Only {len(rows)} valid unique vectors found; {limit} required.")
    vectors = np.ascontiguousarray(np.vstack(rows), dtype=np.float32)
    return tokens, vectors, {
        "schema_version": "metricdb.glove-ingest.v1",
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "dimension": int(dimension),
        "accepted_count": int(len(tokens)),
        "rejected": rejected,
        "normalization": "row-wise L2",
        "parsing_rule": "whitespace token plus exactly dimension finite floats; first unique non-zero rows",
    }


def fixed_glove_split(tokens: list[str], vectors: np.ndarray, seed: int = 42) -> dict[str, Any]:
    """Create the frozen 12,500/12,500 corpus and 256/256 query split."""
    if len(tokens) != 25_512 or np.asarray(vectors).shape[0] != 25_512:
        raise ValueError("Frozen GloVe split requires exactly 25,512 rows.")
    permutation = np.random.default_rng(seed).permutation(25_512)
    a, b, queries = permutation[:12_500], permutation[12_500:25_000], permutation[25_000:]
    return {
        "seed": int(seed),
        "D_A_indices": a,
        "D_B_indices": b,
        "query_A_indices": queries[:256],
        "query_B_indices": queries[256:],
    }
