"""Paired bootstrap utilities; timing repetitions are never quality units."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

BOOTSTRAP_SEED = 20260715


def _paired_arrays(rows: list[dict[str, Any]], baseline: str, method: str, value_key: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    base = {str(row["pair_key"]): float(row[value_key]) for row in rows if row["method"] == baseline}
    candidate = {str(row["pair_key"]): float(row[value_key]) for row in rows if row["method"] == method}
    if set(base) != set(candidate) or not base:
        raise ValueError("Baseline and method must contain the same non-empty pair_key set.")
    keys = sorted(base)
    return keys, np.asarray([base[key] for key in keys]), np.asarray([candidate[key] for key in keys])


def paired_bootstrap(
    rows: list[dict[str, Any]],
    *, baseline: str,
    method: str,
    value_key: str,
    statistic: str = "difference",
    samples: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap paired seed×condition units for deltas or log ratios."""
    if samples <= 0:
        raise ValueError("samples must be positive.")
    keys, base, candidate = _paired_arrays(rows, baseline, method, value_key)
    if statistic == "difference":
        unit_values = candidate - base
        inverse: Callable[[float], float] = float
    elif statistic == "log_ratio":
        if np.any(base <= 0) or np.any(candidate <= 0):
            raise ValueError("log_ratio requires positive values.")
        unit_values = np.log(candidate / base)
        inverse = lambda value: float(np.exp(value))
    else:
        raise ValueError("statistic must be 'difference' or 'log_ratio'.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unit_values), size=(samples, len(unit_values)))
    distribution = unit_values[draws].mean(axis=1)
    low, high = np.percentile(distribution, [2.5, 97.5])
    return {
        "schema_version": "metricdb.paired-bootstrap.v1",
        "pairing_unit": "seed×condition",
        "pair_keys": keys,
        "statistic": statistic,
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "estimate": inverse(float(unit_values.mean())),
        "ci95": [inverse(float(low)), inverse(float(high))],
    }
