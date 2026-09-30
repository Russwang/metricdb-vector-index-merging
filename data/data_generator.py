"""Generate a reproducible synthetic embedding dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase1_common import add_run_metadata, start_run


def generate_synthetic_data(
    num_samples: int = 100,
    dimension: int = 128,
    output_dir: str | Path = "data",
    seed: int = 42,
) -> dict[str, Any]:
    """Generate row-aligned synthetic vectors and metadata with a fixed seed."""
    started_at, start_counter = start_run()
    if num_samples <= 0 or dimension <= 0:
        raise ValueError("num_samples and dimension must be positive.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    print(f"Generating {num_samples} samples with dimension {dimension}...")

    vectors = rng.random((num_samples, dimension), dtype=np.float32)
    vector_path = output_path / "vectors.npy"
    metadata_path = output_path / "metadata.json"
    manifest_path = output_path / "data_generation_manifest.json"
    np.save(vector_path, vectors)

    metadata = []
    categories = ["electronics", "clothing", "home", "beauty", "toys"]
    for i in range(num_samples):
        metadata.append(
            {
                "id": f"item_{i}",
                "category": str(rng.choice(categories)),
                "price": round(float(rng.uniform(10, 1000)), 2),
                "timestamp": "2026-01-19T13:30:00",
            }
        )

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    manifest = add_run_metadata(
        {
            "stage": "synthetic_data_generation",
            "num_samples": num_samples,
            "dimension": dimension,
            "seed": seed,
            "distribution": "uniform_[0,1)",
            "dtype": "float32",
            "vectors": str(vector_path),
            "metadata": str(metadata_path),
        },
        started_at,
        start_counter,
    )
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Dataset generated successfully in '{output_path}/'.")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse reproducible synthetic-data generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_synthetic_data(
        num_samples=args.num_samples,
        dimension=args.dimension,
        seed=args.seed,
        output_dir=args.output_dir,
    )
