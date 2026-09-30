"""Split synthetic embeddings into shared-space ground-truth subsets."""

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


SUBSET_NAMES = ("D_A", "D_B", "D_C")


def _load_source_data(
    vector_path: Path, metadata_path: Path, subset_count: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load and validate the generated vectors and their row-aligned metadata."""
    vectors = np.load(vector_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D vector matrix, got shape {vectors.shape}.")
    if len(vectors) != len(metadata):
        raise ValueError(
            "Vector and metadata counts differ: "
            f"{len(vectors)} vectors versus {len(metadata)} metadata records."
        )
    if len(vectors) < subset_count:
        raise ValueError(
            f"At least {subset_count} vectors are required to create non-empty subsets."
        )

    # FAISS expects a contiguous float32 matrix. The conversion is a no-op when
    # the generated vectors already have the expected representation.
    return np.ascontiguousarray(vectors, dtype=np.float32), metadata


def prepare_phase1_subsets(
    vector_path: Path,
    metadata_path: Path,
    output_dir: Path,
    seed: int = 42,
    subset_names: tuple[str, ...] = SUBSET_NAMES,
) -> dict[str, Any]:
    """Create three reproducible shared-space subsets and persist ID mappings."""
    started_at, start_counter = start_run()
    if len(set(subset_names)) != len(subset_names):
        raise ValueError("subset_names must be unique.")
    vectors, metadata = _load_source_data(
        vector_path, metadata_path, subset_count=len(subset_names)
    )

    # Shuffle before splitting so subsets are not coupled to any ordering in the
    # source file. A fixed seed makes every experiment exactly reproducible.
    rng = np.random.default_rng(seed)
    shuffled_global_ids = rng.permutation(len(vectors)).astype(np.int64)
    global_id_groups = np.array_split(shuffled_global_ids, len(subset_names))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source_vectors": str(vector_path),
        "source_metadata": str(metadata_path),
        "seed": seed,
        "space_role": "shared_ground_truth",
        "distance_metric": "squared_l2",
        "dimension": int(vectors.shape[1]),
        "total_vectors": int(len(vectors)),
        "subset_count": len(subset_names),
        "subset_names": list(subset_names),
        "subsets": {},
    }

    for subset_name, global_ids in zip(subset_names, global_id_groups):
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        shared_vectors = np.ascontiguousarray(vectors[global_ids])
        id_records = [
            {
                "subset_id": subset_name,
                "local_id": local_id,
                "global_id": int(global_id),
                "item_id": metadata[int(global_id)]["id"],
            }
            for local_id, global_id in enumerate(global_ids)
        ]

        np.save(subset_dir / "shared_vectors.npy", shared_vectors)
        with (subset_dir / "id_mapping.json").open("w", encoding="utf-8") as file:
            json.dump(id_records, file, indent=2)

        manifest["subsets"][subset_name] = {
            "size": int(len(global_ids)),
            "shared_vectors": str(subset_dir / "shared_vectors.npy"),
            "id_mapping": str(subset_dir / "id_mapping.json"),
        }

    add_run_metadata(manifest, started_at, start_counter)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reproducible Phase 1 data preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=Path("data/vectors.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/phase1_subsets")
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Prepare subsets and print a concise summary of the generated artifacts."""
    args = parse_args()
    manifest = prepare_phase1_subsets(
        vector_path=args.vectors,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(
        f"Prepared {manifest['total_vectors']} vectors "
        f"({manifest['dimension']} dimensions):"
    )
    for subset_name, subset in manifest["subsets"].items():
        print(
            f"- {subset_name}: {subset['size']} shared-space vectors, "
            f"mapping={subset['id_mapping']}"
        )


if __name__ == "__main__":
    main()
