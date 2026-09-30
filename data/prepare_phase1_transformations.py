"""Create rigorous synthetic alignment experiments for Phase 1.

The generated source vectors are treated as shared-space ground truth. For each
non-reference subset and transformation type, this script constructs local
pre-alignment vectors X such that X @ T recovers the shared vectors Z. Local
HNSW indices are built on X, matching the real setting where existing indices
were created before upstream alignment transformed vectors into a shared space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.phase1_common import add_run_metadata, start_run


SUBSET_NAMES = ("D_A", "D_B", "D_C")
EXPERIMENT_NAMES = ("identity", "orthogonal", "general_linear")
REFERENCE_SUBSET = "D_A"


def _orthogonal_matrix(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a proper orthogonal matrix with determinant +1."""
    random_matrix = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(random_matrix)

    # Correct QR sign ambiguity so the result is deterministic for a given RNG
    # stream, then force a rotation rather than a reflection.
    diagonal_signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    q = q * diagonal_signs
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _general_linear_matrix(
    dimension: int, condition_number: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an invertible non-orthogonal matrix with controlled distortion."""
    if condition_number <= 1.0:
        raise ValueError("condition_number must be greater than 1.")

    # Independent orthogonal factors avoid aligning the distortion with the
    # original coordinate axes. Geometrically spaced singular values give an
    # exact, interpretable condition number and anisotropically change distances.
    left = _orthogonal_matrix(dimension, rng)
    right = _orthogonal_matrix(dimension, rng)
    singular_values = np.geomspace(
        1.0, condition_number, num=dimension, dtype=np.float64
    )
    matrix = left @ np.diag(singular_values) @ right.T
    return matrix, singular_values


def _load_subsets(
    subset_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[str, ...]]:
    """Load shared-space subsets and validate their dimensionality."""
    manifest_path = subset_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        subset_manifest = json.load(file)

    vectors_by_subset: dict[str, np.ndarray] = {}
    expected_dimension = int(subset_manifest["dimension"])
    names = tuple(subset_manifest.get("subset_names", SUBSET_NAMES))
    for subset_name in names:
        vector_path = subset_dir / subset_name / "shared_vectors.npy"
        vectors = np.load(vector_path)
        if vectors.ndim != 2 or vectors.shape[1] != expected_dimension:
            raise ValueError(
                f"{subset_name} has shape {vectors.shape}; expected (*, "
                f"{expected_dimension})."
            )
        vectors_by_subset[subset_name] = np.ascontiguousarray(
            vectors, dtype=np.float32
        )

    return vectors_by_subset, subset_manifest, names


def _load_global_ids(subset_dir: Path, subset_name: str) -> np.ndarray:
    """Load global IDs in the same row order as a subset's vectors."""
    mapping_path = subset_dir / subset_name / "id_mapping.json"
    with mapping_path.open("r", encoding="utf-8") as file:
        id_mapping = json.load(file)
    return np.array([record["global_id"] for record in id_mapping], dtype=np.int64)


def _build_hnsw_index(
    vectors: np.ndarray,
    global_ids: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> faiss.IndexIDMap2:
    """Build a local L2 HNSW index over pre-alignment vectors."""
    dimension = vectors.shape[1]
    hnsw = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_L2)
    hnsw.hnsw.efConstruction = ef_construction
    hnsw.hnsw.efSearch = ef_search
    index = faiss.IndexIDMap2(hnsw)
    index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), global_ids)
    return index


def _transformation_for(
    experiment_name: str,
    subset_name: str,
    dimension: int,
    condition_number: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a matrix and its singular values for one experiment/subset pair."""
    # D_A defines the reference space and therefore stays unchanged in every
    # experiment. Identity is also the no-change control for D_B and D_C.
    if subset_name == REFERENCE_SUBSET or experiment_name == "identity":
        identity = np.eye(dimension, dtype=np.float64)
        return identity, np.ones(dimension, dtype=np.float64)
    if experiment_name == "orthogonal":
        matrix = _orthogonal_matrix(dimension, rng)
        return matrix, np.ones(dimension, dtype=np.float64)
    if experiment_name == "general_linear":
        return _general_linear_matrix(dimension, condition_number, rng)
    raise ValueError(f"Unknown experiment: {experiment_name}")


def prepare_phase1_transformations(
    subset_dir: Path,
    output_dir: Path,
    seed: int = 42,
    condition_number: float = 10.0,
    m: int = 16,
    ef_construction: int = 40,
    ef_search: int = 32,
    experiment_names: tuple[str, ...] = EXPERIMENT_NAMES,
) -> dict[str, Any]:
    """Generate local spaces, transformations, recovered vectors, and HNSW indices."""
    started_at, start_counter = start_run()
    vectors_by_subset, subset_manifest, subset_names = _load_subsets(subset_dir)
    dimension = int(subset_manifest["dimension"])
    if REFERENCE_SUBSET not in subset_names:
        raise ValueError(f"Reference subset {REFERENCE_SUBSET} is missing.")
    unsupported = set(experiment_names) - set(EXPERIMENT_NAMES)
    if unsupported:
        raise ValueError(f"Unsupported experiments: {sorted(unsupported)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source_subsets": str(subset_dir),
        "reference_subset": REFERENCE_SUBSET,
        "seed": seed,
        "dimension": dimension,
        "general_linear_condition_number": condition_number,
        "subset_names": list(subset_names),
        "experiment_names": list(experiment_names),
        "hnsw": {
            "m": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
        },
        "convention": "shared_vectors = local_vectors @ matrix",
        "experiments": {},
    }

    # Separate deterministic RNG streams make D_B and D_C transformations
    # independent while keeping the whole experiment reproducible.
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(experiment_names) * len(subset_names))

    sequence_index = 0
    for experiment_name in experiment_names:
        experiment_dir = output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        experiment_manifest: dict[str, Any] = {"subsets": {}}

        for subset_name in subset_names:
            subset_output_dir = experiment_dir / subset_name
            subset_output_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(child_sequences[sequence_index])
            sequence_index += 1

            matrix, singular_values = _transformation_for(
                experiment_name=experiment_name,
                subset_name=subset_name,
                dimension=dimension,
                condition_number=condition_number,
                rng=rng,
            )
            shared_vectors = vectors_by_subset[subset_name]
            inverse_matrix = np.linalg.inv(matrix)

            # Start from the known shared-space ground truth Z and construct a
            # local pre-alignment space X so that X @ T reconstructs Z. Local
            # HNSW is built on X, while union-level search will use X @ T.
            local_vectors = np.ascontiguousarray(
                shared_vectors.astype(np.float64) @ inverse_matrix,
                dtype=np.float32,
            )
            transformed_vectors = np.ascontiguousarray(
                local_vectors.astype(np.float64) @ matrix,
                dtype=np.float32,
            )
            reconstruction_error = float(
                np.max(
                    np.abs(
                        transformed_vectors.astype(np.float64)
                        - shared_vectors.astype(np.float64)
                    )
                )
            )

            shared_vector_path = subset_output_dir / "shared_vectors.npy"
            local_vector_path = subset_output_dir / "local_vectors.npy"
            vector_path = subset_output_dir / "transformed_vectors.npy"
            parameter_path = subset_output_dir / "transformation.npz"
            index_path = subset_output_dir / "local_hnsw.faiss"
            id_mapping_path = subset_output_dir / "id_mapping.json"

            np.save(shared_vector_path, shared_vectors)
            np.save(local_vector_path, local_vectors)
            np.save(vector_path, transformed_vectors)
            np.savez(
                parameter_path,
                matrix=matrix,
                inverse_matrix=inverse_matrix,
                singular_values=singular_values,
                experiment=np.array(experiment_name),
                subset_id=np.array(subset_name),
                reference_subset=np.array(REFERENCE_SUBSET),
                seed=np.array(seed, dtype=np.int64),
            )

            global_ids = _load_global_ids(subset_dir, subset_name)
            index = _build_hnsw_index(
                local_vectors,
                global_ids,
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
            )
            faiss.write_index(index, str(index_path))

            source_mapping_path = subset_dir / subset_name / "id_mapping.json"
            with source_mapping_path.open("r", encoding="utf-8") as file:
                id_mapping = json.load(file)
            with id_mapping_path.open("w", encoding="utf-8") as file:
                json.dump(id_mapping, file, indent=2)

            is_identity = bool(np.allclose(matrix, np.eye(dimension)))
            experiment_manifest["subsets"][subset_name] = {
                "source_shared_vectors": str(
                    subset_dir / subset_name / "shared_vectors.npy"
                ),
                "shared_vectors": str(shared_vector_path),
                "local_vectors": str(local_vector_path),
                "transformed_vectors": str(vector_path),
                "transformation_parameters": str(parameter_path),
                "id_mapping": str(id_mapping_path),
                "local_hnsw_index": str(index_path),
                "is_reference": subset_name == REFERENCE_SUBSET,
                "is_identity": is_identity,
                "condition_number": float(np.linalg.cond(matrix)),
                "max_reconstruction_error": reconstruction_error,
            }

        manifest["experiments"][experiment_name] = experiment_manifest

    add_run_metadata(manifest, started_at, start_counter)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line options for controlled transformation generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-dir", type=Path, default=Path("data/phase1_subsets")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/phase1_transformations")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition-number", type=float, default=10.0)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=40)
    parser.add_argument("--ef-search", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Generate all transformations and print their key geometric properties."""
    args = parse_args()
    manifest = prepare_phase1_transformations(
        subset_dir=args.subset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        condition_number=args.condition_number,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    print(
        f"Prepared {len(manifest['experiments'])} transformation experiments "
        f"with {manifest['reference_subset']} fixed:"
    )
    for experiment_name, experiment in manifest["experiments"].items():
        descriptions = []
        for subset_name, subset in experiment["subsets"].items():
            descriptions.append(
                f"{subset_name}: cond={subset['condition_number']:.3f}, "
                f"err={subset['max_reconstruction_error']:.2e}"
            )
        print(f"- {experiment_name}: " + ", ".join(descriptions))


if __name__ == "__main__":
    main()
