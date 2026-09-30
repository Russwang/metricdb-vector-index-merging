"""Prepare the ignored GloVe corpus and a compact provenance manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.phase4_glove import fixed_glove_split, parse_glove_text, sha256_file


def prepare(archive: Path, output: Path, config_path: Path, compact_manifest: Path | None = None) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    output.mkdir(parents=True, exist_ok=True)
    archive_sha = sha256_file(archive)
    configured_sha = dataset.get("archive_sha256")
    if configured_sha is not None and archive_sha != configured_sha:
        raise ValueError("GloVe archive SHA256 does not match the frozen config.")
    with zipfile.ZipFile(archive) as bundle:
        candidates = [name for name in bundle.namelist() if name.endswith(".txt") and "50" in name]
        if len(candidates) != 1:
            raise ValueError(f"Expected one 50d text member, found {candidates!r}.")
        member = candidates[0]
        extracted = Path(bundle.extract(member, output))
    tokens, vectors, ingest = parse_glove_text(extracted, dimension=50, limit=25_512)
    split = fixed_glove_split(tokens, vectors, seed=42)
    np.save(output / "vectors.npy", vectors)
    (output / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False) + "\n", encoding="utf-8")
    split_artifacts = {}
    for name, values in split.items():
        if name.endswith("_indices"):
            path = output / f"{name}.npy"
            np.save(path, values)
            split_artifacts[name] = {
                "count": int(len(values)),
                "sha256": sha256_file(path),
            }
    manifest = {
        **ingest,
        "schema_version": "metricdb.glove-cycle25.v1",
        "official_page": dataset["official_page"],
        "download_url": dataset["download_url"],
        "license": dataset["license"],
        "license_url": dataset["license_url"],
        "archive_sha256": archive_sha,
        "archive_bytes": archive.stat().st_size,
        "archive_member": member,
        "split": {"seed": int(split["seed"]), "artifacts": split_artifacts},
    }
    compact = output / "glove_ingest_manifest.json"
    compact.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if compact_manifest is not None:
        compact_manifest.parent.mkdir(parents=True, exist_ok=True)
        compact_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/phase4_cycle25/glove"))
    parser.add_argument("--config", type=Path, default=Path("configs/cycle25_glove_formal.json"))
    parser.add_argument("--compact-manifest", type=Path, default=Path("results/glove_ingest_manifest.json"))
    args = parser.parse_args()
    result = prepare(args.archive, args.output, args.config, args.compact_manifest)
    print(json.dumps({"archive_sha256": result["archive_sha256"], "accepted_count": result["accepted_count"]}, indent=2))
