"""Export a tracked six-pair summary from an ignored synthetic scale manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


COMPACT_SCHEMA = "metricdb.scaling-pairs-compact.v1"


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "recall_at_10": float(payload["quality"]["overall"]["recall_at_10"]),
        "integration_seconds_median": float(payload["cost"]["integration_seconds_median"]),
        "average_query_ms": float(payload["cost"]["average_query_ms"]),
    }


def build(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs: dict[str, dict[str, dict[str, float]]] = {}
    for pair_key, methods in sorted(manifest["method_results"].items()):
        pairs[pair_key] = {
            method: _metrics(value)
            for method, value in methods.items()
            if isinstance(value, dict) and "quality" in value and "cost" in value
        }
    result = {
        "schema_version": COMPACT_SCHEMA,
        "source": {"manifest": str(manifest_path), "manifest_sha256": _checksum(manifest_path)},
        "method_results": pairs,
    }
    validation = validate(result)
    if not validation["valid"]:
        raise ValueError("Invalid scaling compact result: " + "; ".join(validation["errors"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema_version") != COMPACT_SCHEMA:
        errors.append(f"schema_version must be {COMPACT_SCHEMA!r}")
    pairs = payload.get("method_results")
    if not isinstance(pairs, dict) or len(pairs) != 6:
        errors.append("method_results must contain exactly six seed×condition pairs")
        return {"valid": not errors, "errors": errors}
    for pair_key, methods in pairs.items():
        if not methods:
            errors.append(f"{pair_key} has no methods")
        for method, values in methods.items():
            for field in ("recall_at_10", "integration_seconds_median", "average_query_ms"):
                if not isinstance(values.get(field), (int, float)):
                    errors.append(f"{pair_key} {method} has invalid {field}")
    return {"valid": not errors, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(build(args.manifest, args.output)), indent=2, sort_keys=True))
