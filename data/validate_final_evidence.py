"""Validate the compact evidence manifest and its referenced artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.final_manifest import validate_final_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("results/final_manifest.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when any experiment is explicitly marked pending.",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = validate_final_manifest(
        manifest, repo_root=args.repo_root, require_complete=args.require_complete
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)
