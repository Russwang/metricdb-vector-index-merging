"""Generate dissertation CSV/table artifacts from traceable compact rows."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.final_manifest import generate_latex_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.rows.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    print(json.dumps(generate_latex_sources(rows, args.output), indent=2))
