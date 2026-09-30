"""Combine independently executed Cycle 23 case summaries."""

import csv
import json
from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/phase4_cycle23")
    inputs = sorted(root.glob("query_cost_summary_seed*.csv"))
    if len(inputs) != 4:
        raise SystemExit(f"Expected four case summaries, found {len(inputs)}")
    rows = []
    for path in inputs:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "query_cost_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {destination}")
    loaded = []
    for path in sorted((root / "loaded_base").glob("seed_*/*/*.json")):
        loaded.append({"case": str(path.relative_to(root / "loaded_base")), "metrics": json.loads(path.read_text(encoding="utf-8"))})
    loaded_path = destination.with_name("cycle23_loaded_base.json")
    loaded_path.write_text(json.dumps({"schema_version": "metricdb.loaded-base-aggregate.v1", "cases": loaded}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(loaded)} loaded-base cases to {loaded_path}")
