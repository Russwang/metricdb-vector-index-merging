"""Build the compact cross-cycle evidence manifest from checked-in artifacts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.final_manifest import build_final_manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    experiments = [
        {"cycle": 21, "config": "configs/phase3_stage21_50k_validation.json", "source": "formal synthetic 50k stopping gate; release-sized compact summary with source checksums", "status": "complete", "artifacts": ["results/cycle21_50k_compact.json"]},
        {"cycle": 22, "config": "configs/cycle22_system_interface.json", "source": "deterministic interface acceptance", "status": "complete", "artifacts": ["results/cycle22_smoke.json"]},
        {"cycle": 23, "config": "configs/cycle23_query_cost_formal.json", "source": "formal frozen Stage 20 artifacts", "status": "complete", "artifacts": ["results/cycle23_query_cost.csv", "results/cycle23_loaded_base.json"]},
        {"cycle": 24, "config": "configs/cycle24_sequential_formal.json", "source": "formal 15k three-subset sequential evaluation; two seeds, two conditions, three orders, five methods", "status": "complete", "artifacts": ["results/cycle24_formal.json"]},
        {"cycle": 25, "config": "configs/cycle25_glove_formal.json", "source": "formal Stanford GloVe 25k evaluation and generated cross-cycle dissertation rows; fixed split, two transformation seeds, two conditions, six methods, paired bootstrap", "status": "complete", "artifacts": ["results/glove_ingest_manifest.json", "results/cycle25_formal.json", "results/final_plot_rows.json"]},
        {"cycle": 26, "config": "configs/phase4_scaling_statistics.json", "source": "paired bootstrap confidence intervals for frozen 10k, 25k, and 50k synthetic evidence", "status": "complete", "artifacts": ["results/scaling_confidence_intervals.json"]},
    ]
    build_final_manifest(experiments, root / "results/final_manifest.json", repo_root=root)
