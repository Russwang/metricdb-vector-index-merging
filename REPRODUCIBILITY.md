# Transforming and Merging Vector Indices for Unified Retrieval in MetricDB

## Overview

This archive contains the project materials for an MSc dissertation in the School of Informatics at the University of Edinburgh. The dissertation studies how independently built HNSW-derived vector indices can be queried or integrated after their vectors have been mapped into a shared metric space.

The implemented research path compares four operational choices:

- rebuilding one native FAISS HNSW over the transformed union;
- inserting an arriving subset into a maintained shared-space HNSW;
- traversing two stored level-0 graphs separately and globally re-ranking their candidates;
- reusing stored local level-0 topology and adding bounded cross-subset connectivity.

The code evaluates exact and noisy alignment conditions, local-topology repair, random and informed bridges, bounded and unlimited propagation, sequential three-subset integration, query work, and a 25k-vector GloVe evaluation. Exact brute-force neighbours provide the retrieval ground truth.

The completed dissertation contribution is the metric-indexing and experimental path in `data/` and `modules/metric_indexing/`. The other modules under `modules/` are retained because they provide the shared MetricDB scaffold imported by `main.py`; they were not evaluated as completed dissertation contributions.

## Repository structure

```text
metricdb-vector-index-merging/
|-- README.md
|-- README.zh-CN.md
|-- REPRODUCIBILITY.md
|-- MANIFEST.md
|-- LICENSE
|-- requirements.txt
|-- main.py
|-- core/
|-- modules/
|-- data/
|-- configs/
|-- results/
|-- dissertation_assets/
`-- tests/
```

- `core/`: common relation and index interfaces.
- `modules/metric_indexing/`: native FAISS-HNSW and external stored-topology adapters.
- `modules/`: the wider MetricDB scaffold required by the integrated entry point.
- `data/`: dataset preparation, graph extraction and repair, baselines, integration methods, experiment runners, aggregation, statistics, and evidence validation.
- `configs/`: smoke, exploratory, and frozen formal experiment configurations.
- `results/`: compact, checked dissertation evidence. Large arrays and native index files are intentionally excluded.
- `data/phase3_repair_ablation_summary/`: aggregate repair-ablation evidence used in Chapter 6.
- `data/phase3_repair_ablation_phase2_metrics/`: compact Phase 2 metric snapshots used by the repair-ablation summary.
- `dissertation_assets/`: final CSV and LaTeX inputs referenced by the dissertation's results chapter.
- `tests/`: tests for the Phase 1--4 research path and the common index adapter.

See `MANIFEST.md` for a more detailed inventory and the exclusion policy.

## Requirements

The frozen final evidence records the following execution environment:

- macOS 15.7.4 on x86-64;
- Python 3.10.11;
- NumPy 2.2.6;
- FAISS 1.14.3.

The supplied `requirements.txt` contains the dependencies imported by the Python implementation:

```text
numpy>=1.20.0
faiss-cpu>=1.8.0
```

FAISS behaviour and performance may vary by version, architecture, thread count, and installation source. Formal Cycle 23 and sequential measurements used one FAISS thread where stated by their runners; the retained final environment manifest reports eight available threads.

## Installation

From the archive's top-level directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The virtual environment is not included in this archive.

## Quick validation

Validate the retained compact evidence, including each recorded byte size and SHA-256 digest:

```bash
python data/validate_final_evidence.py --require-complete
```

A successful validation reports `release_ready: true`, no pending cycles, and nine verified artifacts.

Run the included research-path tests:

```bash
python -m unittest discover tests
```

The submission copy contains 85 tests. They cover the index adapter, Phase 1 preparation and baselines, Phase 2 graph handling, Phase 3 integration and summaries, and the Phase 4 supplemental evaluations. Tests should be run in a disposable working copy if preserving a completely clean extracted directory is important, because Python creates `__pycache__` directories.

Cycle 22 is a small, self-contained interface check:

```bash
python data/run_cycle22.py --output scratch/cycle22_smoke.json
```

## Experiment reproduction

### Phase 1: preparation and preservation checks

Entry point:

```bash
python data/run_phase1.py \
  --config configs/phase1_smoke.json \
  --output-root scratch/phase1_smoke
```

This generates deterministic synthetic vectors, disjoint subsets, local transformations, exact brute-force neighbours, a native rebuild baseline, and the FAISS-HNSW backend probe. `run_phase1.py` cleans its selected output root by default; use a new scratch directory rather than `results/`.

Relevant implementation files include `data/data_generator.py`, `data/prepare_phase1_subsets.py`, `data/prepare_phase1_transformations.py`, `data/run_phase1_bruteforce_baseline.py`, `data/run_phase1_hnsw_rebuild_baseline.py`, and `data/probe_phase1_hnsw_backend.py`.

### Phase 2: alignment conditions and local topology

Entry point:

```bash
python data/run_phase2.py \
  --config configs/phase2_smoke.json \
  --output-root scratch/phase2_smoke
```

When the configured Phase 1 directories are unavailable, the runner creates the required deterministic Phase 1 inputs inside its selected output root. It then constructs exact and noisy alignment conditions, extracts raw level-0 topology, applies the configured repair variants, and records graph/search diagnostics.

The central implementation is in `data/phase2_conditions.py`, `data/phase2_graph.py`, and `data/run_phase2.py`.

### Phase 3: pairwise integration and baselines

`data/run_phase3.py` executes the enabled pairwise methods only when the Phase 2 paths named by its configuration are present. The supplied smoke configuration expects the default `data/phase2_*` layout. There is therefore no claim that Phase 3 can be rerun from the compact archive alone.

The implementation-to-method mapping is:

| Method | Implementation |
|---|---|
| Exact union ground truth | `data/run_phase1_bruteforce_baseline.py`, `data/phase3_baselines.py` |
| Full native rebuild | `data/phase3_baselines.py` |
| Native incremental insertion | `data/phase3_baselines.py` |
| Dual-index query and global re-ranking | `data/phase3_baselines.py` |
| Disconnected/no-cross control | `data/phase3_merge.py`, `data/phase3_baselines.py` |
| Random direct links | `data/phase3_baselines.py` |
| Sampled cross-kNN | `data/phase3_baselines.py` |
| Indexed portals and bounded/unlimited propagation | `data/phase3_merge.py`, `data/phase3_baselines.py` |
| Recall and graph metrics | `data/phase3_metrics.py` |

The synthetic scaling coordinator is:

```bash
python data/run_phase3_scaling.py --plan configs/phase3_scaling_plan.json
```

This is a full experiment command and can generate several gigabytes; the required full inputs and outputs are not included in this compact submission.

### Cycle 23: query cost and loaded-base insertion

Runner:

```bash
python data/run_cycle23_query_cost.py \
  --config configs/cycle23_query_cost_formal.json
```

Input: the excluded frozen Stage 20 directory `data/phase3_stage20_25k/`.

Retained outputs:

- `results/cycle23_query_cost.csv`;
- `results/cycle23_loaded_base.json`;
- corresponding plot/table inputs in `dissertation_assets/`.

### Cycle 24: sequential three-subset integration

The formal path uses:

```bash
python data/prepare_cycle24_inputs.py --config configs/cycle24_sequential_formal.json
python data/run_cycle24_sequential.py --config configs/cycle24_sequential_formal.json
python data/summarize_cycle24.py --config configs/cycle24_sequential_formal.json
```

Input preparation expects the excluded Stage 20 arrays named in the configuration. The retained compact output is `results/cycle24_formal.json`; `dissertation_assets/cycle24_results_table.tex` is the corresponding dissertation table input.

### Cycle 25: GloVe evaluation

The external dataset is Stanford's 2024 Wikipedia--Gigaword GloVe archive, `glove.2024.wikigiga.50d.zip`. The formal configuration records its source URL, licence, expected member, and SHA-256 digest. The dataset itself is not redistributed here.

With the archive available locally, the implemented stages are:

```bash
python data/prepare_cycle25_glove.py /path/to/glove.2024.wikigiga.50d.zip \
  --config configs/cycle25_glove_formal.json
python data/prepare_cycle25_inputs.py --config configs/cycle25_glove_formal.json
python data/run_cycle25_glove.py --config configs/cycle25_glove_formal.json
python data/summarize_cycle25.py --config configs/cycle25_glove_formal.json
```

These commands use the paths recorded in the configuration and produce large prepared arrays and indices. The retained compact outputs are `results/glove_ingest_manifest.json` and `results/cycle25_formal.json`.

### Regenerating compact table inputs without changing frozen results

Write regenerated files to a new directory:

```bash
mkdir -p scratch/generated
python data/generate_scaling_ci_latex.py \
  --source results/scaling_confidence_intervals.json \
  --output scratch/generated/scaling_ci_table.tex
python data/generate_cycle24_latex.py \
  --source results/cycle24_formal.json \
  --output scratch/generated/cycle24_results_table.tex
python data/generate_cycle25_latex.py \
  --compact results/cycle25_formal.json \
  --output scratch/generated/cycle25_results_table.tex
python data/build_final_plot_rows.py \
  --repo-root . \
  --output scratch/generated/final_plot_rows.json
python data/generate_dissertation_assets.py \
  scratch/generated/final_plot_rows.json \
  scratch/generated
```

Do not use the generators' historical default paths if the goal is to preserve the frozen archive unchanged.

## Evaluation metrics

The formal evaluation uses:

- Recall@1, Recall@10, and Recall@50 against exact squared-L2 neighbours where configured;
- cross-subset Recall@10;
- per-query and aggregate median/p95 latency and throughput;
- FAISS distance/hop counters where exposed;
- explicit external distance evaluations and visited nodes;
- candidate-HNSW calls, re-ranking and propagation distances, proposal counts, and cross-edge counts;
- integration, load, insertion, and rebuild wall-clock measurements;
- weak connectivity, directed reachability, identifier completeness, and prior-edge preservation;
- paired bootstrap confidence intervals for Recall differences and integration-time ratios.

Native FAISS counters and the Python external-backend counters are work indicators from different implementations; they should not be interpreted as equal instruction costs.

## Results and evidence

`results/final_manifest.json` identifies six completed final evidence cycles and records byte sizes and SHA-256 digests for nine compact artifacts. The strict validation command checks those files without regenerating or overwriting them.

The main retained results are:

| Dissertation evidence | Retained material |
|---|---|
| Synthetic 50k stopping gate | `results/cycle21_50k_compact.json` |
| Common adapter acceptance | `results/cycle22_smoke.json` |
| Query-cost and loaded-base experiments | `results/cycle23_query_cost.csv`, `results/cycle23_loaded_base.json` |
| Three-subset sequential integration | `results/cycle24_formal.json` |
| GloVe 25k evaluation | `results/cycle25_formal.json`, `results/glove_ingest_manifest.json` |
| 10k/25k/50k paired scaling intervals | `results/scaling_confidence_intervals.json` and scale compact files |
| Repair ablation | `data/phase3_repair_ablation_summary/` and compact Phase 2 snapshots |
| Dissertation figures and tables | `dissertation_assets/` |

Results are not claimed to reproduce bit-for-bit across arbitrary hardware and dependency versions. Random seeds and formal parameters are fixed in the configurations, but native HNSW timing and approximate search can vary with platform, FAISS build, and thread configuration.

## Dissertation-to-code mapping

| Dissertation component | Implementation or evidence |
|---|---|
| Section 3.3, common index contract | `core/base.py`, `modules/metric_indexing/indexing.py` |
| Sections 4.2--4.3, conversion and repair | `data/phase2_conditions.py`, `data/phase2_graph.py` |
| Sections 4.4--4.7, portals, bridges, propagation and pruning | `data/phase3_merge.py`, `data/phase3_baselines.py` |
| Section 4.8, sequential integration | `data/phase4_sequential.py`, `data/run_cycle24_sequential.py` |
| Section 4.9, external graph query | `data/phase2_graph.py` |
| Section 4.10, operational baselines | `data/phase3_baselines.py`, `data/phase4_loaded_base.py` |
| Sections 5.3--5.4, metrics and measurement | `data/phase3_metrics.py`, `data/phase4_benchmark.py` |
| Section 5.6, paired bootstrap analysis | `data/phase4_statistics.py`, `data/build_scaling_confidence_intervals.py` |
| Sections 6.1--6.8, final evidence | `results/`, repair-ablation summaries, and `dissertation_assets/` |

## Reproducibility limits

This is a compact submission package, not a byte-for-byte copy of the 17 GB working directory. It intentionally excludes raw synthetic arrays, extracted local HNSW indices, full per-case manifests, prepared sequential/GloVe inputs, the external GloVe archive, virtual environments, and caches.

Consequently:

- the included compact results and their checksums can be validated immediately;
- the included smoke paths and unit tests can be run with the stated dependencies;
- formal aggregations and dissertation assets can be inspected from retained outputs;
- a complete formal rerun requires regenerating or reacquiring the excluded inputs described above.

## Academic context

These files are the project materials accompanying an MSc dissertation. They are supplied so examiners can inspect the implementation, configurations, evaluation path, and retained evidence supporting the dissertation's claims.
