# MetricDB Project Materials Manifest

## Submission contents

### Documentation

- `README.md` and `README.zh-CN.md`: public project overview, selected findings, quick start, and scope.
- `REPRODUCIBILITY.md`: retained submission documentation, detailed reproduction notes, evidence map, and limitations.
- `docs/CODE_GUIDE.md` and `docs/RESUME.md`: source navigation, provenance notes, and evidence-backed resume wording.
- `.github/workflows/ci.yml`: automated research tests, strict evidence validation, and interface smoke check.
- `MANIFEST.md`: this archive inventory and exclusion record.
- `LICENSE`: licence retained from the source repository.
- `requirements.txt`: minimal Python dependencies used by the implementation.

### Source and interfaces

- `core/base.py`: shared MetricDB relation, query, and index interfaces.
- `modules/metric_indexing/indexing.py`: native FAISS-HNSW and external stored-topology adapter.
- `modules/entity_linking/`, `modules/predicate_search/`, `modules/query_engine/`, and `modules/union_management/`: shared MetricDB scaffold imported by `main.py`; these are not presented as completed dissertation contributions.
- `main.py`: integrated scaffold entry point and `SimpleMetricRelation` used by adapter tests.

### Research pipeline

`data/` contains the source code for:

- deterministic synthetic-data and subset preparation;
- affine transformations and controlled alignment noise;
- local HNSW creation and level-0 topology extraction;
- graph diagnostics, external traversal, local repair, and connectivity checks;
- exact ground truth, full rebuild, incremental insertion, dual-index, no-cross, random-link, sampled cross-kNN, and indexed cross-kNN baselines;
- bounded N4/C2 and N8/C2 propagation and unlimited JOIN1;
- sequential three-subset integration;
- GloVe ingestion and preparation;
- latency/work benchmarking, paired bootstrap statistics, result aggregation, table generation, and final-evidence validation.

### Configurations

`configs/` retains all checked experiment configurations from the research repository, including smoke tests, formal Phase 1--3 settings, repair ablations, Stage 20/21 settings, Cycles 22--25, and paired scaling statistics.

### Results and dissertation assets

- `results/`: compact results for the synthetic stopping gate, common adapter, query cost, loaded-base insertion, sequential integration, GloVe evaluation, and scaling confidence intervals.
- `results/final_manifest.json`: checksum-bearing final evidence manifest covering nine compact artifacts.
- `data/phase3_repair_ablation_summary/`: aggregate repair-ablation JSON and CSV outputs.
- `data/phase3_repair_ablation_phase2_metrics/`: compact metric snapshots used by the repair-ablation aggregation.
- `dissertation_assets/`: the 14 CSV/TeX plot and table inputs referenced by the final dissertation results chapter.

### Tests

`tests/` contains 85 tests across 16 files covering:

- Phase 1 data preparation, transformations, exact ground truth, native rebuild, backend probe, and reproducibility;
- Phase 2 conditions, graph operations, and pipeline behaviour;
- Phase 3 common infrastructure, indexed cross-kNN, formal summaries, and repair-ablation summaries;
- Phase 4 common adapter and supplemental sequential/GloVe/statistical behaviour;
- the `MetricIndex` interface.

## Intentionally excluded

The submission copy intentionally omits:

- Git history and repository metadata;
- `.venv`, IDE settings, caches, bytecode, logs, and operating-system metadata;
- compiled demonstration binaries and unrelated C++ examples;
- internal development plans, progress logs, scratch files, backups, and old archives;
- the final dissertation PDF and LaTeX source, which are submitted separately;
- the external `glove.2024.wikigiga.50d.zip` archive;
- generated raw vectors, full per-case arrays, native FAISS indices, prepared inputs, and other large intermediate experiment directories;
- unrelated scaffold-module test files that are outside the evaluated dissertation path.

The omitted working data and index artifacts occupy approximately 17 GB in the original workspace. Their sources, preparation logic, formal parameters, compact results, and relevant limitations are documented in `README.md` and the retained configuration/result files.
