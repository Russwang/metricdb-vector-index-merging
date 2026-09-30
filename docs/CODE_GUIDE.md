# Code guide

Start with the adapter and the self-contained Cycle 22 runner, then trace graph integration and the retained evaluation evidence. Commands below run from the repository root.

## 1. Index contract and identifiers

Read [`core/base.py`](../core/base.py) and [`modules/metric_indexing/indexing.py`](../modules/metric_indexing/indexing.py). `MetricIndex` supports `faiss_hnsw` and `external_graph` behind a common build/search interface. Inspect its checks for row alignment, duplicate IDs, dimensions, non-finite vectors, missing graph nodes, and invalid query parameters.

Native row IDs and graph global IDs are mapped back to application item IDs. Search diagnostics expose backend-specific work; serialization retains the representation required by each backend. [`tests/test_metric_indexing.py`](../tests/test_metric_indexing.py) and [`tests/test_phase4_metric_index.py`](../tests/test_phase4_metric_index.py) exercise these paths.

## 2. Alignment and graph representation

[`data/phase2_conditions.py`](../data/phase2_conditions.py) defines controlled alignment conditions. [`data/phase2_graph.py`](../data/phase2_graph.py) handles stored level-0 adjacency, graph diagnostics, repair, and external search over transformed vectors. Trace how node IDs remain stable when the coordinates change.

The graph reuses local topology; it does not mutate complete multi-layer FAISS indices into a single native index. Exact transformations and injected noise let the experiments study when local connectivity remains useful.

## 3. Cross-subset integration

[`data/phase3_merge.py`](../data/phase3_merge.py) implements graph integration, portal/bridge selection, propagation, and pruning. Read the indexed candidate path alongside the random-link control in [`data/phase3_baselines.py`](../data/phase3_baselines.py). Bounded N4/C2 and N8/C2 restrict propagation work; unlimited JOIN1 provides a comparison with fewer budget restrictions.

[`tests/test_phase3_indexed_cross_knn.py`](../tests/test_phase3_indexed_cross_knn.py) checks this integration path. [`data/phase4_sequential.py`](../data/phase4_sequential.py) extends the evaluation to ordered three-subset integration.

## 4. Evaluation and evidence

[`data/phase3_baselines.py`](../data/phase3_baselines.py) includes exact ground truth, native rebuild/insertion, dual-index querying, and graph controls. [`data/phase3_metrics.py`](../data/phase3_metrics.py) computes retrieval metrics; [`data/phase4_benchmark.py`](../data/phase4_benchmark.py) records query work; [`data/phase4_statistics.py`](../data/phase4_statistics.py) implements paired bootstrap analysis.

Verify the resume result using [`results/scaling_confidence_intervals.json`](../results/scaling_confidence_intervals.json): the 50k source / 33,334-vector evaluated pair reports a bounded-N8/C2 versus random-link Recall@10 difference of `0.05634765625`. Multiplying by 100 gives 5.63 percentage points. The frozen stopping-gate results separately document the gap to native baselines.

```bash
python data/validate_final_evidence.py --require-complete
python -m unittest discover tests
python data/run_cycle22.py --output scratch/cycle22_smoke.json
```

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for formal experiment commands, required inputs, measured environment, and the dissertation-to-code mapping.

## Provenance

The publication starts from the curated `MetricDB_Project_Materials` submission package prepared in August 2026. Its audit records source repository HEAD `582ae1ccb81283bd5e273b7a6cad43980b4779e9`. The frozen evidence records an earlier evidence-generation commit; that commit is not a verified execution commit for every experiment. Public repository history starts at this curated snapshot and does not include the shared upstream Git history.

The frozen manifest's repair note combines local/dual wins as `0/55`. The submission audit gives the more precise breakdown: 1/55 pair-averaged local cases, 0/55 dual-index cases, and 55/55 sampled-union cases. Frozen files are retained unchanged so their published checksums remain verifiable.
