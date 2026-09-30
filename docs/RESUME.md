# Resume project wording

Project: **MetricDB: Vector Index Transformation and Merging**  
Repository: https://github.com/Russwang/metricdb-vector-index-merging  
Stack: Python, NumPy, FAISS/HNSW, vector search

## 中文

**MetricDB：向量索引变换与合并｜硕士毕业项目**

- 基于 Python / NumPy / FAISS / HNSW 适配原生与外部索引，实现向量对齐后的第 0 层图拓扑复用、跨子集连边及有限预算传播，处理 ID 映射与序列化。
- 在 10k–50k 合成源数据集及 25k GloVe 向量联合集上对比重建、增量插入与双索引基线；在 33,334 向量合成联合集上，相比随机连边提升 Recall@10 **5.63 个百分点**。

## English

**MetricDB: Vector Index Transformation and Merging | MSc Dissertation**

- Implemented native FAISS/HNSW and external graph adapters, reusing stored level-0 topology after vector alignment with cross-subset linking, bounded propagation, ID mapping, and serialization.
- Benchmarked against rebuilding, insertion, and dual-index retrieval using 10k–50k synthetic source datasets and a 25k GloVe union; improved Recall@10 by **5.63 percentage points** over random linking on a 33,334-vector synthetic union.

## Evidence and interview context

The quantitative result is bounded N8/C2 versus random direct links, with a paired bootstrap 95% interval of [5.36, 5.93] percentage points. It is not a comparison against native full rebuilding. The 50k label names the source dataset; the evaluated pair contains 33,334 vectors. Original configurations and compact results are in [`configs/`](../configs/) and [`results/`](../results/).

The project's engineering scope is the research indexing and evaluation path. Other shared MetricDB modules remain scaffolds. Discuss the native/external backend overhead and the unsuccessful 50k stopping gate alongside the positive comparison; both are supported by the retained evidence.
