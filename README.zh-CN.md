# MetricDB：向量索引变换与合并

**爱丁堡大学硕士毕业项目 · Python / NumPy / FAISS / HNSW**

[English](README.md) · [代码导读](docs/CODE_GUIDE.md) · [复现说明](REPRODUCIBILITY.md) · [简历表述](docs/RESUME.md)

本项目研究：多个数据子集分别建立向量索引后，当其向量被对齐到共同空间，能否复用已有图结构，实现统一检索，并降低索引整合成本？

论文题目为 *Transforming and Merging Vector Indices for Unified Retrieval in MetricDB*。仓库以最终毕业提交材料为基础，保留研究实现、实验配置、精简结果和测试，方便招聘方查看源码与核对简历中的项目描述。

## 实现内容

- **索引适配**：统一封装 FAISS 原生 HNSW 与外部图检索，处理全局/局部 ID、输入校验、序列化和查询诊断。
- **拓扑复用**：提取已存储的 HNSW 第 0 层邻接关系，在对齐后的向量上复用局部拓扑，加入跨子集连边、有限预算传播和剪枝。
- **对照实验**：实现精确 kNN 真值，与全量重建、增量插入、双索引检索、随机连边及采样跨子集 kNN 等方法比较。
- **评估链路**：覆盖 10k–50k 规模的合成源数据集、三子集顺序整合和 25k 向量的 GloVe 联合集；记录 Recall、查询延迟、整合耗时、距离计算和连通性，并计算配对 bootstrap 置信区间。

## 与简历对应的结果

在 50k 合成源数据中选取的 **33,334 向量双子集联合集**上，Bounded N8/C2 相比随机直接连边，**Recall@10 提升 5.63 个百分点**，配对 bootstrap 95% 置信区间为 **[5.36, 5.93] 个百分点**。依据为 [scaling_confidence_intervals.json](results/scaling_confidence_intervals.json) 和 [scaling_ci_table.tex](dissertation_assets/scaling_ci_table.tex)。

该结果说明有信息的连边策略相对随机连边改善了召回。与原生重建、插入比较时仍存在质量和耗时差距，50k 的实验未通过继续扩大规模的预设门槛，详见 [cycle21_50k_compact.json](results/cycle21_50k_compact.json)。项目价值包括明确这些取舍及其适用边界。

## 本地运行

在仓库根目录执行，实验记录使用 Python 3.10：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data/validate_final_evidence.py --require-complete
python -m unittest discover tests
python data/run_cycle22.py --output scratch/cycle22_smoke.json
```

严格证据校验检查 9 个冻结结果文件的大小与 SHA-256；测试套件包含 85 项研究路径测试；Cycle 22 是不依赖大型外部输入的接口演示。完整复现命令及输入限制见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 贡献范围与复现边界

毕业项目贡献集中于 `modules/metric_indexing/` 与 `data/` 中的研究路径。其他模块来自 MetricDB 共享项目框架，保留原团队署名与 MIT 许可证。

复用对象是已存储的 HNSW **第 0 层图拓扑**；完整多层 HNSW 物理合并和对齐模型训练不在本实现范围内。仓库保留正式实验的精简证据，未附大型原始数组、索引、GloVe 下载包、论文 PDF 或个人简历。完整正式实验需要按说明重新生成或获取输入。
