# 科研多模态 RAG 升级记录

> 历史阶段实验（小语料）。当前公开语料、默认切分和 120 题评测结论以 `STAGE3_RAG_SCALE_AND_CITATION.md`、`RAG_PRODUCTION_HARDENING.md` 与 `RETRIEVAL_OPTIMIZATION_V5.md` 为准。

## 范围与边界

本次新增公开 NASA NTRS 资料，而不从受限出版站点抓取全文。`nasa_tufroc_2019` 是公开 TUFROC 技术资料（等级 C）；`nasa_multiwall_1982` 是公开历史扫描专利资料（等级 D）。两者只能解释外部热防护背景与机理，不能支撑项目 MOOSE 的实际指标、运行耗时或参数结论；这些结论仍必须引用等级 A/B 的 Registry、日志、input deck 和项目报告。

每个原始文件保存 URL、访问标记与 SHA-256，见 `data/knowledge_sources/manifest.json` 和 `data/knowledge_sources/parsed/ingest_report.json`。科学语料使用独立命名空间 `scientific:`，导入时不会删除 simulation case、metrics 或原有文档。

## 解析结果

| 资料 | 页数 | 入库页数 | OCR 页数 | 平均 OCR 置信度 |
|---|---:|---:|---:|---:|
| TUFROC Thermal Protection System (2019) | 1 | 1 | 0 | - |
| Multiwall thermal protection system (1982) | 9 | 9 | 5 | 0.9508 |

原生 PDF 使用 PyMuPDF 文本层；页面文本不足时，或 manifest 声明扫描验证时，先渲染为 PNG，再由 RapidOCR 提取。证据卡保留 `source_id`、`page`、`authority`、`sha256`、`ocr_used`、`ocr_confidence` 与页图路径。低置信 OCR 仅能作为需复核的定位证据，不能用于精确数值断言。

## 分块与检索

实现了四种策略：`document`、`fixed`、`structure` 和 `parent_child`。`parent_child` 对 PDF 保留页级 parent metadata、以 1800/260 字符子块索引，适合长论文的段落级检索；固定块保持 850/140 字符。检索为 BM25/Dense/RRF/Reranker 的可切换管线，且对外部 PDF 以“文件+页”去重、对重复 case deck 以“类型+文件名”去重，避免相同模板挤占跨来源证据。

当前公开语料只有两份、其中一份仅一页，实测固定块优于 parent-child，因此默认仍为 `fixed`。这是一项由消融结果决定的配置，不是对 parent-child 的删除。

## 32 题评测与改进

题目独立于原来的 12 题项目检索集，覆盖项目工程证据 10 题、公开资料 14 题、页码定位 14 题、明确 OCR 验证 4 题、跨来源与等级边界 8 题（类别可重叠）。指标由 `eval/scientific_multimodal_retrieval_eval.py` 计算，保存完整逐题结果。

| 方案 | Source Recall@5 | 题目命中 | MRR | nDCG@5 | 页码命中 | 等级正确 | OCR 页码命中 | P50 / P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 首轮 BM25 fixed | 91.15% | 93.75% | 0.9219 | 0.9060 | 76.19% | 87.50% | 100.00% | 0.144 / 0.289 |
| 改进后 BM25 fixed | 97.40% | 100.00% | 0.9844 | 0.9685 | 85.71% | 96.88% | 100.00% | 0.184 / 0.303 |
| 改进后 BM25 parent-child | 97.40% | 100.00% | 0.9688 | 0.9614 | 80.95% | 96.88% | 75.00% | 0.124 / 0.229 |

首轮失败分析表明，问题不在 OCR，而在来源画像不足和重复 deck 占位。改进是：

1. 在科学来源画像中加入实验条件级术语，使未点名论文的热流、剪切、瓦片和电弧喷流查询能路由到公开论文；同时保留扫描、专利、再入和飞行器信号。
2. 将同名 case 模板合并为一个 citation identity，使报告与外部资料能进入同一份证据集。
3. 对旧项目文档在检索输出中推导 A/B 来源等级，补齐可观测性而不重写既有 Registry 数据。

向量 Hybrid/Reranker 仍保留为运行时能力；本轮改进对比固定为低延迟 BM25，以隔离 chunk 与来源策略对多模态资料的影响。完整向量消融应在宿主 CPU 空闲时离线运行，避免与本地模型服务、MOOSE 任务争抢资源。

## 复现

```bash
conda run -n scitime-agent python -m app.cli ingest-scientific
conda run -n scitime-agent python -m eval.scientific_multimodal_retrieval_eval --mode bm25 --chunk-strategy fixed
MOOSE_COPILOT_LLM_ENABLED=0 RETRIEVAL_MODE=bm25 conda run -n scitime-agent python -m app.cli collaborate "检索 TUFROC 在热流和剪切条件下的公开资料"
```
