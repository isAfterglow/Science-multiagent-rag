# 复杂科研文档 RAG：解析、检索与可信回答

> 历史阶段文档解析与切分实验。452 Child 的数字仅对应当时语料；当前 2,234 Child 语料和默认检索请见 `STAGE3_RAG_SCALE_AND_CITATION.md`。

## P0：DocumentIR 与 token-aware Parent-Child

科学语料为 8 份公开 NASA NTRS 技术资料，覆盖演示型技术文档、双栏论文式报告、扫描专利、矢量表格和 OCR 页。解析不再将 PDF 当成整页纯文本：

1. PyMuPDF `get_text("blocks")` 提取文本 block 与 bbox。
2. 过滤重复页眉/页脚，按双栏左列到右列重建阅读顺序。
3. `page.find_tables()` 提取矢量表格为 Markdown 和 CSV；输出表格结构置信度。
4. 低文本页面使用 RapidOCR，并保留置信度、页图和完整页 bbox。
5. 页/章节是 Parent；相邻段落组和表格是可引用 block；检索时使用 BGE tokenizer 生成最大 384 token、64 token overlap 的 Child，保证低于 BGE-M3 512-token passage 上限。最终运行时共有 452 个 Child，长度为 11--384 token，全部满足该预算。

`eval/document_ir_eval.py`：195 页、2216 个 block、解析覆盖率 97.44%、72 个双栏页、9 个表格（3 个高置信、6 个待核验候选）、6 个 OCR block。Docling 已验证为可选高保真解析器，但在本机 CPU 上解析 12 页技术论文超过 3 分钟，因此不作为默认路径；PyMuPDF DocumentIR 是默认快速路径。

## P1：54 题评测与 chunk 消融

原有 44 题扩展为 54 题，新增 FIATC 表格、行星 mole fraction 表、双栏段落、TPS shear 表格和扫描资料题。新增 `block_type_hit_rate`，衡量是否召回段落或表格，而非只命中文件来源。

| BM25 切分 | Source Recall@5 | 页码命中 | Block 类型命中 | P50 / P95 ms |
|---|---:|---:|---:|---:|
| block-document | 93.21% | 74.42% | 100.00% | 2.187 / 2.653 |
| token Parent-Child | 92.28% | 83.72% | 100.00% | 2.700 / 3.582 |

默认选择 token Parent-Child：它牺牲少量来源召回，换来更好的页级定位、严格 token 上限，以及 section/bbox/table 的可追溯证据。来源 Profile 和候选来源配额仍保留在检索层。

## P2：可复现模型与策略选择

本机仅有 BGE-M3 和 BGE Reranker v2-m3；尝试访问 Hugging Face 下载 multilingual-e5-small 未返回，因此没有伪造外部模型对比。现有对照仍覆盖词法、向量、融合和精排四种可运行策略：

| token Parent-Child 策略 | Source Recall@5 | 科学资料 Recall@5 | MRR | 页码命中 | P50 / P95 ms |
|---|---:|---:|---:|---:|---:|
| BM25（最终 452 chunk） | 92.28% | 95.35% | 0.8741 | 83.72% | 2.700 / 3.582 |
| BGE-M3 Dense（最终 452 chunk） | 94.75% | 97.67% | 0.9506 | 72.09% | 143.815 / 178.821 |
| Hybrid RRF（457 chunk 消融） | 94.75% | 95.35% | 0.8701 | 79.07% | 128.962 / 189.081 |
| Hybrid + BGE Reranker（457 chunk 消融） | 97.53% | 100.00% | 0.9213 | 72.09% | 3107.686 / 3839.318 |

最终线上决策只依据严格同语料的前两行：公开论文默认 Dense；扫描/OCR 与精确页定位默认 BM25。Hybrid+Rerank 是短噪声过滤前仅多 5 个 child 的完整策略消融，保留其结果用于展示召回-延迟上界，不作为最终线上默认配置；它们仅用于离线或用户显式高精度模式。模型选择依据是相同 54 题、相同 CPU 环境的准确率-延迟权衡。

## P3：Agent、Reviewer 与工作台

- Supervisor 能识别 `table / page / FIATC / arc jet / shear` 等复杂文档请求；显式扫描意图优先扫描资料。
- Retriever 返回 `block_type`、section、bbox、table CSV、table confidence、OCR confidence、页码和来源等级。
- Reviewer 阻止项目事实由 C/D 外部资料支撑；低置信 OCR 或低置信表格不能回答精确数值。
- 工作台显示节点流转、DocumentIR 摘要、章节、bbox、表格 CSV 预览、表格置信度、扫描原页、审批和执行状态。

端到端验证：FIATC 对照表定位到 `nasa_thermal_response_2011` 第 4 页、高置信 table block；TPS arc-jet shear 数值定位到第 10 页低置信 table candidate，并由 Reviewer 拒绝精确数字结论。

## 复现

```bash
conda run -n scitime-agent python -m app.scientific_ingest --no-download --max-ocr-pages 2
conda run -n scitime-agent python -m eval.document_ir_eval
conda run -n scitime-agent python -m eval.scientific_multimodal_retrieval_eval --mode bm25 --chunk-strategy parent_child
conda run -n scitime-agent python -m eval.scientific_safety_eval
```
