# 阶段 3：科研语料扩展、向量索引对照与原页核验

> 本文前半部分记录 75 题/V2 的扩容过程。当前发布基线为 V5：49 份来源、1,016 页、2,234 chunks、120 题、`dense + parent_child`；Local Dense 的 Source Recall@5 为 0.8694、Page hit 为 0.7890、P95 为 212.173 ms。完整当前选型见 `RAG_PRODUCTION_HARDENING.md` 与 `RETRIEVAL_OPTIMIZATION_V5.md`。

## 目标与边界

本阶段扩大公开科研知识库，并验证两种向量后端在完全相同的语料、embedding 和检索配置下的行为。原始 PDF、解析产物、页图和向量缓存保留在 `data/knowledge_sources/`，不进入版本库；可复现的来源清单位于 `knowledge_sources/manifest.json`。

公开资料仅作为 C/D 级背景或机理证据。项目运行数值仍只能由 A/B 级 Registry 与项目报告证据支撑，低置信 OCR 数字须经人工核验。

## 语料与评测集

初版之后分两批追加了公开下载的 NASA NTRS 资料，覆盖 Arc Jet 设施/试验、Avcoat 与 PICA、炭化烧蚀与热解、多尺度复合材料、PATO 及热物性建模。第二批还加入超高温陶瓷、光学诊断、柔性 TPS、热流表征、计算机视觉自动分析、有限元验证、三维机织材料与多保真建模。批量导入不会在单份 PDF 下载或解析失败时中止：`ingest_report.json` 会记录 `source_id`、失败阶段、异常类型和摘要，成功来源仍可替换 `scientific:` 命名空间。

- 公开来源：47 份，均由版本化 manifest 管理。
- PDF 总页数：999 页；实际纳入 977 页。
- 页级 DocumentIR groups：1,581 个，其中 177 个表格、398 个双栏页。
- 科研文献 Parent-Child 检索块：2,059 个；完整 Registry 重新建索引后会加上项目文档。
- 文档类型：conference paper、presentation、poster、preprint、book chapter、技术资料与扫描报告。`document_kind` 与页码、bbox、表格、OCR 元数据一同保留。
- 页级多模态检索题：75 道。`mm-59` 至 `mm-75` 分别对应第一批新增来源中的实际段落或目录页，不通过替换参数扩题。第二批来源已完成解析，待补充人工页级题后再重建该语料版本的评测基线。
- 扩容批次的失败来源：0；OCR 关闭以控制批量 CPU/内存，既有扫描资料的 OCR 产物不被当作精确数值事实。

## 同语料后端对照

运行条件：BGE-M3 Dense、`parent_child`、Top-5、1,515 chunks、75 个问题。Milvus 使用 Lite 的 `FLAT + IP`，与本地 Numpy 全量内积保持可比；报告文件分别为 `reports/stage3_expanded_final_dense.json` 与 `reports/stage3_expanded_final_milvus.json`。

| 指标 | Local Numpy | Milvus Lite |
| --- | ---: | ---: |
| Source Recall@5 | 0.9311 | 0.9311 |
| MRR | 0.8840 | 0.8840 |
| nDCG@5 | 0.8796 | 0.8796 |
| Page hit rate | 0.7812 | 0.7812 |
| P50 latency | 159.274 ms | 880.577 ms |
| P95 latency | 199.639 ms | 995.911 ms |

结论：在 1,515 个块上，两者结果一致，本地 Numpy 延迟仍明显更低，因而继续作为本地开发和默认在线路径。Milvus 首次同步 1,515 vectors 用时 1,543.764 ms；其工程价值在增量 upsert、metadata filter、collection 生命周期与未来多实例/大语料扩展，而不是宣称小语料性能更好。后续达到数万块后应以新 collection 复跑同一评测，再决定 HNSW 或 AUTOINDEX。

## 扩容后的检索选型

先在 58 条旧题上隔离比较检索策略，再用 75 条扩大题集验证默认路径。所有策略共享同一语料、chunk 和 Top-5，不能把不同数据集的数字横向比较。

| 策略 | Source Recall@5 | Page hit rate | P95 latency | 决策 |
| --- | ---: | ---: | ---: | --- |
| BM25 | 0.8937 | 0.7872 | 15.108 ms | OCR/精确术语的快速候选 |
| Dense | 0.9282 | 0.7447 | 188.862 ms | 语义来源召回基线 |
| RRF Hybrid | 0.9109 | 0.7872 | 204.638 ms | 不作为默认 |
| `dense_page` | 0.9282 | 0.6596 | 188.066 ms | 保留负实验，不启用 |
| Hybrid + BGE reranker | 0.9368 | 0.7234 | 4,040.708 ms | 离线分析候选，不在线启用 |

`dense_page` 是通用两阶段实验：Dense 先选来源，BM25 再在同一来源内选页，并在 EvidenceCard 中记录 `dense_source_rank` 与 `bm25_page_rank`。该实验对英文 PDF 与中文问句的词面错配敏感，页命中变差，故明确不作为默认。75 题最终默认 Dense 得到 Source Recall@5 `0.9311`、Page hit `0.7812`、P95 `199.639 ms`。

## 扩容版本 V2

V2 引入 2 篇开放获取 MOOSE 框架/模块论文，使来源不再局限于 NASA；并将页级题扩至 120 条，新增 `expected_document_kind` 和 `document_kind_hit_rate`。最终实际语料为 49 份来源、1,016 页 PDF、994 个入库页、1,613 个页级 DocumentIR groups、177 个表格、405 个双栏页和 2,234 个完整检索块。

| 指标（120 题，Dense，Local） | 值 |
| --- | ---: |
| Source Recall@5 | 0.8694 |
| MRR / nDCG@5 | 0.8479 / 0.8394 |
| Page hit rate | 0.7890 |
| Block type / document kind hit rate | 0.9655 / 0.9565 |
| P50 / P95 | 164.209 / 209.235 ms |

同一 2,234 chunk、102 题对照中，Milvus Lite FLAT/IP 与 Local 的质量指标一致，但 P50/P95 为 1,226.955/1,305.651 ms，高于 Local 的 180.706/225.397 ms；首次同步 2,234 vectors 耗时 2,015.485 ms。因此 Local Dense 继续是默认在线路径，Milvus 保留给更大规模 collection、增量 upsert 和多实例部署。类型约束参与重排的实验没有提升来源命中且降低页命中，未启用；`document_kind` 仅用于证据展示、人工核验与评测切片。

## 可审计引用与人工核验

EvidenceCard 保留 `source_id`、页码、chunk 行号、PDF bbox 与来源等级。工作台通过 `GET /evidence/page/{source_id}/{page}` 按需渲染受管理的原始 PDF 页，并将相关 bbox 按页面尺寸缩放成叠加框。接口只接受 Registry 中的来源，且仅允许 `data/knowledge_sources/raw` 下的 PDF，禁止用任意路径读取文件。

这使 Reviewer 可以从回答中的证据卡直接回到原页检查段落、双栏阅读顺序或表格位置。该核验能力用于人工审查，不能将模型候选文本或 OCR 自动升级为已证实事实。
