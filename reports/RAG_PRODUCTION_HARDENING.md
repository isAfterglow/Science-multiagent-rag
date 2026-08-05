# RAG Production Hardening: Stages 1-6

评测语料固定为版本化 manifest 的 49 个公开来源、2,234 Parent-Child chunks 和 120 个页级问题。原始 PDF、解析缓存和向量文件不进入 Git；报告中的指标均可由对应命令重建。

## 1. GPU 资源治理

`eval/retrieval_device_benchmark.py` 在同一 16 passage / 8 query / 8 candidate rerank 工作负载上测得：

| Device | Encode texts/s | Dense P50/P95 | Rerank 8 | GPU peak allocation |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.723 | 144.281 / 158.577 ms | 4,705.156 ms | 0 |
| RTX 3050 CUDA | 3.470 | 36.678 / 84.978 ms | 4,301.096 ms | 3,509,726,208 bytes |

CUDA 对 BGE embedding/query 有明显收益；reranker 收益不足以抵消其绝对成本，因此仍不作为在线默认。`resource_limits.acquire_inference` 对 embedding、reranker、local LLM 提供共享 GPU admission（默认并发 1）和独立模型并发上限。它只协调本进程，外部 Ollama 的多进程显存治理必须由部署层/模型服务配置完成。

## 2. 来源摘要两阶段实验

`source_fusion` 先对 `title + topics + aliases + document_kind + first fragment` 形成来源摘要向量，再与全库页级 Dense rank 做 RRF 风格融合。它从不缩小页候选集，因此不是 profile 硬过滤。

| 120 题 CPU | Source Recall@5 | MRR | nDCG@5 | Page hit | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense baseline | 0.8694 | 0.8479 | 0.8394 | 0.7890 | 212.173 ms |
| Source-summary fusion | 0.8778 | 0.8575 | 0.8470 | 0.7890 | 223.002 ms |

融合保留为显式 `precision` 档：Source Recall@5、Question hit 和排名指标提高，但默认 Dense 仍有更低延迟。先前的 profile 硬过滤实验使 Source Recall@5 降到 0.7778，已回退并作为负实验记录。

## 3. 动态检索路由评估

全量 BM25 的 Page hit 为 0.7982，但 Source Recall@5/MRR/nDCG 为 0.8611/0.8232/0.8185，低于 Dense。V5 仅对 4/120 个高置信、精确英文技术术语问题启用 BM25 路由，整体 Source Recall@5 保持 0.8694，Page hit 提升到 0.7982，P95 为 217.114 ms。泛词和不确定问法仍保持 Dense，避免关键词路由损伤语义召回。

## 4. 异步导入

`POST /knowledge/ingest` 创建持久 `ScientificIngestTasks`。任务记录 download/document_ir/registry_replace/completed 阶段、来源进度、失败数、CPU device、耗时与最终结果；`GET /knowledge/ingest/tasks/{task_id}` 可查询。HTTP 请求线程不再承担下载、解析、OCR 和 Registry 替换。

## 5. 回答级可靠性

Claim-Evidence evaluator 从 28 增至 31 个案例，覆盖 chunk 缺失、证据冲突、Registry 分析、待审批草案、可回溯数值表和缺页表格。数值表格仅在 `block_type=table` 且有原 PDF `page` 时可被接受；该校验是确定性来源可追溯，不是语义 NLI。当前 31/31，numeric-table page verification 2/2。

## 6. 回归与发布边界

公共 CI 运行契约、manifest、安全和 Claim-Evidence 测试；self-hosted full regression 在有本地模型/语料的 runner 上额外运行完整 120 题 Dense 回归。Local Numpy 仍是当前默认在线后端；Milvus Lite 的价值保留在更大 collection、增量 upsert、metadata filter 和多实例部署，不宣称其在 2,234 chunks 上更快。
