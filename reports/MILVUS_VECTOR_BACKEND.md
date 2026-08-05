# Milvus 向量后端：实现与对照评测

> 历史阶段对照（452 chunks）。当前 2,234 chunk 的可比基线与选型结论以 `STAGE3_RAG_SCALE_AND_CITATION.md` 和 `README.md` 为准。

## 架构

SQLite Registry 仍是项目事实、审批记录和完整科学证据的事实源；DocumentIR 仍保存 PDF 页码、bbox、表格 CSV、OCR 置信度和原始页图。Milvus 仅保存可再生 Child 的 `chunk_id`、BGE-M3 向量、`source_type`、`document_id` 和 `content_hash`。ANN 命中后由内存中的 SQLite-derived chunk map 回填完整 `EvidenceCard`，避免复制或丢失可审计证据。

`VectorStore` 抽象有两个实现：

- `LocalNpyVectorStore`：小语料、离线评测默认基线。
- `MilvusVectorStore`：Milvus Lite、Standalone 和远程 Milvus 使用同一 PyMilvus API；不可达时明确降级到本地 Numpy，并将原因写入 EvidenceCard 和 LangGraph trace。

## 索引与更新

Collection `scientific_chunks_v2` 使用 `FLOAT_VECTOR(1024)`、`IP` 与 `FLAT` 索引。当前 452 chunk 的规模下，FLAT 用于与 Numpy 全量点积严格对齐；当语料增长后可新建带版本号的 HNSW/AUTOINDEX collection，再做精度-延迟评测，不能静默替换基线。

每个 Child 以 `chunk_id + text + metadata` 计算 `content_hash`，BGE 向量还缓存为 `data/index/chunk_embeddings/<hash>.npy`。知识库更新时：

```text
DocumentIR / SQLite 更新
 -> 仅编码缺失 content_hash 的 Child
 -> Milvus 查询现有 chunk_id/content_hash
 -> upsert 新增或变化 Child
 -> 删除当前语料不再存在的 chunk_id
```

首次同步：452 upsert，888.278ms；无语料变更的第二次同步：0 upsert、0 delete，265.674ms。

## 54 题 Dense 对照

同一 BGE-M3、同一 parent-child（452 chunk）、同一 54 题科研资料评测：

| 后端 | 索引 | Source Recall@5 | MRR | 页码命中 | P50 / P95 ms |
|---|---|---:|---:|---:|---:|
| Local Numpy | 全量 Inner Product | 94.75% | 0.9506 | 72.09% | 137.263 / 160.579 |
| Milvus Lite | FLAT + Inner Product | 94.75% | 0.9506 | 72.09% | 207.774 / 326.159 |

小语料下 Local Numpy 延迟更低，因此仍是默认开发与评测后端。Milvus 的价值是增量 upsert、metadata filtering、索引生命周期与将来 Standalone/多用户部署，而不是在 452 条向量上追求更低延迟。

## 运行方式

本机已安装 PyMilvus 2.6.17 和 Milvus Lite 2.4.12，可直接使用项目内持久文件：

```bash
VECTOR_BACKEND=milvus \
MOOSE_MILVUS_URI=data/milvus/scientific_chunks.db \
conda run -n scitime-agent python -m app.cli vector-sync
```

Docker Standalone 配置位于 `deploy/milvus/docker-compose.yml`，用于服务化部署。当前账号未获 Docker socket 权限，因此未在本轮以容器方式启动；该限制不会影响 Milvus Lite 的真实 API 索引和评测。Standalone 启动后只需将 `MOOSE_MILVUS_URI` 改为 `http://127.0.0.1:19530`。
