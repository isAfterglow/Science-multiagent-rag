# 证据契约、计划顾问与 QoS 策略

评测日期：2026-08-03。

## 类型化证据契约

Supervisor 现在为每个问题声明受限 `EvidenceRequirement`：`registry_analysis`、`report`、`run_log`、`run_status`、`input_deck`、`script`。Retriever 对每个要求的文档类型独立取证，Critic 逐项检查，不能由一个高分 input deck 替代 report。缺失类型触发一次 `typed_evidence_retry` Recovery。

对“cpv_front_scale 对 early_1_2_rmse 的影响，结合运行日志和历史报告解释”，契约要求 Registry、report、run_log，最终三类证据均存在并通过 Reviewer。

## 差异化计划顾问

“下一轮/候选仿真建议”不再复制历史 Top case。`suggest_exploration_plan` 以历史最优 case 为基线，根据 Pearson 绝对值选择最多 3 个参数，并按降低目标指标的相关方向在白名单边界内扰动 5% span。草案同时写入相关性、方向、因果局限和验证要求。

本次 `early_1_2_rmse` 草案选择 `cpv_front_scale`、`ER_scale`、`cpv_mid_scale`。草案 ID 使用 `draft-` 前缀；只有显式调用 `POST /plans/drafts/confirm` 后才进入 `pending`，仍必须经过既有人工审批。

## 30 题质量与服务策略

| 配置 | 通过 | 通过率 | P50 | P95 | 自动降级 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 + fixed | 28/30 | 93.33% | 17.501 ms | 19.961 ms | 0 |
| Hybrid Rerank + fixed | 28/30 | 93.33% | 626.429 ms | 4044.845 ms | 0 |

剩余 `wf-21` 是多证据语义与旧单一路由标注不一致；`wf-27` 已产生探索草案，但旧评测仍要求历史 `case_` 关键词。因此不为提升分数改写业务逻辑。

服务策略为低负载使用 `hybrid_rerank`；Embedding 饱和时降级 BM25，Reranker 饱和时降级 Hybrid。EvidenceCard、Trace 和 `/observability` 记录实际模式与资源排队；测试已验证 Reranker 饱和时切换 Hybrid。

## 工作台

前端新增证据契约、Recovery 动作、探索草案确认、后台任务列表和资源队列面板；SSE 显示节点实时状态。确认草案不会自动批准或执行。
