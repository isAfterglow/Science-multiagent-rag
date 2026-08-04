# Recovery 闭环与默认模型并发

评测日期：2026-08-03。

## 受限 Recovery Agent

Reviewer 出现 `citation_coverage`、`missing_retrieval`、`no_evidence` 时，图从 Reviewer 转入 Recovery，最多一次：

1. 依据失败类型确定动作和来源范围。
2. `citation_coverage` 优先补检索 report。
3. 缺少文档时补检索 report，日志问题同时补 run_log。
4. 无证据时取消来源过滤后重试。
5. Recovery 后重新执行 Critic、Synthesizer、Reviewer。

旧 Critique 不会污染新一轮审核。若仍无法支持结论，Reviewer 以 `requires_revision=false` 明确拒绝，终止图而非循环。

定向验证：

- `cpv_front_scale 的影响，运行日志能说明什么？` 首轮日志证据未覆盖参数，Recovery 报告优先重试后通过。
- `foo_unknown_scale 对 early_1_2_rmse 的影响是否明显？` 重试后仍无来源支持，正确拒绝；只执行 1 次 Recovery。

## 通用工作流能力

- 报告/解释/机理/因果问题优先过滤 report。
- 日志与指标同时出现时要求 run_log + report，避免以日志替代数值解释。
- 包含“下一轮/候选仿真建议”的问题生成仅内存中的受限 `SimulationPlan` 草案；不写入 PlanStore，不执行 MOOSE，仍需人工审批。

30 道科研工作流回归（BM25 + fixed）：`26/30 = 86.67%`，相较上一轮 `25/30` 提升 1 题。剩余失败聚焦于报告未稳定命中、case/输入 deck 与报告之间的证据层级区分、以及更完整的计划性建议，不通过硬编码修题。

## 实时事件与资源隔离

`run_multi_agent` 接受节点事件回调；`/research/stream` 使用线程安全队列在节点完成时推送 SSE，非任务结束后的 trace 回放。

本地模型资源使用有界信号量：Embedding=1、Reranker=1、LLM=2；`/observability` 返回请求数和平均排队时间。

5 用户默认 `hybrid_rerank + fixed` 并发 smoke：

| 指标 | 结果 |
| --- | ---: |
| 完成 / 提交 | 5 / 5 |
| Reviewer 通过 | 5 / 5 |
| 总 wall time | 9196.607 ms |
| P95 请求耗时 | 9195.029 ms |
| Embedding 平均等待 | 3951.588 ms |
| Reranker 平均等待 | 1028.781 ms |

该结果说明当前 CPU 部署以延迟换取内存安全和推理稳定性；下一步若追求多用户吞吐，应为 Embedding/Reranker 配置 GPU worker 或独立推理服务，而不应盲目提高同进程并发。
