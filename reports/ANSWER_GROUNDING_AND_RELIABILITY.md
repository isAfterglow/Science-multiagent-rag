# 回答验证与可靠性

评测日期：2026-08-03。

## 回答级 Grounding

最终回答不再只有一个字符串。`GroundedStatement` 为每个用户可见句子保存：`evidence_kind`、来源路径、chunk ID、行号和支撑内容。

- `analysis`：只可引用 `simulation_registry` 的确定性分析工具输出。
- `document`：只可复用检索到的原始摘录，不把 LLM 摘要升级为事实。
- `limitation`：显式标注 Critic 的因果边界或证据覆盖风险。

Reviewer 增加命名参数/指标覆盖检查。它只在 Supervisor 要求文档检索时启用，因此不会错误拒绝纯 Registry 分析。

## 科研工作流评测

`eval/scientific_workflow_questions.jsonl` 包含 30 道非模板化问题，涵盖：历史排序、参数关联、case 复核、输入 deck、运行日志、状态表、脚本、报告解释、证据冲突和下一轮仿真建议。

本轮在 BM25 + fixed 的低延迟回归配置下：`25 / 30 = 83.33%`。

保留的 5 个失败并非删除或放宽：

- `wf-18`：输入 deck 不能单独支撑“最优 case 参数解释”，严格 Reviewer 拒绝。
- `wf-20`、`wf-21`、`wf-29`：用户需要报告级解释，但当前问题措辞/候选未稳定命中报告；需要下一阶段的查询改写与来源约束。
- `wf-27`：下一轮仿真建议仍只给历史排序，尚未形成带目标和约束的计划性回答。

这批失败为后续 query planner 和 SimulationPlan Agent 提供了明确目标，而不是用评测集反向硬编码规则。

## 并发与可观测性

新增 SQLite `research_tasks`：持久记录 queued/running/completed/failed/cancellation_requested/cancelled 状态、输入、结果、错误和耗时。线程池固定为 4 个 worker，避免并发请求无限加载本地模型。

8 用户只读研究 smoke（BM25 + fixed）结果：全部完成且 Reviewer 通过；总 wall time `144.431 ms`，P95 单请求 `142.848 ms`。这验证 Registry/Graph/SSE 任务通路不会因并发读取产生 SQLite 错误。它不是 GPU/LLM 吞吐压测；Reranker 的 CPU 成本仍以检索消融报告为准。
