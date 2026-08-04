# LLM 多智能体协作：协议、闭环与评测

## 角色与权限

LangGraph 主图包含 Supervisor、Retriever、Simulation Analyst、Research Agent、Experiment Planner、Planner Agent、Critic、Semantic Critic、Synthesizer、Reviewer 与一次受限 Recovery。

- Supervisor LLM：输出 `RoutePlan`，只能建议任务类型、检索 query、指标和证据类型。
- Research LLM：输出 `ResearchSynthesis`，只能基于已召回证据归纳；模型选择 `EVIDENCE_INDEX`，程序映射为稳定 `chunk_id` 并拒绝越界引用。
- Planner LLM：输出 `PlannerProposal`，只能选择参数白名单中的关注项，并且必须声明人工审批；程序仍使用确定性代码生成 `SimulationPlan`、校验范围和模板 hash。
- Semantic Critic LLM：输出 `SemanticCritique`，只指出过度外推、冲突、缺证据或因果混淆；它没有批准权限。

LLM 的结构化协议由 Pydantic 校验，失败后记录 telemetry 并回退模型链。最终事实结论仍只来自 Registry 工具结果与原始 EvidenceCard；确定性 Critic、Reviewer 和人工审批仍是唯一的阻断权威。

```text
START -> Supervisor
      -> Retriever + Simulation Analyst (mixed 时并发)
      -> Research Agent
      -> Evidence Gap / Deterministic Plan Draft
      -> Planner Agent
      -> Deterministic Critic
      -> Synthesizer
      -> Semantic Critic
      -> Reviewer
         -> END
         -> Recovery (最多一次) -> Research Agent
```

## 小模型协议改进

早期让模型直接输出长 `chunk_id` 时，3B 常能输出 JSON 却容易抄错 ID。协议改为让模型输出 0-based `EVIDENCE_INDEX`，运行时严格检查范围后再映射回真实 `chunk_id`。这降低了小模型的格式负担，同时不牺牲证据可追溯性。

真实链路验证中，3B Router 出现 RoutePlan 字段为空或过长时，自动转由 7B-Coder 生成有效协议；7B-Coder 已成功完成 Router、Evidence、Research 和 Semantic Critic 的 JSON 输出。Planner 的真实 3B 输出通过参数白名单与人工审批标志校验。

## 对照评测

36 条现有 Harness 题，BM25、LLM 关闭的可复现基线：

| 方案 | 通过率 | 引用率 | 定量分析覆盖 | Reviewer 通过率 | P95 延迟 |
|---|---:|---:|---:|---:|---:|
| 纯 RAG | 33.33% | 100.00% | 0.00% | 0.00% | 4.062ms |
| RAG + 数据工具 Agent | 61.11% | 100.00% | 50.00% | 0.00% | 227.374ms |
| 多 Agent 协作 | 100.00% | 100.00% | 69.44% | 94.44% | 251.216ms |

30 条科研工作流回归题，LLM 关闭：通过 29/30（96.67%），路由准确率 96.67%，Agent 路径覆盖 100%，Reviewer 通过率 100%，P50/P95 为 237.871/1848.549ms。唯一失败 `wf-21` 是既有人工路由标签与当前规则不一致，不是证据、安全或协议失败。

真实 LLM smoke 采用 3B Router、7B-Coder fallback 和 BM25，对前 3 条工作流题：通过 2/3（66.67%），Agent 路径覆盖 100%，13 次模型调用的 JSON 协议成功率 84.62%，P50/P95 为 32415.328/32647.264ms。该数字说明本机 CPU/本地模型路径应作为可选增强而非默认在线路径；默认回归仍关闭 LLM，确保低延迟和可复现性。

## 运行

```bash
# 无 LLM 的稳定回归
conda run -n scitime-agent python -m eval.multi_agent_collaboration_eval --mode bm25

# 小规模真实 LLM smoke
MOOSE_COPILOT_LLM_ENABLED=true \
LLM_FAST_MODEL=qwen2.5:3b \
LLM_PRIMARY_MODEL=qwen2.5-coder:7b \
conda run -n scitime-agent python -m eval.multi_agent_collaboration_eval \
  --mode bm25 --llm-enabled --limit 3
```
