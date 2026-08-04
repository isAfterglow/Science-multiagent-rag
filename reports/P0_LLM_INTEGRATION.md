# P0：受证据约束的 LLM 接入

执行日期：2026-08-02

## 接入位置与边界

| LLM 角色 | 默认模型 | 输入 | 可影响内容 | 不可影响内容 |
|---|---|---|---|---|
| Planner | `qwen2.5-coder:7b` | 用户问题 | 检索 query 改写 | 任务安全边界、数值工具选择、指标白名单 |
| Evidence Agent | `qwen2.5-coder:7b` | 已检索的摘录 | 证据摘要与局限性表达 | 来源路径、引用、数值、因果结论 |
| 低成本降级 | `qwen2.5:3b` | 同上 | 主模型不可用时重试 | 同上 |
| 语言降级 | `qwen2.5:7b` | 同上 | 后续复杂自然语言表达 | 同上 |
| 最终降级 | DeepSeek API（可选） | 同上 | 本地模型不可用或协议失败时重试 | 同上 |

所有输出均须通过 Pydantic JSON Schema。Planner 无法改变确定性任务分流；Evidence Agent 无法新增来源；参数相关性、Top-K、case 信息仍只能由 Registry 工具生成。模型调用失败时自动回退到阶段二的确定性链路。

LLM Evidence Summary 在当前版本被保存为可观测的候选产物，**不进入最终事实结论**。这是因为 JSON 合法不等于语义蕴含；在加入自动 Citation Entailment 检查前，最终答案只使用原始引用和程序工具结果。

## 模型基准结果

检测到的本地模型：

- `qwen2.5:3b`
- `qwen2.5-coder:7b`
- `qwen2.5:7b`

`3b-coder` 未安装。

真实命令端对三者执行 Planner/Evidence 的 3 个 JSON 协议任务，结果如下：

| 模型 | 有效任务 | 有效率 | 平均延迟 | 结论 |
|---|---:|---:|---:|---|
| `qwen2.5:3b` | 2 / 3 | 66.67% | 2841.697 ms | 第二个 Planner 将 `temp_mean_rmse` 错写为 `early_1_2_rmse` |
| `qwen2.5-coder:7b` | 3 / 3 | 100% | 4578.208 ms | JSON、指标字段和证据摘要均正确 |
| `qwen2.5:7b` | 3 / 3 | 100% | 5715.147 ms | 正确但慢于 Coder 7B |

因此选择 `qwen2.5-coder:7b` 作为 Planner/Evidence 主模型。选择规则是：先比较 JSON Schema 与任务字段有效率，再比较平均延迟。

## 端到端实测

启用 `MOOSE_COPILOT_LLM_ENABLED=true` 后，对“`cpv_front_scale` 对 `early_1_2_rmse` 有何影响，并说明历史报告依据”执行了一次完整多 Agent 链路：

- Planner：`qwen2.5-coder:7b`，8781.909 ms，成功；
- Evidence Agent：`qwen2.5-coder:7b`，7965.276 ms，成功；
- Registry 定量分析、Critic 和 Reviewer 均正常执行；
- 最终答案仍只使用 Registry 数值结果与原始证据路径。

本次实测也验证了为什么候选 LLM 摘要不能直接成为事实：它对截断摘录作出了缺乏完整上下文的判断。系统因此将 LLM 摘要从最终答案中排除并保留为审计产物，直到后续加入 Citation Entailment 检查。

选型脚本仍可复现运行：

```bash
MOOSE_COPILOT_LLM_ENABLED=true \
conda run -n scitime-agent python -m app.cli benchmark-models
```

3B 保留为本地低成本降级；通用 7B 保留给后续更长的叙述性任务。DeepSeek API 仍是可选最终降级。

## 回归验证

- `7 passed`
- 确定性多 Agent 对比 Harness 仍为 `36/36`
- 使用 Fake Router 的测试验证：LLM 结构化摘要会被保留并标记，但无论模型输出如何，文档来源和数据结论仍由 Retriever 与 Registry 控制。
