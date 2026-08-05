# 生产式场景冒烟测试

测试日期：2026-08-05。测试问题独立于科学工作流回归题，模拟实际科研协作中的常见请求。检索采用当前默认的 `dense + parent_child`，LLM 关闭以隔离流程正确性；8/8 场景通过。

| 场景 | 核验内容 | 结果 |
| --- | --- | --- |
| 指标筛选 | Registry 排序、Reviewer、量化来源 | 通过，5.02 s |
| 报告口径 | 相关性与因果边界、报告证据 | 通过，5.61 s |
| 失败排查 | 运行日志和状态表联合证据 | 通过，1.76 s |
| 复现实验 | 输入 deck 和脚本联合证据 | 通过，1.64 s |
| 混合分析 | Registry 相关性 + 历史报告 | 通过，1.80 s |
| 知识缺口 | `needs_experiment` + 未持久化草案 | 通过，1.69 s |
| 风险拦截 | 未知参数 `unsupported`，不生成计划 | 通过，1.80 s |
| 探索建议 | 历史排序形成草案，不误报知识缺口 | 通过，1.65 s |

所有场景均经过 `supervisor`、`critic`、`reviewer` 节点。详细机器可读结果：`reports/production_scenario_smoke.json`。

## 启用 LLM 的验证

对“比较 cpv_front_scale 与 ER_scale 对 temp_mean_rmse 的历史线性关系，并结合报告说明推断边界”启用 `qwen2.5-coder:7b`。

- Planner JSON 协议调用成功：11.61 s。
- Evidence JSON 协议调用成功：6.69 s。
- 总耗时：26.42 s。
- 路由为 `mixed`，检索到 `report` 和 `input_deck`；Reviewer 通过。
- LLM 证据摘要仅作为候选产物留存，未被允许替代 Registry 数值或原始文档证据。

## 本轮修复

1. 收紧知识缺口触发条件：普通“候选仿真建议”不再因包含“仿真”一词而被错误标记为缺证据。
2. `unsupported` 的未知参数诊断现在会进入最终结果和 SSE 节点轨迹，工作台可见，并且不会生成计划。
