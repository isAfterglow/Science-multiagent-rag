# 受限结构化路由

## 目标

将 Supervisor 从纯关键词分流升级为“LLM 语义建议 + 确定性安全合并”。模型不直接决定 SQL、文件、MOOSE 参数或执行权限。

## 协议与决策

`RoutePlan` 是 Pydantic 协议，字段包括任务类型、是否需要 Registry 分析、所需证据类型、实验建议、检索词、指标和原因。路由模型优先使用 `qwen2.5:3b`，协议失败后降级到 `qwen2.5-coder:7b`。

程序始终执行以下策略：

1. 规则识别出的 Registry 分析或文档检索不能被模型移除。
2. 证据类型和指标必须通过白名单；不合法指标被忽略。
3. 最终 `task_type` 由安全合并后的分析/检索需求重新计算，不盲信模型标签。
4. `needs_experiment` 仅是建议；知识缺口诊断、人工确认、审批和真实执行仍由独立受控链路决定。
5. 模型关闭、超时、连接失败或 JSON/Pydantic 校验失败时回退规则路由，并在 `routing.fallback_reason` 中可观测。

## 实测

对混合问题“比较 cpv_front_scale 与 ER_scale 对 temp_mean_rmse 的历史线性关系，并结合报告说明推断边界”进行了真实模型调用。

- `qwen2.5:3b` 在 6.38 s 返回了不合规 `task_type=analysis`，被 Pydantic 拒绝。
- Router 自动降级到 `qwen2.5-coder:7b`，9.24 s 返回合法 `RoutePlan`。
- 该计划仍把任务错误标为 `knowledge` 且给出无效指标；程序重算为 `mixed`、保留 `temp_mean_rmse`，并记录 `invalid_metric_ignored`、`task_type_recomputed_from_safeguards`、`rule_required_registry_analysis`。

这证明模型输出是可替换、可失败的受限输入，而不是不可审计的控制面。
