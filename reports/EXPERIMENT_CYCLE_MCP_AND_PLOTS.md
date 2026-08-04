# 知识缺口闭环、科研绘图与 MCP 验证

## 阶段一：从知识缺口到受控仿真

流程为：问题诊断 -> 历史证据覆盖判断 -> 受限实验草案 -> 显式确认 -> `pending` 人工审批 -> 已批准计划的隔离 dry-run -> 受限结果报告。

- `eval/experiment_gap_questions.jsonl` 覆盖新工况、参数缺失、未知参数和普通历史问答，共 8 题。
- `python -m eval.experiment_gap_eval` 结果为 **8/8（100%）**。
- 自动草案只使用登记的模板、参数白名单和边界；未知参数返回 `unsupported`，不自动扩大执行范围。
- 草案不落库、不执行。确认后也只进入 `pending`；真实 MOOSE 执行需要后续人工审批和显式 real 模式。

## 阶段二：可追溯绘图与 MCP

实际生成并检查了以下 PNG 产物：参数-指标散点图（30 条样本）、参数相关性排序图（30 条样本）、指标排名图（5 条样本）、探索计划参数差异图（3 个候选）。每个工具返回 artifact ID、数据来源、过滤条件、样本数和局限性说明。

标准 stdio MCP 服务共暴露 9 个 typed tools：

1. `search_evidence`
2. `analyze_parameter_correlation`
3. `plot_parameter_scatter`
4. `plot_parameter_correlation_bar`
5. `plot_metric_ranking_bar`
6. `create_experiment_draft`
7. `confirm_experiment_draft`
8. `execute_plan_preview`
9. `get_experiment_report`

安全边界：没有 `execute_sql`、任意代码工具或真实 MOOSE 执行工具；`execute_plan_preview` 强制 `dry_run=True`。

## 阶段三：工作台接入

- FastAPI 提供实验设计、计划报告和绘图端点，并将 `data/artifacts` 只读挂载到 `/artifacts`。
- SSE 研究结果会呈现 `evidence_gap`，工作台显示草案确认、人工审批、后台任务和资源队列。
- 新增相关性排序和指标排名按钮；页面只展示后端返回的受控图表文件，不接受用户提供路径或代码。

## 验证与限制

- `tests/test_experiment_cycle_and_mcp.py`：2 passed。
- `eval.experiment_gap_eval`：8/8 passed。
- 当前环境中的真实 MOOSE/MPI 会受到 socket 权限阻断，因此本阶段以审批链路、隔离 dry-run 和受限报告作为可复现实证；未把环境阻断伪装为仿真成功。
