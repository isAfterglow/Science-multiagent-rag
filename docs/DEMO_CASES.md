# 演示用例

这些用例用于展示证据约束、多 Agent 协作与人工审批边界。它们不是自动执行 MOOSE 的指令。

## 1. 科研 RAG 与扫描页

问题：`FIATC 论文中的方程表在哪一页？扫描件证据的可信边界是什么？`

预期观察：Supervisor 要求论文或扫描报告；Retriever 返回页面、表格或 OCR 元数据；Reviewer 对低置信 OCR 精确数值提示人工核对。最终答案保留原始摘录和页码，而不让 LLM 扩写为未证实结论。

## 2. 文档证据与历史仿真联合分析

问题：`cpv_front_scale 对 early_1_2_rmse 有何影响，并说明历史报告依据。`

预期观察：Retriever 和 Simulation Analyst 并发工作。前者提供报告摘录，后者从 Registry 计算相关性；Critic 明确“相关性不是因果性”，Reviewer 在证据类型完整时通过。

## 3. 知识缺口到审批边界

问题：`针对未覆盖的新工况，为 early_1_2_rmse 生成下一轮候选仿真建议。`

预期观察：系统只生成内存中的 SimulationPlan 草案。用户必须显式确认写入 pending，人工审批后也默认只做隔离 dry-run；真实 MOOSE 执行需要额外显式授权。
