# 秋招面试演示脚本

## 正常链路

1. 输入一个包含参数影响和历史报告依据的科研问题。
2. 展示 Supervisor 的结构化路由，以及 Retriever 与 Simulation Analyst 的并发 Trace。
3. 打开 EvidenceCard，展示来源、页码、chunk、行号和证据等级。
4. 展示 Registry 相关性结果、Critic 的相关性/因果限制和 Reviewer 决策。
5. 对知识缺口问题展示受限 SimulationPlan、人工审批、隔离 dry-run 和结果报告。

## 失败链路

输入未知参数或低置信度 OCR 数值请求，展示 Reviewer 拒绝、一次 Recovery、失败分类和“不能据此形成科学结论”的最终状态。

## 面试口径

离线确定性回归与 LLM-enabled benchmark 分开报告；LLM 只能路由、归纳和提出规划建议，事实、数值、审批与执行权限始终由程序和人工节点控制。
