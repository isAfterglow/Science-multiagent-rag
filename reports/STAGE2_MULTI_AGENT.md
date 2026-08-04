# 阶段二：证据驱动的多智能体协作

执行日期：2026-08-02

## 协作图

```text
Supervisor
  -> Evidence Fan-out
       -> Retriever：Hybrid RAG，生成 EvidenceCard
       -> Simulation Analyst：调用 Registry 定量工具
  -> Critic：标记因果边界、证据不足与参数混杂风险
  -> Synthesizer：仅汇总证据卡和程序分析结果
  -> Reviewer：检查任务要求的证据是否齐全
```

当问题同时要求数值结论和文献/报告依据时，Retriever 与 Simulation Analyst 在独立线程并发运行；结果汇合后才进入 Critic。所有 Agent 间产物使用 Pydantic JSON 校验，避免将未经验证的自然语言直接作为下游事实。

## 角色与边界

| 角色 | 允许做什么 | 不允许做什么 |
|---|---|---|
| Supervisor | 问题分流、选择证据分支 | 生成未经证据支撑的科研结论 |
| Retriever | 混合检索文档、报告、输入文件与日志 | 计算数值指标 |
| Simulation Analyst | 调用 Top-K、相关性等只读工具 | 将相关性表述为因果关系 |
| Critic | 提示因果、样本量、证据缺失风险 | 替代数据工具计算结论 |
| Synthesizer | 汇总带来源证据 | 引入证据卡之外的新事实 |
| Reviewer | 校验证据完整性并决定是否回退 | 绕过安全或数据边界 |

## Harness 对比

评测集共 36 题：原有结构化/检索题 22 题，新增要求同时具备数值与文档证据的混合题 14 题。

| 方案 | 通过 | 通过率 | 引用覆盖率 | 定量分析覆盖率 | Reviewer 覆盖率 | P95 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 纯 Hybrid RAG | 12 / 36 | 33.33% | 100% | 0% | 0% | 9.563 ms |
| RAG + 数据工具单 Agent | 22 / 36 | 61.11% | 100% | 50.00% | 0% | 9.167 ms |
| 完整多 Agent | 36 / 36 | 100% | 100% | 66.67% | 100% | 22.637 ms |

通过条件按题型定义：检索题需要可引用来源；结构化题需要程序分析与来源；混合题同时需要文档证据、定量证据和 Reviewer 放行。因此该结果衡量的是证据闭环完整性，不是对自由生成语言质量的主观评分。

## 可复现命令

```bash
conda run -n scitime-agent python -m app.cli ingest
conda run -n scitime-agent python -m pytest -q
conda run -n scitime-agent python -m app.cli collaborate "cpv_front_scale 对 early_1_2_rmse 有何影响，并说明历史报告依据"
conda run -n scitime-agent python -m app.cli compare
```

阶段二仍是只读历史分析：不会启动 MOOSE、修改输入文件或生成新 case。真实仿真计划、审批和运行将留给阶段三。
