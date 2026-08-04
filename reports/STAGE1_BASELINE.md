# 阶段一基线报告

执行日期：2026-08-02

## 数据导入

数据来自既有 MOOSE LHS 工作区，仅以只读方式解析，没有生成或运行新的 case。

| 项目 | 数量 |
|---|---:|
| LHS 历史仿真 case | 30 |
| 写入 Registry 的指标记录 | 570 |
| 可检索文本证据 | 13 |
| 可调参数 | 10 |

Registry 保存 case 参数、运行状态、返回码、耗时、指标与产物路径；文本索引覆盖 LHS 分析总结、批量运行日志和 MOOSE 输入文件。

## 定量校验

对 `early_1_2_rmse` 的 Pearson 相关性分析复现了已有分析：

| 参数 | Pearson r |
|---|---:|
| `cpv_front_scale` | -0.707181 |
| `ER_scale` | -0.581317 |
| `cpv_mid_scale` | -0.574218 |
| `tbegin2_shift` | -0.453916 |

该数据集中，`case_019` 的 `early_1_2_rmse` 最低，为 `42.946483`。这些数值由 Registry 分析工具生成，不由语言模型计算。

## 基础 Harness

| 类别 | 题数 | 通过 |
|---|---:|---:|
| Top-K case 查询 | 4 | 4 |
| 参数相关性分析 | 4 | 4 |
| case 详情查询 | 2 | 2 |
| 文档/日志混合检索 | 12 | 12 |
| 合计 | 22 | 22（100%） |

该结果验证的是阶段一 Registry、只读分析工具和 Hybrid Retrieval 的离线能力，不代表后续 LLM、多 Agent 或真实仿真执行能力。

## 复现命令

```bash
conda run -n scitime-agent python -m app.cli ingest
conda run -n scitime-agent python -m pytest -q
conda run -n scitime-agent python -m app.cli evaluate
```
