# 阶段三：受控 MOOSE 执行闭环

执行日期：2026-08-02

## 执行流程

```text
历史 case / Registry
  -> 生成受限 SimulationPlan
  -> 参数、模板哈希、case 数与预算校验
  -> 持久化待审批计划
  -> 人工批准或拒绝
  -> 独立 data/runs/<plan_id> 工作区
  -> dry-run 预览（默认）或显式真实 MOOSE 执行
  -> 日志、输入哈希、状态和结果复盘写回 Registry
```

## 安全约束

- 参数白名单固定为现有 10 个 LHS 参数，且逐项检查边界。
- 模板以 SHA-256 固定；模板变化后计划失效。
- 每个计划限制为 1-5 个 case，单 case 超时限制为 30-600 秒。
- 只能从只读源模板构建输入文件，不能让模型写 Shell 命令。
- 未批准计划调用执行器会被拒绝。
- 所有 case 在本项目 `data/runs/<plan_id>/` 内执行，不修改历史 MOOSE 工作区。
- 默认 dry-run；真实运行需已审批状态和显式 `--real`。

## 验证结果

| 验证项 | 结果 |
|---|---:|
| Python 回归测试 | 6 passed |
| 计划与执行安全 Harness | 8 / 8（100%） |
| 审批后的隔离 dry-run | 成功，生成输入、参数、清单与复盘文件 |
| 真实单 case MOOSE P0 | 宿主环境单进程执行成功，返回码 0，84.374 s，原始输出已回写 Registry |

早期 smoke 在受限进程中触发过 MPICH 本地 socket 拒绝：`HYDU_sock_listen ... Operation not permitted`，系统正确分类为 `environment_blocked`。P0 已在允许 MPI socket 的宿主环境对新审批计划完成单进程真实运行，输入 deck、审批记录、输出 CSV、日志、输入哈希、复盘 JSON 和结果指标均已保留。详见 `reports/REAL_MOOSE_EXECUTION_P0.md`。

## 接口

- CLI：`app.cli plan create/list/approve/execute`
- FastAPI：`POST /research`、`POST /plans`、`POST /plans/{id}/approve`、`POST /plans/{id}/execute`
- API 默认 `dry_run=true`。
- 单页工作台：访问 `/` 可查看 Agent Trace、审核结果、计划、审批和隔离预览入口。

阶段三完成后，系统已具备从证据分析到受控仿真执行的完整闭环。MCP 封装仍是可选后续工作，不阻塞当前核心项目。
