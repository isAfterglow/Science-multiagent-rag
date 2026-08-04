# P0：真实 MOOSE 单 Case 执行证据

执行日期：2026-08-03。

## 审批与隔离

- 计划：`plan-3e07141432dc`，目标为 `early_1_2_rmse`。
- 审批人：`user-requested-p0`；审批说明保存在 `simulation_plans` SQLite 表。
- 运行目录：`data/runs/plan-3e07141432dc/candidate-1/`。
- 历史 LHS 工作区未被写入；原始 `run.sh` 仅被复制到隔离目录。
- 输入 SHA-256：`57b15e4bac9f4bf138908ee58908e81b43a460bb129969626bab1acb5efeccc0`。

## 宿主环境与真实执行

- MPICH：3.3.2；MOOSE 可执行文件：`/home/ai4mater/moose/roshan-ablation/roshan-opt`。
- 先以受限进程数 `1` 执行，隔离脚本实际命令为 `mpirun -n 1 ... roshan-opt -i case1_fiat_walltemp_nominal.i`。
- 返回码：`0`；状态：`ok`；耗时：`84.374 s`。
- 计划估计为 `36.9 s`，实际为估计的约 2.29 倍；后续应按 MPI 进程数和实际 case 输出更新耗时预测模型。
- MOOSE 日志确认 `Num Processors: 1`、求解过程收敛，并以 `Job ended` 正常结束。

## 回写结果

pointvalues CSV 共 302 行，末态时间为 `60.0 s`。以下为原始仿真输出特征，已写入 Simulation Registry：

| 指标 | 数值 |
| --- | ---: |
| `final_Tsh` | 1644.000 K |
| `final_T1mmh` | 1526.886 K |
| `final_T2mmh` | 1414.978 K |
| `final_T4mmh` | 1205.807 K |
| `final_T8mmh` | 834.717 K |
| `final_T16mmh` | 411.179 K |
| `final_T24mmh` | 311.799 K |
| `final_Psh` | 99999.990 |

已登记的产物：输入 deck、manifest、运行日志、pointvalues/mass/position CSV，以及温度时序图 `data/artifacts/plot-c1f82a20d904.png`。

## 科学边界

本次证明的是“受控 Agent -> 审批 -> 隔离真实 MOOSE -> 原始输出回写 Registry”的真实闭环，而不是模型参数优于历史 case 的科学结论。当前没有对应实验真值，因此 `final_T*` 不能被标记成 `early_1_2_rmse` 或因果改善。后续如接入实验曲线，可在相同时间点计算 RMSE 并作为新的、可追溯指标登记。
