# Release v2 评测口径

评测环境：本机 `scitime-agent`，BM25/缓存离线回归；LLM-on 使用 Ollama 本地模型，temperature=0。两类结果严格分开，不把确定性流程通过率称为 LLM 准确率。

## 核心结果

| 维度 | 结果 |
|---|---:|
| 确定性多 Agent 工作流 | 29/30（96.67%，历史回归） |
| 同工具 Single Agent | 22/36（61.11%） |
| 完整 Multi-Agent（同一工具集） | 36/36（100%） |
| LLM-enabled 代表性评测（优化后） | 15/15（100%，真实 CPU smoke） |
| LLM Router 首次协议通过率 | 15/15（100%） |
| LLM 最终协议可用率 | 15/15（100%） |
| LLM Grounded statement | 100% |
| LLM Citation coverage | 100% |
| LLM Reviewer pass | 100% |
| Claim-Evidence 分类评测 | 31/31（100%） |
| 科研安全 | 9/9（100%） |
| 执行异常分类 | 3/3（100%） |

LLM-on 优化前基线为 6/15、3B 首次协议通过率 6/15、76 次调用和 43,177 tokens；失败集中在 Router 的非法枚举、空必填字段和输出 Schema 而非实例。优化后同一 15 题为 15/15，3B 首次协议通过 15/15，57 次调用、34,775 tokens，P50/P95 约 38.1/57.3 秒。本地无计费，记录成本为 0 USD。Grounding、Citation 和 Reviewer evidence-policy 均保持 15/15，Unsupported Claim 为 0。

本轮最小 Router Schema 和 Validation-guided Repair 的 A/B 结果证明：字段收缩消除了本批次协议错误；本次没有触发 Retry 或 fallback，因此不能据此声称 Repair 本身贡献了恢复率，后续仍保留其失败闭环能力。

## 独立 Holdout

`eval/llm_holdout_questions.jsonl` 是未参与本轮协议调整的 12 道新问题，结果保存于 `reports/llm_holdout_eval.json`。Holdout 上最终协议可用率为 12/12，Grounded、Citation 和 Reviewer evidence-policy 均为 12/12，Unsupported Claim 为 0；但路由准确率为 5/12（41.67%）。这说明 RouteDecision 的结构化稳定性已经泛化，而新问题中的 knowledge/mixed 语义边界仍需要更明确的任务标签或规则校准。本批结果冻结，不据此继续改代码，避免把 Holdout 变成开发集。

## Repair 契约测试

`tests/test_validation_guided_repair.py` 覆盖非法枚举和“输出 JSON Schema 而非实例”两类故障，验证 `ValidationError → 定向修复提示 → 同模型重试` 路径。真实 15 题回归未触发 Repair，因此没有声称真实 Repair 恢复率提升。

## 消融解释

Pure RAG 为 12/36（33.33%），RAG+工具为 22/36（61.11%），同工具 Single Agent 仍为 22/36，而完整 Multi-Agent 为 36/36。该结果支持“职责拆分、证据审查和最终 Reviewer 带来增益”的判断。`multi_agent_no_critic`/`no_reviewer` 当前脚本复用了完整图，只作为流程近似对照，不作为严格因果消融结论。

## 真实执行与失败边界

已有 1 个真实单进程 MOOSE 成功案例（84.374 s）。新增安全回归验证：非法参数在执行前拒绝、环境错误分类为 `environment_blocked`、运行时错误分类为 `simulation_or_runtime_failure`。dry-run、环境失败和仿真失败不会被写成科学结论。

## RAG 失败分析

`reports/retrieval_failure_slices.json` 保存当前检索报告的按 query/document shape 切片和待排查族：OCR 识别、表格结构、双栏顺序、Parent-Child 定位和术语不一致。当前公开语料检索评测仍以 Dense 为默认，避免用低收益的 Recall 调参替代证据链建设。

## 面试演示

演示脚本见 `docs/INTERVIEW_DEMO.md`，包含正常 Trace、EvidenceCard、Registry 分析、审批 dry-run 和未知参数/低置信 OCR 拒答链路。
