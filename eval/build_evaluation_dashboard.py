"""Refresh the interview-facing evaluation dashboard from reproducible harnesses."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from app.config import DB_PATH, PLAN_DB_PATH, ROOT
from app.registry import SimulationRegistry
from app.simulation_plan import PlanStore
from eval.experiment_gap_eval import run as run_gap
from eval.production_scenario_smoke import run as run_production
from eval.run_comparison import run as run_comparison
from eval.safety_gate_eval import run as run_safety
from eval.scientific_workflow_eval import run as run_workflow


def _read_report(name: str) -> dict | list:
    path = ROOT / "reports" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _best_retrieval(data: dict) -> dict:
    rows = [*data.get("chunking_ablation", []), *data.get("strategy_ablation", [])]
    target = next((row for row in rows if row.get("mode") == "hybrid_rerank" and row.get("chunk_strategy") == "fixed"), None)
    return target or {}


def _table_row(name: str, passed: str, detail: str) -> str:
    return f"| {name} | {passed} | {detail} |"


def _latest_real_execution() -> dict:
    for record in PlanStore(PLAN_DB_PATH).list():
        result = record.get("result", {})
        if record.get("status") == "executed" and result and not result.get("dry_run", True):
            return {"plan_id": record["plan_id"], "successful_cases": result.get("successful_cases", 0), "total_cases": result.get("total_cases", 0), "elapsed_s": result.get("results", [{}])[0].get("elapsed_s"), "mpi_processes": result.get("mpi_processes")}
    return {}


def run(*, refresh_expensive: bool = False) -> dict:
    registry = SimulationRegistry(DB_PATH)
    comparison = run_comparison(registry, retrieval_mode="bm25")
    workflow = run_workflow("hybrid_rerank") if refresh_expensive else _read_report("scientific_workflow_eval.json")
    gap = run_gap()
    production = run_production() if refresh_expensive else _read_report("production_scenario_smoke.json")
    safety = run_safety()
    retrieval = _best_retrieval(_read_report("retrieval_ablation.json"))
    concurrency = _read_report("concurrency_smoke.json")
    real_execution = _latest_real_execution()
    dashboard = {
        "comparison": comparison,
        "scientific_workflow": workflow,
        "knowledge_gap": gap,
        "production_scenarios": production,
        "safety_gates": safety,
        "retrieval_default": retrieval.get("metrics", {}),
        "concurrency_smoke": {key: concurrency.get(key) for key in ("concurrent_users", "completed", "approved", "wall_time_ms", "p95_latency_ms")},
        "real_moose_execution": real_execution,
        "limitations": ["主评测关闭 LLM，保证可复现并将模型输出与流程正确性分离。", "真实 MOOSE 执行仍需审批与显式 real 模式；受限沙箱曾阻断 MPI socket，但 P0 已在宿主环境完成单进程真实运行。"],
    }
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    (reports / "evaluation_dashboard.json").write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")
    multi = comparison["multi_agent"]
    lines = [
        "# 统一评测看板",
        "",
        "主工作流评测使用 `hybrid_rerank + fixed`；基线对比使用相同的轻量 BM25 检索，避免 CPU 模型加载影响公平性和可复现性。LLM 默认关闭。",
        "",
        "| 维度 | 结果 | 说明 |",
        "| --- | --- | --- |",
        _table_row("纯 RAG（36 题）", f"{comparison['rag_only']['passed']}/{comparison['rag_only']['total']} ({comparison['rag_only']['pass_rate']:.2%})", f"引用率 {comparison['rag_only']['citation_rate']:.2%}"),
        _table_row("RAG + 数据工具（36 题）", f"{comparison['tool_agent']['passed']}/{comparison['tool_agent']['total']} ({comparison['tool_agent']['pass_rate']:.2%})", f"分析覆盖 {comparison['tool_agent']['analysis_coverage']:.2%}"),
        _table_row("完整多 Agent（36 题）", f"{multi['passed']}/{multi['total']} ({multi['pass_rate']:.2%})", f"引用率 {multi['citation_rate']:.2%}，Reviewer {multi['reviewed_rate']:.2%}，P95 {multi['p95_latency_ms']:.0f} ms"),
        _table_row("科研工作流", f"{workflow['passed']}/{workflow['questions']} ({workflow['pass_rate']:.2%})", f"P50 {workflow['p50_latency_ms']:.0f} ms，P95 {workflow['p95_latency_ms']:.0f} ms"),
        _table_row("生产式场景", f"{production['passed']}/{production['scenarios']} ({production['pass_rate']:.2%})", "含日志排查、复现、混合证据、缺口、风险拦截和探索建议"),
        _table_row("知识缺口", f"{gap['passed']}/{gap['questions']} ({gap['pass_rate']:.2%})", "覆盖 needs_experiment、unsupported、not_required"),
        _table_row("安全闸门", f"{safety['passed']}/{safety['total']} ({safety['pass_rate']:.2%})", "参数/模板白名单、审批拦截、MCP 工具边界"),
        _table_row("真实 MOOSE P0", f"{real_execution.get('successful_cases', 0)}/{real_execution.get('total_cases', 0)} 成功", f"{real_execution.get('mpi_processes', '-')} MPI 进程，{real_execution.get('elapsed_s', '-')} s，计划 {real_execution.get('plan_id', '-')}"),
        _table_row("默认检索", f"Recall@5 {retrieval.get('metrics', {}).get('recall_at_5', 0):.2%}", f"MRR {retrieval.get('metrics', {}).get('mrr', 0):.3f}，nDCG@5 {retrieval.get('metrics', {}).get('ndcg_at_5', 0):.3f}"),
        _table_row("5 用户并发冒烟", f"{concurrency.get('approved', 0)}/{concurrency.get('completed', 0)} 审核通过", f"端到端 P95 {concurrency.get('p95_latency_ms', 0):.0f} ms"),
        "",
        "## 结构化路由策略",
        "",
        "3B 优先输出 `RoutePlan`，其中包括任务类型、Registry 分析需求、证据类型、检索词、指标和实验建议。程序只接受符合 Pydantic 协议的输出，并以规则下限重算实际路由：模型不能移除必要的 Registry 分析或文档检索，不能选择白名单外指标，也不能授权真实执行。格式无效、超时或关闭模型时自动回退规则路由，并返回回退原因与策略覆盖记录。",
        "",
        "## 解释边界",
        "",
        "- 多 Agent 对比中的通过定义为题目要求的引用、分析和 Reviewer 状态，不等同于通用科研问答的真实世界准确率。",
        "- LLM 只参与受限路由和证据整理；最终数值与事实仍绑定 Registry 或原始文档。",
        "- 受限沙箱会阻断 MPI socket；P0 已在宿主环境完成 1 个真实单进程 case。真实结果不能替代实验真值或因果验证。",
    ]
    (reports / "EVALUATION_DASHBOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-expensive", action="store_true", help="重跑 Hybrid Rerank 工作流和生产场景；CPU 环境耗时和内存开销更高。")
    args = parser.parse_args()
    print(json.dumps(run(refresh_expensive=args.refresh_expensive), ensure_ascii=False, indent=2))
