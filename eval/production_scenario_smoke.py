"""Independent production-style smoke test for the complete research workflow."""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.config import DB_PATH, ROOT
from app.llm_router import LLMRouter
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry


SCENARIOS = [
    {"id": "prod-01", "type": "指标筛选", "question": "请为本周评审列出 early_1_2_rmse 最低的历史候选，并说明数据来源。", "task_type": "simulation_analysis", "review": True, "keyword": "case_"},
    {"id": "prod-02", "type": "报告口径", "question": "向项目负责人汇报时，历史报告如何区分参数相关性与因果推断？", "task_type": "mixed", "review": True, "source": "report"},
    {"id": "prod-03", "type": "失败排查", "question": "某批计算出现非零 return_code，应如何联合运行日志和状态记录定位问题？", "task_type": "knowledge", "review": True, "sources": ["run_log", "run_status"]},
    {"id": "prod-04", "type": "复现实验", "question": "准备复现历史 case 前，输入 deck 和批处理脚本分别需要核验哪些内容？", "task_type": "knowledge", "review": True, "sources": ["input_deck", "script"]},
    {"id": "prod-05", "type": "混合证据分析", "question": "比较 cpv_front_scale 与 ER_scale 对 temp_mean_rmse 的历史线性关系，并结合报告说明推断边界。", "task_type": "mixed", "review": True, "source": "report", "keyword": "Pearson"},
    {"id": "prod-06", "type": "知识缺口", "question": "新热边界工况下 cpv_front_scale 没有历史记录，需要设计受控仿真验证 early_1_2_rmse。", "task_type": "knowledge", "review": True, "gap": "needs_experiment", "plan": True},
    {"id": "prod-07", "type": "风险拦截", "question": "foo_unknown_scale 对 early_1_2_rmse 没有历史记录，请自动设计并执行实验。", "task_type": "knowledge", "gap": "unsupported", "plan": False},
    {"id": "prod-08", "type": "探索建议", "question": "依据 early_1_2_rmse 的历史排序，生成下一轮候选仿真建议。", "task_type": "mixed", "review": True, "plan": True},
]


def run() -> dict:
    registry = SimulationRegistry(DB_PATH)
    results = []
    for scenario in SCENARIOS:
        started = time.perf_counter()
        result = run_multi_agent(scenario["question"], registry, LLMRouter(enabled=False), "dense", "parent_child")
        sources = {card["source_type"] for card in result.get("evidence_cards", [])}
        trace_nodes = [item["node"] for item in result.get("trace", [])]
        checks = {
            "route": result["task_type"] == scenario["task_type"],
            "trace": {"supervisor", "critic", "reviewer"}.issubset(trace_nodes),
            "review": result["review"].get("approved") == scenario.get("review", True),
            "source": not scenario.get("source") or scenario["source"] in sources,
            "sources": set(scenario.get("sources", [])).issubset(sources),
            "keyword": not scenario.get("keyword") or scenario["keyword"].lower() in result["answer"].lower(),
            "gap": not scenario.get("gap") or result.get("evidence_gap", {}).get("status") == scenario["gap"],
            "plan": "plan" not in scenario or bool(result.get("plan_draft")) == scenario["plan"],
        }
        results.append({"id": scenario["id"], "type": scenario["type"], "passed": all(checks.values()), "checks": checks, "task_type": result["task_type"], "review": result["review"], "sources": sorted(sources), "gap": result.get("evidence_gap", {}).get("status"), "plan_id": result.get("plan_draft", {}).get("plan_id"), "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
    passed = sum(item["passed"] for item in results)
    output = {"scenarios": len(results), "passed": passed, "pass_rate": round(passed / len(results), 4), "retrieval_mode": "dense", "chunk_strategy": "parent_child", "llm_enabled": False, "results": results}
    (ROOT / "reports" / "production_scenario_smoke.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
