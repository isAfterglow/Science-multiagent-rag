"""Small concurrent-user smoke test for the read-only research workflow."""
from __future__ import annotations

import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import quantiles

from app.config import DB_PATH, ROOT
from app.llm_router import LLMRouter
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry
from app.resource_limits import metrics as resource_metrics

QUESTIONS = [
    "LHS 分析中基线 early_1_2_rmse 是多少？", "cpv_front_scale 对 early_1_2_rmse 有何影响，并说明历史报告依据。",
    "批量运行日志中如何记录 case 的开始时间？", "MOOSE 输入 deck 包含哪些边界条件设置？",
    "early_1_2_rmse 最好的 case 是哪个？", "运行失败时应该查看什么日志？",
    "ER_scale 与 early_1_2_rmse 的关系及局限性是什么？", "LHS case 的运行状态如何保存？",
]


def _one(question: str, retrieval_mode: str) -> dict:
    started = time.perf_counter()
    result = run_multi_agent(question, SimulationRegistry(DB_PATH), LLMRouter(enabled=False), retrieval_mode, "fixed")
    return {"question": question, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "approved": bool(result["review"].get("approved")), "task_type": result["task_type"]}


def run(users: int = 8, retrieval_mode: str = "bm25") -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=users, thread_name_prefix="load") as pool:
        rows = [future.result() for future in as_completed([pool.submit(_one, QUESTIONS[index % len(QUESTIONS)], retrieval_mode) for index in range(users)])]
    latencies = [row["latency_ms"] for row in rows]
    result = {"concurrent_users": users, "retrieval_mode": retrieval_mode, "completed": len(rows), "approved": sum(row["approved"] for row in rows), "wall_time_ms": round((time.perf_counter() - started) * 1000, 3), "p95_latency_ms": round(quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies), 3), "model_resources": resource_metrics(), "results": rows}
    (ROOT / "reports" / "concurrency_smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument("--mode", default="bm25", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    args = parser.parse_args()
    print(json.dumps(run(args.users, args.mode), ensure_ascii=False, indent=2))
