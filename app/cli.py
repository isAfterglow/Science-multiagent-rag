from __future__ import annotations

import argparse
import json

from app.analysis import parameter_correlation, top_cases
from app.config import DB_PATH, SOURCE_DIR
from app.ingest import ingest
from app.qa import answer
from app.registry import SimulationRegistry
from app.simulation_plan import PlanStore, suggest_plan
from app.config import PLAN_DB_PATH
from app.execution import execute_approved_plan

def main() -> None:
    parser = argparse.ArgumentParser(description="MOOSE Research Copilot stage-one CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("ingest-scientific")
    ask = sub.add_parser("ask"); ask.add_argument("question")
    collaborate = sub.add_parser("collaborate"); collaborate.add_argument("question")
    analyze = sub.add_parser("analyze"); analyze.add_argument("kind", choices=["correlation", "top"]); analyze.add_argument("--metric", default="early_1_2_rmse"); analyze.add_argument("--limit", type=int, default=5)
    sub.add_parser("evaluate")
    sub.add_parser("compare")
    sub.add_parser("benchmark-models")
    sub.add_parser("vector-status")
    sub.add_parser("vector-sync")
    plan = sub.add_parser("plan"); plan_sub = plan.add_subparsers(dest="plan_command", required=True)
    create = plan_sub.add_parser("create"); create.add_argument("--metric", default="early_1_2_rmse"); create.add_argument("--n-cases", type=int, default=3)
    plan_sub.add_parser("list")
    approve = plan_sub.add_parser("approve"); approve.add_argument("plan_id"); approve.add_argument("--actor", required=True); approve.add_argument("--comment", default="")
    execute = plan_sub.add_parser("execute"); execute.add_argument("plan_id"); execute.add_argument("--real", action="store_true", help="Run approved cases with MOOSE; default is safe preview.")
    args = parser.parse_args(); registry = SimulationRegistry(DB_PATH); plan_store = PlanStore(PLAN_DB_PATH)
    if args.command == "ingest": result = ingest(SOURCE_DIR, registry)
    elif args.command == "ingest-scientific":
        from app.scientific_ingest import ingest_scientific_sources
        result = ingest_scientific_sources()
    else:
        if not registry.cases(): ingest(SOURCE_DIR, registry)
        if args.command == "ask": result = answer(args.question, registry)
        elif args.command == "collaborate":
            from app.multi_agent import run_multi_agent
            result = run_multi_agent(args.question, registry)
        elif args.command == "analyze": result = parameter_correlation(registry, args.metric) if args.kind == "correlation" else top_cases(registry, args.metric, args.limit)
        elif args.command == "evaluate":
            from eval.run_eval import run
            result = run(registry)
        elif args.command == "compare":
            from eval.run_comparison import run
            result = run(registry)
        elif args.command == "benchmark-models":
            from eval.model_benchmark import run
            result = run()
        elif args.command in {"vector-status", "vector-sync"}:
            from app.retrieval import HybridRetriever
            # Constructing a dense retriever synchronizes the configured
            # backend; status exposes cache hit/upsert/delete counts.
            result = HybridRetriever(registry, mode="dense").vector_status()
        else:
            if args.plan_command == "create": result = plan_store.create(suggest_plan(registry, args.metric, args.n_cases))
            elif args.plan_command == "list": result = plan_store.list()
            elif args.plan_command == "approve": result = plan_store.decide(args.plan_id, "approved", args.actor, args.comment)
            else: result = execute_approved_plan(args.plan_id, plan_store, registry, dry_run=not args.real)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__": main()
