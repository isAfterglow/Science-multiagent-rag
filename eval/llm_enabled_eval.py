"""Representative LLM-on evaluation with per-call model/token/cost telemetry."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from statistics import quantiles
from app.config import DB_PATH, ROOT
from app.llm_router import LLMRouter
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry

def _error_type(error: str) -> str:
    text = (error or "").lower()
    if "jsondecodeerror" in text: return "invalid_json"
    if "literal_error" in text: return "invalid_enum"
    if "missing" in text and "field required" in text: return "missing_required_field"
    if "string_too_short" in text or "empty" in text: return "empty_required_value"
    if "bool_type" in text or "string_type" in text or "int_type" in text: return "wrong_type"
    if "properties" in text or "schema" in text: return "schema_object_instead_of_instance"
    if "timeout" in text: return "timeout"
    return (error or "unknown").split(":", 1)[0]

def _cases(limit: int | None = None):
    rows = [json.loads(line) for line in (ROOT / "eval" / "scientific_workflow_questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # Keep a stable, representative subset: knowledge, analysis, mixed,
    # evidence gaps and plan requests are all present in the source set.
    preferred = ["wf-01", "wf-02", "wf-03", "wf-04", "wf-05", "wf-06", "wf-07", "wf-08", "wf-09", "wf-10", "wf-11", "wf-12", "wf-13", "wf-14", "wf-15", "wf-16", "wf-17", "wf-18", "wf-19", "wf-20"]
    selected = [next((x for x in rows if x["id"] == ident), None) for ident in preferred]
    selected = [x for x in selected if x is not None]
    return selected[:limit] if limit else selected

def run(limit: int = 20, mode: str = "bm25") -> dict:
    registry = SimulationRegistry(DB_PATH); router = LLMRouter(enabled=True); results = []
    for item in _cases(limit):
        before = len(router.telemetry); started = time.perf_counter()
        result = run_multi_agent(item["question"], registry, router, mode, "parent_child")
        calls = [c.__dict__ for c in router.telemetry[before:]]
        verifications = result.get("claim_verifications", [])
        unsupported = sum(v.get("status") in {"insufficient", "conflicted"} for v in verifications)
        expected = item.get("task_type")
        role_calls = {}
        for call in calls:
            role_calls.setdefault(call.get("role", ""), []).append(call)
        final_protocol = all(any(call.get("success") for call in role_rows) for role_rows in role_calls.values())
        router_calls = role_calls.get("router", [])
        first_router = bool(router_calls and router_calls[0].get("success"))
        retry_router = bool(len(router_calls) > 1 and router_calls[1].get("success"))
        fallback_router = any(call.get("success") and call.get("model") != "qwen2.5:3b" for call in router_calls)
        checks = {
            "route": result.get("task_type") == expected,
            "structured_output": final_protocol,
            "grounded": bool(result.get("grounded_statements")),
            "citation_coverage": unsupported == 0,
            "reviewer_pass": bool(result.get("review", {}).get("approved")),
        }
        errors = [{"role": c.get("role"), "model": c.get("model"), "type": _error_type(c.get("error", "")), "error": c.get("error", "")} for c in calls if not c.get("success")]
        results.append({"id": item["id"], "expected_task_type": expected, "actual_task_type": result.get("task_type"), "checks": checks, "passed": all(checks.values()), "recovery_used": bool(result.get("recovery_actions")), "unsupported_claims": unsupported, "first_pass_success": first_router and all(any(c.get("success") for c in rows[:1]) for rows in role_calls.values()), "router_first_pass": first_router, "router_retry_success": retry_router, "router_fallback_used": fallback_router, "router_final_success": any(c.get("success") for c in router_calls), "protocol_errors": errors, "model_calls": calls, "latency_ms": round((time.perf_counter()-started)*1000, 3), "trace_id": result.get("trace_id")})
    lat = [r["latency_ms"] for r in results]
    calls = [c for r in results for c in r["model_calls"]]
    successful = [c for c in calls if c.get("success")]
    error_counts = {}
    for row in results:
        for error in row["protocol_errors"]: error_counts[error["type"]] = error_counts.get(error["type"], 0) + 1
    out = {"schema_version":"llm_enabled_eval.v2", "llm_enabled":True, "model_chain":{"fast":"qwen2.5:3b","primary":"qwen2.5-coder:7b","fallback":"configured DeepSeek if present"}, "prompt_temperature":0, "questions":len(results), "passed":sum(r["passed"] for r in results), "pass_rate":round(sum(r["passed"] for r in results)/len(results),4) if results else 0, "route_accuracy":round(sum(r["checks"]["route"] for r in results)/len(results),4) if results else 0, "first_pass_protocol_rate":round(sum(r["router_first_pass"] for r in results)/len(results),4) if results else 0, "retry_recovery_rate":round(sum(r["router_retry_success"] for r in results if not r["router_first_pass"])/max(sum(not r["router_first_pass"] for r in results),1),4), "fallback_recovery_rate":round(sum(r["router_fallback_used"] for r in results if not r["router_first_pass"])/max(sum(not r["router_first_pass"] for r in results),1),4), "final_protocol_success_rate":round(sum(r["checks"]["structured_output"] for r in results)/len(results),4) if results else 0, "structured_output_success_rate":round(sum(r["checks"]["structured_output"] for r in results)/len(results),4) if results else 0, "grounded_pass_rate":round(sum(r["checks"]["grounded"] for r in results)/len(results),4) if results else 0, "citation_coverage_rate":round(sum(r["checks"]["citation_coverage"] for r in results)/len(results),4) if results else 0, "reviewer_pass_rate":round(sum(r["checks"]["reviewer_pass"] for r in results)/len(results),4) if results else 0, "recovery_rate":round(sum(r["recovery_used"] for r in results)/len(results),4) if results else 0, "unsupported_claim_count":sum(r["unsupported_claims"] for r in results), "protocol_error_counts":error_counts, "model_calls":len(calls), "successful_calls":len(successful), "input_tokens":sum(c.get("input_tokens",0) for c in calls), "output_tokens":sum(c.get("output_tokens",0) for c in calls), "total_tokens":sum(c.get("total_tokens",0) for c in calls), "cost_usd":round(sum(c.get("cost_usd",0) for c in calls),8), "p50_latency_ms":round(sorted(lat)[len(lat)//2],3) if lat else 0, "p95_latency_ms":round(quantiles(lat,n=20)[18],3) if len(lat)>=20 else (max(lat) if lat else 0), "results":results}
    return out

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--limit",type=int,default=20); p.add_argument("--mode",default="bm25"); p.add_argument("--output",type=Path,default=ROOT/"reports"/"llm_enabled_eval.json"); a=p.parse_args(); out=run(a.limit,a.mode); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({k:out[k] for k in ("questions","passed","pass_rate","structured_output_success_rate","reviewer_pass_rate","model_calls","total_tokens","cost_usd","p95_latency_ms")},ensure_ascii=False,indent=2))
