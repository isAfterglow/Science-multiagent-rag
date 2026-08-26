"""Frozen holdout evaluation; never used to tune the Router contract."""
from __future__ import annotations
import json, time
from pathlib import Path
from statistics import quantiles
from app.config import DB_PATH, ROOT
from app.llm_router import LLMRouter
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry

def run(mode="bm25"):
    questions=[json.loads(x) for x in (ROOT/"eval"/"llm_holdout_questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    registry=SimulationRegistry(DB_PATH); router=LLMRouter(enabled=True); rows=[]
    for item in questions:
        before=len(router.telemetry); start=time.perf_counter(); result=run_multi_agent(item["question"],registry,router,mode,"parent_child"); calls=[c.__dict__ for c in router.telemetry[before:]]
        protocol=all(any(c.get("success") for c in calls if c.get("role")==role) for role in {c.get("role") for c in calls})
        verifications=result.get("claim_verifications",[]); unsupported=sum(v.get("status") in {"insufficient","conflicted"} for v in verifications)
        checks={"route":result.get("task_type")==item["task_type"],"final_protocol":protocol,"grounded":bool(result.get("grounded_statements")),"citation_coverage":unsupported==0,"reviewer_evidence_policy":bool(result.get("review",{}).get("approved"))}
        rows.append({"id":item["id"],"expected_task_type":item["task_type"],"actual_task_type":result.get("task_type"),"checks":checks,"passed":all(checks.values()),"router_first_pass":bool(calls and calls[0].get("success")),"fallback_used":any(c.get("success") and c.get("model")!="qwen2.5:3b" for c in calls if c.get("role")=="router"),"unsupported_claims":unsupported,"model_calls":calls,"latency_ms":round((time.perf_counter()-start)*1000,3)})
    lat=[r["latency_ms"] for r in rows]; calls=[c for r in rows for c in r["model_calls"]]
    out={"schema_version":"llm_holdout_eval.v1","frozen":True,"questions":len(rows),"passed":sum(r["passed"] for r in rows),"pass_rate":round(sum(r["passed"] for r in rows)/len(rows),4),"route_accuracy":round(sum(r["checks"]["route"] for r in rows)/len(rows),4),"first_pass_protocol_rate":round(sum(r["router_first_pass"] for r in rows)/len(rows),4),"final_protocol_success_rate":round(sum(r["checks"]["final_protocol"] for r in rows)/len(rows),4),"grounded_rate":round(sum(r["checks"]["grounded"] for r in rows)/len(rows),4),"citation_coverage_rate":round(sum(r["checks"]["citation_coverage"] for r in rows)/len(rows),4),"reviewer_evidence_policy_rate":round(sum(r["checks"]["reviewer_evidence_policy"] for r in rows)/len(rows),4),"unsupported_claims":sum(r["unsupported_claims"] for r in rows),"fallback_count":sum(r["fallback_used"] for r in rows),"model_calls":len(calls),"total_tokens":sum(c.get("total_tokens",0) for c in calls),"cost_usd":round(sum(c.get("cost_usd",0) for c in calls),8),"p50_latency_ms":round(sorted(lat)[len(lat)//2],3),"p95_latency_ms":round(quantiles(lat,n=20)[18],3) if len(lat)>=20 else max(lat),"results":rows}
    return out

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--mode",default="bm25"); p.add_argument("--output",type=Path,default=ROOT/"reports"/"llm_holdout_eval.json"); a=p.parse_args(); out=run(a.mode); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({k:out[k] for k in ("questions","passed","pass_rate","route_accuracy","first_pass_protocol_rate","final_protocol_success_rate","grounded_rate","citation_coverage_rate","reviewer_evidence_policy_rate","model_calls","total_tokens","p95_latency_ms")},ensure_ascii=False,indent=2))
