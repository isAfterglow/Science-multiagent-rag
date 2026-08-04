"""Compact stage-three workbench API; real execution remains approval-gated."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
import asyncio
import json
from queue import Empty, Queue

from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ARTIFACTS_DIR, DB_PATH, KNOWLEDGE_RENDERED_DIR, PLAN_DB_PATH, ROOT, SOURCE_DIR, TASK_DB_PATH
from app.execution import execute_approved_plan
from app.ingest import ingest
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry
from app.simulation_plan import PlanStore, suggest_plan
from app.models import SimulationPlan
from app.task_manager import ResearchTaskStore
from app.resource_limits import metrics as resource_metrics
from app.experiment_cycle import design_experiment, validate_and_report_result
from app.plotting import plot_metric_ranking_bar, plot_parameter_correlation_bar, plot_parameter_scatter
from app.vector_store import configured_vector_status

app = FastAPI(title="MOOSE Research Copilot", version="0.3.0")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_RENDERED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")
app.mount("/knowledge-pages", StaticFiles(directory=KNOWLEDGE_RENDERED_DIR), name="knowledge-pages")
registry = SimulationRegistry(DB_PATH); plans = PlanStore(PLAN_DB_PATH)
tasks = ResearchTaskStore(TASK_DB_PATH)

class PlanRequest(BaseModel): target_metric: str = "early_1_2_rmse"; n_cases: int = Field(default=3, ge=1, le=5)
class DecisionRequest(BaseModel): actor: str = Field(min_length=2, max_length=80); comment: str = Field(default="", max_length=500)
class QuestionRequest(BaseModel): question: str = Field(min_length=3, max_length=2000)
class DraftConfirmRequest(BaseModel): plan: SimulationPlan
class ScatterPlotRequest(BaseModel): parameter: str; metric: str = "early_1_2_rmse"

def _ensure_data() -> None:
    if not registry.cases(): ingest(SOURCE_DIR, registry)

@app.get("/", include_in_schema=False)
def workbench() -> FileResponse:
    return FileResponse(__import__("pathlib").Path(__file__).resolve().parents[1] / "web" / "index.html")

@app.get("/health")
def health() -> dict: return {"status": "ok", "execution_policy": "approval_required"}

@app.post("/research")
def research(request: QuestionRequest) -> dict:
    _ensure_data(); return run_multi_agent(request.question, registry)


@app.post("/research/tasks", status_code=202)
def submit_research_task(request: QuestionRequest) -> dict:
    _ensure_data()
    return tasks.submit(request.question, lambda: run_multi_agent(request.question, registry))


@app.get("/research/tasks")
def list_research_tasks() -> list[dict]:
    return tasks.list()


@app.get("/research/tasks/{task_id}")
def get_research_task(task_id: str) -> dict:
    result = tasks.get(task_id)
    if not result: raise HTTPException(404, "Research task not found")
    return result


@app.get("/research/tasks/{task_id}/trace")
def get_research_trace(task_id: str) -> dict:
    task = get_research_task(task_id)
    result = dict(task.get("result") or {})
    return {"trace_id": result.get("trace_id", ""), "summary": result.get("trace_summary", {}), "events": result.get("trace", [])}


@app.post("/research/tasks/{task_id}/cancel")
def cancel_research_task(task_id: str) -> dict:
    try: return tasks.cancel(task_id)
    except KeyError as exc: raise HTTPException(404, "Research task not found") from exc


@app.get("/observability")
def observability() -> dict:
    return {"research_tasks": tasks.metrics(), "model_resources": resource_metrics(), "execution_policy": "approval_required", "retrieval_default": {"mode": __import__("app.config", fromlist=["RETRIEVAL_MODE"]).RETRIEVAL_MODE, "chunk_strategy": __import__("app.config", fromlist=["CHUNK_STRATEGY"]).CHUNK_STRATEGY}, "vector_index": configured_vector_status()}


@app.get("/knowledge/summary")
def knowledge_summary() -> dict:
    path = ROOT / "reports" / "document_ir_eval.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_ingested"}


@app.post("/research/stream")
async def research_stream(request: QuestionRequest) -> StreamingResponse:
    """Push LangGraph node events as they occur, then the validated result."""
    async def events():
        yield "event: status\ndata: " + json.dumps({"stage": "accepted"}, ensure_ascii=False) + "\n\n"
        _ensure_data()
        queue: Queue[dict] = Queue()
        work = asyncio.create_task(asyncio.to_thread(run_multi_agent, request.question, registry, None, None, None, queue.put))
        while not work.done():
            try:
                event = await asyncio.to_thread(queue.get, True, 0.2)
                yield "event: node\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            except Empty:
                continue
        result = await work
        while not queue.empty():
            yield "event: node\ndata: " + json.dumps(queue.get_nowait(), ensure_ascii=False) + "\n\n"
        yield "event: result\ndata: " + json.dumps(result, ensure_ascii=False) + "\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

@app.post("/plans", status_code=201)
def create_plan(request: PlanRequest) -> dict:
    _ensure_data(); return plans.create(suggest_plan(registry, request.target_metric, request.n_cases))

@app.post("/plans/drafts/confirm", status_code=201)
def confirm_plan_draft(request: DraftConfirmRequest) -> dict:
    """Only explicit user confirmation turns an in-memory draft into pending approval."""
    _ensure_data()
    return plans.create(request.plan)

@app.post("/experiments/design")
def design_experiment_endpoint(request: QuestionRequest) -> dict:
    _ensure_data()
    return design_experiment(request.question, registry)

@app.get("/experiments/{plan_id}/report")
def experiment_report(plan_id: str) -> dict:
    try: return validate_and_report_result(plan_id, plans, registry)
    except KeyError as exc: raise HTTPException(404, "Plan not found") from exc

@app.post("/plots/scatter")
def scatter_plot(request: ScatterPlotRequest) -> dict:
    _ensure_data(); return plot_parameter_scatter(registry, request.parameter, request.metric)

@app.post("/plots/correlation")
def correlation_plot(request: PlanRequest) -> dict:
    _ensure_data(); return plot_parameter_correlation_bar(registry, request.target_metric)

@app.post("/plots/ranking")
def ranking_plot(request: PlanRequest) -> dict:
    _ensure_data(); return plot_metric_ranking_bar(registry, request.target_metric, request.n_cases)

@app.get("/plans")
def list_plans() -> list[dict]: return plans.list()

@app.get("/plans/{plan_id}")
def get_plan(plan_id: str) -> dict:
    result = plans.get(plan_id)
    if not result: raise HTTPException(404, "Plan not found")
    return result

@app.post("/plans/{plan_id}/approve")
def approve(plan_id: str, request: DecisionRequest) -> dict:
    try: return plans.decide(plan_id, "approved", request.actor, request.comment)
    except KeyError as exc: raise HTTPException(404, "Plan not found") from exc
    except ValueError as exc: raise HTTPException(409, str(exc)) from exc

@app.post("/plans/{plan_id}/reject")
def reject(plan_id: str, request: DecisionRequest) -> dict:
    try: return plans.decide(plan_id, "rejected", request.actor, request.comment)
    except KeyError as exc: raise HTTPException(404, "Plan not found") from exc
    except ValueError as exc: raise HTTPException(409, str(exc)) from exc

@app.post("/plans/{plan_id}/execute")
def execute(plan_id: str, dry_run: bool = True) -> dict:
    try: return execute_approved_plan(plan_id, plans, registry, dry_run=dry_run)
    except KeyError as exc: raise HTTPException(404, "Plan not found") from exc
    except PermissionError as exc: raise HTTPException(409, str(exc)) from exc
    except FileExistsError as exc: raise HTTPException(409, str(exc)) from exc
