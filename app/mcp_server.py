"""Standard stdio MCP server exposing only typed, policy-gated research tools."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from app.analysis import parameter_correlation, top_cases
from app.config import DB_PATH, PLAN_DB_PATH
from app.execution import execute_approved_plan
from app.experiment_cycle import confirm_experiment_draft, design_experiment, validate_and_report_result
from app.plotting import plot_exploration_plan_diff, plot_metric_ranking_bar, plot_parameter_correlation_bar, plot_parameter_scatter
from app.registry import SimulationRegistry
from app.retrieval import HybridRetriever
from app.simulation_plan import PlanStore


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


TOOLS = [
    types.Tool(name="search_evidence", description="只读检索带来源、chunk 和行号的科研证据。", inputSchema=_schema({"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 5}}, ["query"])),
    types.Tool(name="analyze_parameter_correlation", description="只读计算参数与目标指标的 Pearson 相关性。", inputSchema=_schema({"metric": {"type": "string"}}, ["metric"])),
    types.Tool(name="plot_parameter_scatter", description="生成参数-指标散点图与拟合线。", inputSchema=_schema({"parameter": {"type": "string"}, "metric": {"type": "string"}}, ["parameter", "metric"])),
    types.Tool(name="plot_parameter_correlation_bar", description="生成参数相关性排序柱形图。", inputSchema=_schema({"metric": {"type": "string"}}, ["metric"])),
    types.Tool(name="plot_metric_ranking_bar", description="生成历史 case 指标排名柱形图。", inputSchema=_schema({"metric": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 20}}, ["metric"])),
    types.Tool(name="create_experiment_draft", description="基于知识缺口生成未持久化、不可执行的探索性实验草案。", inputSchema=_schema({"question": {"type": "string"}, "metric": {"type": "string"}}, ["question"])),
    types.Tool(name="confirm_experiment_draft", description="显式确认草案后写入 pending；不批准也不执行。", inputSchema=_schema({"plan": {"type": "object"}, "confirmed_by": {"type": "string", "minLength": 2}}, ["plan", "confirmed_by"])),
    types.Tool(name="execute_plan_preview", description="仅执行已批准计划的隔离 dry-run，禁止真实 MOOSE 执行。", inputSchema=_schema({"plan_id": {"type": "string"}}, ["plan_id"])),
    types.Tool(name="get_experiment_report", description="查询计划执行后的受限结果报告。", inputSchema=_schema({"plan_id": {"type": "string"}}, ["plan_id"])),
]


def dispatch_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    args = arguments or {}; registry = SimulationRegistry(DB_PATH); plans = PlanStore(PLAN_DB_PATH)
    if name == "search_evidence":
        return {"evidence": [card.model_dump() for card in HybridRetriever(registry).search(args["query"], int(args.get("limit", 4)))]}
    if name == "analyze_parameter_correlation": return parameter_correlation(registry, args["metric"])
    if name == "plot_parameter_scatter": return plot_parameter_scatter(registry, args["parameter"], args["metric"])
    if name == "plot_parameter_correlation_bar": return plot_parameter_correlation_bar(registry, args["metric"])
    if name == "plot_metric_ranking_bar": return plot_metric_ranking_bar(registry, args["metric"], int(args.get("top_k", 10)))
    if name == "create_experiment_draft": return design_experiment(args["question"], registry, args.get("metric", "early_1_2_rmse"))
    if name == "confirm_experiment_draft":
        result = confirm_experiment_draft(args["plan"], plans)
        return {"confirmed_by": args["confirmed_by"], "plan": result, "policy": "pending_human_approval_required"}
    if name == "execute_plan_preview": return execute_approved_plan(args["plan_id"], plans, registry, dry_run=True)
    if name == "get_experiment_report": return validate_and_report_result(args["plan_id"], plans, registry)
    raise ValueError(f"Unknown or forbidden MCP tool: {name}")


async def _list_tools(_ctx, _params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def _call_tool(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    try:
        result = dispatch_tool(params.name, params.arguments)
        text = json.dumps(result, ensure_ascii=False, default=str)
        return types.CallToolResult(content=[types.TextContent(text=text)], structuredContent=result)
    except Exception as exc:
        return types.CallToolResult(content=[types.TextContent(text=f"{type(exc).__name__}: {exc}")], isError=True)


server = Server("moose-research-tools", version="1.0.0", description="Policy-gated MOOSE research tools", on_list_tools=_list_tools, on_call_tool=_call_tool)


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
