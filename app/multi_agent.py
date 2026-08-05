"""Evidence-constrained LangGraph collaboration over the stage-one tools."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from operator import add
from typing import Annotated, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.analysis import parameter_correlation, top_cases
from app.models import AnalysisEvidence, Critique, EvidenceCard, EvidenceRequirement, GroundedStatement, ReviewDecision
from app.claim_verifier import verify_grounded_statements
from app.llm_protocol import EvidenceSummary, PlannerProposal, ResearchSynthesis, RoutePlan, SemanticCritique
from app.llm_router import LLMRouter
from app.qa import METRICS, _explicit_metric
from app.registry import SimulationRegistry
from app.retrieval import HybridRetriever
from app.resource_limits import saturated
from app.simulation_plan import PARAMETER_BOUNDS, suggest_exploration_plan
from app.experiment_cycle import design_experiment
from app.evidence_policy import review_evidence_policy
from app.trace import new_trace_id, normalize_event, summary as trace_summary


class ResearchState(TypedDict, total=False):
    question: str
    task_type: Literal["knowledge", "simulation_analysis", "mixed"]
    metric: str
    trace: Annotated[list[dict], add]
    evidence_cards: Annotated[list[dict], add]
    analysis_evidence: Annotated[list[dict], add]
    critiques: Annotated[list[dict], add]
    current_critiques: list[dict]
    draft: str
    review: dict
    review_attempts: int
    retrieval_query: str
    llm_evidence_summary: dict
    research_synthesis: dict
    planner_proposal: dict
    semantic_critiques: list[dict]
    llm_calls: Annotated[list[dict], add]
    grounded_statements: Annotated[list[dict], add]
    claim_verifications: list[dict]
    recovered_evidence_cards: list[dict]
    recovery_attempts: int
    recovery_actions: Annotated[list[dict], add]
    plan_draft: dict
    evidence_requirements: list[dict]
    retrieval_mode_used: str
    evidence_gap: dict
    routing: dict
    trace_id: str
    current_span_id: str


def _needs_analysis(question: str) -> bool:
    return bool(re.search(r"case[_ -]?\d+|参数|相关|影响|敏感|最优|最好|最低|关系|调整|排序|结合", question, re.I)) and (
        bool(_explicit_metric(question)) or "case" in question.lower() or "参数" in question
    )


def _needs_retrieval(question: str) -> bool:
    return any(token in question.lower() for token in ("为什么", "原因", "机理", "依据", "文献", "论文", "扫描", "ocr", "表格", "页码", "章节", "双栏", "arc jet", "shear", "日志", "输入", "设置", "报告", "解释", "是否", "基线", "边界", "局限", "不确定", "证据"))


def _needs_plan_draft(question: str) -> bool:
    return any(token in question for token in ("下一轮", "候选仿真", "仿真建议", "生成计划", "建议怎么做"))


def _evidence_requirements(question: str, analysis: bool) -> list[EvidenceRequirement]:
    lowered = question.lower()
    requirements: list[EvidenceRequirement] = []
    if analysis:
        requirements.append(EvidenceRequirement(kind="registry_analysis", reason="问题包含可由历史 case 定量验证的指标或参数。"))
    scan_intent = any(term in lowered for term in ("扫描", "ocr", "multiwall", "多层壁"))
    if not scan_intent and any(term in lowered for term in ("论文", "文献", "外部研究", "表格", "页码", "fiatc", "硅酸盐", "炭化", "热响应", "热防护", "烧蚀", "tufroc", "arc jet", "shear")):
        requirements.append(EvidenceRequirement(kind="paper", reason="问题要求外部公开科研资料或机理证据。"))
    if scan_intent:
        requirements.append(EvidenceRequirement(kind="scan_report", reason="问题明确要求扫描技术报告或 OCR 证据。"))
    if any(term in lowered for term in ("报告", "解释", "机理", "因果", "历史总结")) or (analysis and _needs_retrieval(question)):
        requirements.append(EvidenceRequirement(kind="report", reason="解释性或因果边界结论需要历史报告佐证。"))
    if "日志" in lowered:
        requirements.append(EvidenceRequirement(kind="run_log", reason="问题显式要求运行日志证据。"))
    if any(term in lowered for term in ("状态", "耗时", "return_code")):
        requirements.append(EvidenceRequirement(kind="run_status", reason="问题要求运行状态或耗时记录。"))
    if any(term in lowered for term in ("输入", "deck", "边界", "语法", "设置")):
        requirements.append(EvidenceRequirement(kind="input_deck", reason="问题要求核对输入配置。"))
    if any(term in lowered for term in ("脚本", "命名", "批量运行")):
        requirements.append(EvidenceRequirement(kind="script", reason="问题要求批处理脚本证据。"))
    if _needs_retrieval(question) and not any(item.kind != "registry_analysis" for item in requirements):
        requirements.append(EvidenceRequirement(kind="report", reason="未指定文档类型时，以历史报告作为默认解释性来源。"))
    return requirements


def _preferred_sources(question: str) -> set[str] | None:
    lowered = question.lower()
    sources: set[str] = set()
    if any(term in lowered for term in ("报告", "解释", "机理", "因果", "历史总结")):
        sources.add("report")
    scan_intent = any(term in lowered for term in ("扫描", "ocr", "multiwall", "多层壁"))
    if not scan_intent and any(term in lowered for term in ("论文", "文献", "外部研究", "表格", "页码", "fiatc", "硅酸盐", "炭化", "热响应", "热防护", "烧蚀", "tufroc", "arc jet", "shear")):
        sources.add("paper")
    if scan_intent:
        sources.add("scan_report")
    if "日志" in lowered:
        sources.add("run_log")
    if any(term in lowered for term in ("指标", "rmse", "误差")) and "日志" in lowered:
        sources.add("report")
    if any(term in lowered for term in ("输入", "deck", "边界", "语法")):
        sources.add("input_deck")
    return sources or None


def _decompose_question(question: str, analysis: bool, retrieval: bool) -> dict[str, str]:
    """Expose independent tool and document sub-tasks without fabricating answers."""
    return ({"analysis": question} if analysis else {}) | ({"retrieval": question} if retrieval else {})


def _resolve_route(question: str, route_plan: RoutePlan | None, *, llm_enabled: bool) -> dict:
    """Merge an advisory model plan with deterministic minimum safeguards."""
    rule_analysis, rule_retrieval = _needs_analysis(question), _needs_retrieval(question)
    policy_overrides: list[str] = []
    accepted_sources: list[str] = []
    fallback_reason = ""
    metric = _explicit_metric(question) or "early_1_2_rmse"
    retrieval_query = question
    if route_plan is None:
        analysis, retrieval = rule_analysis, rule_retrieval
        status = "rule_fallback"
        fallback_reason = "llm_unavailable_or_protocol_invalid" if llm_enabled else "llm_disabled"
    else:
        analysis = rule_analysis or route_plan.needs_registry_analysis
        retrieval = rule_retrieval or bool(route_plan.required_sources)
        accepted_sources = list(dict.fromkeys(route_plan.required_sources))
        retrieval_query = route_plan.retrieval_query
        if route_plan.analysis_metric in METRICS:
            metric = route_plan.analysis_metric
        else:
            policy_overrides.append("invalid_metric_ignored")
        status = "llm_validated"
    task_type = "mixed" if analysis and retrieval else "simulation_analysis" if analysis else "knowledge"
    if route_plan:
        if route_plan.task_type != task_type:
            policy_overrides.append("task_type_recomputed_from_safeguards")
        if rule_analysis and not route_plan.needs_registry_analysis:
            policy_overrides.append("rule_required_registry_analysis")
        if rule_retrieval and not route_plan.required_sources:
            policy_overrides.append("rule_required_document_retrieval")
    return {"status": status, "task_type": task_type, "analysis": analysis, "retrieval": retrieval, "metric": metric, "retrieval_query": retrieval_query, "llm_plan_used": route_plan is not None, "accepted_sources": accepted_sources, "advisory_needs_experiment": route_plan.needs_experiment if route_plan else False, "policy_overrides": policy_overrides, "fallback_reason": fallback_reason}


def _citation_coverage(question: str, cards: list[dict]) -> tuple[bool, list[str]]:
    """Check named, verifiable tokens against raw excerpts before approval."""
    named = set(re.findall(r"(?:early_[a-z0-9_]+|temp_mean_rmse|[a-z]+_[a-z]+_scale|tbegin\d+_shift)", question.lower()))
    if not named:
        return True, []
    corpus = " ".join(card.get("excerpt", "").lower() for card in cards)
    missing = sorted(token for token in named if token not in corpus)
    return not missing, missing


def _available_chunk_ids(cards: list[dict]) -> set[str]:
    return {str(card.get("retrieval", {}).get("chunk_id", "")) for card in cards} - {""}


def _validated_research(proposal: ResearchSynthesis | None, cards: list[dict]) -> ResearchSynthesis | None:
    if proposal is None:
        return None
    if not proposal.claims or any(any(index < 0 or index >= len(cards) for index in claim.evidence_indexes) for claim in proposal.claims):
        return None
    return proposal


def _validated_planner(proposal: PlannerProposal | None, metric: str, has_draft: bool) -> PlannerProposal | None:
    if proposal is None or not has_draft or proposal.target_metric != metric or not proposal.requires_human_approval:
        return None
    if any(name not in PARAMETER_BOUNDS for name in proposal.focus_parameters):
        return None
    return proposal


def _validated_semantic_critique(proposal: SemanticCritique | None, cards: list[dict]) -> SemanticCritique | None:
    if proposal is None:
        return None
    if any(any(index < 0 or index >= len(cards) for index in item.evidence_indexes) for item in proposal.issues):
        return None
    return proposal


def _serialize_with_chunk_ids(proposal: ResearchSynthesis | SemanticCritique, cards: list[dict]) -> dict:
    """Models select compact evidence indexes; the runtime restores stable IDs."""
    payload = proposal.model_dump()
    for item in payload.get("claims", []) + payload.get("issues", []):
        item["citation_chunk_ids"] = [str(cards[index].get("retrieval", {}).get("chunk_id", "")) for index in item.pop("evidence_indexes", [])]
    return payload


def build_graph(
    registry: SimulationRegistry,
    router: LLMRouter | None = None,
    retrieval_mode: str | None = None,
    chunk_strategy: str | None = None,
    event_sink: Callable[[dict], None] | None = None,
    trace_id: str | None = None,
):
    router = router or LLMRouter()
    retriever = HybridRetriever(registry, mode=retrieval_mode, chunk_strategy=chunk_strategy)

    active_trace_id = trace_id or new_trace_id()

    def emit(_event: dict) -> None:
        # Node wrappers publish normalized events after the state update is
        # available. This keeps SSE and returned traces identical.
        return None

    def traced(node_name: str, handler: Callable[[ResearchState], dict]):
        def wrapped(state: ResearchState) -> dict:
            started = time.perf_counter()
            output = handler(state)
            raw_events = list(output.get("trace", [])) or [{"node": node_name}]
            parent = str(state.get("current_span_id", ""))
            elapsed = (time.perf_counter() - started) * 1000
            events: list[dict] = []
            for raw in raw_events:
                # Fan-out retains specialist events. Only the owning graph
                # node receives the measured wall time; others are decisions.
                event = normalize_event(raw, trace_id=active_trace_id, parent_span_id=parent,
                                        elapsed_ms=elapsed if raw.get("node") == node_name else 0.0)
                events.append(event)
                if event_sink:
                    event_sink(event)
                parent = event["span_id"]
            return {**output, "trace": events, "trace_id": active_trace_id, "current_span_id": parent}
        return wrapped

    def new_calls(start: int) -> list[dict]:
        return [call.__dict__ for call in router.telemetry[start:]]

    def observed_calls(start: int, role: str, proposal: object | None) -> list[dict]:
        """Keep trace truthfully observable for protocol-compatible routers.

        Production ``LLMRouter`` appends telemetry itself.  Test/adaptor
        routers may expose the same constrained protocol without mutating a
        telemetry list, so record an explicit adapter event instead of
        silently reporting that no model was used.
        """
        calls = new_calls(start)
        if not calls and proposal is not None and bool(getattr(router, "enabled", False)):
            return [{"role": role, "model": "protocol_adapter", "latency_ms": 0.0, "success": True, "error": "telemetry_not_emitted_by_adapter"}]
        return calls

    def supervisor(state: ResearchState) -> dict:
        q = state["question"]
        call_start = len(router.telemetry)
        proposal = router.call_json("router", "你是科研任务路由器。只判断任务所需的受限工具和证据类型；不能作事实判断，不能授权执行仿真。", f"问题：{q}", RoutePlan)
        route = _resolve_route(q, proposal if isinstance(proposal, RoutePlan) else None, llm_enabled=bool(getattr(router, "enabled", False)))
        calls = observed_calls(call_start, "router", proposal)
        requirements = _evidence_requirements(q, route["analysis"])
        existing = {item.kind for item in requirements}
        for source in route["accepted_sources"]:
            if source not in existing:
                requirements.append(EvidenceRequirement(kind=source, reason="经白名单校验的结构化路由建议需要该类型证据。"))
        event = {"node": "supervisor", "task_type": route["task_type"], "decision": {"analysis": route["analysis"], "retrieval": route["retrieval"], "sub_tasks": _decompose_question(q, route["analysis"], route["retrieval"]), "evidence_requirements": [item.model_dump() for item in requirements], "routing": route}}
        emit(event)
        return {"task_type": route["task_type"], "metric": route["metric"], "retrieval_query": route["retrieval_query"], "evidence_requirements": [item.model_dump() for item in requirements], "routing": route, "review_attempts": state.get("review_attempts", 0), "recovery_attempts": state.get("recovery_attempts", 0), "llm_calls": calls, "trace": [event]}

    def retrieve(state: ResearchState) -> dict:
        call_start = len(router.telemetry)
        document_types = [item["kind"] for item in state.get("evidence_requirements", []) if item["kind"] != "registry_analysis"]
        requested_mode = retriever.mode
        # OCR/scanned pages benefit from lexical tokens and exact page cues;
        # public papers benefit from Dense semantic matching in Chinese/English.
        if "scan_report" in document_types and requested_mode != "bm25":
            effective_mode = "bm25"
            downgrade = "scan_page_precision"
        elif requested_mode in {"dense", "hybrid", "hybrid_rerank"} and saturated("embedding"):
            effective_mode = "bm25"
            downgrade = "embedding_busy"
        elif requested_mode == "hybrid_rerank" and saturated("reranker"):
            effective_mode = "hybrid"
            downgrade = "reranker_busy"
        else:
            effective_mode = requested_mode
            downgrade = ""
        active_retriever = retriever if effective_mode == requested_mode else HybridRetriever(registry, mode=effective_mode, chunk_strategy=retriever.chunk_strategy)
        cards: list[EvidenceCard] = []
        # Each required source is independently retrieved once. This prevents
        # a high-scoring input deck from silently substituting for a report.
        for source_type in document_types:
            cards.extend(active_retriever.search(state.get("retrieval_query", state["question"]), limit=1, source_types={source_type}))
        if not cards and _needs_retrieval(state["question"]):
            cards = active_retriever.search(state.get("retrieval_query", state["question"]), limit=4, source_types=_preferred_sources(state["question"]))
        excerpt_bundle = "\n\n".join(f"SOURCE: {card.source_path}\n{card.excerpt[:400]}" for card in cards)
        summary = router.call_json("evidence", "你是证据整理器。只总结给定摘录；不要新增事实、数值、来源或因果结论。", f"原问题：{state['question']}\n证据摘录：\n{excerpt_bundle}", EvidenceSummary) if cards else None
        calls = observed_calls(call_start, "evidence", summary)
        event = {"node": "retriever", "evidence_count": len(cards), "required_source_types": document_types, "retrieval_mode": effective_mode, "vector_backend": active_retriever.vector_status(), "downgrade": downgrade, "llm_evidence_used": bool(summary)}
        emit(event)
        output = {"evidence_cards": [card.model_dump() for card in cards], "retrieval_mode_used": effective_mode, "trace": [event], "llm_calls": calls}
        if summary: output["llm_evidence_summary"] = summary.model_dump()
        return output

    def analyze(state: ResearchState) -> dict:
        metric = state["metric"]
        q = state["question"].lower()
        if any(word in q for word in ("最优", "最好", "最低", "top")):
            result = top_cases(registry, metric, 5)
            claim = f"{metric} 的最优历史 case 可由 Registry 排序得到。"
        else:
            result = parameter_correlation(registry, metric)
            claim = f"参数与 {metric} 的线性相关性已由历史 case 计算。"
        evidence = AnalysisEvidence(claim=claim, metric=metric, source=result["source"], result=result, limitations=["相关性描述历史样本中的线性关联，不能单独证明因果关系。", f"样本量为 {result.get('sample_size', 'N/A')} 个历史 case。"])
        event = {"node": "simulation_analyst", "metric": metric, "tool": "top_cases" if "rows" in result else "parameter_correlation"}
        emit(event)
        return {"analysis_evidence": [evidence.model_dump()], "trace": [event]}

    def plan_draft(state: ResearchState) -> dict:
        experiment = design_experiment(state["question"], registry, state["metric"])
        if experiment["gap"]["status"] == "needs_experiment":
            plan = experiment["plan"]
            event = {"node": "evidence_gap", "status": "needs_experiment", "gap_type": experiment["gap"]["gap_type"], "variables": experiment["gap"]["variable_parameters"], "plan_id": plan["plan_id"]}
            emit(event)
            return {"evidence_gap": experiment["gap"], "plan_draft": plan, "trace": [event]}
        if experiment["gap"]["status"] == "unsupported":
            event = {"node": "evidence_gap", "status": "unsupported", "gap_type": experiment["gap"]["gap_type"], "unknown_parameters": experiment["gap"]["unknown_parameters"]}
            emit(event)
            return {"evidence_gap": experiment["gap"], "trace": [event]}
        if not _needs_plan_draft(state["question"]):
            return {}
        plan = suggest_exploration_plan(registry, state["metric"], n_cases=3).model_dump()
        event = {"node": "plan_draft", "status": "draft_only", "plan_id": plan["plan_id"], "target_metric": plan["target_metric"]}
        emit(event)
        return {"evidence_gap": experiment["gap"], "plan_draft": plan, "trace": [event]}

    def planner_agent(state: ResearchState) -> dict:
        """LLM can prioritize allowed parameters, not construct or execute a plan."""
        plan = state.get("plan_draft")
        gap = state.get("evidence_gap", {})
        call_start = len(router.telemetry)
        proposal = router.call_json(
            "planner",
            "你是仿真实验规划 Agent。只能从给定参数白名单中选择关注项；不得给出参数数值、执行命令或审批结论；requires_human_approval 必须为 true。",
            f"问题：{state['question']}\n目标指标：{state['metric']}\n知识缺口：{gap}\n已有受限草案：{plan}\n参数白名单：{sorted(PARAMETER_BOUNDS)}",
            PlannerProposal,
        ) if plan else None
        validated = _validated_planner(proposal if isinstance(proposal, PlannerProposal) else None, state["metric"], bool(plan))
        calls = observed_calls(call_start, "planner", proposal)
        event = {"node": "planner_agent", "has_draft": bool(plan), "llm_used": proposal is not None, "accepted": validated is not None, "focus_parameters": validated.focus_parameters if validated else []}
        emit(event)
        output = {"trace": [event], "llm_calls": calls}
        if validated:
            output["planner_proposal"] = validated.model_dump()
        return output

    def evidence_fanout(state: ResearchState) -> dict:
        """Fan out independent specialists concurrently and merge validated outputs."""
        if state["task_type"] == "mixed":
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="evidence") as executor:
                retrieval_future = executor.submit(retrieve, state)
                analysis_future = executor.submit(analyze, state)
                retrieval_result, analysis_result = retrieval_future.result(), analysis_future.result()
            output = {
                "evidence_cards": retrieval_result["evidence_cards"],
                "analysis_evidence": analysis_result["analysis_evidence"],
                "retrieval_mode_used": retrieval_result.get("retrieval_mode_used", ""),
                "llm_calls": [*retrieval_result.get("llm_calls", []), *analysis_result.get("llm_calls", [])],
                "trace": [*retrieval_result["trace"], *analysis_result["trace"], {"node": "evidence_fanout", "parallel_agents": ["retriever", "simulation_analyst"]}],
            }
            if retrieval_result.get("llm_evidence_summary"):
                output["llm_evidence_summary"] = retrieval_result["llm_evidence_summary"]
            return output
        result = analyze(state) if state["task_type"] == "simulation_analysis" else retrieve(state)
        return {**result, "trace": [*result.get("trace", []), {"node": "evidence_fanout", "parallel_agents": ["simulation_analyst" if state["task_type"] == "simulation_analysis" else "retriever"]}]}

    def research_agent(state: ResearchState) -> dict:
        """Let an LLM compare retrieved evidence, never create new evidence."""
        cards = state.get("recovered_evidence_cards") or state.get("evidence_cards", [])
        call_start = len(router.telemetry)
        bundle = "\n\n".join(
            f"EVIDENCE_INDEX: {index}\nSOURCE: {card.get('source_path', '')}\nEXCERPT: {card.get('excerpt', '')[:500]}"
            for index, card in enumerate(cards)
        )
        proposal = router.call_json(
            "research",
            "你是科研证据归纳 Agent。只能比较提供的证据，claims 必须填写已有 EVIDENCE_INDEX；不得新增数值、来源、因果或项目事实。",
            f"问题：{state['question']}\n可用证据：\n{bundle}",
            ResearchSynthesis,
        ) if cards else None
        validated = _validated_research(proposal if isinstance(proposal, ResearchSynthesis) else None, cards)
        calls = observed_calls(call_start, "research", proposal)
        event = {"node": "research_agent", "evidence_count": len(cards), "llm_used": proposal is not None, "accepted": validated is not None, "claim_count": len(validated.claims) if validated else 0}
        emit(event)
        output = {"trace": [event], "llm_calls": calls}
        if validated:
            output["research_synthesis"] = _serialize_with_chunk_ids(validated, cards)
        return output

    def critic(state: ResearchState) -> dict:
        critiques: list[Critique] = []
        if state.get("analysis_evidence"):
            critiques.append(Critique(issue_type="causality", severity="warning", message="历史 LHS case 的相关性不能替代因果验证；需要控制变量或新的 MOOSE case 验证。"))
        cards = state.get("recovered_evidence_cards") or state.get("evidence_cards", [])
        if state.get("task_type") == "mixed" and not cards:
            critiques.append(Critique(issue_type="missing_retrieval", severity="warning", message="问题要求机理或依据，但未获得可引用文档证据。"))
        required_types = {item["kind"] for item in state.get("evidence_requirements", []) if item["kind"] != "registry_analysis"}
        present_types = {card["source_type"] for card in cards}
        missing_types = sorted(required_types - present_types)
        if missing_types:
            critiques.append(Critique(issue_type="missing_evidence_type", severity="error", message=f"缺少要求的证据类型：{', '.join(missing_types)}。"))
        # Structured-only answers are grounded in Registry tool output; a
        # document lexical gate applies only when the supervisor requested
        # document evidence.
        covered, missing = _citation_coverage(state["question"], cards) if _needs_retrieval(state["question"]) else (True, [])
        if not covered:
            critiques.append(Critique(issue_type="citation_coverage", severity="error", message=f"引用摘录未覆盖问题中的关键术语：{', '.join(missing)}。"))
        if not state.get("analysis_evidence") and not state.get("evidence_cards"):
            critiques.append(Critique(issue_type="no_evidence", severity="error", message="没有可用证据，不能给出结论。"))
        event = {"node": "critic", "critique_count": len(critiques), "issues": [item.issue_type for item in critiques]}
        emit(event)
        serialized = [item.model_dump() for item in critiques]
        return {"critiques": serialized, "current_critiques": serialized, "trace": [event]}

    def synthesize(state: ResearchState) -> dict:
        sections = [f"问题：{state['question']}"]
        statements: list[GroundedStatement] = []
        for item in state.get("analysis_evidence", []):
            result = item["result"]
            if "correlations" in result:
                strongest = sorted(result["correlations"].items(), key=lambda pair: abs(pair[1]), reverse=True)[:3]
                values = "；".join(f"{name}={value:+.3f}" for name, value in strongest)
                text = f"仿真数据结论：针对 {item['metric']}，绝对相关性最高的参数为 {values}。该结论来自 {result['sample_size']} 个历史 case 的 Pearson 相关性计算。"
            else:
                first = result["rows"][0] if result.get("rows") else {}
                text = f"仿真数据结论：{item['metric']} 当前最优记录为 {first.get('case_id')}，数值为 {first.get('metric_value')}。"
            sections.append(text)
            statements.append(GroundedStatement(text=text, evidence_kind="analysis", source_path=item["source"], support=item["claim"]))
        cards = state.get("recovered_evidence_cards") or state.get("evidence_cards", [])
        if cards:
            card = cards[0]; retrieval = card.get("retrieval", {})
            excerpt = card["excerpt"].strip()[:360]
            text = f"文档摘录（非推理结论）：{excerpt}"
            sections.append(text)
            statements.append(GroundedStatement(text=text, evidence_kind="document", source_path=card["source_path"], chunk_id=retrieval.get("chunk_id", ""), start_line=retrieval.get("start_line"), end_line=retrieval.get("end_line"), support=excerpt))
        # LLM evidence summaries remain observable candidate artifacts. They are
        # deliberately excluded from the final factual answer until an
        # entailment checker is introduced; only raw citations and tool output
        # may support user-facing conclusions.
        if state.get("current_critiques"):
            limitation = "局限性：" + "；".join(item["message"] for item in state["current_critiques"])
            sections.append(limitation)
            statements.append(GroundedStatement(text=limitation, evidence_kind="limitation", source_path="critic", support="Critic 基于证据覆盖和因果边界生成。"))
        if state.get("plan_draft"):
            plan = state["plan_draft"]
            gap = state.get("evidence_gap", {})
            prefix = "知识缺口诊断：" + gap.get("message", "需要新仿真验证。") + "\n" if gap.get("status") == "needs_experiment" else ""
            text = prefix + f"候选仿真计划草案：目标为 {plan['target_metric']}，包含 {len(plan['cases'])} 个候选 case；该草案尚未持久化或执行，需人工审批后才可进入执行流程。"
            sections.append(text)
            statements.append(GroundedStatement(text=text, evidence_kind="analysis", source_path="simulation_plan_draft", support="由历史排序生成的受限 SimulationPlan 草案。"))
        if state.get("planner_proposal"):
            proposal = state["planner_proposal"]
            text = f"规划 Agent 建议优先关注：{', '.join(proposal['focus_parameters']) or '当前草案变量'}；该建议不改变受限计划参数，仍需人工审批。"
            sections.append(text)
            statements.append(GroundedStatement(text=text, evidence_kind="limitation", source_path="planner_agent", support="LLM Planner 建议已通过参数白名单与人工审批标志校验。"))
        event = {"node": "synthesizer", "used_analysis": len(state.get("analysis_evidence", [])), "used_documents": len(cards), "grounded_statement_count": len(statements)}
        emit(event)
        serialized = [item.model_dump() for item in statements]
        verifications = verify_grounded_statements(serialized, cards, state.get("analysis_evidence", []))
        event["claim_verification"] = {"supported": sum(item["status"] == "supported" for item in verifications), "context_only": sum(item["status"] == "context_only" for item in verifications), "insufficient": sum(item["status"] == "insufficient" for item in verifications), "conflicted": sum(item["status"] == "conflicted" for item in verifications)}
        return {"draft": "\n".join(sections), "grounded_statements": serialized, "claim_verifications": verifications, "trace": [event]}

    def semantic_critic(state: ResearchState) -> dict:
        """Advisory semantic review; deterministic gates remain authoritative."""
        cards = state.get("recovered_evidence_cards") or state.get("evidence_cards", [])
        call_start = len(router.telemetry)
        bundle = "\n".join(f"EVIDENCE_INDEX: {index} | {card.get('excerpt', '')[:350]}" for index, card in enumerate(cards))
        proposal = router.call_json(
            "critic",
            "你是科研语义审查 Agent。只识别过度外推、证据冲突、缺证据或相关性当因果的问题；涉及文档时必须填写已有 EVIDENCE_INDEX；不能批准回答。",
            f"问题：{state['question']}\n草稿：{state.get('draft', '')[:1400]}\nResearch Agent 归纳：{state.get('research_synthesis', {})}\n证据：\n{bundle}",
            SemanticCritique,
        )
        validated = _validated_semantic_critique(proposal if isinstance(proposal, SemanticCritique) else None, cards)
        calls = observed_calls(call_start, "critic", proposal)
        issues = _serialize_with_chunk_ids(validated, cards).get("issues", []) if validated else []
        event = {"node": "semantic_critic", "llm_used": proposal is not None, "accepted": validated is not None, "issue_count": len(issues)}
        emit(event)
        return {"semantic_critiques": issues or state.get("semantic_critiques", []), "trace": [event], "llm_calls": calls}

    def reviewer(state: ResearchState) -> dict:
        reasons: list[str] = []
        if state["task_type"] in {"simulation_analysis", "mixed"} and not state.get("analysis_evidence"): reasons.append("缺少程序生成的定量仿真证据。")
        if state["task_type"] == "mixed" and not (state.get("recovered_evidence_cards") or state.get("evidence_cards")): reasons.append("缺少支持机理/依据问题的可引用文档证据。")
        if any(item["issue_type"] == "citation_coverage" for item in state.get("current_critiques", [])): reasons.append("引用摘录未覆盖问题中的关键术语，不能确认其支持结论。")
        if not state.get("draft"): reasons.append("未生成最终草稿。")
        if not state.get("grounded_statements"): reasons.append("最终回答没有可追溯的句级证据链接。")
        unsupported = [item for item in state.get("claim_verifications", []) if item.get("status") in {"insufficient", "conflicted"}]
        if unsupported: reasons.append(f"有 {len(unsupported)} 条最终事实 Claim 缺少可验证证据或存在冲突。")
        if state.get("llm_evidence_summary"):
            reasons.append("LLM 证据摘要已作为候选产物留存，未用于最终事实结论。")
        for issue in state.get("semantic_critiques", []):
            reasons.append(f"语义审查提示（{issue['issue_type']}）：{issue['message']}")
        cards = state.get("recovered_evidence_cards") or state.get("evidence_cards", [])
        reasons.extend(review_evidence_policy(state["question"], cards, state.get("analysis_evidence", [])))
        blocking = [reason for reason in reasons if "未用于最终事实结论" not in reason and not reason.startswith("语义审查提示")]
        review = ReviewDecision(approved=not blocking, reasons=reasons, requires_revision=bool(blocking) and state.get("review_attempts", 0) < 1)
        event = {"node": "reviewer", "approved": review.approved, "reasons": reasons}
        emit(event)
        return {"review": review.model_dump(), "trace": [event]}

    def recovery(state: ResearchState) -> dict:
        issues = {item["issue_type"] for item in state.get("current_critiques", [])}
        query = state.get("retrieval_query", state["question"])
        source_types: set[str] | None = _preferred_sources(state["question"])
        action = "unfiltered_retry"
        if "missing_evidence_type" in issues:
            source_types = {item["kind"] for item in state.get("evidence_requirements", []) if item["kind"] != "registry_analysis"}
            action = "typed_evidence_retry"
        elif "citation_coverage" in issues:
            source_types = {"report"}
            action = "report_priority_retry"
        elif "missing_retrieval" in issues:
            source_types = {"run_log", "report"} if "日志" in state["question"] else {"report"}
            action = "required_source_retry"
        elif "no_evidence" in issues:
            source_types = set()
        cards = retriever.search(query, limit=4, source_types=source_types)
        event = {"node": "recovery", "attempt": state.get("recovery_attempts", 0) + 1, "action": action, "source_filter": sorted(source_types or []), "evidence_count": len(cards), "issues": sorted(issues)}
        emit(event)
        # ``review_attempts`` is the graph's hard revision budget. Advance it
        # together with recovery so unsupported claims terminate explicitly.
        return {"recovery_attempts": state.get("recovery_attempts", 0) + 1, "review_attempts": state.get("review_attempts", 0) + 1, "recovered_evidence_cards": [card.model_dump() for card in cards], "recovery_actions": [event], "trace": [event]}

    def choose_review(state: ResearchState):
        review = state["review"]
        return "recovery" if review["requires_revision"] else END

    graph = StateGraph(ResearchState)
    graph.add_node("supervisor", traced("supervisor", supervisor)); graph.add_node("evidence_fanout", traced("evidence_fanout", evidence_fanout)); graph.add_node("research_agent", traced("research_agent", research_agent))
    graph.add_node("critic", traced("critic", critic)); graph.add_node("synthesizer", traced("synthesizer", synthesize)); graph.add_node("semantic_critic", traced("semantic_critic", semantic_critic)); graph.add_node("reviewer", traced("reviewer", reviewer)); graph.add_node("recovery", traced("recovery", recovery)); graph.add_node("plan_draft", traced("plan_draft", plan_draft)); graph.add_node("planner_agent", traced("planner_agent", planner_agent))
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "evidence_fanout")
    graph.add_edge("evidence_fanout", "research_agent")
    graph.add_edge("research_agent", "plan_draft")
    graph.add_edge("plan_draft", "planner_agent")
    graph.add_edge("planner_agent", "critic")
    graph.add_edge("critic", "synthesizer"); graph.add_edge("synthesizer", "semantic_critic"); graph.add_edge("semantic_critic", "reviewer")
    graph.add_conditional_edges("reviewer", choose_review, {"recovery": "recovery", END: END})
    graph.add_edge("recovery", "research_agent")
    return graph.compile()


def run_multi_agent(
    question: str,
    registry: SimulationRegistry,
    router: LLMRouter | None = None,
    retrieval_mode: str | None = None,
    chunk_strategy: str | None = None,
    event_sink: Callable[[dict], None] | None = None,
) -> dict:
    trace_id = new_trace_id()
    state = build_graph(registry, router, retrieval_mode, chunk_strategy, event_sink, trace_id).invoke({"question": question, "trace_id": trace_id, "current_span_id": ""}, {"recursion_limit": 24})
    events = state.get("trace", [])
    return {"answer": state.get("draft", ""), "trace_id": trace_id, "trace_summary": trace_summary(events), "task_type": state.get("task_type"), "metric": state.get("metric"), "routing": state.get("routing", {}), "evidence_requirements": state.get("evidence_requirements", []), "retrieval_mode_used": state.get("retrieval_mode_used", ""), "evidence_cards": state.get("recovered_evidence_cards") or state.get("evidence_cards", []), "analysis_evidence": state.get("analysis_evidence", []), "grounded_statements": state.get("grounded_statements", []), "claim_verifications": state.get("claim_verifications", []), "critiques": state.get("current_critiques", []), "semantic_critiques": state.get("semantic_critiques", []), "research_synthesis": state.get("research_synthesis", {}), "planner_proposal": state.get("planner_proposal", {}), "recovery_actions": state.get("recovery_actions", []), "evidence_gap": state.get("evidence_gap", {}), "plan_draft": state.get("plan_draft", {}), "review": state.get("review", {}), "trace": events, "llm_calls": state.get("llm_calls", []), "llm_evidence_summary": state.get("llm_evidence_summary", {})}
