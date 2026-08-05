from app.claim_verifier import verify_grounded_statements


def _document(*, excerpt: str = "原始摘录支持该结论。", polarity: str = "supports") -> dict:
    return {"source_path": "report.md", "excerpt": excerpt, "retrieval": {"chunk_id": "chunk-1", "authority": "B", "claim_polarity": polarity}}


def test_document_claim_requires_exact_resolvable_support():
    statement = {"text": "文档摘录：原始摘录支持该结论。", "evidence_kind": "document", "source_path": "report.md", "chunk_id": "chunk-1", "support": "原始摘录支持该结论。"}
    assert verify_grounded_statements([statement], [_document()], [])[0]["status"] == "supported"
    statement["support"] = "模型补充的因果结论。"
    assert verify_grounded_statements([statement], [_document()], [])[0]["status"] == "insufficient"


def test_conflicting_and_registry_claims_are_distinguished():
    statement = {"text": "文档摘录", "evidence_kind": "document", "source_path": "report.md", "chunk_id": "chunk-1", "support": "原始摘录支持该结论。"}
    assert verify_grounded_statements([statement], [_document(polarity="refutes")], [])[0]["status"] == "conflicted"
    analysis = {"text": "定量结论", "evidence_kind": "analysis", "source_path": "simulation_registry", "support": "由 Registry 计算得到。"}
    evidence = [{"source": "simulation_registry", "claim": "由 Registry 计算得到。"}]
    assert verify_grounded_statements([analysis], [], evidence)[0]["status"] == "supported"


def test_plans_and_limitations_are_not_promoted_to_facts():
    plan = {"text": "候选计划", "evidence_kind": "analysis", "source_path": "simulation_plan_draft", "support": "受限草案"}
    limitation = {"text": "相关性不是因果", "evidence_kind": "limitation", "source_path": "critic", "support": "边界"}
    statuses = [item["status"] for item in verify_grounded_statements([plan, limitation], [], [])]
    assert statuses == ["context_only", "context_only"]


def test_numeric_table_claim_requires_page_metadata():
    statement = {"text": "热流为 100", "evidence_kind": "document", "source_path": "thermal_table.pdf", "chunk_id": "one", "support": "热流为 100"}
    cards = [{"source_path": "thermal_table.pdf", "excerpt": "热流为 100", "retrieval": {"chunk_id": "one", "block_type": "table"}}]
    assert verify_grounded_statements([statement], cards, [])[0]["status"] == "insufficient"
