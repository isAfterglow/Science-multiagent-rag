from app.evidence_policy import review_evidence_policy


def test_external_paper_cannot_support_project_metric_without_project_evidence():
    reasons = review_evidence_policy("项目历史仿真的指标是多少？", [{"retrieval": {"authority": "C"}}], [])
    assert any("A/B" in reason for reason in reasons)


def test_low_confidence_ocr_blocks_exact_value_but_not_general_context():
    card = {"retrieval": {"authority": "D", "ocr_used": True, "ocr_confidence": 0.5}}
    assert review_evidence_policy("扫描图中的精确厚度是多少？", [card], [])
    assert not review_evidence_policy("扫描资料中的结构是什么？", [card], [])


def test_low_confidence_table_blocks_exact_value():
    card = {"retrieval": {"authority": "C", "block_type": "table", "table_confidence": 0.65}}
    assert review_evidence_policy("表格中 200-4000 Pa 的精确数值是多少？", [card], [])
