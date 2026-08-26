"""Aggregate retrieval quality by document/query shape for failure analysis."""
from __future__ import annotations
import json
from pathlib import Path
from app.config import ROOT

def run(path: Path = ROOT/"reports"/"scientific_multimodal_retrieval_eval.json"):
    data=json.loads(path.read_text(encoding="utf-8")); rows=data.get("results",[]); out={}
    def tags(r):
        q=r.get("question", "").lower(); out=[]
        if r.get("requires_ocr"): out.append("ocr_scan")
        if any(t in q for t in ("表格","table","单位","变量","equation","mole fractions","编号","专利号")): out.append("exact_table_term")
        if any(t in q for t in ("机理","为什么","影响","局限","驱动力","目的","性能目标")): out.append("mechanism_explanation")
        if any(t in q for t in ("moose","fiat","输入 deck","并行","thermal hydraulics")): out.append("moose_method")
        if len(r.get("expected_sources",[]))>1 or any(t in q for t in ("比较","结合","和外部","多个来源")): out.append("cross_source")
        return out or ["general"]
    for tag in sorted({t for r in rows for t in tags(r)}):
        group=[r for r in rows if tag in tags(r)]
        out[tag]={"questions":len(group),"source_hit_rate":round(sum(r.get("source_hit",False) for r in group)/len(group),4) if group else 0,"page_hit_rate":round(sum(r.get("page_hit",False) for r in group)/len(group),4) if group else 0,"document_kind_hit_rate":round(sum(r.get("document_kind_hit",False) for r in group)/len(group),4) if group else 0,"failed_ids":[r["id"] for r in group if not r.get("source_hit",False)]}
    result={"source_report":str(path),"overall":data.get("metrics",{}),"slices":out,"failure_families":["OCR recognition errors","table structure loss","double-column reading-order errors","parent-child localization mismatch","terminology mismatch"]}
    (ROOT/"reports"/"retrieval_failure_slices.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); return result
if __name__=="__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))
