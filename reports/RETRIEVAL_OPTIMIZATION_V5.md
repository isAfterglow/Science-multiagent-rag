# Retrieval Optimization V5

## P0: Source Fusion on the critical path

The original source-fusion experiment rebuilt a `source_id -> summary` map by scanning 2,234 chunks for every request and also ran a BM25 ranking that fusion did not consume. The new implementation builds `title + topics + aliases + document_kind + first fragment` summaries once, persists source vectors under the corpus fingerprint, reuses the page Dense query vector, and scores only sources represented in the page Dense candidate pool.

| 120 questions, CPU | Source Recall@5 | Question hit | MRR | nDCG@5 | Page hit | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense default | 0.8694 | 0.9000 | 0.8479 | 0.8394 | 0.7890 | 212.173 ms |
| Optimized Source Fusion | 0.8778 | 0.9083 | 0.8575 | 0.8470 | 0.7890 | 223.002 ms |

The P95 increase is 5.1%, inside the 10% target. Source Fusion is therefore exposed as `RETRIEVAL_TIER=precision`; `default` remains Dense and `fast` is BM25.

## P1: Tagged slices

The evaluator now derives generic query-shape tags rather than question-ID rules: `mechanism_explanation`, `exact_table_term`, `ocr_scan`, `cross_source`, `moose_method`, and `general`. The report emits source hit, page hit and MRR for every populated slice. This makes future routing decisions auditable by query form.

## P2: Lexical-confidence routing

The router requires an explicit exact technical shape plus at least two stable Latin technical tokens. Generic Chinese words such as “表格” never force BM25. On the 120-question set it selected BM25 for 4 exact-document queries (FIATC Variables/Equation/Mole Fractions and mARC) and Dense for 116.

| 120 questions, CPU | Source Recall@5 | MRR | nDCG@5 | Page hit | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.8694 | 0.8479 | 0.8394 | 0.7890 | 212.173 ms |
| Lexical-confidence route | 0.8694 | 0.8479 | 0.8394 | 0.7982 | 217.114 ms |

The experiment has no quality regression and improves exact-table-term page hit from 0.8571 to 1.0000. It is retained as a constrained routing policy; it is not applied to generic mechanism, OCR, or Chinese-only questions.

## P3/P4: Cost contract and regression

Evidence cards expose `service_tier`, active retrieval mode, queue waits and backend. `fast/default/precision` map to `bm25/dense/source_fusion`. The self-hosted workflow now runs Dense 120-question regression and routing evaluation. `regression_gate` blocks release when the 120-question count, Source Recall@5 (< 0.8694), Page hit (< 0.7890), or default Dense P95 (> 260 ms) violates its baseline. The P95 limit is intentionally a machine-variance guard, not a claim that every hardware profile has identical latency.
