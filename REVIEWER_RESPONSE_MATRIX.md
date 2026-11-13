# Reviewer-response matrix

| Reviewer concern | Evidence or implementation | Status |
|---|---|---|
| Compare sparse, dense, hybrid, and reranked retrieval | B1-B4 runner, SciFact/NFCorpus raw records and aggregate metrics | Historical measurements; corrected rerun still required |
| Use standard retrieval metrics | Official BEIR aggregate analysis for nDCG@10, Recall@10, and P@10 | Implemented; corrected run pending |
| Align paired statistics with reported metrics | Revised analyzer computes per-query graded-gain metrics from the same qrels and applies Holm correction | Implemented; corrected run pending |
| Evaluate Recall@20 fairly | B4 reranks up to 20 unique parent documents and records returned depth | Implemented; corrected run pending |
| Evaluate cross-platform external memory | New runner reuses ConversationStore and measures Recall@K, MRR, cross-platform Recall@K, latency, and local store size | Harness implemented; labelled representative dataset and execution required |
| Report context degradation and QA quality | No completed controlled experiment or answer-quality evaluator is present | Not completed |
| Evaluate long-term conversational memory | LongMemEval execution and official scoring not integrated or run | Not completed |
| Compare Longformer and advanced RAG systems | No completed fair implementation or run is present | Not completed |
| Measure latency, memory, and scalability | Retrieval latency and local store-size measurements are now available in the memory harness; no RAM/concurrency/scaling study is complete | Partial |
| Address authentication, encryption, retention, and isolation | No completed end-to-end security control suite | Not completed |
| Provide reproducibility information | Commands, model identifiers, configuration, output paths, and provenance rules documented | Implemented; record environment for each new run |

The checked-in `cross_platform_memory.example.jsonl` is a smoke-test fixture,
not a statistically meaningful dataset. Do not cite its scores as evidence.
This matrix deliberately distinguishes implementation from completed evidence.
An item marked as a limitation must not be described as an experimental result.
