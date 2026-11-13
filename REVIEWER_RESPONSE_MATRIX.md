# Reviewer-response matrix

| Reviewer concern | Evidence or implementation | Status |
|---|---|---|
| Compare sparse, dense, hybrid, and reranked retrieval | B1-B4 runner, SciFact/NFCorpus raw records and aggregate metrics | Measured; rerun required after B4 depth fix |
| Use standard retrieval metrics | Official BEIR aggregate analysis for nDCG@10, Recall@10, and P@10 | Implemented |
| Align paired statistics with reported metrics | Revised analyzer computes per-query graded-gain metrics from the same qrels | Implemented; rerun required |
| Evaluate Recall@20 fairly | B4 `rerank_top_n` changed from 8 to 20 and raw `retrieved_count` added | Implemented; rerun required |
| Report context degradation and QA quality | No completed controlled experiment or answer-quality evaluator is present | Not completed; limitation |
| Compare Longformer and advanced RAG systems | No completed fair implementation or run is present | Not completed; limitation |
| Measure latency, memory, and scalability | Existing latency fields are retrieval-run measurements only; no complete scalability report | Partial; do not claim full efficiency study |
| Evaluate persistent conversational memory | Existing `ConversationStore` has unit tests for storage, retrieval, filtering, and deletion | Partial; no benchmark-level memory study |
| Address authentication, encryption, retention, and isolation | Existing API has no completed end-to-end security control suite | Not completed; limitation |
| Provide reproducibility information | Commands, model identifiers, configuration, output paths, and provenance rules documented | Implemented |

The matrix deliberately distinguishes implementation from completed evidence.
An item marked as a limitation must not be described as an experimental result.
