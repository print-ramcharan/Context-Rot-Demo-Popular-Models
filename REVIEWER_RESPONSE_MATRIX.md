# Reviewer-response matrix

This matrix separates completed implementation, executed evidence, and unresolved experiments.

| Reviewer concern | Repository evidence / action | Status |
|---|---|---|
| Standard retrieval benchmarks | Corrected BEIR run on SciFact (300 queries) and NFCorpus (323 queries), four paper-mapped retrieval configurations | Measured |
| Sparse, dense, hybrid, and reranked comparison | B1 BM25; B2 hybrid + cross-encoder; B3 hybrid without reranker; B4 dense FAISS | Measured |
| Quantitative retrieval metrics | nDCG@10, MAP@10, Recall@10, Precision@10, MRR@10 | Measured |
| Statistical validation | Paired per-query tests, bootstrap confidence intervals, Wilcoxon tests, Holm correction | Measured; dataset-dependent outcomes |
| Retrieval latency | 30 sampled queries × 3 warm observations; mean/median/P95/P99 | Reported in manuscript; keep distinct from generation latency |
| Fair reranking implementation | Cross-encoder scores candidate chunks before parent-document collapse; regression coverage exists | Implemented; corrected run completed |
| Reproducibility | Timestamped run, metadata, model IDs, candidate pool, seed, output artifacts, post-processing command | Substantially addressed |
| Long-context / context-degradation QA | Controlled runner supports increasing history sizes and four conditions; only a two-question pilot has been executed | Harness implemented; full evidence pending |
| Longformer / LED / advanced RAG comparison | No matched implementations and measurements | Unresolved |
| Cross-platform memory benchmark | Runner exists; checked-in JSONL is only a smoke fixture | Harness implemented; labelled evaluation data/results pending |
| Memory and scalability | Resource runner measures build/index sizes, RSS and latency at corpus scales | Harness implemented; execution/results pending |
| Privacy and security | Current API has permissive CORS and conversation-store files use pickle; authentication, tenant isolation, encryption and deletion controls are not established by the current code | Unresolved; must not claim secure deployment |
| Novelty | Contribution framed as an integrated, evaluated engineering pipeline, not a new BM25/dense/RRF/reranking algorithm | Reframing required |

## Important result interpretation

- SciFact: B4 (dense FAISS) is highest on all five @10 metrics.
- NFCorpus: B4 is highest on nDCG@10, Recall@10 and Precision@10; B3 is highest on MAP@10; B2 is highest on MRR@10.
- B2 improves over B1 on both datasets, but B2 is not universally best.
- The LongMemEval two-question pilot does not establish end-to-end answer-quality improvement or context-degradation mitigation.
- A runnable benchmark or test harness is evidence of implementation, not evidence that the experiment has been executed.

Use `EXPERIMENTAL_RESULTS.md` for exact reported values and limitations.
