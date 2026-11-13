# Experimental results

## Retrieval results currently available

The validated completed run evaluated 300 SciFact and 323 NFCorpus queries under the standardized 100-candidate pool and 20-parent evaluation depth (`backend/evaluation/results/runs/depth_audit_20261102_033532`).
The official BEIR aggregate output is preserved in `backend/evaluation/results/processed/official_beir_metrics.json`.

The verified official values are:

| Dataset | System | NDCG@10 | Recall@10 | P@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| SciFact | B1 BM25 | 0.59361 | 0.72722 | 0.07900 | 0.55698 |
| SciFact | B2 Dense | 0.67517 | 0.81444 | 0.09100 | 0.63904 |
| SciFact | B3 Hybrid | 0.67834 | 0.81500 | 0.09033 | 0.64407 |
| SciFact | B4 Hybrid + reranker | 0.64913 | 0.78711 | 0.08800 | 0.61385 |
| NFCorpus | B1 BM25 | 0.27369 | 0.13834 | 0.19350 | 0.46141 |
| NFCorpus | B2 Dense | 0.32632 | 0.17010 | 0.24737 | 0.50580 |
| NFCorpus | B3 Hybrid | 0.32860 | 0.17034 | 0.23653 | 0.53325 |
| NFCorpus | B4 Hybrid + reranker | 0.34030 | 0.16544 | 0.24396 | 0.55631 |

## Statistical validation summary

Both the aggregate metrics and per-query significance tests (paired Wilcoxon signed-rank and 95% bootstrap confidence intervals) are computed directly from official BEIR evaluator logic (`pytrec_eval` with standard linear gain $gain(rel) = rel$).

- **B1 (BM25) vs B3 (Hybrid):** Hybrid retrieval demonstrates statistically significant improvements over BM25 across all four metrics on both datasets after Holm-Bonferroni correction ($p_{holm} < 10^{-4}$).
- **B3 (Hybrid) vs B4 (Hybrid + Reranker):** Adding the cross-encoder reranker does **not** demonstrate statistically significant improvement over hybrid retrieval after Holm-Bonferroni correction ($p_{holm} > 0.05$ on all metrics for both datasets). On SciFact, reranking slightly reduces NDCG@10 (-0.0292) and Recall@10 (-0.0279). On NFCorpus, the apparent NDCG gain (+0.0117) has a 95% bootstrap CI of `[-0.0014, 0.0245]` which spans zero, failing statistical significance ($p_{holm} = 0.1148$).

## Unavailable results (Explicit Research Boundaries)

In accordance with our research validity framework, no experimental claims are made for:
- End-to-end LLM answer quality, factual consistency, or grounding metrics.
- Context degradation mitigation curves across varying interaction/context lengths.
- External memory retention and multi-session retrieval benchmarks.
- External RAG framework comparisons or long-context model baselines.

These represent Gates 3–5 in our validation protocol and require isolated, controlled experiments before any claims are asserted.

