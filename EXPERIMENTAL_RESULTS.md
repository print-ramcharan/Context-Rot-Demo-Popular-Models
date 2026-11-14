# Experimental results — corrected paper mapping

## Retrieval benchmark

Validated completed run: `depth_audit_20261114_191803`.

- Dataset: BEIR SciFact, 300 queries; NFCorpus, 323 queries.
- Candidate pool: 100.
- Evaluation depth: up to 20 unique parent documents; use the generated depth-audit report for the exact returned-depth/shortfall counts.
- Benchmark encoder: `BAAI/bge-base-en-v1.5`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Random seed: 42.
- Paper configuration mapping:
  - B1: BM25 only.
  - B2: BM25 + dense FAISS + RRF + cross-encoder reranker.
  - B3: BM25 + dense FAISS + RRF, without reranker.
  - B4: dense FAISS only.

The paper mapping is the analysis mapping. Legacy raw records were not rewritten; the analysis manifest records the validated in-memory mapping. Do not substitute the November 13 17:05 run: it predates the corrected chunk-before-parent reranking path.

### SciFact

| Metric | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| nDCG@10 | 0.646420 | 0.689490 | 0.702862 | 0.739591 |
| MAP@10 | 0.601561 | 0.642996 | 0.658176 | 0.690767 |
| Recall@10 | 0.767167 | 0.815556 | 0.825389 | 0.878000 |
| Precision@10 | 0.084667 | 0.091667 | 0.092667 | 0.098667 |
| MRR@10 | 0.615618 | 0.657766 | 0.669530 | 0.701873 |

### NFCorpus

| Metric | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| nDCG@10 | 0.304701 | 0.353486 | 0.360906 | 0.367779 |
| MAP@10 | 0.118127 | 0.137792 | 0.141657 | 0.138901 |
| Recall@10 | 0.148622 | 0.164587 | 0.177696 | 0.179730 |
| Precision@10 | 0.211765 | 0.248916 | 0.260372 | 0.276471 |
| MRR@10 | 0.515350 | 0.577343 | 0.575020 | 0.563475 |

B4 leads all five reported SciFact metrics. On NFCorpus, B4 leads nDCG@10, Recall@10, and Precision@10; B3 leads MAP@10; B2 leads MRR@10. B2 improves over BM25 on both datasets, but it is not the overall best configuration. These results support dataset-dependent retrieval trade-offs, not a universal claim that reranking wins.

## Paired statistical analysis

Paired tests use per-query differences and Holm correction across a 30-test family.

- SciFact: B1 vs B2 has no metric remaining significant after Holm correction. B3 vs B2 has no significant differences. B4 is significantly better than B2 for nDCG@10 (Holm p=0.010967), MRR@10 (0.038695), and MAP@10 (0.022025); Recall@10 (0.090460) and Precision@10 (0.090460) are not significant.
- NFCorpus: B2 significantly improves over B1 on Recall@10 (Holm p=2.279e-05), nDCG@10 (3.026e-08), Precision@10 (1.157e-05), MRR@10 (0.004317), and MAP@10 (1.008e-06). B2 vs B3 has no significant differences. B4 is significantly better than B2 for Recall@10 (0.004566) and Precision@10 (0.002274); the other three comparisons are not significant.

The statistics do not establish universal superiority of B2. Consult the run's paired-significance JSON for complete test statistics and confidence intervals.

## Retrieval latency

The manuscript's sampled retrieval-latency table reports 30 queries and 3 warm observations per query. Values are milliseconds; these are retrieval-stage timings, not end-to-end LLM response times.

| Dataset | Config | Mean | Median | P95 | P99 |
|---|---|---:|---:|---:|---:|
| NFCorpus | B1 | 2.840 | 1.034 | 14.570 | 17.850 |
| NFCorpus | B2 | 4593.512 | 4362.179 | 6614.010 | 11139.957 |
| NFCorpus | B3 | 3.250 | 2.537 | 8.244 | 10.011 |
| NFCorpus | B4 | 0.958 | 0.872 | 1.899 | 2.142 |
| SciFact | B1 | 12.144 | 10.936 | 27.195 | 29.072 |
| SciFact | B2 | 2920.055 | 2912.610 | 4308.332 | 5093.260 |
| SciFact | B3 | 13.542 | 11.638 | 29.001 | 33.003 |
| SciFact | B4 | 1.018 | 1.005 | 1.616 | 2.095 |

B2's cross-encoder stage has a substantial latency cost. These figures should be tied to the exact raw latency artifact and machine/runtime metadata in the final manuscript; do not present them as end-to-end generation latency.

## Context-degradation and external-memory QA

A two-question LongMemEval pilot was executed (run `20261113T151441Z`). It is a smoke/pilot run only, not a statistically powered benchmark. All four arms had exact match 0. External memory had token F1 0.403846; the other arms had token F1 0. This sample is too small for inferential claims. The full-history arm also did not fit every history session in its prompt budget. Do not use this pilot to claim that context degradation has been mitigated or that answer quality is significantly improved.

The controlled context-degradation runner and official LongMemEval hypothesis export are implemented. A full benchmark run with official LongMemEval LLM-judge scoring has not been completed.

## Comparisons and system-level evidence not yet available

No matched experimental comparison against Longformer/LED, Self-RAG, CRAG, RAPTOR, or another long-context model is reported. No production-scale concurrency study, full RAM/VRAM profile, or end-to-end security-control test suite has been completed. The repository contains runnable resource/scalability and evaluation harnesses, but a harness is not a measurement. Do not convert these pending items into results.

## Reproducibility environment

Recorded benchmark environment: macOS 26.5.2, Apple Silicon ARM64, Python 3.9.6, NumPy 2.0.2, sentence-transformers 5.1.2, FAISS CPU 1.13.0, BEIR 2.2.0, PyTorch 2.8.0. The benchmark encoder is `BAAI/bge-base-en-v1.5`; the application default encoder documented elsewhere is `all-MiniLM-L6-v2`. Keep these roles distinct.
