# Experimental results

## Retrieval results currently available

The previous completed run evaluated 300 SciFact and 323 NFCorpus queries.
The official BEIR aggregate output is in
`backend/evaluation/results/processed/official_beir_metrics.json`.

The previously observed official values were:

| Dataset | System | NDCG@10 | Recall@10 | P@10 |
|---|---|---:|---:|---:|
| SciFact | B1 BM25 | 0.59260 | 0.72389 | 0.07867 |
| SciFact | B2 Dense | 0.67415 | 0.81211 | 0.09033 |
| SciFact | B3 Hybrid | 0.67984 | 0.81189 | 0.08967 |
| SciFact | B4 Hybrid + reranker | 0.66130 | 0.77867 | 0.08600 |
| NFCorpus | B1 BM25 | 0.27246 | 0.13738 | 0.19164 |
| NFCorpus | B2 Dense | 0.32233 | 0.16762 | 0.24056 |
| NFCorpus | B3 Hybrid | 0.32147 | 0.16579 | 0.22910 |
| NFCorpus | B4 Hybrid + reranker | 0.29217 | 0.13540 | 0.17430 |

These values are retained as historical evidence. Because the earlier B4
configuration returned only eight reranked documents, the earlier B4
Recall@20 comparison is superseded. Rerun the benchmark after the depth fix
before using updated values in a paper.

## Statistical reporting rule

Significance tests must be generated from the same per-query metric definition
used for the paper table. The revised analyzer uses graded BEIR nDCG gain
(`2^relevance - 1`) and aligned query IDs. Do not combine old custom-metric
significance files with new official-metric tables.

## Unavailable results

No results are reported here for QA answer quality, grounding, context
degradation, Longformer, external RAG frameworks, scalability, memory
profiling, or Gemini generation. These require additional implementations,
models, API access, or controlled experiments and remain explicit limitations.

