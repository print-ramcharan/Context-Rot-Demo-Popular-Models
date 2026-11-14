# Response to reviewers

**Manuscript:** Mitigating Context Degradation in Large Language Models Using Hybrid Retrieval and External Memory

We thank both reviewers for identifying weaknesses in the evaluation, reproducibility, comparison scope, and security discussion. The revision strengthens the retrieval evaluation and makes the implementation and evidence boundaries explicit. We do not claim that unexecuted experiments have produced results.

## Reviewer 1

### 1. Limited evaluation and lack of standard benchmark validation

**Response:** We added a corrected BEIR evaluation using SciFact (300 queries) and NFCorpus (323 queries). Four configurations are evaluated under a common candidate-pool and parent-document evaluation protocol: BM25, hybrid retrieval with cross-encoder reranking, hybrid retrieval without reranking, and dense FAISS retrieval. The paper-mapped configurations are B1–B4 and are documented in the repository's `EXPERIMENTAL_RESULTS.md`. The run preserves per-query outputs and run metadata.

### 2. Limited quantitative metrics and statistical analysis

**Response:** The revised analysis reports nDCG@10, MAP@10, Recall@10, Precision@10, and MRR@10. Paired per-query comparisons, bootstrap confidence intervals, Wilcoxon signed-rank tests, and Holm correction are included. The results show dataset-dependent trade-offs: dense retrieval is strongest on SciFact, while NFCorpus has different metric leaders. We therefore avoid claiming that the reranked hybrid system is universally superior.

### 3. Limited comparison with Longformer and advanced RAG approaches

**Response:** The manuscript clarifies the relationship to long-context models and advanced retrieval-augmented generation systems in related work. However, no matched implementation and experimental comparison against Longformer/LED, Self-RAG, CRAG, or RAPTOR has been completed. We identify this as a limitation rather than presenting literature discussion as an experimental comparison.

### 4. Reproducibility details

**Response:** The repository now documents the benchmark datasets and query counts, configuration mapping, candidate pool, evaluation depth, embedding and reranker identifiers, random seed, runtime environment, timestamped output structure, and commands for post-processing an existing run. The benchmark encoder used in BEIR is distinguished from the application's default embedding model. Raw per-query records and provenance metadata are retained.

### 5. Privacy and security

**Response:** The implementation branch now adds signed, expiring bearer-token authentication for private API routes, PBKDF2-HMAC-SHA256 password verification, explicit CORS origin allowlisting, upload-size limits, JSON-based conversation persistence, and principal-scoped conversation-memory operations. The development-only demo credential is `demo / 123456`. These are baseline application controls, not a security certification. The configured identity is currently a single application account; the document corpus remains shared, local conversation files are not encrypted at rest, and token revocation, distributed rate limiting, MFA, and retention enforcement remain unimplemented. New regression tests and a static source audit are included but have not yet been executed against the latest implementation commit. No penetration test has been performed.

## Reviewer 2

### 1. Limited novelty

**Response:** We clarify that the contribution is the integration and evaluation of an evidence-oriented pipeline combining sparse and dense retrieval, reciprocal-rank fusion, cross-encoder reranking, context compaction, evidence ordering, token-budget management, and external conversational memory. We do not claim novelty for the individual BM25, FAISS, RRF, or cross-encoder algorithms.

### 2. Lack of standard datasets and retrieval metrics

**Response:** The revised retrieval evaluation uses BEIR SciFact and NFCorpus and reports nDCG@10, MAP@10, Recall@10, Precision@10, and MRR@10 across four retrieval configurations. Dataset sizes and the configuration mapping are stated in the experimental-results record.

### 3. Missing latency, memory, and scalability evaluation

**Response:** Retrieval-stage latency percentiles are reported separately from LLM generation latency. The repository also contains an executable resource/scalability runner that measures index and embedding build time, FAISS footprint, embedding bytes, estimated BM25 object size, process RSS, and retrieval latency across corpus fractions. This runner has not yet produced a completed scalability study; no measured RAM or production-concurrency claim is made.

### 4. Comparison with advanced RAG and long-context systems

**Response:** Related work is expanded and the absence of matched system-level comparisons is explicitly acknowledged. A fair comparison requires the same datasets, query sets, generation model, prompt budgets, and scoring protocol. Such a comparison is not reported as completed.

### 5. Statistical validation

**Response:** The revised per-query analysis uses paired comparisons and Holm correction. The statistical results are reported per dataset and metric. They do not support a universal claim that the reranked hybrid configuration is best.

### 6. Data governance and security

**Response:** The prototype now enforces bearer authentication on private routes and scopes conversation-memory retrieval, listing, deletion, and statistics to the authenticated principal. However, it is still a single configured-account prototype with a shared document corpus; it does not provide full tenant isolation, encryption at rest, enterprise identity, token revocation, or retention enforcement. Regression tests have been added but remain pending execution, and the static source audit is not a substitute for penetration testing.

## Additional limitation: end-to-end context-degradation evidence

The repository includes a controlled context-degradation runner with question-only, recent-history, long-context, and external-memory conditions, configurable history sizes, prompt-token budgets, per-question outputs, and paired statistical analysis. The available execution is only a two-question pilot; it is insufficient for a publication-level answer-quality or context-degradation claim. We therefore restrict the quantitative conclusion to the completed BEIR retrieval results and identify full LongMemEval answer evaluation as future work.

## Summary of changes

- Added corrected BEIR retrieval results and statistical comparisons.
- Documented the paper-aligned B1–B4 mapping and corrected chunk-level reranking procedure.
- Added explicit latency interpretation and reproducibility metadata.
- Distinguished implemented evaluation harnesses from experiments with completed measurements.
- Narrowed novelty and context-degradation claims to what the evidence supports; documented implemented security controls and remaining deployment limitations in `SECURITY_AND_SCALABILITY.md`.
