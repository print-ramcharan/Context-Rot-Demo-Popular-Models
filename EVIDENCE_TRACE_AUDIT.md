# Evidence trace audit — research/ieee-evaluation

## Scope

This audit covers the live API evaluation trace from query embedding through dense retrieval, BM25, RRF, cross-encoder reranking, and context compaction. It is an implementation audit, not a claim that the benchmark has been executed or that the resulting metrics are publication-ready.

## Defects corrected in this revision

1. **Reranker rank provenance**
   - Preserve each candidate's pre-rerank rank as `original_rank`.
   - Assign a new one-based `rank` after cross-encoder sorting.
   - Trace final order and original order separately.

2. **Reranker failure handling**
   - Clear stale trace data before each rerank.
   - Raise an explicit error when the cross-encoder is unavailable.
   - The traced evaluation path now fails closed when reranking is configured but the model is absent; it cannot silently label unchanged candidates as reranked.
   - Trace reports whether reranking was enabled and actually applied.

3. **Passage identity**
   - Preserve the FAISS vector index with each candidate.
   - Fall back to that index when source metadata lacks a chunk index.
   - Keep source-plus-chunk identity in the trace where metadata is available.

4. **Deterministic RRF**
   - Resolve equal RRF scores using the stable vector index as a secondary sort key in both traced and ordinary hybrid retrieval.

5. **Ingestion cache invalidation**
   - Invalidate retrieval-result and BM25 caches after adding uploaded chunks and before rebuilding BM25. Previously, a non-clearing upload could leave BM25 and query caches describing the old corpus.

6. **Metric semantics**
   - Whitespace-delimited counts are now explicitly reported as word counts.
   - Model-token reduction, token budget utilization, and exact textual evidence retention are not reported as measured when the required model tokenizer or gold evidence spans are unavailable.
   - Source-document retention and chunk-identity retention are separate fields.

7. **Evaluation configuration**
   - The evaluation endpoint now passes the configured similarity threshold to the traced retriever.
   - The metric catalog distinguishes measured word-count proxies from unavailable model-token measurements.

## Regression tests added

- `backend/tests/test_reranker_trace.py`: verifies final and original ranks and fail-closed behavior.
- `backend/tests/test_retrieval_trace.py`: verifies reranked order, trace ranks, and passage identity.
- `backend/tests/test_context_quality_metrics.py`: verifies word-count labels, source/chunk retention, and the absence of fabricated token metrics.

## Remaining limitations

- Source-level relevance labels cannot establish which exact supporting sentence survived compaction.
- Chunk-identity retention only proves that a chunk identifier remains represented; it does not prove that the supporting text within that chunk survived sentence extraction.
- Exact model-token counts require the tokenizer for the selected generation model and must be recorded separately from whitespace word counts.
- The new tests have been added but have not been executed in this environment. Run the test commands below on the project environment before relying on this revision.

## Local validation

From `backend/`:

```bash
python -m pytest -q tests/test_reranker_trace.py tests/test_retrieval_trace.py tests/test_context_quality_metrics.py tests/test_metrics.py
python -m pytest -q
```

A successful test run validates software behavior only. It does not replace a fresh BEIR run, per-query trace inspection, or controlled answer-quality experiments.
