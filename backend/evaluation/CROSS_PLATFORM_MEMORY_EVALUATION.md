# Cross-platform external-memory evaluation

## Purpose

The product hypothesis is that a user can create conversational memories on one
platform and retrieve them while interacting through another platform, without
re-explaining the same context. This experiment evaluates the **retrieval
component** of that hypothesis. It does not claim to prove answer-quality gains.

## Run

From `backend/`:

```bash
python -m evaluation.runners.run_memory_benchmark \
  --input evaluation/datasets/cross_platform_memory.example.jsonl \
  --output-dir evaluation/results/memory
```

The checked-in JSONL is a tiny smoke-test example, not a research benchmark.
Replace it with a consented, de-identified and independently labelled dataset
before reporting scientific metrics.

The runner uses the existing `ConversationStore` and
`EmbeddingGenerator`. Memories retain their source-platform metadata. Queries
are issued with no platform filter, matching the shared-memory use case. The
runner reports Recall@K, MRR, cross-platform Recall@K, mean/p50/p95 retrieval
latency, vector count, chunk/session count, and on-disk store size. Per-query
rankings are preserved in `memory_retrieval_results.json`.

## Input contract

One JSON object per line:

```json
{
  "query_id": "q1",
  "query": "Which timezone should be used for my calendar?",
  "query_platform": "gemini",
  "memories": [
    {
      "memory_id": "m1",
      "platform": "chatgpt",
      "text": "The user's preferred timezone is Asia/Kolkata."
    }
  ],
  "relevant_memory_ids": ["m1"]
}
```

The same memory ID must have consistent text and source platform wherever it
appears. Gold relevance must be annotated independently of system output.
Use consented, de-identified examples; do not commit personal conversation
exports or secrets.

## Interpretation and controls

- `Recall@K`: fraction of gold-relevant memory IDs present in the top K.
- `MRR@K`: reciprocal rank of the first relevant memory, averaged over queries.
- `cross_platform_Recall@K`: recall restricted to relevant memories whose
  source platform differs from the query platform.
- Latency is local retrieval latency, not end-to-end LLM response latency.
- Store size is measured after ingestion and excludes the embedding cache.
- Run on a fixed machine and report OS, CPU/RAM, Python, package versions,
  embedding model revision, corpus size, and run timestamp.
- Use a held-out test split. Do not tune chunking, K, or thresholds on the test
  set. Report per-platform breakdowns as well as the overall mean.
- Run at least three independent clean-store repetitions for latency; report
  median and spread. Retrieval metrics are deterministic for fixed embeddings,
  corpus, and settings.

## Limitations

This is a retrieval-only experiment using a labelled JSONL dataset. It does
not test LLM answer correctness, whether a memory is stale or contradictory,
identity/account linking, real provider integrations, privacy controls, or
production-scale concurrency. Those require separate experiments and must not
be inferred from these metrics.

## LongMemEval and answer-quality evaluation

LongMemEval should be evaluated separately with its official dataset and
scoring protocol. The system must use the benchmark's actual conversation
histories, a predeclared memory ingestion/retrieval policy, a fixed LLM and
generation settings, and the same answer judge for every arm. Compare at least:
1. no external memory, with a clearly specified context budget;
2. external memory enabled, with the same model, prompt template, and answer
   budget.

Preserve question-level outputs, retrieved memory IDs, prompts/context hashes,
token counts, latency, model identifier, and judge scores. Do not call the
retrieval Recall@K from this runner an answer-accuracy result.

## Additional reviewer experiments

- Longformer: define a fair long-context baseline with equivalent source
  history, truncation/token budget, and answer-generation settings. Longformer
  is an encoder architecture and should not be described as a drop-in
  generative RAG system.
- Advanced RAG: select reproducible baselines with pinned versions and identical
  corpus, query set, and candidate budget.
- Scalability: repeat memory retrieval at increasing corpus sizes and report
  latency percentiles, throughput, RAM, index bytes, and failure rate.
- Security/governance: separately test tenant/user isolation, authorization,
  deletion propagation (including backups/caches), retention, encryption in
  transit/at rest, auditability, and export. Do not claim these controls from
  the current local FAISS prototype.
