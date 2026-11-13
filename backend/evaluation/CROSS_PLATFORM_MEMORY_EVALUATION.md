# Cross-platform external-memory evaluation

## Purpose

The product hypothesis is that a user can create conversational memories on one
platform and retrieve them while interacting through another platform, without
re-explaining the same context. This experiment evaluates the **retrieval
component** of that hypothesis. It does not claim to prove answer-quality gains.

## Cross-platform labelled-memory benchmark

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
are issued with no platform filter, matching the shared-memory use case. It
reports Recall@K, MRR, cross-platform Recall@K, mean/p50/p95 retrieval latency,
vector count, chunk/session count, and on-disk store size. Per-query rankings
are preserved in `memory_retrieval_results.json`. Each run writes to a
timestamped store directory to avoid contaminating repeated runs.

## LongMemEval evidence-session retrieval

The official LongMemEval dataset contains 500 questions and provides
`haystack_sessions`, `haystack_session_ids`, and `answer_session_ids`.
The latter identifies gold evidence sessions. This repository runner measures
session-level evidence retrieval; it does not generate answers or invoke the
official LLM judge. Obtain the cleaned official dataset from
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned and keep the
dataset file outside version control.

```bash
cd backend
python -m evaluation.runners.run_longmemeval_retrieval \
  --dataset evaluation/data/longmemeval_s_cleaned.json \
  --output-dir evaluation/results/longmemeval \
  --top-k 10
```

The runner creates an isolated store for every question and reports
session-level Recall@K and MRR, by-question-type breakdowns, retrieval latency
percentiles, and store size. Use the official `longmemeval_s_cleaned.json`
rather than an older uncleaned copy. The official answer-quality evaluator is
separate; see the upstream repository's `src/evaluation/evaluate_qa.py`
instructions at https://github.com/xiaowu0162/LongMemEval.

## Input contract for the cross-platform benchmark

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

## Metric definitions and controls

- `Recall@K`: fraction of gold-relevant memory IDs present in the top K.
- `MRR@K`: reciprocal rank of the first relevant memory, averaged over queries.
- `cross_platform_Recall@K`: recall restricted to relevant memories whose
  source platform differs from the query platform.
- LongMemEval session Recall@K measures presence of gold evidence sessions,
  not answer correctness.
- Latency is local retrieval latency, not end-to-end LLM response latency.
- Store size is measured after ingestion and excludes the embedding cache.
- Run on a fixed machine and report OS, CPU/RAM, Python, package versions,
  embedding model revision, corpus size, and run timestamp.
- Use a held-out test split for the custom cross-platform benchmark. Do not tune
  chunking, K, or thresholds on the test set. Report per-platform breakdowns.
- Run at least three independent clean-store repetitions for latency; report
  median and spread.

## Limitations

Neither retrieval runner evaluates answer correctness, stale or contradictory
memories, identity/account linking, real provider integrations, privacy
controls, or production-scale concurrency. The LongMemEval runner does not
replace the official LLM-as-a-judge QA evaluation. Those claims require
separate experiments and must not be inferred from retrieval metrics.

## LongMemEval answer-quality experiment

For answer quality, use the official dataset and scoring protocol. Compare:
1. no external memory, with a clearly specified context budget;
2. external memory enabled, with the same LLM, prompt template, decoding
   settings, and answer budget.

Preserve question-level outputs, retrieved memory IDs, prompt/context hashes,
token counts, latency, model identifier, and judge scores. Use the same judge
for every arm. Do not treat retrieval Recall@K as answer accuracy.

## Additional reviewer experiments

- Longformer: define a fair long-context baseline with equivalent source
  history, truncation/token budget, and answer-generation settings. Longformer
  is an encoder architecture and should not be described as a drop-in
  generative RAG system.
- Advanced RAG: select reproducible baselines with pinned versions and identical
  corpus, query set, and candidate budget.
- Scalability: repeat retrieval at increasing corpus sizes and report latency
  percentiles, throughput, RAM, index bytes, and failure rate.
- Security/governance: separately test tenant/user isolation, authorization,
  deletion propagation (including backups/caches), retention, encryption in
  transit/at rest, auditability, and export. Do not claim these controls from
  the current local FAISS prototype.
