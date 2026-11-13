# Implementation status

## Evidence boundary

This status records only work observed in the repository or produced by a
completed command. Unrun experiments are not represented as results.

## Completed and measured

- SciFact and NFCorpus were downloaded from BEIR.
- Dataset validation completed:
  - SciFact: 5,183 documents, 300 queries.
  - NFCorpus: 3,633 documents, 323 queries.
  - Missing qrels document IDs: 0 for both datasets.
  - Chunks without parent IDs: 0 for both datasets.
- A complete B1-B4 retrieval run produced 1,200 SciFact and 1,292 NFCorpus
  per-query records.
- The November 13 run `depth_audit_20261113_170539` completed, but inspection
  after that run found that its B4 path still collapsed chunks to parent
  documents before reranking. Its B4 scores therefore do NOT evaluate the
  corrected chunk-level reranking implementation added afterward. Preserve
  those outputs as historical results; rerun B1-B4 before making claims about
  the corrected reranker.

## Implemented in this revision

- Dense and hybrid retrieval use a 100-chunk candidate pool. BM25 scans its
  exact score ranking until it has enough positive-scoring unique parents for
  the requested evaluation depth, rather than stopping at 100 chunks.
- All configurations are evaluated at a shared 20-parent depth; any genuine
  shortfall is reported rather than padded.
- B4 now reranks the original candidate chunks before parent-document
  aggregation. It scores title-plus-passage text, then retains the highest
  cross-encoder-scoring passage for each parent. A regression test verifies
  that alternate passages from the same parent reach the cross-encoder.
- Reranking latency is measured during the actual B4 pass rather than by
  running a second, duplicate reranking pass for timing.
- B4 now fails explicitly if the cross-encoder failed to load, rather than
  silently producing an unreranked result under the reranker label.
- The per-query NDCG implementation uses graded gains and the complete query
  qrels for its ideal ranking, aligning the CSV metric with standard BEIR
  NDCG rather than deriving the ideal ranking from retrieved documents.
- Raw records include candidate, parent-deduplication, filtering,
  requested-depth, returned-count, and shortfall diagnostics.
- Analysis validates that all four configurations contain the same query IDs
  and applies Holm-Bonferroni adjustment across the planned paired tests.
- Timestamped run directories preserve corrected benchmark outputs and metadata.
- Cross-platform memory retrieval harness reports Recall@K, MRR,
  cross-platform Recall@K, retrieval latency percentiles, and local store size.
- LongMemEval evidence-session retrieval runner uses isolated per-question
  stores and reports gold session Recall@K/MRR and resource diagnostics.
- A controlled context-degradation QA runner is implemented. It compares
  question-only, recent-history, and retrieved external-memory conditions over
  increasing history sizes, enforces a provider-tokenized prompt budget, and
  saves per-question predictions, token F1/exact-match diagnostics, token use,
  and generation latency. It has not yet been executed on the full benchmark.

## Required execution

The existing raw BEIR run predates the B4 depth fix. Rerun before publication:

```bash
cd backend
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
TOKENIZERS_PARALLELISM=false python -m evaluation.runners.run_retrieval
KMP_DUPLICATE_LIB_OK=TRUE python -m evaluation.analyze_results \
  --run-dir evaluation/results/runs/depth_audit_<timestamp>
python -m evaluation.report_depth_audit \
  --run-dir evaluation/results/runs/depth_audit_<timestamp>
```

Run the cross-platform harness with an independently labelled dataset (the
checked-in example is only a smoke-test fixture):

```bash
cd backend
python -m evaluation.runners.run_memory_benchmark \
  --input evaluation/datasets/cross_platform_memory.example.jsonl \
  --output-dir evaluation/results/memory
```

Run LongMemEval evidence-session retrieval after downloading the official
cleaned S dataset from the upstream Hugging Face dataset:

```bash
cd backend
python -m evaluation.runners.run_longmemeval_retrieval \
  --dataset evaluation/data/longmemeval_s_cleaned.json \
  --output-dir evaluation/results/longmemeval \
  --top-k 10
```

## Newly implemented experiment — execution required

From `backend/`, first run a small smoke test using a valid provider API key
(or a locally available HuggingFace model):

```bash
python -m evaluation.runners.run_context_degradation \\
  --dataset evaluation/data/longmemeval_s_cleaned.json \\
  --output-dir evaluation/results/context_degradation \\
  --history-sizes 5,10,20,all \\
  --max-prompt-tokens 4096 \\
  --limit 5
```

After inspecting the smoke-test outputs, run the full dataset with `--limit 0`.
The runner uses the provider and model from `config.yaml` unless overridden.
OpenAI requires `tiktoken` for exact prompt token counting. Ollama is currently
rejected because this adapter does not have a model-matched tokenizer. Exact
match and token F1 are diagnostics, not substitutes for the official
LongMemEval LLM-judge score.

## Not completed

The context-degradation runner is implemented but has not been executed; no
answer-quality or context-degradation results exist yet. The corrected
chunk-level B4 runner also requires a fresh BEIR run; the November 13 17:05
run predates that final fix. The official
LongMemEval LLM-judge evaluation, Longformer/long-context model comparison,
advanced RAG framework comparisons, RAM profiling, production-scale
concurrency, and end-to-end security controls remain unimplemented or
unmeasured. Do not claim these results until completed.
