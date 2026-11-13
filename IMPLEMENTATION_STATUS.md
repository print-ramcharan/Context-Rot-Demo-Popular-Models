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
- Official BEIR aggregate metrics and paired tests were generated from that
  historical run. These are preserved under
  `backend/evaluation/results/processed/` and are explicitly superseded for
  the corrected B4 depth comparison.

## Implemented in this revision

- Candidate generation is now 100 chunks for BM25, dense, and hybrid
  retrieval, with 100-item reranking candidate pool.
- All configurations collapse candidates to unique parent documents and apply
  a shared 20-parent evaluation depth.
- B4 reranks one representative chunk per unique parent and returns up to 20
  parents. The reranker writes its model score into `score`, and the runner
  preserves that ordering.
- Raw records include candidate, parent-deduplication, filtering,
  requested-depth, returned-count, and shortfall diagnostics.
- Analysis validates that all four configurations contain the same query IDs
  and applies Holm-Bonferroni adjustment across the planned paired tests.
- Timestamped run directories preserve corrected benchmark outputs and metadata.
- Cross-platform memory retrieval harness reports Recall@K, MRR,
  cross-platform Recall@K, retrieval latency percentiles, and local store size.
- LongMemEval evidence-session retrieval runner uses isolated per-question
  stores and reports gold session Recall@K/MRR and resource diagnostics.

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

## Not completed

No measured results currently exist for LongMemEval answer quality, Gemini
answer generation, context-degradation curves, Longformer, advanced RAG
framework comparisons, RAM profiling, production-scale concurrency, or
end-to-end security controls. The memory harnesses are retrieval-only and do
not establish improved generated answers. These remain limitations until
their experiments are executed on appropriate datasets and infrastructure.
