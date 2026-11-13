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
  run. These are preserved under `backend/evaluation/results/processed/`.

## Implemented in this revision

- Candidate generation is now 100 chunks for BM25, dense, and hybrid
  retrieval, with 100-item reranking candidate pool.
- All configurations collapse candidates to unique parent documents and apply
  a shared 20-parent evaluation depth.
- B4 reranks one representative chunk per unique parent and returns up to 20
  parents.
- Raw records now include candidate, parent-deduplication, filtering,
  requested-depth, returned-count, and shortfall diagnostics.
- Analysis validates that all four configurations contain the same query IDs.
- A timestamped run directory preserves each corrected run and its metadata.

## Required rerun

The existing raw run predates the B4 depth fix. It must be treated as
superseded for Recall@20 and rerun before publication:

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

## Not completed

No measured results currently exist in this repository for QA benchmarks,
Gemini answer generation, Longformer, advanced RAG frameworks, context
degradation, scalability, memory profiling, or security controls beyond the
existing application tests. Those items must be reported as limitations until
their experiments are actually run.
