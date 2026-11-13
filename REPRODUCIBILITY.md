# Reproducibility

## Retrieval benchmark

The reproducible retrieval commands are:

```bash
cd backend
python -m evaluation.datasets.download_beir
python -m evaluation.datasets.validate_data --datasets scifact nfcorpus
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
TOKENIZERS_PARALLELISM=false python -m evaluation.runners.run_retrieval
# Copy the timestamped run directory printed by the runner:
KMP_DUPLICATE_LIB_OK=TRUE python -m evaluation.analyze_results \
  --run-dir evaluation/results/runs/depth_audit_<timestamp>
python -m evaluation.report_depth_audit \
  --run-dir evaluation/results/runs/depth_audit_<timestamp>
```

The benchmark uses SciFact and NFCorpus test queries, 400-character chunks
with 80-character overlap, `all-MiniLM-L6-v2`, FAISS cosine search, BM25,
reciprocal-rank fusion, and `cross-encoder/ms-marco-MiniLM-L-6-v2` for B4.
Dense and hybrid first-stage retrieval use a 100-chunk candidate pool. BM25
continues down its exact score ranking until it has enough positive-scoring
unique parent documents for the requested depth. All systems are evaluated
at up to 20 unique parent documents; shortfalls are reported, never padded.

For B4, the cross-encoder scores candidate chunks (using document title plus
passage text) before chunks are collapsed to parent documents. The highest
cross-encoder-scoring passage represents each parent, and the top 20 unique
parents are returned. This preserves alternate passages during reranking.

## Provenance

The benchmark writes `backend/evaluation/results/raw/run_metadata.json`,
including the Git commit, experiment configuration hash, datasets, and UTC
timestamp. Per-query records are retained under the dataset directories.
Generated aggregate files are:

- `backend/evaluation/results/processed/metrics_table.csv`
- `backend/evaluation/results/processed/official_beir_metrics.json`
- `backend/evaluation/results/processed/paired_significance.json`

The working tree was already dirty before this revision. Therefore, the
provenance commit in a result file must be reported together with the working
tree status; a commit hash alone does not prove that the tree was clean.

## Environment limitations

The measured run used macOS Apple Silicon and Python 3.13. OpenMP runtime
settings are scoped to the benchmark command to avoid the observed macOS
duplicate-runtime abort. No claim is made that the same runtime or latency
will be reproduced on other hardware.
