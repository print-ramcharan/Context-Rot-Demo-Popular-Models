# Reproducible BEIR retrieval benchmark

This benchmark evaluates the retrieval stack against the official BEIR SciFact and NFCorpus **test** splits.

## Run it on the application environment

From the repository root:

```bash
git fetch origin
git checkout implementation/evidence-grounded-rag
git pull --ff-only origin implementation/evidence-grounded-rag

cd backend
source venv/bin/activate

python -m pip install "beir>=2.0.0" "scipy>=1.11.0"
python -m evaluation.runners.run_retrieval --batch-size 25
```

The runner now creates the run directory immediately and writes one raw JSON checkpoint after each query/configuration. It prints progress every 25 queries. If the process is interrupted, copy the exact `RUN_DIR=...` path printed at startup and resume without repeating completed query/configuration work:

```bash
python -m evaluation.runners.run_retrieval --batch-size 25 --run-dir "<PASTE_THE_EXACT_RUN_DIR_HERE>"
```

Do not change the experiment configuration or code between the interrupted run and its resume. The final aggregate CSVs, paired statistics, latency summary, depth audit, and metadata are written after all configured datasets finish.

The first run downloads the official SciFact and NFCorpus archives into `backend/evaluation/data/`, and downloads the configured embedding and reranker models if they are not already cached. Keep the machine connected to the internet and allow the run to finish. No OpenRouter/API key is required: this is a retrieval-only benchmark.

The runner evaluates every query in each official test split across four configurations:

| ID | Configuration |
|---|---|
| B1 | BM25 |
| B2 | Dense FAISS |
| B3 | Hybrid BM25 + dense RRF |
| B4 | Hybrid RRF + cross-encoder reranking |

All configurations use the same dataset split, structure-aware chunking, BGE-base embeddings, candidate depth, and document-level qrels. Quality metrics are computed for every test query. Warm latency is repeated three times on a deterministic 30-query sample.

## Output

The command prints a `RESULTS_DIR=...` path. That run directory contains:

- `processed/metrics_table.csv`: mean Recall@5/10/20, MRR@10, nDCG@10 and Precision@10 for each dataset/configuration.
- `paired_significance.csv` and `.json`: paired B4-vs-B1/B2/B3 improvements, percentile-bootstrap 95% CIs, Wilcoxon signed-rank p-values and Holm-adjusted p-values.
- `latency_summary.json`: mean, median, standard deviation, p95 and p99 over the repeated warm timing sample.
- `depth_audit.csv` and `.json`: unique parent documents returned and the number of queries that fall short of the requested top-20 depth.
- `raw/<dataset>/*.json`: per-query retrieved document IDs, relevance grades, stage timings, depth diagnostics and metrics.
- `raw/run_metadata.json`: Git revision, experiment config hash, dataset checksums, package versions and environment details.

## Paper reporting rules

1. Report the actual query count from the run output; do not substitute the nominal dataset size.
2. Report document-level BEIR retrieval metrics, not chunk-level precision.
3. Report the depth audit. A top-20 metric must disclose any query that returned fewer than 20 unique parent documents.
4. Treat the repeated latency sample as descriptive system timing, not as 30 independent datasets or independent observations for significance testing.
5. Use the paired per-query statistics for method comparisons. Report both raw and Holm-adjusted p-values.
6. Do not call token counts, answer accuracy, faithfulness or hallucination rate results of this retrieval-only run. Those require a separate generation benchmark with provider usage and human/validated answer labels.
7. Do not combine these outputs with historical runs that used different embedding models, chunk sizes, or retrieval settings.

The BEIR archives are downloaded using the official BEIR dataset URLs and loader convention: https://github.com/beir-cellar/beir/wiki/Datasets-available
