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
  corrected chunk-level reranking implementation. Preserve those outputs as
  historical results; use the complete November 14 run documented below for
  corrected reranker analysis.

## Implemented in this revision

- Dense and hybrid retrieval use a 100-chunk candidate pool. BM25 scans its
  exact score ranking until it has enough positive-scoring unique parents for
  the requested evaluation depth, rather than stopping at 100 chunks.
- All configurations are evaluated at a shared 20-parent depth; any genuine
  shortfall is reported rather than padded.
- The historical runner mapping assigned reranking to B4. In the current
  paper-aligned mapping, this same chunk-before-parent reranking path is B2.
  It scores title-plus-passage text, then retains the highest
  cross-encoder-scoring passage for each parent. A regression test verifies
  that alternate passages from the same parent reach the cross-encoder.
- Reranking latency is measured during the actual reranking pass rather than
  by running a second, duplicate reranking pass for timing (historical B4;
  current paper-mapped B2).
- The paper-aligned B2 now fails explicitly if the cross-encoder failed to
  load, rather than silently producing an unreranked result under the reranker label.
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

## Historical BEIR results: reuse before rerunning

The complete run `depth_audit_20261114_191803` is preserved with raw per-query
records and `raw/run_metadata.json`. Its recorded code commit contains the
chunk-before-parent cross-encoder reranking path. In that historical runner
mapping, raw B2 is dense-only and raw B4 is hybrid plus reranking. The current
paper mapping reverses those two IDs.

The rank-preserving analyzer validates the legacy run metadata and checks the
reranking-latency signature across both datasets before applying the B2/B4
swap in memory. It does not rename, rewrite, or delete raw JSON. It writes new
metrics and statistics under `processed_rank_preserving/`. The depth audit
uses the same validated mapping and writes
`DEPTH_AUDIT_REPORT_PAPER_MAPPING.md`, preserving any earlier report.

From the repository root, run this single post-processing command. It uses
the existing backend virtual environment when available and does not execute
retrieval, load embedding models, or call an LLM API:

```bash
bash backend/evaluation/run_existing_analysis.sh
```

To analyze another compatible run, pass its run directory as the first argument.

The analyzer checks full query coverage against the installed BEIR datasets
and stops if the run is incomplete or its mapping cannot be proven. The
November 13 run `depth_audit_20261113_170539` predates the final
chunk-before-parent reranking fix and must not be substituted for this run.

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
answer-quality or context-degradation results exist yet. The November 14
`depth_audit_20261114_191803` run contains the corrected chunk-before-parent
reranking implementation under the historical B2/B4 IDs and can be analyzed
without rerunning retrieval. The November 13 17:05 run predates that fix.
The official LongMemEval LLM-judge evaluation, Longformer/long-context model
comparison, advanced RAG framework comparisons, RAM profiling,
production-scale concurrency, and end-to-end security controls remain
unimplemented or unmeasured. Do not claim these results until completed.


## Paper-revision work on the current branch

- The fresh benchmark runner now uses the paper's configuration labels:
  - B1: BM25 only.
  - B2: Hybrid RRF + cross-encoder reranking.
  - B3: Hybrid RRF without reranking.
  - B4: Dense FAISS only.
- The paired-statistics runner and rank-preserving analyzer now compare B2 against B1, B3, and B4.
- Every new BEIR run records its exact B1-B4 mapping in `raw/run_metadata.json`. For pre-manifest runs, the analyzer supports only the known legacy B2/B4 reversal after validating the recorded reranker setting, matching query IDs, and per-query reranking-latency signature on both datasets.
- Legacy mapping is applied only to the in-memory analysis view. Raw per-query records and historical summaries remain unchanged; the applied paper-to-raw mapping is recorded in `processed_rank_preserving/analysis_manifest.json`.
- The complete November 14 run `depth_audit_20261114_191803` can be reused for official retrieval metrics, paired bootstrap confidence intervals, Wilcoxon tests, Holm correction, and the corrected depth audit. No retrieval rerun is required to produce these outputs.
- Added `evaluation/runners/run_resource_scalability.py` to measure index build time, embedding build time, BM25 build time, FAISS disk footprint, embedding bytes, estimated BM25 Python-object size, process RSS, and per-configuration retrieval latency over increasing SciFact/NFCorpus corpus fractions. It runs each scale in a separate subprocess. This is an experiment harness, not a completed result.

### Commands for the completed BEIR run

These are post-processing only. Do not run `run_retrieval` for this existing
benchmark:

```bash
cd backend
python -m evaluation.analyze_results \\
  --run-dir evaluation/results/runs/depth_audit_20261114_191803
python -m evaluation.report_depth_audit \\
  --run-dir evaluation/results/runs/depth_audit_20261114_191803
```

The analyzer writes official BEIR metrics, per-query metrics, paired bootstrap
confidence intervals, Wilcoxon tests, Holm-adjusted p-values, and a provenance
manifest to `processed_rank_preserving/`. The depth audit writes a separate
paper-mapped report. Neither command performs retrieval or makes API calls.

### Remaining evidence not yet produced

- LongMemEval answer-generation experiment: runner exists, but a full benchmark execution and official LLM-judge scoring are still required. Exact match and token F1 are diagnostics only.
- Longformer/LED and Self-RAG/CRAG/RAPTOR comparisons: no matched experimental results are currently present. Related-work discussion is not an experimental comparison.
- Cross-platform memory: the checked-in example JSONL is a smoke fixture, not a publication dataset. Supply a consented, de-identified, independently labelled evaluation set before reporting results.
- Security: the paper can describe required safeguards and limitations, but actual tenant-isolation, authorization, deletion, prompt-injection, and encryption tests must be run against the deployed authenticated storage/API. The current conversation store's session metadata alone does not prove user/tenant isolation.
- Resource/scalability results: the new harness has not been executed. Its RSS measurements include the loaded model/runtime, and its BM25 object size is an estimate; report these definitions with any resulting table.

No new benchmark values are asserted in this status file. Only completed runs with preserved raw per-query records, run metadata, and analysis outputs may be used as results.
