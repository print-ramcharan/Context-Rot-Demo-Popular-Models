# LongMemEval-S evaluation protocol

This experiment evaluates the paper's central claim: hybrid retrieval and external
conversation memory can reduce context degradation under a constrained reader
context budget.

## Dataset

Use the cleaned official LongMemEval-S release, not the oracle split. The downloader
validates the 500-question JSON array, required fields, unique question IDs, and
aligned session arrays, then prints the SHA-256 digest.

From `backend/`:

```bash
python evaluation/download_longmemeval.py
```

Do not commit the dataset file or API keys.

## Smoke test

Run from `backend/` after configuring the selected provider in `config.yaml`:

```bash
python -m evaluation.runners.run_context_degradation \
  --dataset evaluation/data/longmemeval_s_cleaned.json \
  --output-dir evaluation/results/context_degradation_smoke \
  --history-sizes all \
  --max-prompt-tokens 4096 \
  --long-context-max-prompt-tokens 32768 \
  --limit 1
```

This is an integration check only; do not report smoke-test metrics as benchmark
results.

## Main run

The comparison uses the five most recent sessions for `recent_history`, all
available sessions for `long_context`, and a full-history index for
`external_memory`. The context packer distributes the available budget across
sessions and records the exact context text and source-session provenance.
Choose a long-context budget that the configured reader model supports.

The default quota guard allows only four generation calls. A complete 500-question
run with one history-size condition needs 2,000 calls (500 questions x four arms),
so explicitly raise the cap:

```bash
python -m evaluation.runners.run_context_degradation \
  --dataset evaluation/data/longmemeval_s_cleaned.json \
  --output-dir evaluation/results/context_degradation_main \
  --history-sizes 5 \
  --max-prompt-tokens 2048 \
  --long-context-max-prompt-tokens 32768 \
  --candidate-pool-size 100 \
  --retrieval-k 5 \
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --limit 0 \
  --max-generation-calls 2000
```

Before launching this run, validate the selected long-context budget with the
provider and run a small smoke test. The full run is an API-cost-consuming
experiment. Do not run it until the four-arm smoke test, context provenance,
and token-budget checks pass.

## Official answer judging

Exact match and token F1 in the runner are diagnostics, not the benchmark's
official answer score. Use the official LongMemEval evaluator with GPT-4o and the
same cleaned reference file. Clone the official repository separately:

```bash
git clone https://github.com/xiaowu0162/LongMemEval.git /tmp/LongMemEval
```

For each arm, run the evaluator against its matching JSONL hypothesis file. Example:

```bash
export OPENAI_API_KEY="YOUR_KEY"
python /tmp/LongMemEval/src/evaluation/evaluate_qa.py \
  gpt-4o \
  evaluation/results/context_degradation_main/hypotheses_all_external_memory_<RUN_ID>.jsonl \
  evaluation/data/longmemeval_s_cleaned.json
```

Repeat for `question_only`, `recent_history`, and `long_context`. The official
script writes a judged JSONL file with per-question `autoeval_label` values.
After all four arms are judged, run the paired analysis (replace `<RUN_ID>` with
the UTC run ID in the generated hypothesis filenames):

```bash
python -m evaluation.analyze_longmemeval_judged \
  --run-dir evaluation/results/context_degradation_main \
  --run-id <RUN_ID> \
  --history-size all \
  --dataset evaluation/data/longmemeval_s_cleaned.json
```

This produces overall and question-type accuracy, abstention accuracy, paired
accuracy differences, bootstrap confidence intervals, Wilcoxon tests, and
Holm-adjusted p-values. Preserve the model identifier, judged output, dataset
checksum, run configuration, and judge logs. Do not describe a non-GPT-4o judge
as the official judge.

## Reporting rules

- Report overall and question-type accuracy from the official judged output.
- Report retrieval Recall@K / Precision@K / MRR@K separately from answer accuracy.
- Report paired comparisons on the same question IDs; correct the planned family
  of p-values with Holm-Bonferroni.
- Keep the constrained-budget and long-context budgets explicit in tables.
- Report reader model, embedding model, reranker model, temperature, token budgets,
  hardware, latency percentiles, memory/index size, and dataset SHA-256.
- Never compare scores from different reader models or judge models as if they
  were controlled comparisons.
- The long-context arm is a valid full-history baseline only when its token budget
  is high enough that the provider tokenizer confirms the entire history fits.
  Otherwise describe it as a truncated long-context baseline.

## Standard BEIR retrieval benchmarks

Run the repository's standard BEIR SciFact and NFCorpus retrieval experiments
(B1 BM25, B2 dense, B3 hybrid RRF, B4 hybrid plus reranker) in a dedicated run
directory. These are standard BEIR datasets; do not mix their retrieval metrics
with LongMemEval answer-quality metrics.

```bash
python -m evaluation.runners.run_retrieval \
  --run-dir evaluation/results/runs/beir_paper

python -m evaluation.analyze_results \
  --run-dir evaluation/results/runs/beir_paper/raw \
  --output-dir evaluation/results/runs/beir_paper/processed_rank_preserving
```

The analysis verifies full query coverage and writes aggregate official BEIR
metrics, paired significance tests, and per-query metrics.

## One JSON for paper analysis

After the LongMemEval run, official judging, and BEIR analysis are complete,
build one combined, auditable JSON bundle:

```bash
python -m evaluation.build_paper_metrics \
  --longmemeval-json evaluation/results/context_degradation_main/context_degradation_<RUN_ID>.json \
  --official-longmemeval-analysis evaluation/results/context_degradation_main/longmemeval_official_analysis_<RUN_ID>.json \
  --beir-processed-dir evaluation/results/runs/beir_paper/processed_rank_preserving \
  --output evaluation/results/paper_metrics.json
```

The JSON contains benchmark provenance, per-question LongMemEval contexts and
predictions, official judged per-question labels (when supplied), BEIR
per-query and aggregate metrics, paired statistical results, token accounting,
and indexing/retrieval/generation timing. If official judging or BEIR results
are omitted, the JSON marks them as unavailable; it never fills in or fabricates
scores. The JSON is the artifact to share for paper writing.
