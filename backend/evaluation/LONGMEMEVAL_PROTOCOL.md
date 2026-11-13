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

Choose a long-context budget supported by the reader model. To test a model with
a 128k context window, start conservatively (for example, 120000) and let the
provider's tokenizer enforce the actual prompt limit.

```bash
python -m evaluation.runners.run_context_degradation \
  --dataset evaluation/data/longmemeval_s_cleaned.json \
  --output-dir evaluation/results/context_degradation_main \
  --history-sizes all \
  --max-prompt-tokens 4096 \
  --long-context-max-prompt-tokens 120000 \
  --candidate-pool-size 100 \
  --retrieval-k 30 \
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --limit 0
```

The four arms are:
- `question_only`: no conversation history.
- `recent_history`: most recent history sessions under the constrained prompt budget.
- `long_context`: all available history sessions up to the separately declared long-context budget.
- `external_memory`: BM25 + FAISS dense retrieval + reciprocal-rank fusion + cross-encoder reranking, packed under the constrained prompt budget.

The external-memory arm uses the repository's existing ConversationStore chunking,
FAISS cosine index, BM25 implementation, RRF implementation, and cross-encoder.
The retrieved chunk's `session_id` metadata is retained when packing evidence.
The runner also records session Recall@K, Precision@K, MRR@K, retrieval latency,
prompt tokens, generation latency, and answer diagnostics. Retrieval metrics are
not defined for no-evidence/abstention items.

The main run above uses one history-size condition (`all`) to avoid multiplying
the 500-question generation cost. Run `5,10,20,all` as a separate sensitivity
experiment only if compute/API budget permits.

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
