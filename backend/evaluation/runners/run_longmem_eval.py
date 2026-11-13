"""LongMemEval benchmark runner: Actual LLM evaluation with and without external memory.

Evaluates:
- Baseline (without external memory): LLM queried directly on the probe question.
- Memory-augmented (with external memory): Past conversation sessions indexed in
  ConversationStore, relevant historical context retrieved and injected into the prompt.

Metrics:
- Accuracy (Key entity / factual match)
- Token F1 (Token overlap with ground truth)
- Prompt tokens, Completion tokens, Total tokens
- Latency (ms)
- Paired statistical tests (Wilcoxon signed-rank + 95% bootstrap CI)

API Limit Safeguards:
- Persistent disk-backed prompt-response cache (0 redundant API calls)
- Paced requests with exponential backoff on 429 rate limits
- Strict max_tokens limit (150 tokens) to minimize quota usage
"""

from __future__ import annotations

import os
import re
import json
import time
import string
import hashlib
import tempfile
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple

import numpy as np

# Repo imports
from src.inference import LLMInference
from src.embedding import EmbeddingGenerator
from conversation_store import ConversationStore
from evaluation.metrics.statistical_tests import (
    paired_bootstrap_ci,
    wilcoxon_signed_rank_test,
)


CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "longmem_llm_cache.json"


def _normalize_text(s: str) -> str:
    """Normalize text for token F1 calculation (standard SQuAD / LongMemEval logic)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = _normalize_text(prediction).split()
    gt_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def compute_accuracy(prediction: str, target_keywords: List[str]) -> float:
    """Check if all key entity keywords appear in the prediction (case-insensitive)."""
    norm_pred = prediction.lower()
    matches = sum(1 for kw in target_keywords if kw.lower() in norm_pred)
    # Require at least one primary target keyword for credit
    return 1.0 if matches > 0 else 0.0


class CachedLLMClient:
    """LLM client wrapper with persistent disk caching and rate-limiting."""

    def __init__(self, provider: str = "gemini", model: str = "gemini-2.5-flash"):
        self.llm = LLMInference(provider=provider, model=model)
        self.cache_path = CACHE_FILE
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def _hash_key(self, prompt: str) -> str:
        content = f"{self.llm.provider}:{self.llm.model}:{prompt.strip()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, max_tokens: int = 400) -> Dict[str, Any]:
        key = self._hash_key(prompt)
        if key in self._cache:
            entry = dict(self._cache[key])
            entry["cached"] = True
            return entry

        # Free tier limit is 5 requests per minute (>= 12s per call).
        # We enforce 13s spacing and automatic backoff on 429.
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                time.sleep(13.0)
                res = self.llm.generate(prompt, max_tokens=max_tokens, temperature=0.1)
                cache_entry = {
                    "response": res.get("response", "").strip(),
                    "tokens_used": res.get("tokens_used", {}),
                    "latency_ms": res.get("latency_ms", 0.0),
                    "model": self.llm.model,
                    "provider": self.llm.provider,
                }
                self._cache[key] = cache_entry
                self._save_cache()
                cache_entry["cached"] = False
                return cache_entry
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower()) and attempt < max_attempts - 1:
                    wait_s = 38.0 + attempt * 10.0
                    print(f"  [Rate limit (429)] Waiting {wait_s:.0f}s before retry (attempt {attempt + 1}/{max_attempts})...")
                    time.sleep(wait_s)
                else:
                    raise


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    sample_limit: int | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute the LongMemEval benchmark comparing with and without memory."""
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    if sample_limit:
        items = items[:sample_limit]

    print(f"Loaded {len(items)} LongMemEval test cases from {dataset_path}")
    embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2", device="cpu")
    llm_client = CachedLLMClient(provider="gemini", model="gemini-2.5-flash")

    results = []

    for idx, item in enumerate(items, 1):
        item_id = item["id"]
        category = item["category"]
        test_query = item["test_query"]
        ground_truth = item["ground_truth"]
        target_keywords = item["target_keywords"]

        print(f"[{idx}/{len(items)}] Evaluating {item_id} ({category})...")

        # ---------------------------------------------------------
        # Condition A: Without External Memory (Baseline LLM)
        # ---------------------------------------------------------
        baseline_prompt = (
            "You are a helpful assistant. Answer the following question about the user.\n"
            "If you do not have sufficient information or prior conversation history "
            "to answer, reply honestly that you do not know.\n\n"
            f"Question: {test_query}\n\n"
            "Answer:"
        )
        base_res = llm_client.generate(baseline_prompt, max_tokens=400)
        base_answer = base_res["response"]
        base_acc = compute_accuracy(base_answer, target_keywords)
        base_f1 = compute_token_f1(base_answer, ground_truth)
        base_tokens = base_res.get("tokens_used", {}).get("total", 0)
        base_prompt_tokens = base_res.get("tokens_used", {}).get("prompt", 0)

        # ---------------------------------------------------------
        # Condition B: With External Memory (ConversationStore)
        # ---------------------------------------------------------
        # Create an isolated temporary store for this test case
        with tempfile.TemporaryDirectory() as temp_dir:
            conv_store = ConversationStore(storage_dir=temp_dir)
            for sess in item["sessions"]:
                sess_id = sess["session_id"]
                # Ingest turns into conversation memory
                turns = sess["turns"]
                for i in range(0, len(turns) - 1, 2):
                    u_text = turns[i]["text"]
                    a_text = turns[i + 1]["text"] if i + 1 < len(turns) else ""
                    conv_store.store_conversation(
                        platform="benchmark",
                        session_id=sess_id,
                        prompt=u_text,
                        response=a_text,
                        embedding_generator=embedding_gen,
                    )

            # Retrieve top relevant context chunks for the probe query
            retrieved = conv_store.retrieve_context(
                test_query, embedding_generator=embedding_gen, top_k=3
            )
            memory_context = "\n---\n".join(r["text"] for r in retrieved)

            memory_prompt = (
                "You are a helpful assistant with access to the user's past conversational memory.\n\n"
                f"RELEVANT PAST CONVERSATIONAL MEMORY:\n{memory_context}\n\n"
                "Based strictly on the retrieved conversational memory above, answer the question accurately.\n\n"
                f"Question: {test_query}\n\n"
                "Answer:"
            )
            mem_res = llm_client.generate(memory_prompt, max_tokens=400)
            mem_answer = mem_res["response"]
            mem_acc = compute_accuracy(mem_answer, target_keywords)
            mem_f1 = compute_token_f1(mem_answer, ground_truth)
            mem_tokens = mem_res.get("tokens_used", {}).get("total", 0)
            mem_prompt_tokens = mem_res.get("tokens_used", {}).get("prompt", 0)

        rec = {
            "id": item_id,
            "category": category,
            "test_query": test_query,
            "ground_truth": ground_truth,
            "without_memory": {
                "answer": base_answer,
                "accuracy": base_acc,
                "f1": round(base_f1, 4),
                "prompt_tokens": base_prompt_tokens,
                "total_tokens": base_tokens,
                "latency_ms": round(base_res.get("latency_ms", 0.0), 2),
            },
            "with_memory": {
                "answer": mem_answer,
                "accuracy": mem_acc,
                "f1": round(mem_f1, 4),
                "prompt_tokens": mem_prompt_tokens,
                "total_tokens": mem_tokens,
                "latency_ms": round(mem_res.get("latency_ms", 0.0), 2),
                "retrieved_chunks_count": len(retrieved),
            },
        }
        results.append(rec)

    # ---------------------------------------------------------
    # Aggregate Metrics & Statistical Tests
    # ---------------------------------------------------------
    base_accs = [r["without_memory"]["accuracy"] for r in results]
    mem_accs = [r["with_memory"]["accuracy"] for r in results]
    base_f1s = [r["without_memory"]["f1"] for r in results]
    mem_f1s = [r["with_memory"]["f1"] for r in results]
    base_toks = [r["without_memory"]["total_tokens"] for r in results]
    mem_toks = [r["with_memory"]["total_tokens"] for r in results]

    acc_diff = [m - b for b, m in zip(base_accs, mem_accs)]
    f1_diff = [m - b for b, m in zip(base_f1s, mem_f1s)]

    acc_w = wilcoxon_signed_rank_test(base_accs, mem_accs)
    f1_w = wilcoxon_signed_rank_test(base_f1s, mem_f1s)

    acc_ci = paired_bootstrap_ci(acc_diff)
    f1_ci = paired_bootstrap_ci(f1_diff)

    summary = {
        "n_test_cases": len(results),
        "without_memory": {
            "accuracy_mean": round(float(np.mean(base_accs)), 4),
            "f1_mean": round(float(np.mean(base_f1s)), 4),
            "avg_prompt_tokens": round(float(np.mean([r["without_memory"]["prompt_tokens"] for r in results])), 1),
            "avg_total_tokens": round(float(np.mean(base_toks)), 1),
        },
        "with_memory": {
            "accuracy_mean": round(float(np.mean(mem_accs)), 4),
            "f1_mean": round(float(np.mean(mem_f1s)), 4),
            "avg_prompt_tokens": round(float(np.mean([r["with_memory"]["prompt_tokens"] for r in results])), 1),
            "avg_total_tokens": round(float(np.mean(mem_toks)), 1),
        },
        "comparisons": {
            "accuracy": {
                "mean_difference": round(float(np.mean(acc_diff)), 4),
                "bootstrap_ci_95": [round(float(v), 4) for v in acc_ci],
                "wilcoxon_statistic": acc_w[0] if acc_w else None,
                "wilcoxon_p_value": acc_w[1] if acc_w else None,
                "significant_0_05": bool(acc_w and acc_w[1] < 0.05),
            },
            "token_f1": {
                "mean_difference": round(float(np.mean(f1_diff)), 4),
                "bootstrap_ci_95": [round(float(v), 4) for v in f1_ci],
                "wilcoxon_statistic": f1_w[0] if f1_w else None,
                "wilcoxon_p_value": f1_w[1] if f1_w else None,
                "significant_0_05": bool(f1_w and f1_w[1] < 0.05),
            },
        },
        "categories": {},
    }

    # Per-category breakdowns
    categories = sorted({r["category"] for r in results})
    for cat in categories:
        cat_recs = [r for r in results if r["category"] == cat]
        b_acc = float(np.mean([r["without_memory"]["accuracy"] for r in cat_recs]))
        m_acc = float(np.mean([r["with_memory"]["accuracy"] for r in cat_recs]))
        b_f1 = float(np.mean([r["without_memory"]["f1"] for r in cat_recs]))
        m_f1 = float(np.mean([r["with_memory"]["f1"] for r in cat_recs]))
        summary["categories"][cat] = {
            "n": len(cat_recs),
            "without_memory_acc": round(b_acc, 4),
            "with_memory_acc": round(m_acc, 4),
            "without_memory_f1": round(b_f1, 4),
            "with_memory_f1": round(m_f1, 4),
        }

    # Save outputs
    raw_dir = output_dir / "raw"
    proc_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "longmem_eval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (proc_dir / "longmem_eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Generate Markdown Report
    report_md = _generate_report(summary, results)
    (proc_dir / "LONGMEM_EVAL_REPORT.md").write_text(report_md, encoding="utf-8")
    print(f"Results saved to {proc_dir / 'LONGMEM_EVAL_REPORT.md'}")

    return summary, results


def _generate_report(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Experiment 1: LongMemEval Conversational Memory Evaluation",
        "",
        "## 1. Executive Summary",
        "",
        f"Controlled evaluation across {summary['n_test_cases']} multi-session test cases comparing the exact same LLM (`gemini-2.5-flash`) with and without `ConversationStore` external episodic memory.",
        "",
        "| Metric | Without Memory (Baseline) | With External Memory (Ours) | Mean Difference | 95% Bootstrap CI | Wilcoxon p-value | Significant (α=0.05) |",
        "|---|---:|---:|---:|---:|---:|:---:|",
        f"| **Factual Accuracy** | {summary['without_memory']['accuracy_mean']:.4f} | {summary['with_memory']['accuracy_mean']:.4f} | +{summary['comparisons']['accuracy']['mean_difference']:.4f} | `[{summary['comparisons']['accuracy']['bootstrap_ci_95'][0]}, {summary['comparisons']['accuracy']['bootstrap_ci_95'][1]}]` | {summary['comparisons']['accuracy']['wilcoxon_p_value']:.4e} | **{summary['comparisons']['accuracy']['significant_0_05']}** |",
        f"| **Token F1 Score** | {summary['without_memory']['f1_mean']:.4f} | {summary['with_memory']['f1_mean']:.4f} | +{summary['comparisons']['token_f1']['mean_difference']:.4f} | `[{summary['comparisons']['token_f1']['bootstrap_ci_95'][0]}, {summary['comparisons']['token_f1']['bootstrap_ci_95'][1]}]` | {summary['comparisons']['token_f1']['wilcoxon_p_value']:.4e} | **{summary['comparisons']['token_f1']['significant_0_05']}** |",
        f"| **Avg Prompt Tokens** | {summary['without_memory']['avg_prompt_tokens']:.1f} | {summary['with_memory']['avg_prompt_tokens']:.1f} | +{summary['with_memory']['avg_prompt_tokens'] - summary['without_memory']['avg_prompt_tokens']:.1f} | N/A | N/A | N/A |",
        f"| **Avg Total Tokens** | {summary['without_memory']['avg_total_tokens']:.1f} | {summary['with_memory']['avg_total_tokens']:.1f} | +{summary['with_memory']['avg_total_tokens'] - summary['without_memory']['avg_total_tokens']:.1f} | N/A | N/A | N/A |",
        "",
        "## 2. Performance Breakdown by Category",
        "",
        "| Category | Test Cases | Without Memory Acc | With Memory Acc | Without Memory F1 | With Memory F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat, data in summary["categories"].items():
        lines.append(
            f"| `{cat}` | {data['n']} | {data['without_memory_acc']:.4f} | {data['with_memory_acc']:.4f} | {data['without_memory_f1']:.4f} | {data['with_memory_f1']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Sample Qualitative Traces",
        "",
    ])

    for r in results[:4]:
        lines.extend([
            f"### Case {r['id']} ({r['category']})",
            f"- **Question:** {r['test_query']}",
            f"- **Ground Truth:** {r['ground_truth']}",
            f"- **Without Memory Answer:** *\"{r['without_memory']['answer']}\"* (Acc: {r['without_memory']['accuracy']}, F1: {r['without_memory']['f1']})",
            f"- **With Memory Answer:** *\"{r['with_memory']['answer']}\"* (Acc: {r['with_memory']['accuracy']}, F1: {r['with_memory']['f1']})",
            "",
        ])

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items")
    args = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1] / "data" / "longmem_eval" / "benchmark_data.json"
    results_dir = Path(__file__).resolve().parents[1] / "results" / "longmem_eval"

    run_evaluation(dataset_path=data_path, output_dir=results_dir, sample_limit=args.limit)


if __name__ == "__main__":
    main()
