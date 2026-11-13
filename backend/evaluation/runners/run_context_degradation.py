"""Controlled LongMemEval context-degradation and external-memory QA experiment.

Compares question-only, recent-history (long-context/truncation), and retrieved
external-memory prompts under one fixed prompt-token budget and identical
generation settings. Writes predictions plus exact-match/token-F1 diagnostics.
For publication, also run the official LongMemEval LLM-judge evaluator on the
saved predictions; token F1 is only a transparent automatic diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from conversation_store import ConversationStore
from src.embedding import EmbeddingGenerator
from src.inference import LLMInference
from src.retrieval import SemanticRetriever
from src.vector_store import VectorStore
from src.reranker import CrossEncoderReranker
from evaluation.metrics.statistical_tests import (
    paired_bootstrap_ci,
    wilcoxon_signed_rank_test,
    holm_bonferroni,
)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not pred or not gold:
        return float(pred == gold)
    common: dict[str, int] = {}
    for token in pred:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold:
        if common.get(token, 0):
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def session_to_text(session: list[dict[str, Any]]) -> str:
    lines = []
    for turn in session:
        role = str(turn.get("role", "unknown")).strip().lower()
        text = str(turn.get("content", "")).strip()
        if text:
            lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def build_hybrid_memory_retriever(
    store: ConversationStore,
    embedder: EmbeddingGenerator,
    reranker: CrossEncoderReranker,
    candidate_pool_size: int,
    retrieval_k: int,
) -> SemanticRetriever:
    """Reuse the stored FAISS memory index with BM25 + dense RRF + cross-encoder."""
    vector_store = VectorStore(
        dimension=embedder.get_embedding_dimension(),
        index_type="cosine",
    )
    # ConversationStore already built and normalized this FAISS index.
    vector_store.index = store._index
    vector_store.chunks = store._chunks
    vector_store.metadata = store._metadata
    retriever = SemanticRetriever(
        vector_store=vector_store,
        embedding_generator=embedder,
        top_k=retrieval_k,
        similarity_threshold=0.0,
        mode="hybrid",
        dense_k=candidate_pool_size,
        bm25_k=candidate_pool_size,
        alpha=0.5,
        enable_rerank=True,
        rerank_candidate_pool=candidate_pool_size,
        rerank_top_n=retrieval_k,
    )
    retriever.set_reranker(reranker)
    return retriever


def build_prompt(question: str, context: str, question_date: str = "") -> str:
    date_line = f"\nCurrent date: {question_date}" if question_date else ""
    return (
        "Answer the question using the supplied conversation history when it is "
        "relevant. If the history does not contain the answer, say that you do "
        "not know. Do not invent facts."
        f"{date_line}\n\n"
        f"CONVERSATION HISTORY:\n{context if context else '[No history supplied]'}\n\n"
        f"QUESTION:\n{question}\n\nANSWER:"
    )


class PromptTokenCounter:
    """Use the selected provider's tokenizer/count-token endpoint."""

    def __init__(self, llm: LLMInference):
        self.llm = llm
        self._encoding = None
        if llm.provider == "openai":
            try:
                import tiktoken
                try:
                    self._encoding = tiktoken.encoding_for_model(llm.model)
                except KeyError as exc:
                    raise RuntimeError(
                        f"No exact tiktoken encoding is registered for model {llm.model!r}; "
                        "choose a supported model or add its documented tokenizer."
                    ) from exc
            except ImportError as exc:
                raise RuntimeError(
                    "Install tiktoken for OpenAI token-budget enforcement: pip install tiktoken"
                ) from exc
        elif llm.provider == "ollama":
            raise RuntimeError(
                "Ollama does not expose a reliable generic tokenizer through this adapter. "
                "Use Gemini/OpenAI/Anthropic/HuggingFace, or add a model-matched tokenizer."
            )
        elif llm.provider == "huggingface" and llm.tokenizer is None:
            raise RuntimeError("HuggingFace tokenizer was not initialized")

    def count(self, prompt: str) -> int:
        if self.llm.provider == "huggingface":
            return len(self.llm.tokenizer.encode(prompt, add_special_tokens=True))
        if self.llm.provider == "gemini":
            result = self.llm.client.count_tokens(prompt)
            return int(result.total_tokens)
        if self.llm.provider == "anthropic":
            result = self.llm.client.messages.count_tokens(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return int(result.input_tokens)
        if self.llm.provider == "openai":
            return len(self._encoding.encode(prompt))
        raise RuntimeError(f"No exact prompt token counter for provider: {self.llm.provider}")


def fit_context(
    question: str,
    candidates: list[str],
    max_prompt_tokens: int,
    counter: PromptTokenCounter,
    question_date: str = "",
) -> tuple[str, int, list[str]]:
    """Pack whole evidence units, then verify the final prompt with the real tokenizer.

    A conservative character-based estimate avoids making a provider token-count
    API call for every candidate. The final prompt is always checked with the
    provider's actual tokenizer/count endpoint; units are removed until it fits.
    """
    def conservative_estimate(text: str) -> int:
        return math.ceil(len(text) / 3)

    chosen: list[str] = []
    for candidate in candidates:
        trial = "\n\n".join(chosen + [candidate])
        if conservative_estimate(build_prompt(question, trial, question_date)) <= max_prompt_tokens:
            chosen.append(candidate)

    while True:
        context = "\n\n".join(chosen)
        actual_tokens = counter.count(build_prompt(question, context, question_date))
        if actual_tokens <= max_prompt_tokens:
            return context, actual_tokens, chosen
        if not chosen:
            raise ValueError(
                f"Question/instructions alone require {actual_tokens} tokens, "
                f"exceeding budget {max_prompt_tokens}"
            )
        chosen.pop()


def make_llm(provider: str | None, model: str | None, config_path: Path) -> LLMInference:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    llm_cfg = config.get("llm", {})
    selected_provider = (provider or llm_cfg.get("provider") or "gemini").lower()
    provider_cfg = dict(llm_cfg.get(selected_provider, {}))
    selected_model = model or provider_cfg.pop("model", None)
    # Explicitly disable semantic caching: every arm must invoke the model.
    provider_cfg["semantic_cache"] = {"enabled": False}
    return LLMInference(provider=selected_provider, model=selected_model, config=provider_cfg)


def select_history(
    sessions: list[list[dict[str, Any]]],
    session_ids: list[str],
    size: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    if len(sessions) != len(session_ids):
        raise ValueError("haystack_sessions and haystack_session_ids length mismatch")
    count = len(sessions) if size == "all" else min(int(size), len(sessions))
    return [(str(sid), session) for sid, session in zip(session_ids, sessions)][-count:]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    items = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset must be a non-empty LongMemEval JSON array")
    if args.limit:
        items = items[:args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    llm = make_llm(args.provider, args.model, args.config)
    counter = PromptTokenCounter(llm)
    embedder = EmbeddingGenerator(
        model_name=args.embedding_model,
        device=args.device,
        cache_path=str(args.output_dir / "embedding_cache.sqlite"),
        cache_enabled=True,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    history_sizes = args.history_sizes.split(",")
    reranker = CrossEncoderReranker(model_name=args.rerank_model)
    if reranker.model is None:
        raise RuntimeError(
            f"Could not load required cross-encoder {args.rerank_model!r}; "
            "refusing to label dense-only retrieval as hybrid+rereanked memory."
        )

    for item_index, item in enumerate(items):
        qid = str(item["question_id"])
        question = str(item["question"])
        question_date = str(item.get("question_date", ""))
        answer = str(item.get("answer", ""))
        sessions = item.get("haystack_sessions", [])
        session_ids = list(map(str, item.get("haystack_session_ids", [])))

        for history_size in history_sizes:
            selected = select_history(sessions, session_ids, history_size)
            store_path = (
                args.output_dir / "stores" / run_id /
                f"{item_index:04d}_{re.sub(r'[^A-Za-z0-9_-]', '_', qid)}_{history_size}"
            )
            store = ConversationStore(str(store_path))
            for sid, session in selected:
                text = session_to_text(session)
                if text:
                    store.store_conversation(
                        platform="longmemeval",
                        session_id=sid,
                        prompt=f"LongMemEval session {sid}",
                        response=text,
                        embedding_generator=embedder,
                        chunk_size=args.chunk_size,
                        overlap=args.overlap,
                    )

            recent_units = [
                f"[session_id:{sid}]\n{session_to_text(session)}"
                for sid, session in reversed(selected)
                if session_to_text(session)
            ]
            memory_retriever = build_hybrid_memory_retriever(
                store=store,
                embedder=embedder,
                reranker=reranker,
                candidate_pool_size=args.candidate_pool_size,
                retrieval_k=args.retrieval_k,
            )
            retrieval_started = time.perf_counter()
            hits = memory_retriever.retrieve(
                query=question,
                k=args.retrieval_k,
                threshold=0.0,
                mode="hybrid",
                use_cache=False,
            )
            retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
            memory_units = []
            seen_sessions = set()
            for hit in hits:
                metadata = hit.get("metadata", {})
                sid = str(metadata.get("session_id", ""))
                if not sid:
                    raise RuntimeError("Retrieved memory chunk has no session_id metadata")
                if sid in seen_sessions:
                    continue
                seen_sessions.add(sid)
                memory_units.append(f"[session_id:{sid}]\n{hit.get('text', '')}")

            arms = {
                "question_only": ([], args.max_prompt_tokens),
                "recent_history": (recent_units, args.max_prompt_tokens),
                "long_context": (recent_units, args.long_context_max_prompt_tokens),
                "external_memory": (memory_units, args.max_prompt_tokens),
            }
            for arm, (units, prompt_budget) in arms.items():
                context, prompt_tokens, selected_units = fit_context(
                    question, units, prompt_budget, counter, question_date
                )
                prompt = build_prompt(question, context, question_date)
                included_session_ids = []
                for unit in selected_units:
                    match = re.search(r"\[session_id:([^\]]+)\]", unit)
                    if match and match.group(1) not in included_session_ids:
                        included_session_ids.append(match.group(1))
                start = time.perf_counter()
                generated = llm.generate(
                    prompt,
                    max_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                prediction = str(generated.get("response", "")).strip()
                retrieved_session_ids = list(seen_sessions) if arm == "external_memory" else []
                gold_session_ids = set(map(str, item.get("answer_session_ids", [])))
                retrieved_relevant = [
                    sid for sid in retrieved_session_ids if sid in gold_session_ids
                ]
                retrieval_recall = (
                    len(set(retrieved_relevant)) / len(gold_session_ids)
                    if arm == "external_memory" and gold_session_ids else None
                )
                retrieval_precision = (
                    len(retrieved_relevant) / len(retrieved_session_ids)
                    if arm == "external_memory" and retrieved_session_ids else None
                )
                retrieval_mrr = None
                if arm == "external_memory" and gold_session_ids:
                    retrieval_mrr = next(
                        (1.0 / rank for rank, sid in enumerate(retrieved_session_ids, 1)
                         if sid in gold_session_ids),
                        0.0,
                    )
                row = {
                    "question_id": qid,
                    "question_type": item.get("question_type"),
                    "history_size": history_size,
                    "history_sessions_available": len(selected),
                    "arm": arm,
                    "prediction": prediction,
                    "gold_answer": answer,
                    "exact_match": exact_match(prediction, answer),
                    "token_f1": token_f1(prediction, answer),
                    "prompt_tokens": prompt_tokens,
                    "prompt_token_budget": prompt_budget,
                    "context_characters": len(context),
                    "context_session_ids": included_session_ids,
                    "retrieved_session_ids": retrieved_session_ids,
                    "gold_session_count": len(gold_session_ids),
                    "retrieval_recall_at_k": retrieval_recall,
                    "retrieval_precision_at_k": retrieval_precision,
                    "retrieval_mrr_at_k": retrieval_mrr,
                    "generation_latency_ms": round(elapsed_ms, 3),
                    "reported_latency_ms": generated.get("latency_ms"),
                    "tokens_used": generated.get("tokens_used", {}),
                    "provider": llm.provider,
                    "model": llm.model,
                    "cache_hit": generated.get("cache_hit", False),
                    "memory_retrieval_latency_ms": round(retrieval_latency_ms, 3),
                    "retrieval_method": "BM25+FAISS+RRF+cross-encoder",
                }
                rows.append(row)

        if (item_index + 1) % 5 == 0:
            print(f"Processed {item_index + 1}/{len(items)} questions")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["history_size"], row["arm"])].append(row)
    significance: dict[str, list[dict[str, Any]]] = {}
    all_tests: list[dict[str, Any]] = []
    row_lookup = {
        (row["question_id"], row["history_size"], row["arm"]): row
        for row in rows
    }
    for history_size in history_sizes:
        significance[history_size] = []
        for baseline in ("recent_history", "long_context", "question_only"):
            for metric in ("token_f1", "exact_match"):
                differences = [
                    row_lookup[(str(item["question_id"]), history_size, "external_memory")][metric]
                    - row_lookup[(str(item["question_id"]), history_size, baseline)][metric]
                    for item in items
                ]
                before = [
                    row_lookup[(str(item["question_id"]), history_size, baseline)][metric]
                    for item in items
                ]
                after = [
                    row_lookup[(str(item["question_id"]), history_size, "external_memory")][metric]
                    for item in items
                ]
                test = wilcoxon_signed_rank_test(before, after)
                result = {
                    "comparison": f"external_memory_vs_{baseline}",
                    "metric": metric,
                    "n_questions": len(differences),
                    "mean_difference": statistics.mean(differences) if differences else 0.0,
                    "bootstrap_ci_95": list(paired_bootstrap_ci(differences)),
                    "wilcoxon_statistic": test[0] if test else None,
                    "wilcoxon_p_value": test[1] if test else None,
                }
                significance[history_size].append(result)
                all_tests.append(result)

    adjusted = holm_bonferroni([test["wilcoxon_p_value"] for test in all_tests])
    for test, adjusted_p in zip(all_tests, adjusted):
        test["wilcoxon_p_value_holm"] = adjusted_p
        test["significant_holm_0_05"] = adjusted_p is not None and adjusted_p < 0.05

    summary = {
        "benchmark": "LongMemEval controlled context-degradation",
        "run_id_utc": run_id,
        "dataset": str(args.dataset),
        "n_questions": len(items),
        "history_sizes": history_sizes,
        "max_prompt_tokens": args.max_prompt_tokens,
        "long_context_max_prompt_tokens": args.long_context_max_prompt_tokens,
        "long_context_baseline_note": (
            "The long_context arm receives all selected history sessions up to its "
            "separately declared token budget; for history_size=all this is the full "
            "LongMemEval-S history subject to that budget."
        ),
        "retrieval_method": "BM25 + FAISS dense retrieval + RRF + cross-encoder reranking",
        "retrieval_candidate_pool_size": args.candidate_pool_size,
        "retrieval_top_k": args.retrieval_k,
        "reranker_model": args.rerank_model,
        "provider": llm.provider,
        "model": llm.model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "metrics": {},
        "paired_significance": significance,
        "per_question": rows,
        "scoring_note": (
            "Exact match and token F1 are automatic diagnostics only. Use the official "
            "LongMemEval LLM-judge protocol for publication-quality answer scoring."
        ),
    }
    for (history_size, arm), group in grouped.items():
        latencies = [float(r["generation_latency_ms"]) for r in group]
        summary["metrics"][f"{history_size}:{arm}"] = {
            "n": len(group),
            "exact_match": round(statistics.mean(r["exact_match"] for r in group), 6),
            "token_f1": round(statistics.mean(r["token_f1"] for r in group), 6),
            "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in group), 2),
            "mean_generation_latency_ms": round(statistics.mean(latencies), 2),
            "mean_memory_retrieval_latency_ms": round(
                statistics.mean(float(r["memory_retrieval_latency_ms"]) for r in group), 2
            ),
            "retrieval_recall_at_k": (
                round(statistics.mean([r["retrieval_recall_at_k"] for r in group
                                       if r["retrieval_recall_at_k"] is not None]), 6)
                if any(r["retrieval_recall_at_k"] is not None for r in group) else None
            ),
            "retrieval_precision_at_k": (
                round(statistics.mean([r["retrieval_precision_at_k"] for r in group
                                       if r["retrieval_precision_at_k"] is not None]), 6)
                if any(r["retrieval_precision_at_k"] is not None for r in group) else None
            ),
            "retrieval_mrr_at_k": (
                round(statistics.mean([r["retrieval_mrr_at_k"] for r in group
                                       if r["retrieval_mrr_at_k"] is not None]), 6)
                if any(r["retrieval_mrr_at_k"] is not None for r in group) else None
            ),
            "p50_generation_latency_ms": round(sorted(latencies)[math.floor(0.50 * (len(latencies)-1))], 2),
            "p95_generation_latency_ms": round(sorted(latencies)[math.ceil(0.95 * (len(latencies)-1))], 2),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"context_degradation_{run_id}.json"
    csv_path = args.output_dir / f"context_degradation_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "question_id", "question_type", "history_size", "arm", "exact_match",
            "token_f1", "prompt_tokens", "prompt_token_budget", "context_characters",
            "generation_latency_ms", "memory_retrieval_latency_ms", "retrieval_method",
            "retrieval_recall_at_k", "retrieval_precision_at_k", "retrieval_mrr_at_k",
            "provider", "model",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    # Official LongMemEval evaluator input: one JSONL per experimental arm.
    hypothesis_paths = []
    for history_size in history_sizes:
        for arm in ("question_only", "recent_history", "long_context", "external_memory"):
            hypothesis_path = args.output_dir / (
                f"hypotheses_{history_size}_{arm}_{run_id}.jsonl"
            )
            with hypothesis_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    if row["history_size"] == history_size and row["arm"] == arm:
                        handle.write(json.dumps({
                            "question_id": row["question_id"],
                            "hypothesis": row["prediction"],
                        }) + "\n")
            hypothesis_paths.append(str(hypothesis_path))

    summary["official_evaluator_hypothesis_files"] = hypothesis_paths
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, indent=2))
    print(f"RESULT_JSON={output}")
    print(f"RESULT_CSV={csv_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/context_degradation"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--provider", choices=["gemini", "openai", "anthropic", "huggingface", "ollama"])
    parser.add_argument("--model")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--history-sizes", default="5,10,20,all")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--long-context-max-prompt-tokens", type=int, default=32768,
        help="Separate larger context budget for the full-history baseline; must fit the reader model."
    )
    parser.add_argument("--retrieval-k", type=int, default=30)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=5, help="Smoke test limit; use 0 for all questions")
    args = parser.parse_args()
    if (
        args.max_prompt_tokens < 128
        or args.long_context_max_prompt_tokens < args.max_prompt_tokens
        or args.retrieval_k < 1
        or args.candidate_pool_size < args.retrieval_k
    ):
        parser.error(
            "Require long-context-max-prompt-tokens >= max-prompt-tokens >=128 "
            "and candidate-pool-size >= retrieval-k >=1"
        )
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        parser.error("Require 0 <= overlap < chunk-size")
    if not args.history_sizes or any(
        part != "all" and (not part.isdigit() or int(part) < 1)
        for part in args.history_sizes.split(",")
    ):
        parser.error("history-sizes must be comma-separated positive integers and/or all")
    evaluate(args)


if __name__ == "__main__":
    main()
