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
import hashlib
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        rerank_top_n=candidate_pool_size,
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
            # Count tokens with the exact same model used for generation.
            expected_model = "gemini-3.5-flash-lite"
            if self.llm.model != expected_model:
                raise RuntimeError(
                    f"Gemini model mismatch: expected {expected_model}, got {self.llm.model}"
                )
            if not hasattr(self, "_gemini_token_cache"):
                self._gemini_token_cache = {}
            if prompt not in self._gemini_token_cache:
                result = self.llm.client.models.count_tokens(
                    model=expected_model,
                    contents=prompt,
                )
                self._gemini_token_cache[prompt] = int(result.total_tokens)
            return self._gemini_token_cache[prompt]
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
    """Pack newest evidence first, truncating an oversized session instead of dropping all history.

    Candidates are supplied oldest-to-newest. The returned context remains in
    chronological order. The provider tokenizer verifies the final prompt.
    """
    base_prompt = build_prompt(question, "", question_date)
    base_estimate = math.ceil(len(base_prompt) / 3)
    if base_estimate > max_prompt_tokens:
        raise ValueError("Question and instructions exceed the configured prompt budget")

    chosen_newest_first: list[str] = []
    used_chars = 0
    char_budget = max(0, max_prompt_tokens * 3 - len(base_prompt))
    for candidate in reversed(candidates):
        separator_chars = 2 if chosen_newest_first else 0
        remaining = char_budget - used_chars - separator_chars
        if remaining <= 0:
            break
        if len(candidate) <= remaining:
            chosen_newest_first.append(candidate)
            used_chars += len(candidate) + separator_chars
            continue

        # Preserve the session identifier/date header and the newest part of
        # an oversized session, rather than silently discarding the session.
        header, sep, body = candidate.partition("\n")
        if not sep:
            header, body = "", candidate
        header_cost = len(header) + (1 if sep else 0)
        body_budget = max(0, remaining - header_cost)
        if body_budget > 0:
            partial = header + ("\n" if sep else "") + body[-body_budget:]
            chosen_newest_first.append(partial)
        break

    chosen = list(reversed(chosen_newest_first))
    while True:
        context = "\n\n".join(chosen)
        actual_tokens = counter.count(build_prompt(question, context, question_date))
        if actual_tokens <= max_prompt_tokens:
            return context, actual_tokens, chosen
        if not chosen:
            raise ValueError("Unable to fit prompt within the declared token budget")
        if len(chosen) > 1:
            chosen.pop(0)  # discard oldest evidence first
            continue
        header, sep, body = chosen[0].partition("\n")
        if not sep or not body:
            raise ValueError("Unable to fit the newest history unit within the token budget")
        keep = max(1, int(len(body) * 0.8))
        chosen[0] = header + "\n" + body[-keep:]


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
    session_dates: list[str],
    size: str,
) -> list[tuple[str, str, list[dict[str, Any]], int]]:
    if not (len(sessions) == len(session_ids) == len(session_dates)):
        raise ValueError("haystack_sessions, haystack_session_ids, and haystack_dates must align")
    count = len(sessions) if size == "all" else min(int(size), len(sessions))
    return [
        (str(sid), str(date), session, index)
        for index, (sid, date, session) in enumerate(zip(session_ids, session_dates, sessions))
    ][-count:]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    items = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset must be a non-empty LongMemEval JSON array")
    if args.limit:
        items = items[:args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_sizes = args.history_sizes.split(",")
    if len(set(history_sizes)) != len(history_sizes):
        raise ValueError("--history-sizes must not contain duplicate windows")
    arms_per_question = 4
    planned_generation_calls = len(items) * len(history_sizes) * arms_per_question
    if planned_generation_calls > args.max_generation_calls:
        raise ValueError(
            f"This run requires {planned_generation_calls} generation calls "
            f"({len(items)} questions x {len(history_sizes)} history size(s) x "
            f"{arms_per_question} arms), exceeding --max-generation-calls "
            f"{args.max_generation_calls}. Reduce --limit/--history-sizes or "
            "raise the cap deliberately."
        )
    print(
        f"Quota guard: at most {planned_generation_calls} generation calls; "
        f"max output tokens per call: {args.max_output_tokens}; "
        f"theoretical output-token ceiling: "
        f"{planned_generation_calls * args.max_output_tokens}."
    )

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
        session_dates = list(map(str, item.get("haystack_dates", [])))

        # Build the memory corpus from the complete available history once per
        # question. The recent-window parameter must never restrict retrieval.
        all_history = select_history(sessions, session_ids, session_dates, "all")
        source_session_by_memory_id = {
            f"{sid}__occ{index}": sid for sid, _, _, _ in all_history
        }
        date_by_memory_id = {
            f"{sid}__occ{index}": date for sid, date, _, index in all_history
        }
        store_path = (
            args.output_dir / "stores" / run_id /
            f"{item_index:04d}_{re.sub(r'[^A-Za-z0-9_-]', '_', qid)}_all"
        )
        store = ConversationStore(str(store_path))
        indexed_session_ids = []
        for sid, session_date, session, session_index in all_history:
            memory_id = f"{sid}__occ{session_index}"
            text = session_to_text(session)
            if text:
                store.store_conversation(
                    platform="longmemeval",
                    session_id=memory_id,
                    prompt=f"LongMemEval session {sid}; date {session_date}",
                    response=f"[session_date:{session_date}]\\n{text}",
                    embedding_generator=embedder,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                )
                indexed_session_ids.append(sid)

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
            k=args.candidate_pool_size,
            threshold=0.0,
            mode="hybrid",
            use_cache=False,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
        memory_units = []
        seen_memory_ids = set()
        retrieved_session_ids_ordered = []
        seen_source_ids = set()
        for hit in hits:
            metadata = hit.get("metadata", {})
            memory_id = str(metadata.get("session_id", ""))
            if not memory_id or memory_id not in source_session_by_memory_id:
                raise RuntimeError("Retrieved memory chunk has unknown session_id metadata")
            if memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory_id)
            source_sid = source_session_by_memory_id[memory_id]
            if source_sid in seen_source_ids:
                continue
            seen_source_ids.add(source_sid)
            retrieved_session_ids_ordered.append(source_sid)
            session_date = date_by_memory_id[memory_id]
            memory_units.append(
                f"[session_id:{memory_id}] [source_session_id:{source_sid}] "
                f"[session_date:{session_date}]\\n{hit.get('text', '')}"
            )
            if len(memory_units) >= args.retrieval_k:
                break

        if len(indexed_session_ids) != len(set(indexed_session_ids)):
            raise RuntimeError("Duplicate source session IDs detected in memory index")

        for history_size in history_sizes:
            selected = select_history(sessions, session_ids, session_dates, history_size)
            recent_units = [
                (
                    f"[session_id:{sid}__occ{session_index}] "
                    f"[source_session_id:{sid}] [session_date:{session_date}]\\n"
                    f"{session_to_text(session)}"
                )
                for sid, session_date, session, session_index in selected
                if session_to_text(session)
            ]
            all_history_units = [
                (
                    f"[session_id:{sid}__occ{session_index}] "
                    f"[source_session_id:{sid}] [session_date:{session_date}]\\n"
                    f"{session_to_text(session)}"
                )
                for sid, session_date, session, session_index in all_history
                if session_to_text(session)
            ]

            arms = {
                "question_only": ([], args.max_prompt_tokens),
                "recent_history": (recent_units, args.max_prompt_tokens),
                "long_context": (all_history_units, args.long_context_max_prompt_tokens),
                "external_memory": (memory_units, args.max_prompt_tokens),
            }
            for arm, (units, prompt_budget) in arms.items():
                context, prompt_tokens, selected_units = fit_context(
                    question, units, prompt_budget, counter, question_date
                )
                included_session_ids = []
                for unit in selected_units:
                    match = re.search(r"\\[source_session_id:([^\\]]+)\\]", unit)
                    if match and match.group(1) not in included_session_ids:
                        included_session_ids.append(match.group(1))
                if arm == "recent_history" and selected and not context.strip():
                    raise RuntimeError(
                        f"Recent-history context is empty for question {qid}; "
                        "refusing to generate an invalid comparison."
                    )
                start = time.perf_counter()
                generated = llm.generate(
                    build_prompt(question, context, question_date),
                    max_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                prediction = str(generated.get("response", "")).strip()
                retrieved_session_ids = (
                    retrieved_session_ids_ordered if arm == "external_memory" else []
                )
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
                    if arm == "external_memory" and retrieved_session_ids else
                    (0.0 if arm == "external_memory" else None)
                )
                retrieval_mrr = None
                if arm == "external_memory" and gold_session_ids:
                    retrieval_mrr = next(
                        (1.0 / rank for rank, sid in enumerate(retrieved_session_ids, 1)
                         if sid in gold_session_ids),
                        0.0,
                    )
                rows.append({
                    "question_id": qid,
                    "question_type": item.get("question_type"),
                    "history_size": history_size,
                    "history_sessions_available": len(selected),
                    "full_history_sessions_available": len(all_history),
                    "full_history_sessions_indexed": len(indexed_session_ids),
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
                    "retrieved_session_count": len(retrieved_session_ids),
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
                    "memory_retrieval_latency_ms": (
                        round(retrieval_latency_ms, 3) if arm == "external_memory" else None
                    ),
                    "retrieval_method": "BM25+FAISS+RRF+cross-encoder",
                })

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
        "dataset_sha256": sha256_file(args.dataset),
        "n_questions": len(items),
        "chunk_size_words": args.chunk_size,
        "chunk_overlap_words": args.overlap,
        "embedding_model": args.embedding_model,
        "retrieval_k_unique_sessions": args.retrieval_k,
        "history_sizes": history_sizes,
        "max_prompt_tokens": args.max_prompt_tokens,
        "long_context_max_prompt_tokens": args.long_context_max_prompt_tokens,
        "history_condition_definition": {
            "recent_history": "Most recent N sessions, packed newest-first and presented chronologically; oversized sessions are tail-truncated to fit.",
            "long_context": "All available sessions, packed newest-first and presented chronologically under the separately declared larger token budget.",
            "external_memory": "Index and retrieval corpus built from every non-empty available session, independent of recent-history window.",
            "question_only": "No conversation history."
        },
        "long_context_baseline_note": (
            "The long_context arm receives all available history sessions subject to "
            "the separately declared token budget; the newest evidence is prioritized "
            "if the complete history cannot fit."
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
            "mean_memory_retrieval_latency_ms": (
                round(statistics.mean(
                    float(r["memory_retrieval_latency_ms"]) for r in group
                    if r["memory_retrieval_latency_ms"] is not None
                ), 2)
                if any(r["memory_retrieval_latency_ms"] is not None for r in group) else None
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
    print(f"RUN_ID={run_id}")
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
    parser.add_argument("--history-sizes", default="5", help="History window(s); default runs only the 5-session window")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048, help="Small default prompt budget to limit input usage")
    parser.add_argument(
        "--long-context-max-prompt-tokens", type=int, default=4096,
        help="Separate larger context budget for the full-history baseline; must fit the reader model."
    )
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=20)
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=64, help="Small response cap per generation call")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=1, help="Number of questions; default is one")
    parser.add_argument("--max-generation-calls", type=int, default=4, help="Hard cap on total LLM generation calls; default permits one question across four arms")
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
