"""Run session-level evidence retrieval on the official LongMemEval JSON dataset.

This runner evaluates retrieval only (gold evidence-session Recall@K and MRR).
It does not generate answers or call an LLM judge. Each question gets an
isolated ConversationStore so evidence from other questions cannot leak in.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conversation_store import ConversationStore
from src.embedding import EmbeddingGenerator


def _session_text(session: list[dict[str, Any]]) -> tuple[str, str]:
    user_parts, assistant_parts = [], []
    for turn in session:
        role = str(turn.get("role", "")).lower()
        text = str(turn.get("content", "")).strip()
        if not text:
            continue
        if role == "user":
            user_parts.append(text)
        elif role == "assistant":
            assistant_parts.append(text)
        else:
            # Preserve unexpected roles instead of silently dropping content.
            user_parts.append(f"[{role or 'unknown'}] {text}")
    return "\n".join(user_parts), "\n".join(assistant_parts)


def _recall(retrieved: list[str], relevant: set[str]) -> float:
    return len(set(retrieved) & relevant) / len(relevant) if relevant else 0.0


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))], 3)


def evaluate_item(
    item: dict[str, Any],
    dataset_index: int,
    dataset_path: Path,
    output_root: Path,
    embedder: EmbeddingGenerator,
    top_k: int,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    question_id = str(item["question_id"])
    # Per-question isolation is required: one benchmark item must not retrieve
    # memories ingested for another item.
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in question_id)
    store_path = output_root / "stores" / run_id / f"{dataset_index:04d}_{safe_id}"
    store = ConversationStore(str(store_path))

    sessions = item.get("haystack_sessions", [])
    session_ids = item.get("haystack_session_ids", [])
    if len(sessions) != len(session_ids):
        raise ValueError(f"{question_id}: haystack_sessions/session_ids length mismatch")

    for sid, session in zip(session_ids, sessions):
        user_text, assistant_text = _session_text(session)
        marker = f"[longmemeval_session_id:{sid}]"
        # The marker is present in both sides so it survives the standard
        # prompt/response exchange representation for ordinary session sizes.
        store.store_conversation(
            platform="longmemeval",
            session_id=str(sid),
            prompt=f"{marker}\n{user_text}",
            response=f"{marker}\n{assistant_text}",
            embedding_generator=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    started = time.perf_counter()
    hits = store.retrieve_context(
        query=str(item["question"]),
        embedding_generator=embedder,
        top_k=top_k,
        platform_filter=None,
        similarity_threshold=0.0,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    ranked_sessions = []
    for hit in hits:
        text = hit.get("text", "")
        marker = "[longmemeval_session_id:"
        if marker not in text:
            continue
        sid = text.split(marker, 1)[1].split("]", 1)[0]
        if sid not in ranked_sessions:
            ranked_sessions.append(sid)

    relevant = set(map(str, item.get("answer_session_ids", [])))
    first_rank = next(
        (rank for rank, sid in enumerate(ranked_sessions, 1) if sid in relevant),
        None,
    )
    stats = store.get_stats()
    storage_bytes = sum(p.stat().st_size for p in store_path.rglob("*") if p.is_file())
    return {
        "question_id": question_id,
        "question_type": item.get("question_type"),
        "retrieved_session_ids": ranked_sessions,
        "answer_session_ids": sorted(relevant),
        f"session_recall@{top_k}": _recall(ranked_sessions, relevant),
        f"session_mrr@{top_k}": 1.0 / first_rank if first_rank else 0.0,
        "retrieval_latency_ms": round(latency_ms, 3),
        "stored_chunks": stats["total_chunks"],
        "index_vectors": stats["index_size"],
        "store_bytes": storage_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Official LongMemEval JSON file (S, M, or oracle)")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/longmemeval"))
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all questions")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.top_k < 1 or args.chunk_size < 1 or args.overlap < 0 or args.overlap >= args.chunk_size:
        parser.error("Require top-k >= 1 and 0 <= overlap < chunk-size")

    items = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("LongMemEval dataset must be a non-empty JSON array")
    if args.limit:
        items = items[:args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedder = EmbeddingGenerator(
        model_name=args.embedding_model,
        device=args.device,
        cache_path=str(args.output_dir / "embedding_cache.sqlite"),
        cache_enabled=True,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for index, item in enumerate(items):
        rows.append(evaluate_item(
            item, index, args.dataset, args.output_dir, embedder,
            args.top_k, args.chunk_size, args.overlap,
        ))
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(items)} questions")

    recall_key = f"session_recall@{args.top_k}"
    mrr_key = f"session_mrr@{args.top_k}"
    latencies = [row["retrieval_latency_ms"] for row in rows]
    summary = {
        "benchmark": "LongMemEval session evidence retrieval",
        "dataset_path": str(args.dataset),
        "run_id_utc": run_id,
        "n_questions": len(rows),
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "metrics": {
            recall_key: round(statistics.mean(row[recall_key] for row in rows), 6),
            mrr_key: round(statistics.mean(row[mrr_key] for row in rows), 6),
            "retrieval_latency_ms_mean": round(statistics.mean(latencies), 3),
            "retrieval_latency_ms_p50": _percentile(latencies, 0.50),
            "retrieval_latency_ms_p95": _percentile(latencies, 0.95),
            "total_store_bytes": sum(row["store_bytes"] for row in rows),
        },
        "by_question_type": {},
        "per_question": rows,
        "limitations": [
            "Retrieval-only; no answer generation or official LLM-as-judge scoring.",
            "Session-level recall uses answer_session_ids; it is not answer accuracy.",
            "Results depend on chunking, top-k, embedding model, and compute environment.",
        ],
    }
    for question_type in sorted({str(row.get("question_type")) for row in rows}):
        subset = [row for row in rows if str(row.get("question_type")) == question_type]
        summary["by_question_type"][question_type] = {
            "n_questions": len(subset),
            recall_key: round(statistics.mean(row[recall_key] for row in subset), 6),
            mrr_key: round(statistics.mean(row[mrr_key] for row in subset), 6),
        }

    (args.output_dir / "longmemeval_retrieval_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
