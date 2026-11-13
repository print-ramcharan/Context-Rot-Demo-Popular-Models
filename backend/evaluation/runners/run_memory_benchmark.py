"""Evaluate whether conversation memories can be retrieved across platforms.

Input is a JSONL file. Each line:
{"query_id":"q1","query":"What timezone do I use?","query_platform":"gemini",
 "memories":[{"memory_id":"m1","platform":"chatgpt","text":"User timezone is Asia/Kolkata."}],
 "relevant_memory_ids":["m1"]}

This is a retrieval benchmark, not an LLM answer-quality benchmark. It never
sends data to a model provider. Use consented/de-identified memories only.
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


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("Input JSONL contains no cases")
    for i, case in enumerate(cases, 1):
        for field in ("query_id", "query", "query_platform", "memories", "relevant_memory_ids"):
            if field not in case:
                raise ValueError(f"Case {i} is missing required field: {field}")
        ids = {m.get("memory_id") for m in case["memories"]}
        if None in ids or not set(case["relevant_memory_ids"]).issubset(ids):
            raise ValueError(f"Case {i} has missing or unknown memory_id values")
        for memory in case["memories"]:
            if not memory.get("platform") or not memory.get("text", "").strip():
                raise ValueError(f"Case {i} memories require non-empty platform and text")
    return cases


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return round(ordered[index], 3)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store_dir = args.output_dir / "stores" / run_id
    store = ConversationStore(str(store_dir))
    embedder = EmbeddingGenerator(
        model_name=args.embedding_model,
        device=args.device,
        cache_path=str(args.output_dir / "embedding_cache.sqlite"),
        cache_enabled=True,
    )

    # Each memory is written under its originating platform. Retrieval below
    # deliberately omits platform_filter to model shared cross-platform memory.
    seen: set[str] = set()
    memory_definitions: dict[str, tuple[str, str]] = {}
    for case in cases:
        for memory in case["memories"]:
            memory_id = str(memory["memory_id"])
            definition = (str(memory["platform"]), str(memory["text"]))
            if memory_id in memory_definitions and memory_definitions[memory_id] != definition:
                raise ValueError(f"Inconsistent platform/text for memory_id: {memory_id}")
            memory_definitions[memory_id] = definition
            if memory_id in seen:
                continue
            seen.add(memory_id)
            store.store_conversation(
                platform=str(memory["platform"]),
                session_id=f"memory-{memory_id}",
                prompt=f"[memory_id:{memory_id}]",
                response=str(memory["text"]),
                embedding_generator=embedder,
            )

    per_query = []
    latencies = []
    for case in cases:
        start = time.perf_counter()
        hits = store.retrieve_context(
            query=str(case["query"]),
            embedding_generator=embedder,
            top_k=args.k * 3,
            platform_filter=None,
            similarity_threshold=args.similarity_threshold,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        # Store chunks include the explicit ID marker, allowing deterministic
        # mapping back to gold memories without an LLM judge.
        ranked_ids = []
        for hit in hits:
            session_id = str(hit.get("session_id", ""))
            memory_id = session_id.removeprefix("memory-")
            if memory_id and memory_id not in ranked_ids:
                ranked_ids.append(memory_id)
            if len(ranked_ids) >= args.k:
                break
        relevant = set(map(str, case["relevant_memory_ids"]))
        retrieved_relevant = [mid for mid in ranked_ids if mid in relevant]
        first_rank = next((i + 1 for i, mid in enumerate(ranked_ids) if mid in relevant), None)
        source_platforms = {
            str(m["memory_id"]): str(m["platform"]) for m in case["memories"]
        }
        cross_platform_relevant = {
            mid for mid in relevant
            if source_platforms[mid] != str(case["query_platform"])
        }
        cross_platform_hits = [mid for mid in ranked_ids if mid in cross_platform_relevant]
        per_query.append({
            "query_id": case["query_id"],
            "query_platform": case["query_platform"],
            "retrieved_memory_ids": ranked_ids,
            "relevant_memory_ids": sorted(relevant),
            "retrieved_relevant_count": len(retrieved_relevant),
            "recall_at_k": len(retrieved_relevant) / len(relevant) if relevant else None,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "cross_platform_relevant_count": len(cross_platform_relevant),
            "cross_platform_recall_at_k": (
                len(cross_platform_hits) / len(cross_platform_relevant)
                if cross_platform_relevant else None
            ),
            "retrieval_latency_ms": round(latency_ms, 3),
            "retrieved_platforms": [hit.get("platform") for hit in hits],
        })

    stats = store.get_stats()
    storage_bytes = sum(p.stat().st_size for p in store_dir.rglob("*") if p.is_file())
    recalls = [r["recall_at_k"] for r in per_query if r["recall_at_k"] is not None]
    reciprocal_ranks = [r["reciprocal_rank"] for r in per_query]
    cross_recalls = [
        r["cross_platform_recall_at_k"] for r in per_query
        if r["cross_platform_recall_at_k"] is not None
    ]
    result = {
        "benchmark": "cross_platform_conversation_memory_retrieval",
        "note": "Retrieval-only; does not measure generated answer quality.",
        "embedding_model": args.embedding_model,
        "run_id_utc": run_id,
        "k": args.k,
        "n_queries": len(per_query),
        "n_unique_memories": len(seen),
        "metrics": {
            f"Recall@{args.k}": round(statistics.mean(recalls), 6) if recalls else None,
            f"MRR@{args.k}": round(statistics.mean(reciprocal_ranks), 6),
            f"cross_platform_Recall@{args.k}": round(statistics.mean(cross_recalls), 6) if cross_recalls else None,
            "retrieval_latency_ms_mean": round(statistics.mean(latencies), 3) if latencies else None,
            "retrieval_latency_ms_p50": percentile(latencies, 0.50),
            "retrieval_latency_ms_p95": percentile(latencies, 0.95),
            "conversation_store_bytes": storage_bytes,
            "conversation_store_mib": round(storage_bytes / (1024 * 1024), 4),
            "index_vectors": stats["index_size"],
            "stored_chunks": stats["total_chunks"],
            "stored_sessions": stats["total_sessions"],
        },
        "per_query": per_query,
    }
    (args.output_dir / f"memory_retrieval_results_{run_id}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Benchmark JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/memory"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be >= 1")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
