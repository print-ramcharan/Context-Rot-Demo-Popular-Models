"""Profile BEIR retrieval resource use as corpus size increases.

Each scale runs in a fresh subprocess so peak RSS is not contaminated by
previous scale measurements. This is a retrieval/index scalability experiment;
it does not measure hosted generation cost or multi-user concurrency.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import yaml

from evaluation.datasets.load_beir import chunk_corpus, load_beir_dataset
from evaluation.runners.retrieval_configs import CONFIGS
from evaluation.runners.run_retrieval import (
    _embed_chunks,
    _make_retriever,
    _run_config_with_depth,
)
from src.embedding import EmbeddingGenerator
from src.retrieval import BM25Index
from src.vector_store import VectorStore


def current_rss_mb() -> float | None:
    """Read current process RSS in MB using ps (macOS/Linux)."""
    try:
        value = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        return int(value) / 1024.0
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def peak_rss_mb() -> float:
    """Return process high-water RSS; ru_maxrss is bytes on macOS, KiB on Linux."""
    import resource
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def deep_size_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate Python-managed object size; excludes native allocator overhead."""
    import sys as _sys
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = _sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size_bytes(k, seen) + deep_size_bytes(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(deep_size_bytes(item, seen) for item in value)
    return size


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * p + 0.5)))
    return round(ordered[index], 3)


def worker(dataset: str, fraction: float, query_count: int, output: Path) -> dict:
    backend_dir = Path(__file__).resolve().parents[2]
    evaluation_dir = backend_dir / "evaluation"
    config = yaml.safe_load((backend_dir / "config.yaml").read_text(encoding="utf-8"))
    experiment = yaml.safe_load(
        (evaluation_dir / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    corpus, queries, _ = load_beir_dataset(evaluation_dir / "data" / dataset, dataset)
    ordered_ids = sorted(corpus)
    selected_count = max(1, min(len(ordered_ids), int(len(ordered_ids) * fraction)))
    selected_ids = ordered_ids[:selected_count]
    selected_corpus = {doc_id: corpus[doc_id] for doc_id in selected_ids}

    embedding_cfg = experiment["embedding"]
    retrieval_cfg = experiment["retrieval"]
    chunk_cfg = experiment["chunking"]
    embedder = EmbeddingGenerator(
        model_name=embedding_cfg["model_name"],
        device=embedding_cfg.get("device", "cpu"),
        query_instruction=embedding_cfg.get("query_instruction", ""),
        cache_enabled=False,
    )
    rss_after_model = current_rss_mb()

    chunks = chunk_corpus(
        selected_corpus,
        int(chunk_cfg["chunk_size"]),
        int(chunk_cfg["overlap"]),
    )
    texts = [item["text"] for item in chunks]
    metadata = [item["metadata"] for item in chunks]
    embedding_texts = [
        ("Section: " + " > ".join(item.get("metadata", {}).get("section_path", []))
         + " " + item["text"]).strip()
        for item in chunks
    ]

    tracemalloc.start()
    embedding_started = time.perf_counter()
    embeddings = _embed_chunks(embedder, embedding_texts)
    embedding_build_ms = (time.perf_counter() - embedding_started) * 1000.0

    index_started = time.perf_counter()
    vector_store = VectorStore(
        dimension=embedder.get_embedding_dimension(),
        index_type=experiment["index"].get("type", "cosine"),
    )
    vector_store.add(embeddings, texts, metadata)
    faiss_build_ms = (time.perf_counter() - index_started) * 1000.0

    bm25_started = time.perf_counter()
    bm25 = BM25Index(texts, k1=1.5, b=0.75)
    bm25_build_ms = (time.perf_counter() - bm25_started) * 1000.0
    _, python_heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_build = current_rss_mb()
    process_peak = peak_rss_mb()

    with tempfile.TemporaryDirectory(prefix="rag-scale-") as temp_dir:
        index_dir = Path(temp_dir) / "faiss"
        vector_store.save(str(index_dir))
        faiss_disk_bytes = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file())

    retrievers = {
        config_id: _make_retriever(
            vector_store, embedder, retrieval_cfg, enable_rerank=False
        )
        for config_id, cfg in CONFIGS.items()
        if cfg["mode"] == "hybrid"
    }
    reranker = None
    if CONFIGS["B2"]["rerank"]:
        from src.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(
            model_name=retrieval_cfg["rerank_model"],
        )
        if reranker.model is None:
            raise RuntimeError("B2 cross-encoder did not load; refusing a mislabeled profile")

    selected_queries = [queries[qid] for qid in sorted(queries)[:query_count]]
    timing: dict[str, list[float]] = {key: [] for key in CONFIGS}
    rerank_timing: list[float] = []
    returned_counts: dict[str, list[int]] = {key: [] for key in CONFIGS}
    for query in selected_queries:
        for config_id, cfg in CONFIGS.items():
            started = time.perf_counter()
            results, diagnostics = _run_config_with_depth(
                query, cfg, chunks, vector_store, embedder, bm25,
                retrieval_cfg, retrievers.get(config_id), reranker,
            )
            timing[config_id].append((time.perf_counter() - started) * 1000.0)
            returned_counts[config_id].append(len(results))
            if config_id == "B2":
                rerank_timing.append(float(diagnostics["reranking_latency_ms"]))

    result = {
        "dataset": dataset,
        "document_fraction": fraction,
        "documents": len(selected_corpus),
        "chunks": len(chunks),
        "queries_profiled": len(selected_queries),
        "embedding_model": embedding_cfg["model_name"],
        "reranker_model": retrieval_cfg["rerank_model"],
        "embedding_build_ms": round(embedding_build_ms, 3),
        "faiss_build_ms": round(faiss_build_ms, 3),
        "bm25_build_ms": round(bm25_build_ms, 3),
        "embedding_matrix_bytes": int(embeddings.nbytes),
        "faiss_index_disk_bytes": int(faiss_disk_bytes),
        "bm25_python_object_bytes_estimate": int(
            deep_size_bytes(bm25.inverted_index)
            + deep_size_bytes(bm25.doc_freq)
            + deep_size_bytes(bm25.doc_len)
        ),
        "python_heap_peak_bytes_during_build": int(python_heap_peak),
        "rss_after_model_load_mb": rss_after_model,
        "rss_after_index_build_mb": rss_after_build,
        "process_peak_rss_mb": round(process_peak, 3),
        "retrieval": {
            config_id: {
                "mean_ms": round(sum(values) / len(values), 3) if values else None,
                "p50_ms": percentile(values, 0.50),
                "p95_ms": percentile(values, 0.95),
                "mean_returned_unique_parents": round(
                    sum(returned_counts[config_id]) / len(returned_counts[config_id]), 3
                ) if returned_counts[config_id] else None,
            }
            for config_id, values in timing.items()
        },
        "B2_reranking_mean_ms": (
            round(sum(rerank_timing) / len(rerank_timing), 3) if rerank_timing else None
        ),
        "limitations": [
            "Single-process, single-query-at-a-time measurements; not a concurrency benchmark.",
            "RSS includes model/runtime memory; it is not an isolated index-only measurement.",
            "BM25 Python object size is an estimate and excludes interpreter/native allocator overhead.",
            "Document subsets are deterministic sorted-ID prefixes, not random samples.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="scifact,nfcorpus")
    parser.add_argument("--scales", default="0.10,0.25,0.50,1.00")
    parser.add_argument("--query-count", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/scalability"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    parser.add_argument("--fraction", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.dataset or args.fraction is None or args.output is None:
            parser.error("worker mode requires --dataset, --fraction, and --output")
        worker(args.dataset, args.fraction, args.query_count, args.output)
        return
    if args.query_count < 1:
        parser.error("--query-count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets.split(","):
        for fraction_text in args.scales.split(","):
            fraction = float(fraction_text)
            if not 0 < fraction <= 1:
                parser.error("each scale must be in (0, 1]")
            output = args.output_dir / f"{dataset}_{int(fraction * 100):03d}pct.json"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--dataset", dataset, "--fraction", str(fraction),
                "--query-count", str(args.query_count), "--output", str(output),
            ]
            subprocess.run(command, check=True)
            rows.append(json.loads(output.read_text(encoding="utf-8")))

    manifest = {
        "benchmark": "BEIR retrieval resource and corpus-scaling profile",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration_mapping": CONFIGS,
        "scale_runs": [str(row) for row in sorted(args.output_dir.glob("*.json"))],
        "results": rows,
        "note": "No result values are generated unless the worker processes complete successfully.",
    }
    (args.output_dir / "scalability_summary.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved scalability summary: {args.output_dir / 'scalability_summary.json'}")


if __name__ == "__main__":
    main()
