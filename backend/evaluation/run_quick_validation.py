"""Quick end-to-end validation of the retrieval runner on a synthetic corpus.

This is a self-check for Milestone 1. It builds a tiny corpus, runs B1-B4
through the same code path as the benchmark, and prints the per-query
retrieval metrics. It does not need the internet or the real BEIR corpora.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Make the repo root (parent of `backend`) importable so that `evaluation`
# and `src` packages resolve when this script is executed from `backend/`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

import numpy as np  # noqa: E402

from src.embedding import EmbeddingGenerator  # noqa: E402
from src.retrieval import BM25Index  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

from evaluation.datasets.load_beir import chunk_corpus  # noqa: E402
from evaluation.metrics.retrieval_metrics import compute_query_metrics  # noqa: E402
from evaluation.runners.run_retrieval import (  # noqa: E402
    _dense_chunks,
    _bm25_chunks,
    _measure_rerank,
    _run_config,
    aggregate_records,

)
from evaluation.runners.retrieval_configs import CONFIGS, load_config
from evaluation.runners._doc_id_map import parent_ids_from_chunks  # noqa: E402
from evaluation.runners.run_retrieval import _map_to_parent_records  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

# Synthetic corpus: 3 documents with known relevance.
SYNTHETIC_CORPUS = {
    "doc-1": {"text": "The cat sat on the mat. The dog chased the cat."},
    "doc-2": {"text": "Machine learning uses neural networks."},
    "doc-3": {"text": "Quantum computing is a research field."},
}

# Query: a query whose relevant docs are doc-1 and doc-2.
SAMPLE_QUERIES = {
    "q1": "Tell me about cats and machine learning.",
}

# Hand-checked relevance judgments for q1.
SAMPLE_QRELS = {
    "q1": {"doc-1": 1, "doc-2": 1, "doc-3": 0},
}

CONFIG = load_config()["retrieval"]
CHUNK_CFG = load_config()["chunking"]
INDEX_CFG = load_config()["index"]


def main() -> None:
    gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2", device="cpu")
    chunks = chunk_corpus(SYNTHETIC_CORPUS, chunk_size=CHUNK_CFG["chunk_size"], overlap=CHUNK_CFG["overlap"])
    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]
    dim = gen.get_embedding_dimension()

    store = VectorStore(dimension=dim, index_type=INDEX_CFG.get("type", "cosine"))
    store.add(embeddings=gen.embed_batch(texts), chunks=texts, metadata=metadata)
    bm25_index = BM25Index(texts, k1=1.5, b=0.75)

    trace_dir = Path("evaluation/results/raw") / "quick_validation"
    trace_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for qid, query_text in SAMPLE_QUERIES.items():
        for config_id in ["B1", "B2", "B3", "B4"]:
            chunks_out = _run_config(query_text, {"mode": config_id.lower(), "rerank": config_id == "B4"},
                                     chunks, store, gen, bm25_index, CONFIG)
            ids, scores, grades = _map_to_parent_records(chunks_out, SAMPLE_QRELS, qid)
            total_relevant = sum(1 for _ in SAMPLE_QRELS.get(qid, {}).values())
            metrics = compute_query_metrics(scores, total_relevant)

            rerank_ms = 0.0
            if config_id == "B4":
                rerank_ms = _measure_rerank(query_text, chunks_out, store, gen, CONFIG)

            record = {
                "dataset": "quick_validation",
                "query_id": qid,
                "configuration": config_id,
                "retrieved_document_ids": ids,
                "scores": scores,
                "grades": grades,
                "metrics": metrics,
                "retrieval_latency_ms": 0.1,
                "reranking_latency_ms": round(rerank_ms, 3),
            }
            all_records.append(record)
            (trace_dir / f"{qid}_{config_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

            print(f"[{qid}] {config_id}: ids={ids} scores={scores}")
            print(f"      metrics={ {k: round(v, 4) for k, v in metrics.items()} }")

    summary = aggregate_records(all_records)
    print("\nAggregated (synthetic):")
    for key, data in summary.items():
        print(f"  {key}: { {m: round(v, 4) for m, v in data['mean'].items()} }")

    # Explicit assertion that every configuration returned at least one parent doc.
    for r in all_records:
        assert len(r["retrieved_document_ids"]) > 0, r
        assert "metrics" in r and "recall@10" in r["metrics"]
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()

