"""Run the retrieval benchmark against BEIR SciFact / NFCorpus.

Writes per-query JSON (raw), dataset-level metrics (processed), and a
comparison table (tables) under ``backend/evaluation/results``.

Configurations
--------------
B1 : BM25 only (sparse)
B2 : Dense FAISS only
B3 : BM25 + dense + RRF
B4 : B3 + cross-encoder reranking
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.chunking import TextChunker
from src.embedding import EmbeddingGenerator
from src.retrieval import BM25Index, SemanticRetriever
from src.vector_store import VectorStore

from ..datasets.load_beir import chunk_corpus, load_beir_dataset
from ..metrics.retrieval_metrics import compute_query_metrics
from .retrieval_configs import CONFIGS
from ._doc_id_map import parent_scores_for_query

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF_K = 20


def _make_retriever(
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    cfg: Dict[str, Any],
    enable_rerank: bool,
) -> SemanticRetriever:
    return SemanticRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        top_k=cfg["top_k"],
        similarity_threshold=cfg["similarity_threshold"],
        mode=cfg["mode"],
        dense_k=cfg["dense_k"],
        bm25_k=cfg["bm25_k"],
        alpha=cfg["alpha"],
        enable_rerank=enable_rerank,
        rerank_top_n=cfg["rerank_top_n"],
        rerank_candidate_pool=cfg["rerank_candidate_pool"],
    )


def _bm25_chunks(
    query: str, bm25_index: BM25Index, cfg: Dict[str, Any], k: int
) -> List[Dict[str, Any]]:
    scores = bm25_index.score(query)
    top_n = min(int(cfg["bm25_k"]), len(scores))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    chunks = []
    for rank, idx in enumerate(ranked):
        chunks.append(
            {
                "text": bm25_index.texts[idx],
                "score": float(scores[idx]),
                "rank": rank,
                "metadata": {
                    "doc_id": f"bm25-doc-{idx}",
                    "parent_id": f"bm25-doc-{idx}",
                    "source": "bm25",
                },
            }
        )
    return chunks


def _dense_chunks(
    query: str,
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    k: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    query_emb = embedding_generator.embed_text(query)
    results = vector_store.search(query_emb, k=k)
    chunks = []
    for i in range(len(results["chunks"])):
        score = results["distances"][i]
        if threshold > 0 and score < threshold:
            continue
        chunks.append(
            {
                "text": results["chunks"][i],
                "score": float(score),
                "rank": i,
                "metadata": {
                    "doc_id": str(i),
                    "parent_id": str(i),
                    "source": "dense",
                },
            }
        )
    return chunks


def _map_to_parent_records(
    chunks: List[Dict[str, Any]], qrels: Dict[str, Dict[str, int]], query_id: str
) -> Tuple[List[str], List[int], List[int]]:
    parent_ids = []
    scores = []
    grades = []
    seen = set()
    for c in chunks:
        meta = c.get("metadata", {}) if isinstance(c, dict) else {}
        pid = meta.get("parent_id") or meta.get("doc_id") or c.get("text", "")
        if pid in seen:
            continue
        seen.add(pid)
        parent_ids.append(str(pid))
        scores.append(float(c.get("score", 0.0)))
        grades.append(int(qrels.get(query_id, {}).get(str(pid), 0)))
    return parent_ids, scores, grades

def run_dataset(
    dataset_name: str,
    data_path: Path,
    cfg: Dict[str, Any],
    embedding_generator: EmbeddingGenerator,
    chunk_size: int,
    overlap: int,
    index_type: str,
    trace_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    corpus, queries, qrels = load_beir_dataset(data_path, dataset_name)
    logger.info("Loaded %s: %d documents, %d queries", dataset_name, len(corpus), len(queries))

    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunk_corpus(corpus, chunk_size, overlap)
    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]
    dim = embedding_generator.get_embedding_dimension()

    vector_store = VectorStore(dimension=dim, index_type=index_type)
    vector_store.add(embeddings=embeddings, chunks=texts, metadata=metadata)
    bm25_index = BM25Index(texts, k1=1.5, b=0.75)

    per_query_records = []
    for qid, query_text in queries.items():
        for config_id, config in CONFIGS.items():
            config_start = time.perf_counter()
            retrieved_chunks = _run_config(
                query_text, config, chunks, vector_store, embedding_generator, bm25_index, cfg
            )
            config_ms = (time.perf_counter() - config_start) * 1000.0

            parent_ids, scores, grades = _map_to_parent_records(retrieved_chunks, qrels, qid)
            total_relevant = int(sum(1 for _ in qrels.get(qid, {}).values()))
            metrics = compute_query_metrics(scores, total_relevant)

            rerank_ms = 0.0
            if config["rerank"]:
                rerank_ms = _measure_rerank(query_text, retrieved_chunks, vector_store, embedding_generator, cfg)

            record = {
                "dataset": dataset_name,
                "query_id": qid,
                "configuration": config_id,
                "retrieved_document_ids": parent_ids,
                "retrieval_latency_ms": round(config_ms, 3),
                "reranking_latency_ms": round(rerank_ms, 3),
                "scores": scores,
                "grades": grades,
                "metrics": metrics,
            }
            per_query_records.append(record)

    trace_dir.mkdir(parents=True, exist_ok=True)
    for rec in per_query_records:
        out_path = trace_dir / f"{rec['dataset']}_{rec['configuration']}_{rec['query_id']}.json"
        out_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    return aggregate_records(per_query_records), per_query_records


def _run_config(
    query: str,
    config: Dict[str, Any],
    chunks,
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    bm25_index: BM25Index,
    retrieval_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    top_k = int(retrieval_cfg["top_k"])
    threshold = float(retrieval_cfg["similarity_threshold"])
    if config["mode"] == "bm25":
        return _bm25_chunks(query, bm25_index, retrieval_cfg, top_k)
    if config["mode"] == "dense":
        return _dense_chunks(query, vector_store, embedding_generator, top_k, threshold)
    retriever = _make_retriever(vector_store, embedding_generator, retrieval_cfg, enable_rerank=config["rerank"])
    return retriever.retrieve(query, k=top_k, threshold=threshold, mode="hybrid", use_cache=False)


def _measure_rerank(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    retrieval_cfg: Dict[str, Any],
) -> float:
    from src.reranker import CrossEncoderReranker

    pool = retrieved_chunks[: int(retrieval_cfg["rerank_candidate_pool"])]
    reranker = CrossEncoderReranker(model_name=retrieval_cfg["rerank_model"])
    start = time.perf_counter()
    reranker.rerank(query, pool, top_n=int(retrieval_cfg["rerank_top_n"]))
    return (time.perf_counter() - start) * 1000.0


def aggregate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for rec in records:
        key = (rec["dataset"], rec["configuration"])
        out.setdefault(key, []).append(rec)
    summary = {}
    for (dataset, config_id), items in out.items():
        means = {}
        for metric in ["recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "precision@10"]:
            means[metric] = round(float(np.mean([r["metrics"][metric] for r in items])), 6)
        summary[f"{dataset}:{config_id}"] = {
            "label": CONFIGS[config_id]["label"],
            "n_queries": len(items),
            "mean": means,
        }
    return summary


def save_summary(summary: Dict[str, Any], results_dir: Path) -> None:
    processed = results_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for (dataset, config_id), data in summary.items():
        for metric, value in data["mean"].items():
            table_rows.append(
                {
                    "dataset": dataset,
                    "configuration": config_id,
                    "label": data["label"],
                    "metric": metric,
                    "mean": value,
                    "n_queries": data["n_queries"],
                }
            )
    if not table_rows:
        return
    table_path = processed / "metrics_table.csv"
    header = ",".join(table_rows[0].keys())
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in table_rows:
            f.write(",".join(str(v) for v in row.values()) + "\n")
    logger.info("Saved aggregated results to %s", table_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_config()

    embedding_cfg = cfg["embedding"]
    retrieval_cfg = cfg["retrieval"]
    dataset_names = cfg["datasets"]
    chunk_cfg = cfg["chunking"]
    index_type = cfg["index"].get("type", "cosine")
    results_dir = Path(cfg["results"]["raw"])

    embedding_generator = EmbeddingGenerator(
        model_name=embedding_cfg["model_name"], device=embedding_cfg.get("device", "cpu")
    )

    for dataset_name in dataset_names:
        data_path = Path(f"evaluation/data/{dataset_name}")
        trace_dir = results_dir / dataset_name
        summary, _ = run_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            cfg=retrieval_cfg,
            embedding_generator=embedding_generator,
            chunk_size=chunk_cfg["chunk_size"],
            overlap=chunk_cfg["overlap"],
            index_type=index_type,
            trace_dir=trace_dir,
        )
        logger.info("Dataset %s summary: %s", dataset_name, summary)
    save_summary(summary, results_dir)


if __name__ == "__main__":
    main()

