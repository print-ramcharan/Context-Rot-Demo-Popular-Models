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
import subprocess
import time
import hashlib
import argparse
import importlib.metadata
import csv
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.embedding import EmbeddingGenerator
from src.retrieval import BM25Index, SemanticRetriever
from src.vector_store import VectorStore

from ..datasets.load_beir import chunk_corpus, load_beir_dataset
from ..metrics.retrieval_metrics import compute_query_metrics
from ..metrics.statistical_tests import (
    paired_bootstrap_ci,
    wilcoxon_signed_rank_test,
    holm_bonferroni,
)
from .retrieval_configs import CONFIGS, load_config
logger = logging.getLogger(__name__)

DEFAULT_CUTOFF_K = 20


def _make_retriever(
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    cfg: Dict[str, Any],
    enable_rerank: bool,
) -> SemanticRetriever:
    retriever = SemanticRetriever(
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
    if enable_rerank:
        from src.reranker import CrossEncoderReranker

        retriever.set_reranker(CrossEncoderReranker(model_name=cfg["rerank_model"]))
    return retriever


def _embed_chunks(embedding_generator: EmbeddingGenerator, texts: List[str], batch_size: int = 128) -> np.ndarray:
    """Embed large corpora in bounded batches to avoid native allocator failures."""
    batches = []
    for start in range(0, len(texts), batch_size):
        batches.append(
            embedding_generator.embed_batch(
                texts[start : start + batch_size],
                batch_size=min(32, batch_size),
                show_progress=False,
            )
        )
    return np.vstack(batches) if batches else np.empty((0, embedding_generator.get_embedding_dimension()), dtype=np.float32)


def _bm25_chunks(
    query: str,
    bm25_index: BM25Index,
    source_chunks: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    k: int,
) -> List[Dict[str, Any]]:
    scores = bm25_index.score(query)
    top_n = min(int(cfg["bm25_k"]), k, len(scores))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    results = []
    for rank, idx in enumerate(ranked):
        results.append(
            {
                "text": bm25_index.texts[idx],
                "score": float(scores[idx]),
                "rank": rank,
                "metadata": {
                    **source_chunks[idx].get("metadata", {}),
                    "source": "bm25",
                },
            }
        )
    return results


def _dense_chunks(
    query: str,
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    k: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    query_emb = embedding_generator.embed_query(query)
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
                    **results["metadata"][i],
                    "source": "dense",
                },
            }
        )
    return chunks


def _map_to_parent_records(
    chunks: List[Dict[str, Any]],
    qrels: Dict[str, Dict[str, int]],
    query_id: str,
    evaluation_depth: int = DEFAULT_CUTOFF_K,
    include_diagnostics: bool = False,
) -> Tuple[List[str], List[float], List[int]] | Tuple[List[str], List[float], List[int], Dict[str, int]]:
    parent_candidates = _collapse_parent_candidates(chunks)
    parent_candidates = parent_candidates[:evaluation_depth]
    parent_ids = [candidate["parent_id"] for candidate in parent_candidates]
    scores = [candidate["score"] for candidate in parent_candidates]
    grades = [int(qrels.get(query_id, {}).get(pid, 0)) for pid in parent_ids]
    diagnostics = {
        "unique_parent_count_before_reranking": len(_collapse_parent_candidates(chunks)),
        "parent_deduplication_count": len(chunks) - len(_collapse_parent_candidates(chunks)),
    }
    if include_diagnostics:
        return parent_ids, scores, grades, diagnostics
    return parent_ids, scores, grades


def _collapse_parent_candidates(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the highest-scoring representative for each parent in rank order."""
    by_parent: Dict[str, Dict[str, Any]] = {}
    for rank, c in enumerate(chunks):
        meta = c.get("metadata", {}) if isinstance(c, dict) else {}
        pid = meta.get("parent_id") or meta.get("doc_id") or c.get("text", "")
        if pid is None:
            continue
        pid = str(pid)
        candidate = dict(c)
        candidate["parent_id"] = pid
        candidate["rank"] = rank
        candidate["score"] = float(c.get("score", 0.0))
        existing = by_parent.get(pid)
        if existing is None or candidate["score"] > existing["score"]:
            by_parent[pid] = candidate
    return sorted(by_parent.values(), key=lambda item: (-item["score"], item["rank"]))

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

    chunks = chunk_corpus(corpus, chunk_size, overlap)
    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]
    embedding_texts = [
        ("Section: " + " > ".join(item.get("metadata", {}).get("section_path", []))
         + "\\n" + item["text"]).strip()
        for item in chunks
    ]
    dim = embedding_generator.get_embedding_dimension()

    vector_store = VectorStore(dimension=dim, index_type=index_type)
    embeddings = _embed_chunks(embedding_generator, embedding_texts)
    vector_store.add(embeddings=embeddings, chunks=texts, metadata=metadata)
    bm25_index = BM25Index(texts, k1=1.5, b=0.75)
    retrievers = {
        config_id: _make_retriever(
            vector_store, embedding_generator, cfg, enable_rerank=False
        )
        for config_id, config in CONFIGS.items()
        if config["mode"] == "hybrid"
    }
    reranker = None
    if cfg.get("rerank_enabled", False):
        from src.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=cfg["rerank_model"])

    per_query_records = []
    latency_cfg = cfg.get("latency", {})
    latency_sample_size = max(0, int(latency_cfg.get("sample_size", 30)))
    latency_repeats = max(1, int(latency_cfg.get("repeats", 3)))
    query_ids = sorted(queries)
    sampled_ids = set(random.Random(int(cfg.get("seed", 42))).sample(
        query_ids, min(latency_sample_size, len(query_ids))
    )) if latency_sample_size else set()

    for qid in query_ids:
        query_text = queries[qid]
        for config_id, config in CONFIGS.items():
            config_start = time.perf_counter()
            retrieved_chunks, depth = _run_config_with_depth(
                query_text,
                config,
                chunks,
                vector_store,
                embedding_generator,
                bm25_index,
                cfg,
                retrievers.get(config_id),
                reranker,
            )
            config_ms = (time.perf_counter() - config_start) * 1000.0
            latency_samples_ms = [config_ms]
            if qid in sampled_ids:
                for _ in range(latency_repeats - 1):
                    repeat_start = time.perf_counter()
                    _run_config_with_depth(
                        query_text,
                        config,
                        chunks,
                        vector_store,
                        embedding_generator,
                        bm25_index,
                        cfg,
                        retrievers.get(config_id),
                        reranker,
                    )
                    latency_samples_ms.append(
                        (time.perf_counter() - repeat_start) * 1000.0
                    )

            parent_ids, scores, grades, parent_diag = _map_to_parent_records(
                retrieved_chunks, qrels, qid, int(cfg["evaluation_depth"]), include_diagnostics=True
            )
            total_relevant = int(sum(grade > 0 for grade in qrels.get(qid, {}).values()))
            qrel_grades = list(qrels.get(qid, {}).values())
            metrics = compute_query_metrics(grades, total_relevant, qrel_grades=qrel_grades)

            rerank_ms = float(depth.get("reranking_latency_ms", 0.0))

            record = {
                "dataset": dataset_name,
                "query_id": qid,
                "configuration": config_id,
                "retrieved_document_ids": parent_ids,
                "retrieval_latency_ms": round(max(0.0, config_ms - rerank_ms), 3),
                "reranking_latency_ms": round(rerank_ms, 3),
                "scores": scores,
                "grades": grades,
                "retrieved_count": len(parent_ids),
                "latency_samples_ms": [round(value, 3) for value in latency_samples_ms]
                if qid in sampled_ids else [],
                "latency_repeated": qid in sampled_ids,
                "candidate_chunk_count": depth["candidate_chunk_count"],
                "unique_parent_count_before_reranking": depth["unique_parent_count_before_reranking"],
                "unique_parent_count_after_reranking": depth["unique_parent_count_after_reranking"],
                "requested_depth": int(cfg["evaluation_depth"]),
                "depth_shortfall": max(0, int(cfg["evaluation_depth"]) - len(parent_ids)),
                "filter_rejection_count": depth["filter_rejection_count"],
                "parent_deduplication_count": depth["candidate_chunk_count"] - depth["unique_parent_count_before_reranking"],
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
    retriever: SemanticRetriever | None = None,
) -> List[Dict[str, Any]]:
    results, _ = _run_config_with_depth(
        query, config, chunks, vector_store, embedding_generator, bm25_index,
        retrieval_cfg, retriever, None
    )
    return results


def _run_config_with_depth(
    query: str,
    config: Dict[str, Any],
    chunks,
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    bm25_index: BM25Index,
    retrieval_cfg: Dict[str, Any],
    retriever: SemanticRetriever | None = None,
    reranker: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_pool_size = int(retrieval_cfg.get("candidate_pool_size", retrieval_cfg.get("top_k", 20)))
    threshold = float(retrieval_cfg["similarity_threshold"])
    if config["mode"] == "bm25":
        candidate_chunks = _bm25_chunks(query, bm25_index, chunks, retrieval_cfg, candidate_pool_size)
    elif config["mode"] == "dense":
        candidate_chunks = _dense_chunks(
            query, vector_store, embedding_generator, candidate_pool_size, threshold
        )
    else:
        retriever = retriever or _make_retriever(
            vector_store, embedding_generator, retrieval_cfg, enable_rerank=False
        )
        candidate_chunks = retriever.retrieve(
            query, k=candidate_pool_size, threshold=threshold, mode="hybrid", use_cache=False
        )

    filter_rejection_count = (
        max(0, candidate_pool_size - len(candidate_chunks))
        if config["mode"] != "bm25"
        else 0
    )
    parent_candidates = _collapse_parent_candidates(candidate_chunks)
    reranking_latency_ms = 0.0
    rerank_candidates: List[Dict[str, Any]] = []
    if config["rerank"]:
        if reranker is None or getattr(reranker, "model", None) is None:
            raise RuntimeError(
                "B4 requires a loaded cross-encoder; refusing to label an unreranked run as B4."
            )
        # Rerank chunk candidates BEFORE parent aggregation. This lets the
        # cross-encoder choose the most relevant passage when a document has
        # multiple candidate chunks.
        rerank_candidates = candidate_chunks[: int(retrieval_cfg["rerank_candidate_pool"])]
        rerank_inputs = []
        for chunk in rerank_candidates:
            candidate = dict(chunk)
            metadata = candidate.get("metadata", {})
            title = str(metadata.get("title") or metadata.get("source") or "")
            passage = str(candidate.get("text", ""))
            candidate["rerank_text"] = f"{title}\n{passage}".strip() if title else passage
            rerank_inputs.append(candidate)
        rerank_started = time.perf_counter()
        reranked_chunks = reranker.rerank(
            query,
            rerank_inputs,
            top_n=len(rerank_inputs),
        )
        reranking_latency_ms = (time.perf_counter() - rerank_started) * 1000.0
        final_chunks = _collapse_parent_candidates(reranked_chunks)
        final_chunks = final_chunks[: int(retrieval_cfg["evaluation_depth"])]
        after_count = len(final_chunks)
    else:
        final_chunks = parent_candidates
        after_count = len(parent_candidates)
    diagnostics = {
        "candidate_chunk_count": len(candidate_chunks),
        "unique_parent_count_before_reranking": len(parent_candidates),
        "unique_parent_count_after_reranking": after_count,
        "filter_rejection_count": filter_rejection_count,
        "rerank_candidate_chunk_count": len(rerank_candidates),
        "reranking_latency_ms": reranking_latency_ms,
    }
    return final_chunks, diagnostics


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
    for summary_key, data in summary.items():
        dataset, config_id = summary_key.split(":", 1)
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



def save_paired_statistics(records: List[Dict[str, Any]], run_dir: Path) -> None:
    """Compare B4 against B1-B3 with paired per-query bootstrap CIs and Wilcoxon tests."""
    by_dataset: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for record in records:
        by_dataset.setdefault(record["dataset"], {}).setdefault(
            record["configuration"], {}
        )[record["query_id"]] = record["metrics"]

    metrics = ("recall@10", "ndcg@10", "mrr@10", "precision@10")
    rows = []
    for dataset, configurations in sorted(by_dataset.items()):
        proposed = configurations.get("B4", {})
        for metric in metrics:
            metric_rows = []
            for baseline_id in ("B1", "B2", "B3"):
                baseline = configurations.get(baseline_id, {})
                paired_ids = sorted(set(proposed) & set(baseline))
                before = [float(baseline[qid][metric]) for qid in paired_ids]
                after = [float(proposed[qid][metric]) for qid in paired_ids]
                differences = [right - left for left, right in zip(before, after)]
                ci_low, ci_high = paired_bootstrap_ci(
                    differences,
                    n_resamples=10000,
                    confidence=0.95,
                    seed=42,
                ) if differences else (None, None)
                test = wilcoxon_signed_rank_test(before, after)
                metric_rows.append({
                    "dataset": dataset,
                    "comparison": f"B4-{baseline_id}",
                    "metric": metric,
                    "n_queries": len(paired_ids),
                    "mean_paired_improvement": float(np.mean(differences)) if differences else None,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wilcoxon_statistic": test[0] if test else None,
                    "p_value": test[1] if test else None,
                    "holm_adjusted_p": None,
                })
            adjusted = holm_bonferroni([row["p_value"] for row in metric_rows])
            for row, corrected in zip(metric_rows, adjusted):
                row["holm_adjusted_p"] = corrected
                rows.append(row)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "paired_significance.json").write_text(
        json.dumps({
            "method": "paired bootstrap percentile 95% CI and two-sided Wilcoxon signed-rank",
            "multiple_testing": "Holm-Bonferroni across B4-vs-B1/B2/B3 comparisons within each dataset and metric",
            "positive_improvement": "B4 metric minus baseline metric",
            "results": rows,
        }, indent=2),
        encoding="utf-8",
    )
    if rows:
        with open(run_dir / "paired_significance.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def save_latency_summary(records: List[Dict[str, Any]], run_dir: Path) -> None:
    """Summarize repeated timings on a deterministic sample; quality uses all queries."""
    grouped: Dict[tuple, List[float]] = {}
    for record in records:
        if not record.get("latency_repeated"):
            continue
        key = (record["dataset"], record["configuration"])
        grouped.setdefault(key, []).extend(record.get("latency_samples_ms", []))

    output = []
    for (dataset, configuration), values in sorted(grouped.items()):
        ordered = sorted(values)
        output.append({
            "dataset": dataset,
            "configuration": configuration,
            "sampled_queries": len({
                record["query_id"] for record in records
                if record["dataset"] == dataset
                and record["configuration"] == configuration
                and record.get("latency_repeated")
            }),
            "timing_observations": len(values),
            "mean_ms": float(np.mean(values)) if values else None,
            "median_ms": float(np.median(values)) if values else None,
            "std_dev_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "p95_ms": float(np.percentile(values, 95)) if values else None,
            "p99_ms": float(np.percentile(values, 99)) if values else None,
        })

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "latency_summary.json").write_text(
        json.dumps({
            "note": "Three repeated warm retrieval timings for a deterministic sample of queries; timings are descriptive and are not independent query samples.",
            "results": output,
        }, indent=2),
        encoding="utf-8",
    )



def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_run_metadata(cfg: Dict[str, Any], results_dir: Path, repo_root: Path) -> None:
    config_path = repo_root / "backend" / "evaluation" / "configs" / "experiments.yaml"
    dataset_checksums = {}
    for dataset in cfg["datasets"]:
        dataset_dir = repo_root / "backend" / "evaluation" / "data" / dataset
        files = sorted(path for path in dataset_dir.rglob("*") if path.is_file())
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.relative_to(dataset_dir)).encode("utf-8"))
            digest.update(path.read_bytes())
        dataset_checksums[dataset] = digest.hexdigest()
    metadata = {
        "git_commit": _git_commit(repo_root),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config": cfg,
        "datasets": cfg["datasets"],
        "dataset_checksums_sha256": dataset_checksums,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": __import__("platform").python_version(),
        "platform": __import__("platform").platform(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "sentence-transformers", "faiss-cpu", "beir", "torch")
            if _package_available(name)
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _package_available(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, help="Output directory for this benchmark run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_config()

    embedding_cfg = cfg["embedding"]
    retrieval_cfg = cfg["retrieval"]
    dataset_names = cfg["datasets"]
    chunk_cfg = cfg["chunking"]
    index_type = cfg["index"].get("type", "cosine")
    repo_root = Path(__file__).resolve().parents[3]
    results_root = repo_root / "backend" / "evaluation" / "results"
    run_dir = args.run_dir or (
        results_root / "runs" / f"depth_audit_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir = run_dir / "raw"
    print(f"RUN_DIR={run_dir}")

    embedding_generator = EmbeddingGenerator(
        model_name=embedding_cfg["model_name"],
        device=embedding_cfg.get("device", "cpu"),
        query_instruction=embedding_cfg.get("query_instruction", ""),
    )
    all_summary = {}
    all_records = []
    base_dir = Path(__file__).resolve().parents[2]
    for dataset_name in dataset_names:
        data_path = base_dir / "evaluation" / "data" / dataset_name
        trace_dir = results_dir / dataset_name
        summary, dataset_records = run_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            cfg=retrieval_cfg,
            embedding_generator=embedding_generator,
            chunk_size=chunk_cfg["chunk_size"],
            overlap=chunk_cfg["overlap"],
            index_type=index_type,
            trace_dir=trace_dir,
        )
        all_summary.update(summary)
        all_records.extend(dataset_records)
        logger.info("Dataset %s summary: %s", dataset_name, summary)
    save_summary(all_summary, run_dir)
    save_paired_statistics(all_records, run_dir)
    save_latency_summary(all_records, run_dir)
    _write_run_metadata(cfg, results_dir, repo_root)
    print(f"RESULTS_DIR={run_dir}")


if __name__ == "__main__":
    main()
