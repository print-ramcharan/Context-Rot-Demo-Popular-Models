"""Document-level retrieval metrics computed from ranked parent document IDs.

Metrics are computed per query, then aggregated (mean over queries) by the
caller. The ranked list is expected to contain parent BEIR document IDs in
retriever rank order, with duplicates removed (a doc that appears through
multiple chunks is counted once).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Sequence

DEFAULT_CUTOFF_K = 20
DEFAULT_MRR_K = 10
DEFAULT_NDCG_K = 10


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def dcg_at_k(rel_scores: Sequence[float], k: int) -> float:
    scores = _as_float_array(rel_scores)[:k]
    if scores.size == 0:
        return 0.0
    positions = np.arange(2, scores.size + 2, dtype=float)
    return float(np.sum(scores / np.log2(positions)))


def ndcg_at_k(rel_scores: Sequence[float], k: int) -> float:
    scores = _as_float_array(rel_scores)[:k]
    dcg = dcg_at_k(scores, k)
    ideal = dcg_at_k(sorted(scores.tolist(), reverse=True), k)
    if ideal <= 0.0:
        return 0.0
    return float(dcg / ideal)


def recall_at_k(rel_scores: Sequence[float], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    relevant_in_top_k = int(np.sum(_as_float_array(rel_scores[:k]) > 0))
    return relevant_in_top_k / total_relevant


def precision_at_k(rel_scores: Sequence[float], k: int) -> float:
    """Fraction of the top-k retrieved documents that are relevant.

    The denominator is the cutoff ``k`` (standard IR definition), so a run
    that returns fewer than ``k`` items is still measured against the full
    top-k slot budget.
    """
    if k <= 0:
        return 0.0
    top_k = _as_float_array(rel_scores[:k])
    return float(np.sum(top_k > 0) / k)


def reciprocal_rank(rel_scores: Sequence[float], k: int | None = None) -> float:
    for rank, score in enumerate(rel_scores[:k], start=1):
        if score > 0:
            return 1.0 / rank
    return 0.0


def compute_query_metrics(retrieved_grades: Sequence[float], total_relevant: int) -> Dict[str, float]:
    """Compute document-level metrics from ranked relevance grades."""
    rel = _as_float_array(retrieved_grades)
    return {
        "recall@5": recall_at_k(rel, 5, total_relevant),
        "recall@10": recall_at_k(rel, 10, total_relevant),
        "recall@20": recall_at_k(rel, DEFAULT_CUTOFF_K, total_relevant),
        "mrr@10": reciprocal_rank(rel, DEFAULT_MRR_K),
        "ndcg@10": ndcg_at_k(rel, DEFAULT_NDCG_K),
        "precision@10": precision_at_k(rel, 10),
    }
