import statistics
import re
import math
from collections import Counter
from collections import defaultdict
from typing import Optional

class MetricsCollector:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timings = defaultdict(list)

    def increment(self, name: str, value: int = 1):
        self.counters[name] += value

    def observe(self, name: str, value: float):
        self.timings[name].append(float(value))

    def _percentile(self, values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = (len(values) - 1) * pct
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return values[int(k)]
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return d0 + d1

    def snapshot(self) -> dict:
        timing_stats = {}
        for name, values in self.timings.items():
            if not values:
                continue
            timing_stats[name] = {
                "count": len(values),
                "avg_ms": statistics.mean(values),
                "p50_ms": self._percentile(values, 0.5),
                "p95_ms": self._percentile(values, 0.95),
                "p99_ms": self._percentile(values, 0.99),
                "max_ms": max(values)
            }
        return {
            "counters": dict(self.counters),
            "timings": timing_stats
        }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def rank_metrics(retrieved: list[dict], relevant: set[str]) -> dict:
    """Calculate standard ranking metrics from result IDs or source names."""
    ranked = []
    for item in retrieved:
        metadata = item.get("metadata", {})
        value = str(
            item.get("id") or item.get("chunk_id") or item.get("source")
            or metadata.get("source", "")
        )
        if value and value not in ranked:
            ranked.append(value)
    hits = [index for index, value in enumerate(ranked, start=1) if value in relevant]
    return {
        "hit_rate": float(bool(hits)),
        "precision_at_k": len(hits) / len(ranked) if ranked else 0.0,
        "recall_at_k": len(hits) / len(relevant) if relevant else 0.0,
        "mrr": 1.0 / hits[0] if hits else 0.0,
        "ndcg_at_k": (
            sum(1.0 / __import__("math").log2(rank + 1) for rank in hits)
            / sum(1.0 / __import__("math").log2(rank + 1)
                  for rank in range(1, min(len(relevant), len(ranked)) + 1))
            if hits and relevant else 0.0
        ),
        "retrieved_count": len(ranked),
        "relevant_count": len(relevant),
    }


def text_metrics(answer: str, expected: str, context: str) -> dict:
    """Dependency-free answer quality proxies; use an LLM judge for semantic scores."""
    answer_tokens = _tokens(answer)
    expected_tokens = _tokens(expected)
    context_tokens = _tokens(context)
    overlap = len(answer_tokens & expected_tokens)
    token_precision = overlap / len(answer_tokens) if answer_tokens else 0.0
    token_recall = overlap / len(expected_tokens) if expected_tokens else 0.0
    expected_overlap = (
        overlap / len(expected_tokens)
        if expected_tokens else 0.0
    )
    context_overlap = (
        len(answer_tokens & context_tokens) / len(answer_tokens)
        if answer_tokens else 0.0
    )
    return {
        "answer_token_recall": expected_overlap,
        "answer_token_precision": token_precision,
        "answer_relevancy_proxy": token_precision,
        "answer_token_f1": (
            2 * token_precision * token_recall / (token_precision + token_recall)
            if token_precision + token_recall else 0.0
        ),
        "faithfulness_proxy": context_overlap,
        "hallucination_rate_proxy": 1.0 - context_overlap if answer_tokens else 0.0,
        "exact_match": float(answer.strip().lower() == expected.strip().lower())
        if expected else 0.0,
        "answer_tokens": len(answer_tokens),
        "expected_tokens": len(expected_tokens),
    }


def advanced_ranking_metrics(retrieved: list[dict], relevant: set[str], k: int) -> dict:
    """Return binary relevance metrics with a mathematically bounded nDCG."""
    identifiers = []
    for item in retrieved[:k]:
        metadata = item.get("metadata", {})
        identifier = str(
            item.get("source") or metadata.get("source")
            or item.get("id") or item.get("chunk_id", "")
        )
        if identifier and identifier not in identifiers:
            identifiers.append(identifier)
    relevance = [int(identifier in relevant) for identifier in identifiers]
    relevant_found = sum(relevance)
    precision_values = []
    hits = 0
    for rank, is_relevant in enumerate(relevance, start=1):
        hits += is_relevant
        if is_relevant:
            precision_values.append(hits / rank)
    average_precision = (
        sum(precision_values) / min(len(relevant), k)
        if relevant else None
    )
    dcg = sum(
        (2 ** value - 1) / math.log2(rank + 1)
        for rank, value in enumerate(relevance, start=1)
    )
    ideal = sorted(relevance, reverse=True)
    idcg = sum(
        (2 ** value - 1) / math.log2(rank + 1)
        for rank, value in enumerate(ideal, start=1)
    )
    return {
        "precision_at_k": (
            relevant_found / len(identifiers) if identifiers else 0.0
        ),
        "recall_at_k": relevant_found / len(relevant) if relevant else None,
        "hit_rate_at_k": float(bool(relevant_found)) if relevant else None,
        "mrr_at_k": next(
            (1.0 / rank for rank, value in enumerate(relevance, 1) if value),
            0.0,
        ) if relevant else None,
        "average_precision_at_k": average_precision,
        "ndcg_at_k": dcg / idcg if idcg else (None if relevant else None),
        "unique_document_coverage": len(set(identifiers)),
        "retrieved_count": len(identifiers),
        "relevant_count": len(relevant),
    }


def context_quality_metrics(
    before: list[dict],
    after: list[dict],
    relevant: set[str],
    budget_tokens: Optional[int] = None,
) -> dict:
    """Measure evidence retention and compression using source/chunk evidence units."""
    def evidence_ids(items: list[dict]) -> set[str]:
        values = set()
        for item in items:
            metadata = item.get("metadata", {})
            value = str(
                item.get("source") or metadata.get("source")
                or item.get("id") or item.get("chunk_id")
                or metadata.get("chunk_index", "")
            )
            if value:
                values.add(value)
        return values

    before_ids = evidence_ids(before)
    after_ids = evidence_ids(after)
    relevant_before = before_ids & relevant if relevant else set()
    relevant_after = after_ids & relevant if relevant else set()
    before_tokens = len(" ".join(item.get("text", "") for item in before).split())
    after_tokens = len(" ".join(item.get("text", "") for item in after).split())
    retention = (
        len(relevant_after) / len(relevant_before)
        if relevant_before else None
    )
    reduction = (
        (before_tokens - after_tokens) / before_tokens
        if before_tokens else 0.0
    )
    density_before = len(relevant_before) / before_tokens if before_tokens else 0.0
    density_after = len(relevant_after) / after_tokens if after_tokens else 0.0
    return {
        "context_precision": (
            len(relevant_after) / len(after_ids)
            if relevant and after_ids else None
        ),
        "context_recall": (
            len(relevant_after) / len(relevant)
            if relevant else None
        ),
        "evidence_retention": retention,
        "evidence_loss_rate": 1.0 - retention if retention is not None else None,
        "token_reduction": reduction,
        "token_reduction_percent": reduction * 100,
        "compression_ratio": before_tokens / after_tokens if after_tokens else None,
        "evidence_density_before": density_before,
        "evidence_density_after": density_after,
        "evidence_density_gain": (
            (density_after - density_before) / density_before
            if density_before else None
        ),
        "budget_utilization": (
            after_tokens / budget_tokens if budget_tokens else None
        ),
        "context_preservation_score": (
            retention * reduction if retention is not None else None
        ),
        "tokens_before": before_tokens,
        "tokens_after": after_tokens,
    }


def summarize_metric(values: list[float]) -> dict:
    """Summarize repeated observations for reproducible evaluation reports."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0}
    mean = statistics.mean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    se = sd / math.sqrt(len(clean)) if clean else 0.0
    return {
        "count": len(clean),
        "mean": mean,
        "median": statistics.median(clean),
        "std_dev": sd,
        "standard_error": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "p95": sorted(clean)[min(len(clean) - 1, math.ceil(len(clean) * 0.95) - 1)],
        "p99": sorted(clean)[min(len(clean) - 1, math.ceil(len(clean) * 0.99) - 1)],
    }


def paired_comparison(proposed: list[float], baseline: list[float]) -> dict:
    """Report paired improvement and effect size without claiming significance."""
    differences = [
        float(left) - float(right)
        for left, right in zip(proposed, baseline)
    ]
    summary = summarize_metric(differences)
    return {
        **summary,
        "effect_size_dz": (
            summary["mean"] / summary["std_dev"]
            if summary.get("std_dev") else None
        ),
        "note": "Use a paired t-test or Wilcoxon test for a p-value.",
    }
