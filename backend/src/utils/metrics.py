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
                "p75_ms": self._percentile(values, 0.75),
                "p90_ms": self._percentile(values, 0.90),
                "p95_ms": self._percentile(values, 0.95),
                "p99_ms": self._percentile(values, 0.99),
                "min_ms": min(values),
                "std_dev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
                "max_ms": max(values)
            }
        return {
            "counters": dict(self.counters),
            "timings": timing_stats
        }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _identifier_matches_relevance(identifier: str, relevant: set[str]) -> bool:
    """Match either chunk-level labels or source-level labels."""
    if identifier in relevant:
        return True
    source = identifier.split(":", 1)[0]
    return source in relevant


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
    hits = [
        index for index, value in enumerate(ranked, start=1)
        if _identifier_matches_relevance(value, relevant)
    ]
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
    raw_identifiers = []
    for item in retrieved[:k]:
        metadata = item.get("metadata", {})
        identifier = str(
            item.get("source") or metadata.get("source")
            or item.get("id") or item.get("chunk_id", "")
        )
        if identifier and identifier not in raw_identifiers:
            raw_identifiers.append(identifier)
    exact_label_match = any(identifier in relevant for identifier in raw_identifiers)
    source_label_match = any(
        identifier.split(":", 1)[0] in relevant
        for identifier in raw_identifiers
    )
    source_level_labels = source_label_match and not exact_label_match
    identifiers = []
    for identifier in raw_identifiers:
        metric_identifier = (
            identifier.split(":", 1)[0]
            if source_level_labels else identifier
        )
        if metric_identifier not in identifiers:
            identifiers.append(metric_identifier)
    relevance = [
        int(_identifier_matches_relevance(identifier, relevant))
        for identifier in identifiers
    ]
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
    ideal = [1] * min(len(relevant), k)
    idcg = sum(
        (2 ** value - 1) / math.log2(rank + 1)
        for rank, value in enumerate(ideal, start=1)
    )
    return {
        "precision_at_k": relevant_found / k if k else 0.0,
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
    """
    Measure source/chunk retention and a whitespace-word compression proxy.

    This function deliberately does not claim model-token counts: exact token
    counts require the tokenizer for the generation model. Source labels also
    cannot establish whether the actual supporting sentence survived.
    """
    def source_ids(items: list[dict]) -> set[str]:
        values = set()
        for item in items:
            metadata = item.get("metadata", {})
            value = item.get("source") or metadata.get("source")
            if value:
                values.add(str(value))
        return values

    def chunk_ids(items: list[dict]) -> set[str]:
        values = set()
        for item in items:
            metadata = item.get("metadata", {})
            explicit = item.get("chunk_id") or item.get("id")
            source = item.get("source") or metadata.get("source")
            chunk_index = metadata.get("chunk_index")
            if explicit:
                values.add(str(explicit))
            elif source is not None and chunk_index is not None:
                values.add(f"{source}:{chunk_index}")
        return values

    before_sources = source_ids(before)
    after_sources = source_ids(after)
    relevant_sources = {
        label.split(":", 1)[0] for label in relevant
    } if relevant else set()
    relevant_before = before_sources & relevant_sources
    relevant_after = after_sources & relevant_sources

    before_chunks = chunk_ids(before)
    after_chunks = chunk_ids(after)
    chunk_retention = (
        len(before_chunks & after_chunks) / len(before_chunks)
        if before_chunks else None
    )
    source_retention = (
        len(relevant_after) / len(relevant_before)
        if relevant_before else None
    )

    before_words = len(" ".join(item.get("text", "") for item in before).split())
    after_words = len(" ".join(item.get("text", "") for item in after).split())
    word_reduction = (
        (before_words - after_words) / before_words
        if before_words else 0.0
    )
    density_before = len(relevant_before) / before_words if before_words else 0.0
    density_after = len(relevant_after) / after_words if after_words else 0.0

    return {
        "context_precision": (
            len(relevant_after) / len(after_sources)
            if relevant and after_sources else None
        ),
        "context_recall": (
            len(relevant_after) / len(relevant_sources)
            if relevant_sources else None
        ),
        # Backward-compatible aliases, explicitly documented as source-level.
        "evidence_retention": source_retention,
        "evidence_retention_unit": "source_document",
        "source_retention": source_retention,
        "evidence_loss_rate": (
            1.0 - source_retention if source_retention is not None else None
        ),
        "chunk_retention": chunk_retention,
        "chunk_retention_unit": "chunk_identity",
        "textual_evidence_retention": None,
        "textual_evidence_retention_note": (
            "Requires gold evidence spans or sentence IDs; source labels are insufficient."
        ),
        # These values are whitespace-word proxies, not model token counts.
        "word_count_reduction": word_reduction,
        "word_count_reduction_percent": word_reduction * 100,
        "word_compression_ratio": (
            before_words / after_words if after_words else None
        ),
        "evidence_density_before_per_word": density_before,
        "evidence_density_after_per_word": density_after,
        "evidence_density_gain": (
            (density_after - density_before) / density_before
            if density_before else None
        ),
        "word_count_before": before_words,
        "word_count_after": after_words,
        # Exact token metrics are unavailable here; never substitute word counts.
        "token_reduction": None,
        "token_reduction_percent": None,
        "tokens_before": None,
        "tokens_after": None,
        "budget_utilization": None,
        "context_preservation_score": None,
        "token_count_status": "not_measured_no_model_tokenizer",
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
