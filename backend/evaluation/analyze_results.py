"""Verify and analyze completed BEIR retrieval runs."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval

from evaluation.metrics.statistical_tests import (
    paired_bootstrap_ci,
    wilcoxon_signed_rank_test,
    holm_bonferroni,
)

METRICS = ("NDCG@10", "Recall@10", "P@10", "MRR@10")


def _records(root: Path, dataset: str, config: str) -> List[dict]:
    directory = root / dataset
    records = []
    for path in sorted(directory.glob(f"{dataset}_{config}_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _result_map(records: List[dict]) -> dict[str, dict[str, float]]:
    results = {
        record["query_id"]: {
            doc_id: float(score)
            for doc_id, score in zip(record["retrieved_document_ids"], record["scores"])
        }
        for record in records
    }
    return results


def _evaluate_system(
    qrels: Dict[str, Dict[str, int]],
    records: List[dict],
    k_values: Sequence[int] = (5, 10, 20),
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute official BEIR aggregate metrics and aligned per-query metrics using pytrec_eval."""
    import pytrec_eval

    results = _result_map(records)
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    # Compute per-query MRR@10 according to official BEIR logic
    # (reciprocal rank of first relevant doc within top-k)
    per_query_metrics = {}
    for qid, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:10]
        q_rels = set(doc for doc, g in qrels.get(qid, {}).items() if g > 0)
        rr = 0.0
        for rank, (doc_id, _) in enumerate(top_hits):
            if doc_id in q_rels:
                rr = 1.0 / (rank + 1)
                break

        q_score = scores.get(qid, {})
        per_query_metrics[qid] = {
            "ndcg@10": float(q_score.get("ndcg_cut_10", 0.0)),
            "recall@10": float(q_score.get("recall_10", 0.0)),
            "precision@10": float(q_score.get("P_10", 0.0)),
            "mrr@10": float(rr),
        }

    n = len(per_query_metrics) if per_query_metrics else 1
    aggregate = {
        "NDCG@10": round(sum(m["ndcg@10"] for m in per_query_metrics.values()) / n, 5),
        "Recall@10": round(sum(m["recall@10"] for m in per_query_metrics.values()) / n, 5),
        "P@10": round(sum(m["precision@10"] for m in per_query_metrics.values()) / n, 5),
        "MRR@10": round(sum(m["mrr@10"] for m in per_query_metrics.values()) / n, 5),
    }
    return aggregate, per_query_metrics


def _paired(
    per_query_a: Dict[str, Dict[str, float]],
    per_query_b: Dict[str, Dict[str, float]],
    metric: str,
) -> dict:
    ids = sorted(set(per_query_a.keys()) & set(per_query_b.keys()))
    before = [per_query_a[qid][metric] for qid in ids]
    after = [per_query_b[qid][metric] for qid in ids]
    differences = [b - a for a, b in zip(before, after)]
    ci = paired_bootstrap_ci(differences)
    test = wilcoxon_signed_rank_test(before, after)
    return {
        "metric": metric,
        "n_queries": len(ids),
        "mean_difference": sum(differences) / len(differences) if differences else 0.0,
        "bootstrap_ci_95": list(ci),
        "wilcoxon_statistic": test[0] if test else None,
        "wilcoxon_p_value": test[1] if test else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    base = args.run_dir or (Path(__file__).resolve().parent / "results")
    raw = base / "raw" if (base / "raw").exists() else base
    processed = base / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    qrels_by_dataset = {}
    from evaluation.datasets.load_beir import load_beir_dataset

    for dataset in ("scifact", "nfcorpus"):
        _, _, qrels_by_dataset[dataset] = load_beir_dataset(
            Path(__file__).resolve().parent / "data" / dataset, dataset
        )

    official = {}
    significance = {}
    for dataset, qrels in qrels_by_dataset.items():
        records_by_config = {
            config: _records(raw, dataset, config) for config in ("B1", "B2", "B3", "B4")
        }
        query_ids = {
            config: {record["query_id"] for record in records}
            for config, records in records_by_config.items()
        }
        if (
            any(not ids for ids in query_ids.values())
            or len({frozenset(ids) for ids in query_ids.values()}) != 1
        ):
            counts = {config: len(ids) for config, ids in query_ids.items()}
            raise RuntimeError(f"Incomplete or misaligned results for {dataset}: {counts}")

        dataset_eval = {
            config: _evaluate_system(qrels, records)
            for config, records in records_by_config.items()
        }
        official[dataset] = {config: res[0] for config, res in dataset_eval.items()}
        per_query = {config: res[1] for config, res in dataset_eval.items()}

        significance[dataset] = {
            f"{baseline}_vs_{candidate}": [
                _paired(per_query[baseline], per_query[candidate], metric)
                for metric in ("recall@10", "ndcg@10", "precision@10", "mrr@10")
            ]
            for baseline, candidate in (("B1", "B3"), ("B3", "B4"))
        }

    # Apply Holm-Bonferroni across all planned tests.
    all_tests = [
        result
        for dataset_results in significance.values()
        for comparison_results in dataset_results.values()
        for result in comparison_results
    ]

    adjusted_p_values = holm_bonferroni([
        result["wilcoxon_p_value"]
        for result in all_tests
    ])

    for result, adjusted_p in zip(all_tests, adjusted_p_values):
        result["wilcoxon_p_value_holm"] = adjusted_p
        result["significant_holm_0_05"] = (
            adjusted_p is not None and adjusted_p < 0.05
        )

    (processed / "official_beir_metrics.json").write_text(
        json.dumps(official, indent=2), encoding="utf-8"
    )
    (processed / "paired_significance.json").write_text(
        json.dumps(significance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
