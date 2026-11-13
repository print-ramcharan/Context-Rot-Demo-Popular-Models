"""Deterministic, rank-preserving analysis of completed BEIR retrieval runs.

The raw retrieval order is authoritative. Tied retrieval scores are replaced only
inside the evaluator with unique descending rank scores so pytrec_eval cannot
reorder tied documents. Raw run files are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from evaluation.metrics.statistical_tests import (
    paired_bootstrap_ci,
    wilcoxon_signed_rank_test,
    holm_bonferroni,
)


def _records(root: Path, dataset: str, config: str) -> List[dict]:
    directory = root / dataset
    paths = sorted(directory.glob(f"{dataset}_{config}_*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not records:
        raise RuntimeError(f"No raw records found for {dataset}/{config} under {directory}")
    return records


def _result_map(records: List[dict]) -> dict[str, dict[str, float]]:
    """Convert stored rank order to unique scores without changing that order."""
    results: dict[str, dict[str, float]] = {}
    for record in records:
        qid = str(record["query_id"])
        if qid in results:
            raise RuntimeError(f"Duplicate query record: {qid}")
        doc_ids = [str(doc_id) for doc_id in record["retrieved_document_ids"]]
        raw_scores = record.get("scores", [])
        if len(doc_ids) != len(raw_scores):
            raise RuntimeError(f"ID/score length mismatch for query {qid}")
        if len(doc_ids) != len(set(doc_ids)):
            raise RuntimeError(f"Duplicate retrieved document ID for query {qid}")
        # Higher values mean earlier rank; every score is unique within a query.
        results[qid] = {
            doc_id: float(len(doc_ids) - rank)
            for rank, doc_id in enumerate(doc_ids)
        }
    return results


def _evaluate_system(
    qrels: Dict[str, Dict[str, int]],
    records: List[dict],
    k_values: Sequence[int] = (5, 10, 20),
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compute pytrec_eval metrics and MRR from the preserved stored ranking."""
    import pytrec_eval

    results = _result_map(records)
    map_string = "map_cut." + ",".join(map(str, k_values))
    ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
    recall_string = "recall." + ",".join(map(str, k_values))
    precision_string = "P." + ",".join(map(str, k_values))
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    per_query_metrics: dict[str, dict[str, float]] = {}
    for record in records:
        qid = str(record["query_id"])
        doc_ids = [str(doc_id) for doc_id in record["retrieved_document_ids"]]
        relevant = {doc for doc, grade in qrels.get(qid, {}).items() if int(grade) > 0}
        rr = 0.0
        for rank, doc_id in enumerate(doc_ids[:10], start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        q_score = scores.get(qid, {})
        per_query_metrics[qid] = {
            "ndcg@10": float(q_score.get("ndcg_cut_10", 0.0)),
            "recall@10": float(q_score.get("recall_10", 0.0)),
            "precision@10": float(q_score.get("P_10", 0.0)),
            "mrr@10": float(rr),
        }

    n = len(per_query_metrics)
    if n == 0:
        raise RuntimeError("No per-query metrics were produced")
    aggregate = {
        "NDCG@10": sum(m["ndcg@10"] for m in per_query_metrics.values()) / n,
        "Recall@10": sum(m["recall@10"] for m in per_query_metrics.values()) / n,
        "P@10": sum(m["precision@10"] for m in per_query_metrics.values()) / n,
        "MRR@10": sum(m["mrr@10"] for m in per_query_metrics.values()) / n,
    }
    return {key: round(value, 8) for key, value in aggregate.items()}, per_query_metrics


def _paired(
    per_query_a: Dict[str, Dict[str, float]],
    per_query_b: Dict[str, Dict[str, float]],
    metric: str,
) -> dict:
    ids = sorted(set(per_query_a) & set(per_query_b))
    if not ids or set(per_query_a) != set(per_query_b):
        raise RuntimeError("Paired comparisons require identical, non-empty query sets")
    before = [per_query_a[qid][metric] for qid in ids]
    after = [per_query_b[qid][metric] for qid in ids]
    differences = [b - a for a, b in zip(before, after)]
    test = wilcoxon_signed_rank_test(before, after)
    return {
        "metric": metric,
        "n_queries": len(ids),
        "mean_difference": sum(differences) / len(differences),
        "bootstrap_ci_95": list(paired_bootstrap_ci(differences)),
        "wilcoxon_statistic": test[0] if test else None,
        "wilcoxon_p_value": test[1] if test else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Separate analysis output directory (default: RUN_DIR/processed_rank_preserving)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    raw = run_dir / "raw" if (run_dir / "raw").is_dir() else run_dir
    processed = args.output_dir or (run_dir / "processed_rank_preserving")
    processed.mkdir(parents=True, exist_ok=True)

    from evaluation.datasets.load_beir import load_beir_dataset

    data_root = Path(__file__).resolve().parent / "data"
    official: dict[str, dict] = {}
    significance: dict[str, dict] = {}
    query_counts: dict[str, int] = {}

    for dataset in ("scifact", "nfcorpus"):
        _, queries, qrels = load_beir_dataset(data_root / dataset, dataset)
        expected_ids = set(map(str, queries))
        if expected_ids != set(qrels):
            raise RuntimeError(
                f"{dataset}: query/qrels coverage mismatch "
                f"(queries={len(expected_ids)}, qrels={len(qrels)})"
            )
        query_counts[dataset] = len(expected_ids)

        records_by_config = {
            config: _records(raw, dataset, config) for config in ("B1", "B2", "B3", "B4")
        }
        for config, records in records_by_config.items():
            found = {str(record["query_id"]) for record in records}
            if found != expected_ids:
                raise RuntimeError(
                    f"{dataset}/{config}: expected {len(expected_ids)} queries, "
                    f"found {len(found)}, missing={len(expected_ids - found)}, "
                    f"unexpected={len(found - expected_ids)}"
                )

        evaluated = {
            config: _evaluate_system(qrels, records)
            for config, records in records_by_config.items()
        }
        official[dataset] = {
            config: values[0] for config, values in evaluated.items()
        }
        per_query = {config: values[1] for config, values in evaluated.items()}
        significance[dataset] = {
            f"{baseline}_vs_{candidate}": [
                _paired(per_query[baseline], per_query[candidate], metric)
                for metric in ("recall@10", "ndcg@10", "precision@10", "mrr@10")
            ]
            for baseline, candidate in (("B1", "B3"), ("B3", "B4"))
        }

    all_tests = [
        result
        for dataset_results in significance.values()
        for comparison_results in dataset_results.values()
        for result in comparison_results
    ]
    adjusted = holm_bonferroni([result["wilcoxon_p_value"] for result in all_tests])
    for result, adjusted_p in zip(all_tests, adjusted):
        result["wilcoxon_p_value_holm"] = adjusted_p
        result["significant_holm_0_05"] = adjusted_p is not None and adjusted_p < 0.05

    manifest = {
        "analysis": "rank-preserving BEIR evaluation",
        "raw_results_modified": False,
        "ranking_source": "retrieved_document_ids order in each raw query record",
        "tie_handling": "unique descending synthetic scores derived from stored rank",
        "ndcg_implementation": "pytrec_eval ndcg_cut (standard exponential gain)",
        "significance": "paired bootstrap 95% CI; Wilcoxon signed-rank; Holm correction",
        "holm_family_size": len(all_tests),
        "query_counts": query_counts,
        "run_dir": str(run_dir),
    }
    (processed / "official_beir_metrics.json").write_text(
        json.dumps(official, indent=2), encoding="utf-8"
    )
    (processed / "paired_significance.json").write_text(
        json.dumps(significance, indent=2), encoding="utf-8"
    )
    (processed / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metrics": official, "significance": significance,
                      "manifest": manifest, "output_dir": str(processed)}, indent=2))


if __name__ == "__main__":
    main()
