"""Verify and analyze completed BEIR retrieval runs."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval

from evaluation.metrics.statistical_tests import paired_bootstrap_ci, wilcoxon_signed_rank_test


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


def _official(qrels: Dict[str, Dict[str, int]], records: List[dict]) -> dict:
    results = _result_map(records)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [5, 10, 20]
    )
    return {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@10": recall["Recall@10"],
        "P@10": precision["P@10"],
    }


def _official_query_metrics(
    qrels: Dict[str, Dict[str, int]], record: dict
) -> dict[str, float]:
    ranked_ids = record["retrieved_document_ids"]
    grades = [qrels.get(record["query_id"], {}).get(doc_id, 0) for doc_id in ranked_ids]
    relevant = sum(grade > 0 for grade in qrels.get(record["query_id"], {}).values())
    cutoff = 10
    top = grades[:cutoff]
    dcg = sum((2**grade - 1) / np.log2(rank + 2) for rank, grade in enumerate(top))
    ideal = sorted(qrels.get(record["query_id"], {}).values(), reverse=True)[:cutoff]
    idcg = sum((2**grade - 1) / np.log2(rank + 2) for rank, grade in enumerate(ideal))
    first_relevant = next(
        (rank + 1 for rank, grade in enumerate(top) if grade > 0), None
    )
    return {
        "recall@10": sum(grade > 0 for grade in top) / relevant if relevant else 0.0,
        "ndcg@10": dcg / idcg if idcg else 0.0,
        "precision@10": sum(grade > 0 for grade in top) / cutoff,
        "mrr@10": 1.0 / first_relevant if first_relevant else 0.0,
    }


def _paired(
    qrels: Dict[str, Dict[str, int]],
    records_a: List[dict],
    records_b: List[dict],
    metric: str,
) -> dict:
    by_id_a = {
        r["query_id"]: _official_query_metrics(qrels, r)[metric] for r in records_a
    }
    by_id_b = {
        r["query_id"]: _official_query_metrics(qrels, r)[metric] for r in records_b
    }
    ids = sorted(set(by_id_a) & set(by_id_b))
    before = [by_id_a[qid] for qid in ids]
    after = [by_id_b[qid] for qid in ids]
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
        if any(not ids for ids in query_ids.values()) or len(set(query_ids.values())) != 1:
            counts = {config: len(ids) for config, ids in query_ids.items()}
            raise RuntimeError(f"Incomplete or misaligned results for {dataset}: {counts}")
        official[dataset] = {
            config: _official(qrels, records)
            for config, records in records_by_config.items()
        }
        significance[dataset] = {
            f"{baseline}_vs_{candidate}": [
                _paired(qrels, records_by_config[baseline], records_by_config[candidate], metric)
                for metric in ("recall@10", "ndcg@10", "precision@10", "mrr@10")
            ]
            for baseline, candidate in (("B1", "B3"), ("B3", "B4"))
        }

    (processed / "official_beir_metrics.json").write_text(
        json.dumps(official, indent=2), encoding="utf-8"
    )
    (processed / "paired_significance.json").write_text(
        json.dumps(significance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
