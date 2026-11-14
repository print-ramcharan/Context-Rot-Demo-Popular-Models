"""Tests for retrieval metric calculations.

These values are hand-computed, so they double as the guaranteed
validation set for the evaluation harness.
"""

import unittest

from src.utils.metrics import advanced_ranking_metrics, text_metrics
from evaluation.metrics.retrieval_metrics import (
    compute_query_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Hand-computed example:
#   graded relevance in rank order: [1, 0, 1]  (doc_a=1, doc_c=0, doc_b=1)
#   total_relevant = 2
#   recall@5    = 2 / 2 = 1.0
#   recall@10   = 2 / 2 = 1.0
#   recall@20   = 2 / 2 = 1.0
#   mrr@10      = 1 / 1 = 1.0
#   precision@10 = 2 / 10 = 0.2
#   DCG@10 = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
#   IDCG@10 = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
#   ndcg@10 = 1.5 / 1.6309 = 0.91980...
_EXPECTED = {
    "recall@5": 1.0,
    "recall@10": 1.0,
    "recall@20": 1.0,
    "mrr@10": 1.0,
    "ndcg@10": 0.9197,
    "precision@10": 0.2,
}


class TestRetrievalMetrics(unittest.TestCase):
    def setUp(self):
        self.scores = [1, 0, 1]

    def test_recall(self):
        total_relevant = 2
        self.assertAlmostEqual(recall_at_k(self.scores, 5, total_relevant), 1.0)
        self.assertAlmostEqual(recall_at_k(self.scores, 10, total_relevant), 1.0)
        self.assertAlmostEqual(recall_at_k(self.scores, 20, total_relevant), 1.0)

    def test_precision(self):
        self.assertAlmostEqual(precision_at_k(self.scores, 10), 0.2)

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank(self.scores), 1.0)

    def test_ndcg(self):
        self.assertAlmostEqual(ndcg_at_k(self.scores, 10), 0.9197, places=4)

    def test_ndcg_with_ideal_scores(self):
        # Retrieved [1, 0] when total 3 relevant documents exist
        # DCG@10 = 1 / log2(2) = 1.0
        # IDCG@10 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63092975 + 0.5 = 2.13092975
        # NDCG@10 = 1.0 / 2.13092975 = 0.469278...
        scores = [1, 0]
        ideal_scores = [1, 1, 1]
        self.assertAlmostEqual(ndcg_at_k(scores, 10, ideal_scores=ideal_scores), 0.4693, places=4)

    def test_ndcg_graded_matches_trec_eval(self):
        # Graded relevance: query has docs with grades 3, 2, 1
        # System retrieved in reverse order: [1, 2, 3]
        # DCG@10 = 1/log2(2) + 2/log2(3) + 3/log2(4) = 1.0 + 1.2618595 + 1.5 = 3.7618595
        # IDCG@10 = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3.0 + 1.2618595 + 0.5 = 4.7618595
        # NDCG@10 = 3.7618595 / 4.7618595 = 0.789998...
        retrieved = [1, 2, 3]
        qrels = [3, 2, 1]
        self.assertAlmostEqual(ndcg_at_k(retrieved, 10, ideal_scores=qrels), 0.7900, places=4)

    def test_compute_query_metrics(self):
        metrics = compute_query_metrics(self.scores, total_relevant=2)
        for key, expected in _EXPECTED.items():
            self.assertAlmostEqual(metrics[key], expected, places=4)

    def test_compute_query_metrics_graded(self):
        retrieved = [1, 2, 3]
        qrel_grades = [3, 2, 1]
        metrics = compute_query_metrics(retrieved, total_relevant=3, qrel_grades=qrel_grades)
        self.assertAlmostEqual(metrics["ndcg@10"], 0.7900, places=4)
        self.assertAlmostEqual(metrics["recall@10"], 1.0, places=4)
        self.assertAlmostEqual(metrics["precision@10"], 0.3, places=4)
        self.assertAlmostEqual(metrics["mrr@10"], 1.0, places=4)

    def test_advanced_metrics_match_source_labels_and_use_requested_cutoff(self):
        retrieved = [
            {"chunk_id": "paper-a.txt:0"},
            {"chunk_id": "paper-b.txt:0"},
            {"chunk_id": "paper-a.txt:1"},
        ]
        metrics = advanced_ranking_metrics(
            retrieved,
            {"paper-a.txt", "paper-c.txt"},
            5,
        )
        self.assertEqual(metrics["retrieved_count"], 2)
        self.assertEqual(metrics["relevant_count"], 2)
        self.assertAlmostEqual(metrics["recall_at_k"], 0.5)
        self.assertAlmostEqual(metrics["precision_at_k"], 0.2)
        self.assertLessEqual(metrics["recall_at_k"], 1.0)

    def test_advanced_metrics_do_not_report_retrieved_count_as_recall(self):
        retrieved = [{"source": "doc.txt"} for _ in range(20)]
        metrics = advanced_ranking_metrics(retrieved, {"doc.txt"}, 20)
        self.assertAlmostEqual(metrics["recall_at_k"], 1.0)
        self.assertLessEqual(metrics["recall_at_k"], 1.0)

    def test_answer_metrics_distinguish_correct_and_wrong_reference(self):
        correct = text_metrics("Alice", "Alice", "Alice falls down the rabbit hole.")
        wrong = text_metrics("Alice", "Bob", "Alice falls down the rabbit hole.")
        self.assertEqual(correct["exact_match"], 1.0)
        self.assertEqual(wrong["exact_match"], 0.0)
        self.assertGreater(correct["answer_token_f1"], wrong["answer_token_f1"])


if __name__ == "__main__":
    unittest.main()
