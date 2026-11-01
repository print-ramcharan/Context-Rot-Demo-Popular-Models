"""Tests for retrieval metric calculations.

These values are hand-computed, so they double as the guaranteed
validation set for the evaluation harness.
"""

import unittest

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

    def test_compute_query_metrics(self):
        metrics = compute_query_metrics(self.scores, total_relevant=2)
        for key, expected in _EXPECTED.items():
            self.assertAlmostEqual(metrics[key], expected, places=4)


if __name__ == "__main__":
    unittest.main()
