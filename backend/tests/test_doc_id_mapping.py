"""Tests for parent-document ID mapping (chunk -> BEIR doc id)."""

import unittest

from evaluation.runners._doc_id_map import (
    parent_ids_from_chunks,
    parent_scores_for_query,
)

# (retrieved chunk dicts, qrels, query_id) -> expected parent ids / scores
SAMPLE_CHUNKS = [
    {"text": "chunk-a-1", "metadata": {"parent_id": "d1", "source": "t1"}},
    {"text": "chunk-a-2", "metadata": {"parent_id": "d1", "source": "t1"}},
    {"text": "chunk-b", "metadata": {"doc_id": "d2", "source": "t2"}},
    {"text": "chunk-c", "metadata": {}},
]

QRELS = {"q1": {"d1": 1, "d2": 1, "d3": 0}}

EXPECTED_PARENT_IDS = ["d1", "d2", "chunk-c"]
EXPECTED_SCORES = [1, 1, 0]


class TestDocIdMapping(unittest.TestCase):
    def test_parent_ids_from_chunks(self):
        result = parent_ids_from_chunks(SAMPLE_CHUNKS)
        self.assertEqual(result, EXPECTED_PARENT_IDS)

    def test_parent_scores_for_query(self):
        scores = parent_scores_for_query(SAMPLE_CHUNKS, QRELS, "q1")
        self.assertEqual(scores, EXPECTED_SCORES)

    def test_missing_qrels_returns_zero(self):
        scores = parent_scores_for_query(SAMPLE_CHUNKS, {}, "unknown")
        self.assertEqual(scores, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
