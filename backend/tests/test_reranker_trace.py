import pytest

from src.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        # The second candidate is more relevant to the query.
        return [0.1, 0.9]


def test_reranker_records_final_and_original_ranks():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "fake-cross-encoder"
    reranker.model = FakeCrossEncoder()
    reranker.last_trace = {}

    candidates = [
        {
            "text": "first passage",
            "score": 0.8,
            "rank": 1,
            "metadata": {"source": "doc.txt", "chunk_index": 4},
        },
        {
            "text": "second passage",
            "score": 0.7,
            "rank": 2,
            "metadata": {"source": "doc.txt", "chunk_index": 9},
        },
    ]

    results = reranker.rerank("query", candidates, top_n=2)

    assert [item["text"] for item in results] == [
        "second passage", "first passage"
    ]
    assert [item["rank"] for item in results] == [1, 2]
    assert [item["original_rank"] for item in results] == [2, 1]
    assert reranker.last_trace["scored_chunk_ids"] == [
        "doc.txt:4", "doc.txt:9"
    ]
    assert reranker.last_trace["raw_scores"] == [0.1, 0.9]


def test_unavailable_reranker_fails_closed():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "missing-model"
    reranker.model = None
    reranker.last_trace = {"stale": "must be cleared"}

    with pytest.raises(RuntimeError, match="unavailable"):
        reranker.rerank(
            "query",
            [{"text": "passage", "metadata": {"source": "doc.txt"}}],
        )

    assert reranker.last_trace == {"scored_chunk_ids": [], "raw_scores": []}
