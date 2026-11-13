import pytest

from evaluation.runners.run_retrieval import _run_config_with_depth


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, *args, **kwargs):
        return self.chunks


class FakeReranker:
    model = object()

    def __init__(self):
        self.seen = []

    def rerank(self, query, candidates, top_n):
        self.seen = candidates
        scored = []
        for candidate in candidates:
            item = dict(candidate)
            text = item.get("rerank_text", item.get("text", ""))
            # The second passage of parent A is deliberately the best passage.
            score = 3.0 if "best passage" in text else (2.0 if "parent B" in text else 1.0)
            item["score"] = score
            item["rerank_score"] = score
            scored.append(item)
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_n]


def test_b4_reranks_chunks_before_parent_deduplication():
    chunks = [
        {"text": "weak passage", "score": 0.9, "metadata": {"parent_id": "A", "title": "Title A"}},
        {"text": "parent B passage", "score": 0.8, "metadata": {"parent_id": "B", "title": "Title B"}},
        {"text": "best passage", "score": 0.7, "metadata": {"parent_id": "A", "title": "Title A"}},
    ]
    reranker = FakeReranker()
    cfg = {
        "candidate_pool_size": 3,
        "similarity_threshold": 0.0,
        "rerank_candidate_pool": 3,
        "evaluation_depth": 2,
    }

    results, diagnostics = _run_config_with_depth(
        query="query",
        config={"mode": "hybrid", "rerank": True},
        chunks=[],
        vector_store=None,
        embedding_generator=None,
        bm25_index=None,
        retrieval_cfg=cfg,
        retriever=FakeRetriever(chunks),
        reranker=reranker,
    )

    assert len(reranker.seen) == 3
    assert "Title A" in reranker.seen[2]["rerank_text"]
    assert [item["parent_id"] for item in results] == ["A", "B"]
    assert results[0]["text"] == "best passage"
    assert diagnostics["unique_parent_count_before_reranking"] == 2
    assert diagnostics["unique_parent_count_after_reranking"] == 2
    assert diagnostics["rerank_candidate_chunk_count"] == 3
    assert diagnostics["reranking_latency_ms"] >= 0


def test_b4_fails_if_reranker_is_unavailable():
    cfg = {
        "candidate_pool_size": 1,
        "similarity_threshold": 0.0,
        "rerank_candidate_pool": 1,
        "evaluation_depth": 1,
    }
    with pytest.raises(RuntimeError, match="requires a loaded cross-encoder"):
        _run_config_with_depth(
            query="query",
            config={"mode": "hybrid", "rerank": True},
            chunks=[],
            vector_store=None,
            embedding_generator=None,
            bm25_index=None,
            retrieval_cfg=cfg,
            retriever=FakeRetriever([
                {"text": "passage", "score": 1.0, "metadata": {"parent_id": "A"}}
            ]),
            reranker=None,
        )
