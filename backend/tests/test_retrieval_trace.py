from src.retrieval import SemanticRetriever


class FakeIndex:
    ntotal = 2


class FakeVectorStore:
    index = FakeIndex()
    chunks = ["needle appears in first passage", "needle appears in second passage"]
    metadata = [
        {"source": "paper.txt", "chunk_index": 3},
        {"source": "paper.txt", "chunk_index": 8},
    ]

    def search(self, query_embedding, k):
        indices = [0, 1][:k]
        return {
            "indices": indices,
            "distances": [0.8, 0.7][:k],
            "chunks": [self.chunks[i] for i in indices],
            "metadata": [self.metadata[i] for i in indices],
        }


class FakeEmbeddingGenerator:
    def embed_text(self, query):
        return [1.0, 0.0]


class FakeReranker:
    model = object()

    def __init__(self):
        self.last_trace = {}

    def rerank(self, query, candidates, top_n):
        self.last_trace = {
            "scored_chunk_ids": [
                f"{item['metadata']['source']}:{item['metadata']['chunk_index']}"
                for item in candidates
            ],
            "raw_scores": [0.1, 0.9],
        }
        results = []
        for item, score in zip(candidates, [0.1, 0.9]):
            updated = dict(item)
            updated["original_rank"] = item["rank"]
            updated["score"] = score
            updated["rank"] = 1 if score == 0.9 else 2
            results.append(updated)
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_n]


def test_trace_uses_final_reranked_order_and_preserves_chunk_identity():
    retriever = SemanticRetriever(
        FakeVectorStore(),
        FakeEmbeddingGenerator(),
        dense_k=2,
        bm25_k=2,
        enable_rerank=True,
        rerank_candidate_pool=2,
    )
    retriever.set_reranker(FakeReranker())

    results, trace = retriever.retrieve_with_trace(
        "needle", evaluation_k=(2,)
    )

    rerank_trace = trace["stages"]["reranker"]
    assert [item["metadata"]["chunk_index"] for item in results] == [8, 3]
    assert rerank_trace["reranked_chunk_ids"] == [
        "paper.txt:8", "paper.txt:3"
    ]
    assert rerank_trace["final_ranks"] == [1, 2]
    assert rerank_trace["original_ranks"] == [2, 1]
    assert rerank_trace["scored_chunk_ids"] == [
        "paper.txt:3", "paper.txt:8"
    ]
    assert rerank_trace["applied"] is True
