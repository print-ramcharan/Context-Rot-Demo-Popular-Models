"""
tests/test_conversation_store.py
─────────────────────────────────
Pytest tests for the ConversationStore module.

Run from the backend/ directory:
    pytest tests/test_conversation_store.py -v
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock


# ── Minimal EmbeddingGenerator mock ──────────────────────────────────────────
# We mock the embedding generator so tests don't need a GPU / model download.

def _make_mock_gen(dim: int = 384):
    """Return a mock EmbeddingGenerator that produces deterministic vectors."""
    gen = MagicMock()

    def embed_text(text: str) -> np.ndarray:
        # Hash the text to get a stable, unique vector
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        vec = rng.standard_normal(dim).astype("float32")
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    def embed_batch(texts, **kwargs) -> np.ndarray:
        return np.stack([embed_text(t) for t in texts])

    gen.embed_text.side_effect = embed_text
    gen.embed_batch.side_effect = embed_batch
    return gen


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    """Fresh ConversationStore backed by a temp directory."""
    # Import here so the module path resolution doesn't depend on CWD
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from conversation_store import ConversationStore
    return ConversationStore(storage_dir=str(tmp_path / "conversations"))


@pytest.fixture
def gen():
    return _make_mock_gen()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestStoreConversation:

    def test_store_returns_correct_keys(self, tmp_store, gen):
        result = tmp_store.store_conversation(
            platform="chatgpt",
            session_id="sess-001",
            prompt="What is the capital of France?",
            response="The capital of France is Paris.",
            embedding_generator=gen,
        )
        assert "session_id" in result
        assert "chunk_count" in result
        assert result["session_id"] == "sess-001"
        assert result["chunk_count"] >= 1

    def test_store_creates_session_entry(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="gemini",
            session_id="sess-002",
            prompt="Tell me about Python.",
            response="Python is a programming language.",
            embedding_generator=gen,
        )
        sessions = tmp_store.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "sess-002" in ids

    def test_store_multiple_messages_same_session(self, tmp_store, gen):
        for i in range(3):
            tmp_store.store_conversation(
                platform="claude",
                session_id="sess-003",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                embedding_generator=gen,
            )
        sessions = {s["session_id"]: s for s in tmp_store.list_sessions()}
        assert sessions["sess-003"]["message_count"] == 3

    def test_store_empty_prompt_and_response_raises(self, tmp_store, gen):
        with pytest.raises(Exception):
            tmp_store.store_conversation(
                platform="chatgpt",
                session_id="sess-empty",
                prompt="   ",
                response="   ",
                embedding_generator=gen,
            )

    def test_chunks_persisted_to_disk(self, tmp_path, gen):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from conversation_store import ConversationStore

        store_dir = str(tmp_path / "persist_test")
        s1 = ConversationStore(storage_dir=store_dir)
        s1.store_conversation(
            platform="chatgpt",
            session_id="sess-persist",
            prompt="Hello",
            response="Hi there",
            embedding_generator=gen,
        )
        del s1

        # Reload from disk
        s2 = ConversationStore(storage_dir=store_dir)
        assert len(s2._chunks) > 0
        assert "sess-persist" in s2._sessions


class TestRetrieveContext:

    def test_retrieve_returns_results(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="chatgpt",
            session_id="sess-r1",
            prompt="Explain neural networks",
            response="Neural networks are computational models inspired by the brain.",
            embedding_generator=gen,
        )
        results = tmp_store.retrieve_context(
            query="What are neural networks?",
            embedding_generator=gen,
            top_k=3,
            similarity_threshold=0.0,
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_retrieve_empty_store_returns_empty(self, tmp_store, gen):
        results = tmp_store.retrieve_context(
            query="anything",
            embedding_generator=gen,
        )
        assert results == []

    def test_retrieve_result_has_required_fields(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="gemini",
            session_id="sess-r2",
            prompt="What is FAISS?",
            response="FAISS is a library for efficient similarity search.",
            embedding_generator=gen,
        )
        results = tmp_store.retrieve_context(
            query="FAISS similarity search",
            embedding_generator=gen,
            top_k=5,
        )
        assert len(results) >= 1
        r = results[0]
        for key in ("text", "score", "session_id", "platform", "timestamp"):
            assert key in r, f"Missing key: {key}"

    def test_platform_filter_works(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="chatgpt",
            session_id="s-gpt",
            prompt="From ChatGPT session",
            response="ChatGPT answer here",
            embedding_generator=gen,
        )
        tmp_store.store_conversation(
            platform="gemini",
            session_id="s-gem",
            prompt="From Gemini session",
            response="Gemini answer here",
            embedding_generator=gen,
        )
        results = tmp_store.retrieve_context(
            query="session answer",
            embedding_generator=gen,
            top_k=10,
            platform_filter="chatgpt",
        )
        for r in results:
            assert r["platform"] == "chatgpt"

    def test_similarity_threshold_filters_low_scores(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="claude",
            session_id="sess-thresh",
            prompt="Completely unrelated gardening tips",
            response="Water your plants regularly.",
            embedding_generator=gen,
        )
        # With a very high threshold almost nothing should pass
        results = tmp_store.retrieve_context(
            query="quantum physics equations",
            embedding_generator=gen,
            top_k=5,
            similarity_threshold=0.99,
        )
        # Either empty or only very high-similarity results
        for r in results:
            assert r["score"] >= 0.99

    def test_top_k_respected(self, tmp_store, gen):
        for i in range(10):
            tmp_store.store_conversation(
                platform="chatgpt",
                session_id=f"sess-topk-{i}",
                prompt=f"Message {i} about machine learning",
                response=f"Response {i} about machine learning topics",
                embedding_generator=gen,
            )
        results = tmp_store.retrieve_context(
            query="machine learning",
            embedding_generator=gen,
            top_k=3,
        )
        assert len(results) <= 3


class TestListSessions:

    def test_list_sessions_empty(self, tmp_store):
        assert tmp_store.list_sessions() == []

    def test_list_sessions_sorted_newest_first(self, tmp_store, gen):
        import time
        for platform in ("chatgpt", "gemini", "claude"):
            tmp_store.store_conversation(
                platform=platform,
                session_id=f"sess-{platform}",
                prompt="hello",
                response="hi",
                embedding_generator=gen,
            )
            time.sleep(0.01)  # ensure distinct timestamps

        sessions = tmp_store.list_sessions()
        timestamps = [s.get("last_updated", s.get("created_at")) for s in sessions]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_sessions_contains_metadata(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="chatgpt",
            session_id="sess-meta",
            prompt="test",
            response="test response",
            embedding_generator=gen,
        )
        sessions = tmp_store.list_sessions()
        s = sessions[0]
        for key in ("session_id", "platform", "message_count", "chunk_count"):
            assert key in s, f"Missing key: {key}"


class TestDeleteSession:

    def test_delete_existing_session(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="chatgpt",
            session_id="sess-del",
            prompt="delete me",
            response="sure",
            embedding_generator=gen,
        )
        result = tmp_store.delete_session("sess-del")
        assert result["deleted"] is True

        ids = [s["session_id"] for s in tmp_store.list_sessions()]
        assert "sess-del" not in ids

    def test_delete_nonexistent_session(self, tmp_store):
        result = tmp_store.delete_session("nonexistent-id")
        assert result["deleted"] is False

    def test_delete_removes_chunks_from_index(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="chatgpt",
            session_id="sess-del2",
            prompt="some content about python",
            response="python is great",
            embedding_generator=gen,
        )
        before = tmp_store.get_stats()["total_chunks"]
        tmp_store.delete_session("sess-del2")
        after = tmp_store.get_stats()["total_chunks"]
        assert after < before

    def test_delete_does_not_affect_other_sessions(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="chatgpt",
            session_id="sess-keep",
            prompt="keep this",
            response="keeping",
            embedding_generator=gen,
        )
        tmp_store.store_conversation(
            platform="gemini",
            session_id="sess-remove",
            prompt="remove this",
            response="removing",
            embedding_generator=gen,
        )
        tmp_store.delete_session("sess-remove")

        ids = [s["session_id"] for s in tmp_store.list_sessions()]
        assert "sess-keep" in ids
        assert "sess-remove" not in ids

    def test_retrieve_after_delete_excludes_deleted(self, tmp_store, gen):
        tmp_store.store_conversation(
            platform="chatgpt",
            session_id="sess-del3",
            prompt="very specific deleted content about flying elephants",
            response="flying elephants are fictional",
            embedding_generator=gen,
        )
        tmp_store.delete_session("sess-del3")

        results = tmp_store.retrieve_context(
            query="flying elephants",
            embedding_generator=gen,
            top_k=10,
        )
        for r in results:
            assert r["session_id"] != "sess-del3"


class TestGetStats:

    def test_stats_structure(self, tmp_store):
        stats = tmp_store.get_stats()
        for key in ("total_chunks", "total_sessions", "index_size", "storage_dir"):
            assert key in stats

    def test_stats_reflect_stored_data(self, tmp_store, gen):
        assert tmp_store.get_stats()["total_sessions"] == 0

        tmp_store.store_conversation(
            platform="chatgpt",
            session_id="sess-stats",
            prompt="test",
            response="test",
            embedding_generator=gen,
        )
        stats = tmp_store.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["total_chunks"] >= 1
        assert stats["index_size"] == stats["total_chunks"]