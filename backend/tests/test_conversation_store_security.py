from pathlib import Path

import numpy as np

from conversation_store import ConversationStore


class FakeEmbeddingGenerator:
    def embed_batch(self, texts):
        vectors = np.zeros((len(texts), 384), dtype="float32")
        for index, text in enumerate(texts):
            vectors[index, 0] = 1.0
        return vectors

    def embed_text(self, text):
        vector = np.zeros((1, 384), dtype="float32")
        vector[0, 0] = 1.0
        return vector


def test_conversation_sessions_are_isolated_by_user(tmp_path: Path):
    store = ConversationStore(str(tmp_path / "conversations"))
    embeddings = FakeEmbeddingGenerator()

    store.store_conversation(
        user_id="alice", platform="chatgpt", session_id="shared-id",
        prompt="Alice private note", response="Alice response",
        embedding_generator=embeddings,
    )
    store.store_conversation(
        user_id="bob", platform="chatgpt", session_id="shared-id",
        prompt="Bob private note", response="Bob response",
        embedding_generator=embeddings,
    )

    alice_results = store.retrieve_context(
        user_id="alice", query="private note",
        embedding_generator=embeddings, top_k=10,
    )
    bob_results = store.retrieve_context(
        user_id="bob", query="private note",
        embedding_generator=embeddings, top_k=10,
    )

    assert alice_results
    assert bob_results
    assert all("Alice" in item["text"] for item in alice_results)
    assert all("Bob" in item["text"] for item in bob_results)
    assert len(store.list_sessions("alice")) == 1
    assert len(store.list_sessions("bob")) == 1

    assert store.delete_session("shared-id", "bob")["deleted"] is True
    assert len(store.list_sessions("alice")) == 1
    assert store.list_sessions("bob") == []


def test_conversation_store_persists_json_not_pickle(tmp_path: Path):
    directory = tmp_path / "conversations"
    store = ConversationStore(str(directory))
    store.store_conversation(
        user_id="alice", platform="chatgpt", session_id="s1",
        prompt="persist this", response="saved",
        embedding_generator=FakeEmbeddingGenerator(),
    )

    assert (directory / "chunks.json").exists()
    assert (directory / "metadata.json").exists()
    assert (directory / "sessions.json").exists()
    assert not (directory / "chunks.pkl").exists()
    assert not (directory / "metadata.pkl").exists()

    reloaded = ConversationStore(str(directory))
    assert len(reloaded.list_sessions("alice")) == 1
