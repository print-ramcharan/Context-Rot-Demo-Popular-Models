"""User-scoped conversation memory backed by FAISS and schema-checked JSON.

Legacy pickle files are deliberately not loaded. Pickle deserialization can execute
arbitrary code; back up old .pkl files and re-import conversations through the API.
"""
from __future__ import annotations

import json
import os
import logging
import faiss
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from src.chunking import TextChunker

EMBEDDING_DIM = 384
logger = logging.getLogger(__name__)


class ConversationStore:
    """FAISS conversation memory with user ownership enforced at every operation."""

    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = Path(storage_dir)
        self.index_file = self.storage_dir / "index.faiss"
        self.chunks_file = self.storage_dir / "chunks.json"
        self.metadata_file = self.storage_dir / "metadata.json"
        self.sessions_file = self.storage_dir / "sessions.json"
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: list[str] = []
        self._metadata: list[dict] = []
        self._sessions: dict[str, dict] = {}
        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.storage_dir, 0o700)
        except OSError:
            logger.warning("Could not restrict permissions on conversation directory")
        if not self.chunks_file.exists() and any(
            (self.storage_dir / name).exists()
            for name in ("chunks.pkl", "metadata.pkl")
        ):
            logger.warning(
                "Legacy pickle conversation files were found and intentionally ignored. "
                "Back them up and re-import trusted conversations through the API."
            )
        self._load()

    def _init_index(self):
        return faiss.IndexFlatIP(EMBEDDING_DIM)

    @staticmethod
    def _read_json(path: Path, expected_type):
        if not path.exists():
            return expected_type()
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, expected_type):
            raise ValueError("Invalid conversation store schema in " + path.name)
        return value

    def _load(self):
        self._chunks = self._read_json(self.chunks_file, list)
        self._metadata = self._read_json(self.metadata_file, list)
        self._sessions = self._read_json(self.sessions_file, dict)
        if len(self._chunks) != len(self._metadata):
            raise ValueError("Conversation chunks and metadata are inconsistent")
        if any(not isinstance(chunk, str) for chunk in self._chunks):
            raise ValueError("Conversation chunk records must be strings")
        if any(
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("user_id"), str)
            or not isinstance(metadata.get("session_id"), str)
            for metadata in self._metadata
        ):
            raise ValueError("Conversation metadata is missing valid ownership fields")
        if any(
            not isinstance(session, dict)
            or not isinstance(session.get("user_id"), str)
            or not isinstance(session.get("session_id"), str)
            for session in self._sessions.values()
        ):
            raise ValueError("Conversation session registry has an invalid schema")
        if self.index_file.exists() and self.chunks_file.exists() and self.metadata_file.exists():
            self._index = faiss.read_index(str(self.index_file))
            if self._index.ntotal != len(self._chunks):
                raise ValueError("Conversation FAISS index does not match JSON records")
        else:
            self._index = self._init_index()
            self._chunks = []
            self._metadata = []
            self._sessions = {}

    @staticmethod
    def _atomic_json(path: Path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(str(temporary), str(path))
        try:
            os.chmod(path, 0o600)
        except OSError:
            logger.warning("Could not restrict permissions on %s", path.name)

    def _save(self):
        temporary_index = self.index_file.with_suffix(".faiss.tmp")
        faiss.write_index(self._index, str(temporary_index))
        os.replace(str(temporary_index), str(self.index_file))
        try:
            os.chmod(self.index_file, 0o600)
        except OSError:
            logger.warning("Could not restrict permissions on conversation FAISS index")
        self._atomic_json(self.chunks_file, self._chunks)
        self._atomic_json(self.metadata_file, self._metadata)
        self._atomic_json(self.sessions_file, self._sessions)

    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        # JSON tuple encoding avoids delimiter-collision between principals and IDs.
        return json.dumps([user_id, session_id], separators=(",", ":"))

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype="float32")
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != EMBEDDING_DIM:
            raise ValueError("Unexpected embedding dimension")
        faiss.normalize_L2(vecs)
        return vecs

    def store_conversation(
        self,
        user_id: str,
        platform: str,
        session_id: str,
        prompt: str,
        response: str,
        embedding_generator,
        chunk_size: int = 220,
        overlap: int = 30,
    ) -> dict:
        if not user_id or not session_id:
            raise ValueError("Authenticated user and session ID are required")
        if not prompt.strip() and not response.strip():
            raise ValueError("Both prompt and response are empty")
        exchange = f"[user]: {prompt.strip()}\n[assistant]: {response.strip()}"
        raw_chunks = [
            item["text"] for item in TextChunker(
                chunk_size=chunk_size, overlap=overlap
            ).chunk_document(exchange)
        ]
        if not raw_chunks:
            return {"chunk_count": 0, "session_id": session_id}
        key = self._session_key(user_id, session_id)
        existing_session = self._sessions.get(key)
        if existing_session and existing_session.get("platform") != platform:
            raise ValueError("A session cannot change platform")
        embeddings = self._normalize(embedding_generator.embed_batch(raw_chunks))
        timestamp = datetime.now(timezone.utc).isoformat()
        start_idx = len(self._chunks)
        new_metadata = [{
            "user_id": user_id,
            "session_id": session_id,
            "platform": platform,
            "timestamp": timestamp,
            "chunk_index": start_idx + offset,
            "role": "exchange",
        } for offset in range(len(raw_chunks))]
        self._index.add(embeddings)
        self._chunks.extend(raw_chunks)
        self._metadata.extend(new_metadata)
        if key not in self._sessions:
            self._sessions[key] = {
                "session_id": session_id,
                "user_id": user_id,
                "platform": platform,
                "created_at": timestamp,
                "message_count": 0,
                "chunk_count": 0,
            }
        session = self._sessions[key]
        session["message_count"] += 1
        session["chunk_count"] += len(raw_chunks)
        session["last_updated"] = timestamp
        self._save()
        return {
            "session_id": session_id,
            "chunk_count": len(raw_chunks),
            "total_chunks_stored": len(self._chunks),
        }

    def retrieve_context(
        self,
        user_id: str,
        query: str,
        embedding_generator,
        top_k: int = 5,
        platform_filter: Optional[str] = None,
        similarity_threshold: float = 0.0,
    ) -> list[dict]:
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50")
        if self._index.ntotal == 0:
            return []
        query_embedding = self._normalize(embedding_generator.embed_text(query))
        # Fetch all IDs so another user's vectors cannot crowd this user's top-k.
        scores, indices = self._index.search(query_embedding, self._index.ntotal)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            metadata = self._metadata[index]
            if metadata.get("user_id") != user_id:
                continue
            if platform_filter and metadata.get("platform") != platform_filter:
                continue
            if similarity_threshold > 0 and score < similarity_threshold:
                continue
            results.append({
                "text": self._chunks[index],
                "score": float(score),
                "session_id": metadata.get("session_id", ""),
                "platform": metadata.get("platform", ""),
                "timestamp": metadata.get("timestamp", ""),
            })
            if len(results) >= top_k:
                break
        return results

    def list_sessions(self, user_id: str) -> list[dict]:
        return sorted(
            (session for session in self._sessions.values()
             if session.get("user_id") == user_id),
            key=lambda session: session.get("last_updated", session.get("created_at", "")),
            reverse=True,
        )

    def delete_session(self, session_id: str, user_id: str) -> dict:
        key = self._session_key(user_id, session_id)
        if key not in self._sessions:
            return {"deleted": False, "reason": "session_id not found"}
        keep_indices = [
            index for index, metadata in enumerate(self._metadata)
            if not (
                metadata.get("user_id") == user_id
                and metadata.get("session_id") == session_id
            )
        ]
        if self._chunks:
            all_embeddings = np.zeros((len(self._chunks), EMBEDDING_DIM), dtype="float32")
            self._index.reconstruct_n(0, len(self._chunks), all_embeddings)
            self._chunks = [self._chunks[index] for index in keep_indices]
            self._metadata = [self._metadata[index] for index in keep_indices]
            self._index = self._init_index()
            if keep_indices:
                self._index.add(all_embeddings[keep_indices])
        del self._sessions[key]
        self._save()
        return {"deleted": True, "session_id": session_id}

    def get_stats(self, user_id: str) -> dict:
        owned_sessions = self.list_sessions(user_id)
        owned_chunks = sum(
            1 for metadata in self._metadata if metadata.get("user_id") == user_id
        )
        return {
            "total_chunks": owned_chunks,
            "total_sessions": len(owned_sessions),
            "storage_dir": str(self.storage_dir),
        }
