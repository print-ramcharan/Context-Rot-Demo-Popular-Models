from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import hashlib
import sqlite3
import time
import threading
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Persistent embedding cache backed by SQLite."""

    def __init__(self, path: str, max_items: int = 200000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._initialize()

    def _initialize(self):
        cursor = self._conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_updated ON embeddings(updated_at)")
        self._conn.commit()

    def get(self, cache_key: str) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding, dim FROM embeddings WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        return np.frombuffer(row[0], dtype=np.float32).reshape(row[1],).copy()

    def set(self, cache_key: str, embedding: np.ndarray):
        embedding = np.asarray(embedding, dtype=np.float32)
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO embeddings
                (cache_key, embedding, dim, updated_at) VALUES (?, ?, ?, ?)""",
                (cache_key, embedding.tobytes(), embedding.shape[0], time.time()),
            )
            if self.max_items > 0:
                self._conn.execute(
                    """DELETE FROM embeddings WHERE cache_key IN (
                        SELECT cache_key FROM embeddings ORDER BY updated_at ASC
                        LIMIT MAX(0, (SELECT COUNT(*) FROM embeddings) - ?)
                    )""",
                    (self.max_items,),
                )
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()


class EmbeddingGenerator:
    """
    Dense retrieval embeddings. Documents and queries use explicit separate
    methods so instruction-aware models can apply query prompts consistently.
    """

    def __init__(
        self,
        model_name="BAAI/bge-base-en-v1.5",
        device=None,
        cache_path: str | None = "cache/embeddings.sqlite",
        cache_max_items: int = 200000,
        cache_enabled: bool = True,
        query_instruction: str = "",
        query_prompt_name: str | None = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.query_prompt_name = query_prompt_name
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = {}
        self.cache_enabled = cache_enabled
        self.persistent_cache = (
            EmbeddingCache(cache_path, max_items=cache_max_items)
            if cache_enabled and cache_path else None
        )

    def _hash_text(self, text: str, role: str) -> str:
        query_config = (
            f"{self.query_instruction}::{self.query_prompt_name}"
            if role == "query" else ""
        )
        payload = f"{self.model_name}::{role}::{query_config}::{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _encode_one(self, text: str, role: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimension(), dtype=np.float32)
        cache_key = self._hash_text(text, role)
        if self.persistent_cache:
            cached = self.persistent_cache.get(cache_key)
            if cached is not None:
                return cached
        cache_text = f"{self.query_instruction}{text}" if role == "query" and self.query_instruction else text
        kwargs = {"convert_to_numpy": True, "normalize_embeddings": True}
        if role == "query" and self.query_prompt_name:
            kwargs["prompt_name"] = self.query_prompt_name
        vector = self.model.encode(cache_text, **kwargs).astype(np.float32)
        if self.persistent_cache:
            self.persistent_cache.set(cache_key, vector)
        return vector

    def embed_text(self, text: str) -> np.ndarray:
        return self._encode_one(text, "document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode_one(text, "query")

    def embed_batch(self, texts: list[str], batch_size=16, show_progress=False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.get_embedding_dimension()), dtype=np.float32)
        vectors = []
        missing = []
        missing_positions = []
        for position, text in enumerate(texts):
            key = self._hash_text(str(text), "document")
            cached = self.persistent_cache.get(key) if self.persistent_cache else None
            if cached is not None:
                vectors.append((position, cached))
            else:
                missing.append(str(text))
                missing_positions.append(position)
        if missing:
            encoded = self.model.encode(
                missing,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            for position, text, vector in zip(missing_positions, missing, encoded):
                vectors.append((position, vector))
                if self.persistent_cache:
                    self.persistent_cache.set(self._hash_text(text, "document"), vector)
        vectors.sort(key=lambda item: item[0])
        return np.vstack([vector for _, vector in vectors]).astype(np.float32)

    def get_embedding_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim == 1:
            norm = np.linalg.norm(values)
            return values if norm == 0 else values / norm
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return values / norms

    def close(self):
        if self.persistent_cache:
            self.persistent_cache.close()
