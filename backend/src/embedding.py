from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import hashlib
import sqlite3
import time
import threading
from pathlib import Path

class EmbeddingCache:
    """
    Simple persistent embedding cache backed by SQLite.
    """

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
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_updated ON embeddings(updated_at)")
        self._conn.commit()

    def get(self, cache_key: str) -> np.ndarray | None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT embedding, dim FROM embeddings WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
        if not row:
            return None
        blob, dim = row
        arr = np.frombuffer(blob, dtype=np.float32)
        return arr.reshape(dim,)

    def set(self, cache_key: str, embedding: np.ndarray):
        embedding = embedding.astype(np.float32)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings (cache_key, embedding, dim, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, embedding.tobytes(), embedding.shape[0], time.time())
            )
            self._conn.commit()
            self._prune()

    def _prune(self):
        if self.max_items <= 0:
            return
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        if count <= self.max_items:
            return
        to_delete = count - self.max_items
        cursor.execute(
            """
            DELETE FROM embeddings
            WHERE cache_key IN (
                SELECT cache_key FROM embeddings
                ORDER BY updated_at ASC
                LIMIT ?
            )
            """,
            (to_delete,)
        )
        self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

class EmbeddingGenerator:
    """
    Generates dense vector embeddings for text using Sentence Transformers.
    """
    
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        device=None,
        cache_path: str | None = "cache/embeddings.sqlite",
        cache_max_items: int = 200000,
        cache_enabled: bool = True
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): HuggingFace model identifier
            device (str): "cpu" or "cuda" or "mps"
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = {}  # Simple in-memory cache
        self.cache_enabled = cache_enabled
        self.persistent_cache = None
        if cache_enabled and cache_path:
            self.persistent_cache = EmbeddingCache(cache_path, max_items=cache_max_items)

    def _hash_text(self, text: str) -> str:
        payload = f"{self.model_name}::{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Embedding vector of shape (dimension,)
        """
        if not text.strip():
            # Return zero vector if text is empty
            dim = self.get_embedding_dimension()
            return np.zeros(dim, dtype=np.float32)
            
        if text in self.cache:
            return self.cache[text]

        cache_key = self._hash_text(text)
        if self.persistent_cache:
            cached = self.persistent_cache.get(cache_key)
            if cached is not None:
                self.cache[text] = cached
                return cached

        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)
        self.cache[text] = embedding
        if self.persistent_cache:
            self.persistent_cache.set(cache_key, embedding)
        return embedding
    
    def embed_batch(self, texts: list[str], batch_size=32, 
                    show_progress=False) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts (list[str]): List of input texts
            batch_size (int): Number of texts to process at once
            show_progress (bool): Show progress bar
            
        Returns:
            np.ndarray: Embedding matrix of shape (num_texts, dimension)
        """
        if not texts:
            return np.array([], dtype=np.float32)

        results: list[np.ndarray] = [None] * len(texts)
        missing_texts = []
        missing_indices = []

        for i, text in enumerate(texts):
            if text is None or not str(text).strip():
                dim = self.get_embedding_dimension()
                results[i] = np.zeros(dim, dtype=np.float32)
                continue
            if text in self.cache:
                results[i] = self.cache[text]
                continue
            cache_key = self._hash_text(text)
            if self.persistent_cache:
                cached = self.persistent_cache.get(cache_key)
                if cached is not None:
                    self.cache[text] = cached
                    results[i] = cached
                    continue
            missing_texts.append(text)
            missing_indices.append(i)

        if missing_texts:
            new_embeddings = self.model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            ).astype(np.float32)
            for idx, emb in zip(missing_indices, new_embeddings):
                text = texts[idx]
                results[idx] = emb
                self.cache[text] = emb
                if self.persistent_cache:
                    self.persistent_cache.set(self._hash_text(text), emb)

        if all(r is None for r in results):
            dim = self.get_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        return np.vstack(results)
    
    def get_embedding_dimension(self) -> int:
        """
        Return the dimensionality of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings for cosine similarity computation.
        
        Args:
            embeddings (np.ndarray): Embedding vectors
            
        Returns:
            np.ndarray: Normalized embeddings
        """
        if embeddings.ndim == 1:
            norm = np.linalg.norm(embeddings)
            if norm == 0:
                return embeddings
            return embeddings / norm
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            return embeddings / norms

    def close(self):
        if self.persistent_cache:
            self.persistent_cache.close()
