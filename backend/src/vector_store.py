import json
import pickle
import os
import shutil
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """FAISS vector store with strict row-to-chunk metadata alignment."""

    def __init__(self, dimension: int, index_type="L2"):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.chunks = []
        self.metadata = []
        self._initialize_index()

    def _initialize_index(self):
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError("Unsupported index_type: {}".format(self.index_type))

    def _validate_alignment(self, index=None, chunks=None, metadata=None):
        index = self.index if index is None else index
        chunks = self.chunks if chunks is None else chunks
        metadata = self.metadata if metadata is None else metadata
        if index.d != self.dimension:
            raise ValueError("FAISS index dimension does not match configured dimension")
        if index.ntotal != len(chunks) or len(chunks) != len(metadata):
            raise ValueError(
                "Vector store alignment error: index={}, chunks={}, metadata={}".format(
                    index.ntotal, len(chunks), len(metadata)
                )
            )

    def add(self, embeddings: np.ndarray, chunks: list, metadata: list = None):
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension does not match store dimension")
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        if metadata is not None and len(metadata) != len(chunks):
            raise ValueError("Number of metadata items must match number of chunks")
        if any(not isinstance(chunk, str) for chunk in chunks):
            raise ValueError("All chunks must be strings")
        vectors = np.array(embeddings, dtype="float32", copy=True)
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values")
        chunk_metadata = [{} for _ in chunks] if metadata is None else list(metadata)
        if any(not isinstance(item, dict) for item in chunk_metadata):
            raise ValueError("Each metadata item must be a dictionary")
        if not chunks:
            return
        if self.index_type == "cosine":
            faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.metadata.extend(chunk_metadata)
        self._validate_alignment()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> dict:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        query = np.asarray(query_embedding)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query_embedding must contain exactly one query vector")
        if query.shape[1] != self.dimension:
            raise ValueError("Query dimension does not match store dimension")
        query = np.array(query, dtype="float32", copy=True)
        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain only finite values")
        self._validate_alignment()
        if self.index.ntotal == 0:
            return {"chunks": [], "distances": [], "indices": [], "metadata": []}
        if self.index_type == "cosine":
            faiss.normalize_L2(query)
        distances, indices = self.index.search(query, min(k, self.index.ntotal))
        valid = [int(idx) for idx in indices[0] if idx >= 0]
        return {
            "chunks": [self.chunks[idx] for idx in valid],
            "distances": distances[0][:len(valid)].tolist(),
            "indices": valid,
            "metadata": [self.metadata[idx] for idx in valid],
        }

    def contains_document(self, document_id: str) -> bool:
        return any(item.get("document_id") == document_id for item in self.metadata)

    def save(self, directory: str):
        """Persist a complete, aligned index snapshot using directory swap."""
        self._validate_alignment()
        path = Path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        backup_path = path.with_name(path.name + ".bak")

        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self.index, str(temp_path / "index.faiss"))
            with open(temp_path / "chunks.pkl", "wb") as handle:
                pickle.dump(self.chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(temp_path / "metadata.pkl", "wb") as handle:
                pickle.dump(self.metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(temp_path / "config.json", "w", encoding="utf-8") as handle:
                json.dump({
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "total_vectors": self.index.ntotal,
                    "format_version": 2,
                }, handle)
            if backup_path.exists():
                shutil.rmtree(backup_path)
            if path.exists():
                os.replace(path, backup_path)
            os.replace(temp_path, path)
            if backup_path.exists():
                shutil.rmtree(backup_path)
        except Exception:
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
            if not path.exists() and backup_path.exists():
                os.replace(backup_path, path)
            raise

    def load(self, directory: str):
        path = Path(directory)
        backup_path = path.with_name(path.name + ".bak")
        if not path.exists() and backup_path.exists():
            os.replace(backup_path, path)
        with open(path / "config.json", "r") as handle:
            config = json.load(handle)
        loaded_dimension = int(config["dimension"])
        loaded_type = config["index_type"]
        if loaded_dimension != self.dimension:
            raise ValueError("Saved index dimension does not match configured dimension")
        if loaded_type not in ("L2", "cosine"):
            raise ValueError("Unsupported saved index_type: {}".format(loaded_type))
        loaded_index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as handle:
            loaded_chunks = pickle.load(handle)
        with open(path / "metadata.pkl", "rb") as handle:
            loaded_metadata = pickle.load(handle)
        if loaded_index.d != loaded_dimension:
            raise ValueError("Saved FAISS index dimension does not match config")
        if "total_vectors" in config and loaded_index.ntotal != int(config["total_vectors"]):
            raise ValueError("Saved FAISS vector count does not match snapshot metadata")
        if loaded_index.ntotal != len(loaded_chunks) or len(loaded_chunks) != len(loaded_metadata):
            raise ValueError(
                "Saved vector store is inconsistent: index={}, chunks={}, metadata={}".format(
                    loaded_index.ntotal, len(loaded_chunks), len(loaded_metadata)
                )
            )
        self.index_type = loaded_type
        self.index = loaded_index
        self.chunks = loaded_chunks
        self.metadata = loaded_metadata

    def get_statistics(self) -> dict:
        self._validate_alignment()
        return {
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "index_type": self.index_type,
        }

    def clear(self):
        self.chunks = []
        self.metadata = []
        self._initialize_index()

    def get_all_texts(self):
        return self.chunks if self.chunks else []
