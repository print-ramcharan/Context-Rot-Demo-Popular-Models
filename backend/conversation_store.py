import json
import uuid
import faiss
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from src.chunking import TextChunker


# ── Storage layout ────────────────────────────────────────────────────────────
#
#   backend/
#     conversations/
#       index.faiss          ← FAISS inner-product index (cosine, dim=384)
#       chunks.pkl           ← list[str]  — raw chunk texts
#       metadata.pkl         ← list[dict] — per-chunk metadata
#       sessions.json        ← {session_id: SessionMeta} master registry
#
# ─────────────────────────────────────────────────────────────────────────────

CONV_DIR = Path("conversations")
INDEX_FILE = CONV_DIR / "index.faiss"
CHUNKS_FILE = CONV_DIR / "chunks.pkl"
META_FILE = CONV_DIR / "metadata.pkl"
SESSIONS_FILE = CONV_DIR / "sessions.json"
EMBEDDING_DIM = 384


class ConversationStore:
    """
    Manages a FAISS vector index dedicated to conversation memory.

    Keeps its own on-disk storage completely separate from the
    existing document index so the two never interfere.
    """

    def __init__(self, storage_dir: str = "conversations"):
        global CONV_DIR, INDEX_FILE, CHUNKS_FILE, META_FILE, SESSIONS_FILE
        CONV_DIR = Path(storage_dir)
        INDEX_FILE = CONV_DIR / "index.faiss"
        CHUNKS_FILE = CONV_DIR / "chunks.pkl"
        META_FILE = CONV_DIR / "metadata.pkl"
        SESSIONS_FILE = CONV_DIR / "sessions.json"

        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: list[str] = []
        self._metadata: list[dict] = []
        self._sessions: dict[str, dict] = {}

        CONV_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_index(self) -> faiss.IndexFlatIP:
        """Create a fresh cosine-similarity FAISS index."""
        return faiss.IndexFlatIP(EMBEDDING_DIM)

    def _load(self):
        """Load all data from disk, or initialise empty structures."""
        if INDEX_FILE.exists():
            self._index = faiss.read_index(str(INDEX_FILE))
        else:
            self._index = self._init_index()

        self._chunks = (
            pickle.loads(CHUNKS_FILE.read_bytes()) if CHUNKS_FILE.exists() else []
        )
        self._metadata = (
            pickle.loads(META_FILE.read_bytes()) if META_FILE.exists() else []
        )
        self._sessions = (
            json.loads(SESSIONS_FILE.read_text()) if SESSIONS_FILE.exists() else {}
        )

    def _save(self):
        """Persist index + metadata to disk atomically enough for a demo."""
        faiss.write_index(self._index, str(INDEX_FILE))
        CHUNKS_FILE.write_bytes(pickle.dumps(self._chunks))
        META_FILE.write_bytes(pickle.dumps(self._metadata))
        SESSIONS_FILE.write_text(json.dumps(self._sessions, indent=2))

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalize rows so inner product == cosine similarity."""
        vecs = vecs.astype("float32")
        faiss.normalize_L2(vecs)
        return vecs

    # ── Public API ────────────────────────────────────────────────────────────

    def store_conversation(
        self,
        platform: str,
        session_id: str,
        prompt: str,
        response: str,
        embedding_generator,          # pass in the existing EmbeddingGenerator
        chunk_size: int = 220,
        overlap: int = 30,
    ) -> dict:
        """
        Chunk a prompt+response exchange, embed it, and store in FAISS.

        Parameters
        ----------
        platform          : "chatgpt" | "gemini" | "claude" | …
        session_id        : stable ID for this browser session
        prompt            : user message text
        response          : assistant message text
        embedding_generator: the EmbeddingGenerator already in main.py
        chunk_size        : words per chunk
        overlap           : word overlap between chunks

        Returns
        -------
        dict with chunk_count and session_id
        """
        # Guard: reject fully-empty exchanges
        if not prompt.strip() and not response.strip():
            raise ValueError("Both prompt and response are empty — nothing to store")

        # Combine prompt + response into one block with role tags
        exchange = f"[user]: {prompt.strip()}\n[assistant]: {response.strip()}"

        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        raw_chunks = [c["text"] for c in chunker.chunk_document(exchange)]

        if not raw_chunks:
            return {"chunk_count": 0, "session_id": session_id}

        # Embed
        embeddings = embedding_generator.embed_batch(raw_chunks)
        embeddings = self._normalize(embeddings)

        # Build per-chunk metadata
        timestamp = datetime.now(timezone.utc).isoformat()
        start_idx = len(self._chunks)
        chunk_meta = [
            {
                "session_id": session_id,
                "platform": platform,
                "timestamp": timestamp,
                "chunk_index": start_idx + j,
                "role": "exchange",
            }
            for j in range(len(raw_chunks))
        ]

        # Store
        self._index.add(embeddings)
        self._chunks.extend(raw_chunks)
        self._metadata.extend(chunk_meta)

        # Update session registry
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "session_id": session_id,
                "platform": platform,
                "created_at": timestamp,
                "message_count": 0,
                "chunk_count": 0,
            }
        self._sessions[session_id]["message_count"] += 1
        self._sessions[session_id]["chunk_count"] += len(raw_chunks)
        self._sessions[session_id]["last_updated"] = timestamp

        self._save()

        return {
            "session_id": session_id,
            "chunk_count": len(raw_chunks),
            "total_chunks_stored": len(self._chunks),
        }

    def retrieve_context(
        self,
        query: str,
        embedding_generator,
        top_k: int = 5,
        platform_filter: Optional[str] = None,
        similarity_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Find the top-K conversation chunks most relevant to the query.

        Parameters
        ----------
        query             : the user's current question / prompt
        embedding_generator: the EmbeddingGenerator already in main.py
        top_k             : number of results to return
        platform_filter   : if set, only return chunks from this platform
        similarity_threshold: minimum cosine similarity (0.0 = no filter)

        Returns
        -------
        list of dicts: {text, score, session_id, platform, timestamp}
        """
        if self._index.ntotal == 0:
            return []

        q_emb = embedding_generator.embed_text(query)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        q_emb = q_emb.astype("float32")
        faiss.normalize_L2(q_emb)

        # Fetch more than top_k so we can filter by platform if needed
        fetch_k = min(top_k * 3, self._index.ntotal)
        scores, indices = self._index.search(q_emb, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < similarity_threshold and similarity_threshold > 0.0:
                continue
            meta = self._metadata[idx]
            if platform_filter and meta.get("platform") != platform_filter:
                continue
            results.append(
                {
                    "text": self._chunks[idx],
                    "score": float(score),
                    "session_id": meta.get("session_id", ""),
                    "platform": meta.get("platform", ""),
                    "timestamp": meta.get("timestamp", ""),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def list_sessions(self) -> list[dict]:
        """Return all stored sessions sorted newest-first."""
        return sorted(
            self._sessions.values(),
            key=lambda s: s.get("last_updated", s.get("created_at", "")),
            reverse=True,
        )

    def delete_session(self, session_id: str) -> dict:
        """
        Remove all chunks belonging to a session from the FAISS index.

        Because FAISS flat indexes don't support in-place deletion,
        we rebuild the index from the remaining chunks. This is fine
        for a demo-scale store (thousands of chunks, not millions).
        """
        if session_id not in self._sessions:
            return {"deleted": False, "reason": "session_id not found"}

        # Partition chunks: keep vs. delete
        keep_chunks, keep_meta, keep_embs = [], [], []

        if len(self._chunks) > 0:
            # Reconstruct embeddings from index (flat IP stores vectors)
            all_embs = np.zeros(
                (len(self._chunks), EMBEDDING_DIM), dtype="float32"
            )
            self._index.reconstruct_n(0, len(self._chunks), all_embs)

            for i, (chunk, meta) in enumerate(zip(self._chunks, self._metadata)):
                if meta.get("session_id") != session_id:
                    keep_chunks.append(chunk)
                    keep_meta.append(meta)
                    keep_embs.append(all_embs[i])

        # Rebuild index
        self._index = self._init_index()
        self._chunks = keep_chunks
        self._metadata = keep_meta

        if keep_embs:
            emb_matrix = np.stack(keep_embs).astype("float32")
            self._index.add(emb_matrix)

        # Remove from session registry
        del self._sessions[session_id]

        # Update chunk_counts in remaining sessions (they didn't change,
        # but the registry numbers are still valid — nothing to adjust)
        self._save()

        return {"deleted": True, "session_id": session_id}

    def get_stats(self) -> dict:
        """Return a summary of what's stored."""
        return {
            "total_chunks": len(self._chunks),
            "total_sessions": len(self._sessions),
            "index_size": self._index.ntotal,
            "storage_dir": str(CONV_DIR),
        }
