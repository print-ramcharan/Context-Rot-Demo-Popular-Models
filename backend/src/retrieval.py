from src.vector_store import VectorStore
from src.embedding import EmbeddingGenerator
import numpy as np
import math
import re
import time
from collections import OrderedDict
import logging
import hashlib

logger = logging.getLogger(__name__)

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

class BM25Index:
    """
    BM25 lexical index for fast keyword-based scoring.
    Uses an inverted index for O(1) term lookups instead of linear scans.
    Standard k1/b parameters for term saturation and length normalization.
    """
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.k1 = k1
        self.b = b
        self.doc_freq = {}
        self.doc_len = []
        self.avgdl = 0.0
        self.inverted_index = {}  # {token: {doc_idx: tf}}
        self._build()

    def _build(self):
        total_len = 0
        for doc_idx, text in enumerate(self.texts):
            tokens = _tokenize(text)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)

            # Count term frequencies for this document in a single pass
            tf_map = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            # Populate inverted index and document frequency
            for token, tf in tf_map.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                    self.doc_freq[token] = 0
                self.inverted_index[token][doc_idx] = tf
                self.doc_freq[token] += 1

        self.avgdl = (total_len / len(self.texts)) if self.texts else 0.0

    def score(self, query: str) -> list[float]:
        if not self.texts:
            return []
        if self.avgdl == 0.0:
            return [0.0] * len(self.texts)
        tokens = _tokenize(query)
        scores = [0.0] * len(self.texts)
        if not tokens:
            return scores
        total_docs = len(self.texts)
        for token in tokens:
            if token not in self.inverted_index:
                continue
            df = self.doc_freq[token]
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            # Only iterate over documents that actually contain this token
            for doc_idx, tf in self.inverted_index[token].items():
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[doc_idx] / self.avgdl))
                scores[doc_idx] += idf * ((tf * (self.k1 + 1)) / denom)
        return scores

class QueryCache:
    """
    LRU-style cache with TTL for retrieval results.
    """
    def __init__(self, max_items: int = 10000, ttl_s: int = 300):
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._store = OrderedDict()

    def get(self, key):
        now = time.time()
        if key not in self._store:
            return None
        value, timestamp = self._store.pop(key)
        if now - timestamp > self.ttl_s:
            return None
        self._store[key] = (value, timestamp)
        return value

    def set(self, key, value):
        now = time.time()
        self._store[key] = (value, now)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear()

class SemanticRetriever:
    """
    Retrieves relevant text chunks using semantic similarity.
    """
    
    def __init__(self, vector_store: VectorStore, 
                 embedding_generator: EmbeddingGenerator,
                 top_k: int = 3,
                 similarity_threshold: float = 0.0,
                 mode: str = "semantic",
                 dense_k: int = 30,
                 bm25_k: int = 30,
                 alpha: float = 0.65,
                 enable_rerank: bool = False,
                 rerank_top_n: int = 6,
                 rerank_candidate_pool: int = 30,
                 query_cache_ttl_s: int = 300,
                 query_cache_max_items: int = 10000):
        """
        Initialize retriever with vector store and embedding generator.
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            top_k: Default number of chunks to retrieve
            similarity_threshold: Minimum similarity score (0-1 for cosine)
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.mode = mode
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.alpha = alpha
        self.enable_rerank = enable_rerank
        self.rerank_top_n = rerank_top_n
        self.rerank_candidate_pool = rerank_candidate_pool
        self.reranker = None
        self._query_cache = QueryCache(max_items=query_cache_max_items, ttl_s=query_cache_ttl_s)
        self._cache_generation = 0
        self._bm25_index = None
        self._bm25_generation = -1

    def set_reranker(self, reranker):
        self.reranker = reranker

    @staticmethod
    def _trace_chunk_id(index: int, metadata: dict) -> str:
        source = metadata.get("source") or "index"
        chunk_index = metadata.get("chunk_index")
        if chunk_index is None:
            chunk_index = index
        return f"{source}:{chunk_index}"

    def retrieve_with_trace(
        self,
        query: str,
        evaluation_k: tuple[int, ...] = (5, 10, 20),
        threshold: float = 0.0,
    ) -> tuple[list[dict], dict]:
        """Run uncached hybrid retrieval and retain every measurable stage."""
        import concurrent.futures

        max_k = max(evaluation_k)
        dense_k = min(self.dense_k, self.vector_store.index.ntotal or 0)
        bm25_k = self.bm25_k
        query_embedding_start = time.perf_counter()
        query_embedding = self.embedding_generator.embed_text(query)
        embedding_ms = (time.perf_counter() - query_embedding_start) * 1000

        faiss_start = time.perf_counter()
        dense_results = self.vector_store.search(query_embedding, k=dense_k or max_k)
        faiss_ms = (time.perf_counter() - faiss_start) * 1000

        bm25_start = time.perf_counter()
        self.build_lexical_index()
        bm25_scores = self._bm25_index.score(query) if self._bm25_index else []
        positive_bm25 = [
            (idx, score) for idx, score in enumerate(bm25_scores) if score > 0
        ]
        positive_bm25.sort(key=lambda pair: pair[1], reverse=True)
        positive_bm25 = positive_bm25[:bm25_k]
        bm25_ms = (time.perf_counter() - bm25_start) * 1000

        dense_entries = []
        for rank, index in enumerate(dense_results["indices"], start=1):
            if index < 0:
                continue
            metadata = self.vector_store.metadata[index]
            dense_entries.append({
                "chunk_id": self._trace_chunk_id(index, metadata),
                "index": int(index),
                "rank": rank,
                "score": float(dense_results["distances"][rank - 1]),
            })
        bm25_entries = [{
            "chunk_id": self._trace_chunk_id(index, self.vector_store.metadata[index]),
            "index": int(index),
            "rank": rank,
            "score": float(score),
        } for rank, (index, score) in enumerate(positive_bm25, start=1)]

        rrf_start = time.perf_counter()
        rrf_scores = {}
        dense_by_index = {entry["index"]: entry for entry in dense_entries}
        bm25_by_index = {entry["index"]: entry for entry in bm25_entries}
        for index in set(dense_by_index) | set(bm25_by_index):
            score = 0.0
            if index in dense_by_index:
                score += 1.0 / (60 + dense_by_index[index]["rank"])
            if index in bm25_by_index:
                score += 1.0 / (60 + bm25_by_index[index]["rank"])
            rrf_scores[index] = score
        # Stable index tie-break makes equal-score RRF rankings reproducible.
        rrf_entries = sorted(
            rrf_scores.items(), key=lambda item: (-item[1], item[0])
        )[: max(self.dense_k + self.bm25_k, max_k)]
        rrf_ms = (time.perf_counter() - rrf_start) * 1000

        candidate_items = []
        for rank, (index, score) in enumerate(rrf_entries, start=1):
            metadata = self.vector_store.metadata[index]
            dense_score = dense_by_index.get(index, {}).get("score")
            if threshold > 0 and dense_score is not None and dense_score < threshold:
                continue
            candidate_items.append({
                "text": self.vector_store.chunks[index],
                "score": float(score),
                "metadata": metadata,
                "rank": rank,
                "vector_index": int(index),
                "dense_score": dense_score,
                "bm25_score": bm25_by_index.get(index, {}).get("score", 0.0),
            })

        rerank_input = candidate_items[:max(self.rerank_candidate_pool, max_k)]
        rerank_start = time.perf_counter()
        reranker_applied = False
        if self.enable_rerank:
            if self.reranker is None or getattr(self.reranker, "model", object()) is None:
                raise RuntimeError(
                    "Reranking is enabled, but the cross-encoder is unavailable. "
                    "The evaluation was stopped rather than silently labelling "
                    "unreranked candidates as reranked."
                )
            if rerank_input:
                final_items = self.reranker.rerank(
                    query, rerank_input, top_n=max_k
                )
                reranker_applied = True
            else:
                final_items = []
        else:
            final_items = rerank_input[:max_k]
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        final_items = final_items[:max_k]

        def trace_item(item: dict, final_rank: int | None = None) -> dict:
            metadata = item.get("metadata", {})
            fallback_index = item.get("vector_index")
            if fallback_index is None:
                fallback_index = item.get("original_rank")
            if fallback_index is None:
                fallback_index = item.get("rank", 0)
            return {
                "chunk_id": self._trace_chunk_id(fallback_index, metadata),
                "score": item.get("score"),
                "rank": final_rank if final_rank is not None else item.get("rank"),
                "original_rank": item.get("original_rank"),
                "metadata": metadata,
            }

        reranked_ids = [
            trace_item(item, final_rank)["chunk_id"]
            for final_rank, item in enumerate(final_items, start=1)
        ]
        input_ids = [
            trace_item(item)["chunk_id"] for item in rerank_input
        ]
        trace = {
            "configuration": {
                "dense_k": self.dense_k,
                "bm25_k": self.bm25_k,
                "rrf_k": 60,
                "rrf_pool": len(rrf_entries),
                "reranker_limit": self.rerank_candidate_pool,
                "evaluation_k": list(evaluation_k),
            },
            "stages": {
                "embedding": {
                    "latency_ms": embedding_ms,
                    "query_vector_dimensions": len(query_embedding),
                },
                "bm25": {
                    "retrieved_chunk_ids": [entry["chunk_id"] for entry in bm25_entries],
                    "scores": [entry["score"] for entry in bm25_entries],
                    "ranks": [entry["rank"] for entry in bm25_entries],
                    "latency_ms": bm25_ms,
                },
                "dense": {
                    "retrieved_chunk_ids": [entry["chunk_id"] for entry in dense_entries],
                    "scores": [entry["score"] for entry in dense_entries],
                    "ranks": [entry["rank"] for entry in dense_entries],
                    "latency_ms": faiss_ms,
                },
                "rrf": {
                    "candidate_chunk_ids": [
                        self._trace_chunk_id(index, self.vector_store.metadata[index])
                        for index, _ in rrf_entries
                    ],
                    "rrf_scores": [score for _, score in rrf_entries],
                    "latency_ms": rrf_ms,
                },
                "reranker": {
                    "enabled": self.enable_rerank,
                    "applied": reranker_applied,
                    "input_chunk_ids": input_ids,
                    "scored_chunk_ids": (
                        self.reranker.last_trace.get("scored_chunk_ids", [])
                        if reranker_applied else []
                    ),
                    "reranked_chunk_ids": reranked_ids,
                    "final_ranks": list(range(1, len(reranked_ids) + 1)),
                    "original_ranks": [
                        item.get("original_rank", item.get("rank"))
                        for item in final_items
                    ],
                    "discarded_chunk_ids": [
                        chunk_id for chunk_id in input_ids if chunk_id not in reranked_ids
                    ],
                    "raw_scores": (
                        self.reranker.last_trace.get("raw_scores", [])
                        if reranker_applied else []
                    ),
                    "latency_ms": rerank_ms,
                },
            },
        }
        return final_items, trace

    def invalidate_cache(self):
        self._cache_generation += 1
        self._query_cache.clear()
    
    def retrieve(self, query: str, k: int = None,
                 threshold: float = None, mode: str = None,
                 use_cache: bool = True) -> list[dict]:
        """
        Retrieve most relevant chunks for a query.
        """
        k = k if k is not None else self.top_k
        threshold = threshold if threshold is not None else self.similarity_threshold
        mode = mode or self.mode

        # Cache key includes threshold/mode to keep results consistent with filters.
        query_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        cache_key = (query_key, k, threshold, mode, self._cache_generation)
        if use_cache:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                return cached

        if mode == "hybrid":
            results = self.retrieve_hybrid(query, k=k, threshold=threshold)
        else:
            results = self.retrieve_semantic(query, k=k, threshold=threshold)

        if use_cache:
            self._query_cache.set(cache_key, results)
        return results

    def retrieve_semantic(self, query: str, k: int, threshold: float) -> list[dict]:
        query_embedding = self.embedding_generator.embed_text(query)
        results = self.vector_store.search(query_embedding, k=k)
        retrieved_items = []
        for i in range(len(results['chunks'])):
            score = results['distances'][i]
            if self.vector_store.index_type == "cosine":
                if score < threshold:
                    continue
            retrieved_items.append({
                'text': results['chunks'][i],
                'score': score,
                'metadata': results['metadata'][i],
                'rank': i
            })
        return self._apply_rerank(query, retrieved_items)

    def build_lexical_index(self):
        if self._bm25_generation == self._cache_generation and self._bm25_index:
            return
        self._bm25_index = BM25Index(self.vector_store.chunks)
        self._bm25_generation = self._cache_generation

    def _normalize_scores(self, scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        max_val = max(values)
        min_val = min(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    def retrieve_hybrid(self, query: str, k: int, threshold: float) -> list[dict]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion (RRF) to combine
        dense (semantic) and sparse (BM25) rankings without broken
        score normalization.
        """
        import concurrent.futures

        # Phase 1: Prepare for search (Parallelizable)
        # We need the embedding for dense search and the index for lexical search
        query_emb = None
        
        def run_dense_search():
            nonlocal query_emb
            query_emb = self.embedding_generator.embed_text(query)
            index_total = self.vector_store.index.ntotal if self.vector_store.index else 0
            dense_k = min(self.dense_k, index_total or 0)
            dense_k = dense_k if dense_k > 0 else k
            return self.vector_store.search(query_emb, k=dense_k)

        def run_lexical_search():
            self.build_lexical_index()
            return self._bm25_index.score(query) if self._bm25_index else []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(run_dense_search)
            future_lexical = executor.submit(run_lexical_search)
            
            dense_results = future_dense.result()
            bm25_scores_list = future_lexical.result()

        # Build dense rank and score maps
        dense_rank_map = {}
        dense_score_map = {}
        for rank, idx in enumerate(dense_results['indices'], start=1):
            if idx != -1:
                dense_rank_map[idx] = rank
                dense_score_map[idx] = dense_results['distances'][rank - 1]

        # Build BM25 rank map
        bm25_rank_map = {}
        if bm25_scores_list:
            bm25_ranked_indices = sorted(
                range(len(bm25_scores_list)),
                key=lambda i: bm25_scores_list[i],
                reverse=True
            )
            positive_indices = [
                idx for idx in bm25_ranked_indices if bm25_scores_list[idx] > 0
            ][:self.bm25_k]
            for rank, idx in enumerate(positive_indices, start=1):
                bm25_rank_map[idx] = rank

        # --- Reciprocal Rank Fusion ---
        # RRF_K is the standard constant (typically 60) that prevents
        # top-ranked documents from dominating the fusion score.
        RRF_K = 60
        all_candidate_ids = set(dense_rank_map.keys()) | set(bm25_rank_map.keys())

        rrf_scores = {}
        for idx in all_candidate_ids:
            rrf_scores[idx] = 0.0
            if idx in dense_rank_map:
                rrf_scores[idx] += 1.0 / (RRF_K + dense_rank_map[idx])
            if idx in bm25_rank_map:
                rrf_scores[idx] += 1.0 / (RRF_K + bm25_rank_map[idx])

        # Stable index tie-break makes equal-score RRF rankings reproducible.
        ranked = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))

        retrieved_items = []
        for rank, (idx, score) in enumerate(ranked[:k]):
            if idx >= len(self.vector_store.chunks):
                continue

            # Apply similarity threshold on the original dense score if available
            dense_score = dense_score_map.get(idx, None)
            if threshold > 0 and dense_score is not None:
                if self.vector_store.index_type == "cosine" and dense_score < threshold:
                    continue

            retrieved_items.append({
                'text': self.vector_store.chunks[idx],
                'score': score,
                'metadata': self.vector_store.metadata[idx],
                'rank': rank,
                'dense_score': dense_score,
                'bm25_score': bm25_scores_list[idx] if idx < len(bm25_scores_list) else 0.0
            })

        return self._apply_rerank(query, retrieved_items)

    def _apply_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        if not self.enable_rerank or not self.reranker:
            return candidates
        pool = candidates[: self.rerank_candidate_pool]
        try:
            reranked = self.reranker.rerank(query, pool, top_n=self.rerank_top_n)
            return reranked
        except Exception:
            logger.exception("Reranking failed")
            return candidates
    
    def retrieve_multi_query(self, queries: list[str], 
                            k: int = None) -> list[dict]:
        """
        Retrieve chunks relevant to multiple queries and merge results.
        
        Args:
            queries (list[str]): Multiple query strings
            k (int): Total chunks to return
            
        Returns:
            list[dict]: Deduplicated and ranked chunks
        """
        k = k if k is not None else self.top_k
        all_results = []
        
        for query in queries:
            all_results.extend(self.retrieve(query, k=k))
            
        # Deduplicate by text content
        seen_texts = set()
        unique_results = []
        
        # Sort by score (assuming higher is better for ranking consistency)
        # If L2, we should sort by score ASC.
        # Hybrid retrieval always uses higher-is-better combined scores.
        reverse_sort = True if self.mode == "hybrid" else (self.vector_store.index_type == "cosine")
        all_results.sort(key=lambda x: x['score'], reverse=reverse_sort)
        
        for res in all_results:
            if res['text'] not in seen_texts:
                seen_texts.add(res['text'])
                unique_results.append(res)
            if len(unique_results) >= k:
                break
                
        return unique_results
    
    def deduplicate_chunks(self, chunks: list[dict], overlap_threshold: float = 0.8) -> list[dict]:
        """
        Remove duplicate or highly overlapping chunks.
        
        Args:
            chunks (list[dict]): Retrieved chunks
            overlap_threshold: Jaccard similarity threshold for deduplication
            
        Returns:
            list[dict]: Deduplicated chunks
        """
        if not chunks:
            return []
            
        import re
        def get_words(text):
            # Strip punctuation and normalize whitespace
            text = re.sub(r'[^\w\s]', '', text.lower())
            return set(text.split())
            
        unique_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            is_duplicate = False
            words_i = get_words(chunks[i]['text'])
            
            for existing in unique_chunks:
                words_existing = get_words(existing['text'])
                intersection = words_i.intersection(words_existing)
                union = words_i.union(words_existing)
                
                if len(union) == 0:
                    continue
                    
                jaccard = len(intersection) / len(union)
                if jaccard > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunks[i])
                
        return unique_chunks
    
    def explain_retrieval(self, query: str, k: int = 5) -> dict:
        """
        Retrieve with detailed explanation of why chunks were selected.
        """
        query_embedding = self.embedding_generator.embed_text(query)
        results = self.retrieve(query, k=k)
        
        explanation = {
            'query': query,
            'embedding_norm': float(np.linalg.norm(query_embedding)),
            'retrieved_chunks': results
        }
        return explanation
