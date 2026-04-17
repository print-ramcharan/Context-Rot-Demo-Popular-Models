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
    Uses standard k1/b parameters for term saturation and length normalization.
    """
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.k1 = k1
        self.b = b
        self.doc_freq = {}
        self.doc_len = []
        self.avgdl = 0.0
        self.doc_tokens = []
        self._build()

    def _build(self):
        total_len = 0
        for text in self.texts:
            tokens = _tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
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
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for i, doc_tokens in enumerate(self.doc_tokens):
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl))
                scores[i] += idf * ((tf * (self.k1 + 1)) / denom)
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
        index_total = self.vector_store.index.ntotal if self.vector_store.index else 0
        dense_k = min(self.dense_k, index_total or 0)
        dense_k = dense_k if dense_k > 0 else k
        dense_results = self.vector_store.search(
            self.embedding_generator.embed_text(query),
            k=dense_k
        )
        dense_raw_scores = {
            idx: dense_results['distances'][i]
            for i, idx in enumerate(dense_results['indices'])
            if idx != -1
        }
        dense_scores = dict(dense_raw_scores)
        if self.vector_store.index_type != "cosine":
            # Convert L2 distance to similarity using a reciprocal transform.
            # This maps smaller distances to higher similarity while bounding values in (0, 1].
            dense_scores = {idx: 1 / (1 + score) for idx, score in dense_scores.items()}

        self.build_lexical_index()
        bm25_scores_list = self._bm25_index.score(query) if self._bm25_index else []
        bm25_scores = {}
        if bm25_scores_list:
            top_indices = sorted(
                range(len(bm25_scores_list)),
                key=lambda i: bm25_scores_list[i],
                reverse=True
            )[: self.bm25_k]
            bm25_scores = {
                idx: bm25_scores_list[idx]
                for idx in top_indices
                if bm25_scores_list[idx] > 0
            }

        dense_norm = self._normalize_scores(dense_scores)
        bm25_norm = self._normalize_scores(bm25_scores)

        combined = {}
        for idx in set(dense_norm) | set(bm25_norm):
            combined[idx] = (self.alpha * dense_norm.get(idx, 0.0)) + ((1 - self.alpha) * bm25_norm.get(idx, 0.0))

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        retrieved_items = []
        for rank, (idx, score) in enumerate(ranked[:k]):
            if idx >= len(self.vector_store.chunks):
                continue
            dense_score = dense_scores.get(idx, None)
            if threshold > 0 and dense_score is not None:
                if self.vector_store.index_type == "cosine":
                    if dense_score < threshold:
                        continue
                else:
                    raw_distance = dense_raw_scores.get(idx, None)
                    if raw_distance is not None and raw_distance > threshold:
                        continue
            retrieved_items.append({
                'text': self.vector_store.chunks[idx],
                'score': score,
                'metadata': self.vector_store.metadata[idx],
                'rank': rank,
                'dense_score': dense_score,
                'bm25_score': bm25_scores.get(idx, 0.0)
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
