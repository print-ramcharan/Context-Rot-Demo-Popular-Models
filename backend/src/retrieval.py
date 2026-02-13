from src.vector_store import VectorStore
from src.embedding import EmbeddingGenerator
import numpy as np

class SemanticRetriever:
    """
    Retrieves relevant text chunks using semantic similarity.
    """
    
    def __init__(self, vector_store: VectorStore, 
                 embedding_generator: EmbeddingGenerator,
                 top_k: int = 3,
                 similarity_threshold: float = 0.0):
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
    
    def retrieve(self, query: str, k: int = None, 
                 threshold: float = None) -> list[dict]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query (str): User query text
            k (int, optional): Override default top_k
            threshold (float, optional): Override similarity threshold
            
        Returns:
            list[dict]: Retrieved chunks with scores
        """
        k = k if k is not None else self.top_k
        threshold = threshold if threshold is not None else self.similarity_threshold
        
        # 1. Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        
        # 2. Search in vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # 3. Format and filter results
        retrieved_items = []
        for i in range(len(results['chunks'])):
            score = results['distances'][i]
            
            # If using L2, distances are squared Euclidean. Lower is better.
            # Convert to a similarity score if needed, but let's keep raw for now
            # or normalize depending on index_type.
            if self.vector_store.index_type == "cosine":
                # For cosine (Inner Product), higher is better.
                if score < threshold:
                    continue
            else:
                # For L2, lower is better. Thresholding is trickier.
                # Let's assume threshold is a similarity floor if provided.
                pass
                
            retrieved_items.append({
                'text': results['chunks'][i],
                'score': score,
                'metadata': results['metadata'][i],
                'rank': i
            })
            
        return retrieved_items
    
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
        reverse_sort = (self.vector_store.index_type == "cosine")
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
