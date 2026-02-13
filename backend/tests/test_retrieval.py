import pytest
import numpy as np
from src.vector_store import VectorStore
from src.embedding import EmbeddingGenerator
from src.retrieval import SemanticRetriever

class TestSemanticRetriever:
    
    @pytest.fixture(scope="class")
    def retriever(self):
        # Using a small dimension to speed up. But EmbeddingGenerator uses 384.
        # Let's use 384 consistently since it's already downloaded.
        gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        store = VectorStore(dimension=384, index_type="cosine")
        
        # Add some sample data
        chunks = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "The capital of France is Paris.",
            "Python is a popular programming language."
        ]
        embeddings = gen.embed_batch(chunks)
        store.add(embeddings, chunks)
        
        return SemanticRetriever(store, gen, top_k=2)
    
    def test_basic_retrieval(self, retriever):
        """Test basic retrieval for a query."""
        results = retriever.retrieve("What is ML?")
        assert len(results) == 2
        assert "Machine learning" in results[0]['text']
        assert results[0]['rank'] == 0
        assert results[1]['rank'] == 1
    
    def test_threshold_retrieval(self, retriever):
        """Test retrieval with similarity threshold."""
        # Query totally unrelated to Paris
        results = retriever.retrieve("Python programming", threshold=0.5)
        # Should find Python, maybe French capital is too low
        assert any("Python" in r['text'] for r in results)
        assert all(r['score'] >= 0.5 for r in results)
    
    def test_multi_query(self, retriever):
        """Test merging results from multiple queries."""
        queries = ["Who is Machine Learning?", "Tell me about France."]
        results = retriever.retrieve_multi_query(queries, k=3)
        
        assert len(results) <= 3
        # Should have both ML and Paris info
        texts = [r['text'] for r in results]
        assert any("Machine learning" in t for t in texts)
        assert any("France" in t for t in texts)
    
    def test_deduplication(self, retriever):
        """Test deduplicating highly overlapping chunks."""
        chunks = [
            {'text': "This is a test chunk for deduplication.", 'score': 0.9},
            {'text': "This is a test chunk for deduplication!", 'score': 0.85}, # Almost same
            {'text': "Something completely different.", 'score': 0.5}
        ]
        unique = retriever.deduplicate_chunks(chunks, overlap_threshold=0.8)
        assert len(unique) == 2
        assert unique[0]['text'] == chunks[0]['text']
        assert unique[1]['text'] == chunks[2]['text']
