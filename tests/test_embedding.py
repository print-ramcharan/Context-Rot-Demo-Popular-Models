import pytest
import numpy as np
from src.embedding import EmbeddingGenerator

class TestEmbeddingGenerator:
    
    @pytest.fixture(scope="class")
    def generator(self):
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    def test_single_embedding(self, generator):
        """Test single text embedding."""
        emb = generator.embed_text("hello world")
        assert emb.shape == (384,)
        assert emb.dtype == np.float32
    
    def test_batch_embedding(self, generator):
        """Test batch embedding."""
        texts = ["text1", "text2", "text3"]
        embs = generator.embed_batch(texts)
        assert embs.shape == (3, 384)
    
    def test_embedding_determinism(self, generator):
        """Test that same input gives same output."""
        text = "deterministic test"
        emb1 = generator.embed_text(text)
        emb2 = generator.embed_text(text)
        assert np.allclose(emb1, emb2)
    
    def test_normalization(self, generator):
        """Test L2 normalization."""
        embs = np.random.randn(10, 384).astype('float32')
        normalized = generator.normalize_embeddings(embs)
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_empty_text(self, generator):
        """Test handling of empty text."""
        # Current implementation returns zero vector or handles it
        emb = generator.embed_text("")
        assert emb.shape == (384,)
        assert np.all(emb == 0)
    
    def test_get_dimension(self, generator):
        """Test getting embedding dimension."""
        assert generator.get_embedding_dimension() == 384
