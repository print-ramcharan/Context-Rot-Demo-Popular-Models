import pytest
import numpy as np
from src.vector_store import VectorStore
import shutil
import os

class TestVectorStore:
    
    @pytest.fixture
    def store(self):
        return VectorStore(dimension=384, index_type="L2")
    
    def test_add_and_search(self, store):
        """Test adding vectors and searching."""
        # Create reproducible random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(10, 384).astype('float32')
        chunks = [f"chunk_{i}" for i in range(10)]
        metadata = [{'id': i} for i in range(10)]
        
        store.add(embeddings, chunks, metadata)
        
        # Search for first embedding
        query = embeddings[0]
        results = store.search(query, k=3)
        
        assert len(results['chunks']) == 3
        # Should find itself first for L2 (distance 0)
        assert results['chunks'][0] == "chunk_0"
        assert results['metadata'][0]['id'] == 0
        assert results['distances'][0] < 1e-5
    
    def test_cosine_similarity(self):
        """Test cosine similarity index."""
        store = VectorStore(dimension=384, index_type="cosine")
        embeddings = np.array([
            [1.0, 0.0] + [0.0]*382,
            [0.0, 1.0] + [0.0]*382
        ]).astype('float32')
        chunks = ["chunk_x", "chunk_y"]
        store.add(embeddings, chunks)
        
        # Query similar to chunk_x
        query = np.array([1.0, 0.1] + [0.0]*382).astype('float32')
        results = store.search(query, k=1)
        
        assert results['chunks'][0] == "chunk_x"
        # For inner product on normalized vectors, higher is more similar
        assert results['distances'][0] > 0.9
    
    def test_save_and_load(self, store, tmp_path):
        """Test persistence."""
        embeddings = np.random.randn(5, 384).astype('float32')
        chunks = [f"chunk_{i}" for i in range(5)]
        
        save_dir = str(tmp_path / "test_index")
        store.add(embeddings, chunks)
        store.save(save_dir)
        
        new_store = VectorStore(dimension=384)
        new_store.load(save_dir)
        
        stats = new_store.get_statistics()
        assert stats['total_chunks'] == 5
        assert new_store.chunks == chunks
    
    def test_clear(self, store):
        """Test clearing the index."""
        embeddings = np.random.randn(5, 384).astype('float32')
        chunks = [f"chunk_{i}" for i in range(5)]
        
        store.add(embeddings, chunks)
        store.clear()
        
        assert store.get_statistics()['total_chunks'] == 0
        assert len(store.chunks) == 0

    def test_rejects_embedding_dimension_mismatch(self, store):
        with pytest.raises(ValueError, match="dimension"):
            store.add(np.zeros((1, 12), dtype="float32"), ["chunk"])

    def test_rejects_metadata_length_mismatch_even_when_empty(self, store):
        with pytest.raises(ValueError, match="metadata"):
            store.add(np.zeros((1, 384), dtype="float32"), ["chunk"], metadata=[])

    def test_rejects_non_finite_embeddings(self, store):
        vectors = np.zeros((1, 384), dtype="float32")
        vectors[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            store.add(vectors, ["chunk"])

    def test_empty_search_and_invalid_k(self, store):
        result = store.search(np.zeros(384, dtype="float32"), k=5)
        assert result == {"chunks": [], "distances": [], "indices": [], "metadata": []}
        with pytest.raises(ValueError, match="positive"):
            store.search(np.zeros(384, dtype="float32"), k=0)

    def test_document_identity_lookup(self, store):
        vectors = np.ones((1, 384), dtype="float32")
        store.add(vectors, ["chunk"], [{"document_id": "doc-a"}])
        assert store.contains_document("doc-a")
        assert not store.contains_document("doc-b")

    def test_load_rejects_misaligned_metadata(self, store, tmp_path):
        import pickle
        save_dir = tmp_path / "broken_index"
        store.add(np.ones((1, 384), dtype="float32"), ["chunk"], [{"id": 1}])
        store.save(str(save_dir))
        with open(save_dir / "metadata.pkl", "wb") as handle:
            pickle.dump([], handle)
        fresh = VectorStore(dimension=384)
        with pytest.raises(ValueError, match="inconsistent"):
            fresh.load(str(save_dir))
