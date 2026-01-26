import pytest
from src.chunking import TextChunker

class TestTextChunker:
    
    def test_basic_chunking(self):
        """Test basic word-based chunking."""
        chunker = TextChunker(chunk_size=10, overlap=2)
        # 30 words
        text = " ".join([f"word{i}" for i in range(30)])
        chunks = chunker.chunk_by_words(text)
        
        assert len(chunks) > 1
        assert len(chunks[0].split()) == 10
        # Check overlap: last 2 words of chunk 0 should be first 2 words of chunk 1
        c0_words = chunks[0].split()
        c1_words = chunks[1].split()
        assert c0_words[-2:] == c1_words[:2]
    
    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = TextChunker()
        assert chunker.chunk_by_words("") == []
        assert chunker.chunk_by_words("   ") == []
    
    def test_overlap_validation(self):
        """Test that overlap < chunk_size is enforced."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=10, overlap=15)
    
    def test_short_text(self):
        """Test text shorter than chunk size."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "short text"
        chunks = chunker.chunk_by_words(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_metadata_generation(self):
        """Test chunk metadata is accurate."""
        chunker = TextChunker(chunk_size=10, overlap=0)
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = chunker.chunk_by_words(text)
        metadata = chunker.get_chunk_metadata(chunks)
        
        assert len(metadata) == len(chunks)
        assert all('chunk_id' in m for m in metadata)
        assert all('word_count' in m for m in metadata)
        assert metadata[0]['word_count'] == 10

    def test_sentence_chunking(self):
        """Test chunking by sentences."""
        chunker = TextChunker()
        text = "Sentence one. Sentence two! Sentence three? Sentence four."
        chunks = chunker.chunk_by_sentences(text, max_words=5)
        # Each "Sentence x." is 2 words. 
        # Sent 1 + Sent 2 = 4 words
        # Sent 3 + Sent 4 = 4 words
        assert len(chunks) == 2
        assert "Sentence one. Sentence two!" in chunks[0]
        assert "Sentence three? Sentence four." in chunks[1]
