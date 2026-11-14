import re
import uuid

class TextChunker:
    """
    Splits text into overlapping chunks for embedding and retrieval.
    """
    
    def __init__(self, chunk_size=220, overlap=30):
        """
        Initialize chunker with size and overlap parameters.
        
        Args:
            chunk_size (int): Number of words per chunk
            overlap (int): Number of overlapping words between chunks
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def _normalize_and_split(self, text: str) -> list[str]:
        """Normalize whitespace and split into words."""
        if not text or not text.strip():
            return []
        text = " ".join(text.split())
        return text.split()

    def chunk_document(self, text: str) -> list[dict]:
        """
        Split text into word-based chunks with overlap, returning chunk metadata.

        Returns:
            list[dict]: [{'text', 'offset', 'length', 'word_count', 'chunk_index'}]
        """
        # Preserve offsets into the original, unnormalized source text.
        spans = [(match.group(0), match.start(), match.end())
                 for match in re.finditer(r"\\S+", text or "")]
        if not spans:
            return []

        chunks = []
        start = 0
        chunk_index = 0
        while start < len(spans):
            end = min(start + self.chunk_size, len(spans))
            selected = spans[start:end]
            chunk_text = " ".join(item[0] for item in selected)
            offset = selected[0][1]
            source_end = selected[-1][2]
            chunks.append({
                "text": chunk_text,
                "offset": offset,
                "length": source_end - offset,
                "word_count": len(selected),
                "chunk_index": chunk_index
            })
            chunk_index += 1

            if end >= len(words):
                break
            start += (self.chunk_size - self.overlap)

        return chunks
    
    def chunk_by_words(self, text: str) -> list[str]:
        """
        Split text into word-based chunks with overlap.
        
        Args:
            text (str): Input text to chunk
            
        Returns:
            list[str]: List of text chunks
        """
        words = self._normalize_and_split(text)
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [" ".join(words)]

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            if end >= len(words):
                break

            start += (self.chunk_size - self.overlap)

        return chunks
    
    def chunk_by_sentences(self, text: str, max_words=300) -> list[str]:
        """
        Split text into chunks at sentence boundaries.
        Combines sentences until reaching max_words limit.
        
        Args:
            text (str): Input text to chunk
            max_words (int): Maximum words per chunk
            
        Returns:
            list[str]: List of text chunks
        """
        if not text or not text.strip():
            return []

        # Simple sentence splitter (can be improved with nltk/spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            if current_word_count + sentence_word_count > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def get_chunk_metadata(self, chunks: list[str]) -> list[dict]:
        """
        Generate metadata for each chunk including position, length, and ID.
        
        Args:
            chunks (list[str]): List of text chunks
            
        Returns:
            list[dict]: Metadata for each chunk
        """
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                'chunk_id': str(uuid.uuid4()),
                'position': i,
                'word_count': len(chunk.split()),
                'char_count': len(chunk)
            })
        return metadata
