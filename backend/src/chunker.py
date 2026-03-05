import re
from typing import List, Dict

class TextChunker:
    """
    Splits long text into overlapping chunks for semantic retrieval.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size (int): Target chunk size in characters
            overlap (int): Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict]: Chunks with metadata
        """
        # Clean text
        text = text.strip()
        if not text:
            return []
        
        # Split by sentences first (better boundaries)
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        offset = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'offset': offset,
                        'length': len(current_chunk)
                    })
                    offset += len(current_chunk)
                
                # Start new chunk with overlap
                current_chunk = current_chunk[-self.overlap:] + " " + sentence if len(current_chunk) > self.overlap else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'offset': offset,
                'length': len(current_chunk)
            })
        
        return chunks
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]