import faiss
import numpy as np
import pickle
import json
from pathlib import Path

class VectorStore:
    """
    FAISS-based vector database for semantic similarity search.
    """
    
    def __init__(self, dimension: int, index_type="L2"):
        """
        Initialize vector store with specified dimension.
        
        Args:
            dimension (int): Embedding dimension
            index_type (str): "L2" or "cosine"
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.chunks = []  # Store actual text chunks
        self.metadata = []  # Store chunk metadata
        self._initialize_index()
    
    def _initialize_index(self):
        """
        Create FAISS index based on index_type.
        """
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "cosine":
            # For cosine, use inner product on normalized vectors
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")
    
    def add(self, embeddings: np.ndarray, chunks: list[str], 
            metadata: list[dict] = None):
        """
        Add embeddings and associated chunks to the index.
        
        Args:
            embeddings (np.ndarray): Embedding vectors (n, dimension)
            chunks (list[str]): Corresponding text chunks
            metadata (list[dict], optional): Additional metadata per chunk
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
            
        if metadata and len(metadata) != len(chunks):
            raise ValueError("Number of metadata items must match number of chunks")
            
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        if self.index_type == "cosine":
            # Normalize for inner product to get cosine similarity
            faiss.normalize_L2(embeddings)
            
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in range(len(chunks))])
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> dict:
        """
        Find k most similar chunks to the query embedding.
        
        Args:
            query_embedding (np.ndarray): Query vector (dimension,)
            k (int): Number of results to return
            
        Returns:
            dict: {
                'chunks': list[str],
                'distances': list[float],
                'indices': list[int],
                'metadata': list[dict]
            }
        """
        # Ensure query is float32 and has correct shape (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        if self.index_type == "cosine":
            faiss.normalize_L2(query_embedding)
            
        distances, indices = self.index.search(query_embedding, k)
        
        # Filter out invalid indices (FAISS returns -1 if fewer than k items)
        valid_indices = [idx for idx in indices[0] if idx != -1]
        
        return {
            'chunks': [self.chunks[idx] for idx in valid_indices],
            'distances': distances[0][:len(valid_indices)].tolist(),
            'indices': valid_indices,
            'metadata': [self.metadata[idx] for idx in valid_indices]
        }
    
    def save(self, directory: str):
        """
        Save index and associated data to disk.
        
        Args:
            directory (str): Path to save directory
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        
        # Save chunks and metadata
        with open(dir_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(dir_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
            
        # Save config
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        with open(dir_path / "config.json", "w") as f:
            json.dump(config, f)
    
    def load(self, directory: str):
        """
        Load index and associated data from disk.
        
        Args:
            directory (str): Path to load directory
        """
        dir_path = Path(directory)
        
        # Load config first to check compatibility
        with open(dir_path / "config.json", "r") as f:
            config = json.load(f)
            self.dimension = config['dimension']
            self.index_type = config['index_type']
            
        # Load FAISS index
        self.index = faiss.read_index(str(dir_path / "index.faiss"))
        
        # Load chunks and metadata
        with open(dir_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        with open(dir_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
    
    def get_statistics(self) -> dict:
        """
        Return statistics about the vector store.
        
        Returns:
            dict: {
                'total_chunks': int,
                'dimension': int,
                'index_type': str
            }
        """
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'index_type': self.index_type
        }
    
    def clear(self):
        """
        Clear all data and reset the index.
        """
        self.chunks = []
        self.metadata = []
        self._initialize_index()

    def get_all_texts(self):
        """Get all stored text chunks."""
        return self.chunks if self.chunks else []
   
