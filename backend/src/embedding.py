from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class EmbeddingGenerator:
    """
    Generates dense vector embeddings for text using Sentence Transformers.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): HuggingFace model identifier
            device (str): "cpu" or "cuda" or "mps"
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = {} # Simple in-memory cache
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Embedding vector of shape (dimension,)
        """
        if not text.strip():
            # Return zero vector if text is empty
            dim = self.get_embedding_dimension()
            return np.zeros(dim, dtype=np.float32)
            
        if text in self.cache:
            return self.cache[text]
            
        embedding = self.model.encode(text, convert_to_numpy=True)
        self.cache[text] = embedding
        return embedding
    
    def embed_batch(self, texts: list[str], batch_size=32, 
                    show_progress=False) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts (list[str]): List of input texts
            batch_size (int): Number of texts to process at once
            show_progress (bool): Show progress bar
            
        Returns:
            np.ndarray: Embedding matrix of shape (num_texts, dimension)
        """
        if not texts:
            return np.array([], dtype=np.float32)
            
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Return the dimensionality of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings for cosine similarity computation.
        
        Args:
            embeddings (np.ndarray): Embedding vectors
            
        Returns:
            np.ndarray: Normalized embeddings
        """
        if embeddings.ndim == 1:
            norm = np.linalg.norm(embeddings)
            if norm == 0:
                return embeddings
            return embeddings / norm
        else:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            return embeddings / norms
