import yaml
from pathlib import Path
from src.chunking import TextChunker
from src.embedding import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retrieval import SemanticRetriever
from src.context_assembly import ContextAssembler
from src.inference import LLMInference

class ExternalMemorySystem:
    """
    Complete external memory system for LLMs.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize system with configuration.
        """
        self.config = self._load_config(config_path)
        self.persistence_path = self.config.get('storage', {}).get('index_path', 'memory_index')
        self._initialize_components()
        
        # Auto-load if index exists
        if Path(self.persistence_path).exists() or Path(f"{self.persistence_path}.index").exists():
            self.load_memory(self.persistence_path)
    
    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_components(self):
        """Initialize all system components."""
        # 1. Chunker
        chunk_cfg = self.config.get('chunking', {})
        self.chunker = TextChunker(
            chunk_size=chunk_cfg.get('chunk_size', 300),
            overlap=chunk_cfg.get('overlap', 50)
        )
        
        # 2. Embedding Generator
        emb_cfg = self.config.get('embedding', {})
        self.embedding_generator = EmbeddingGenerator(
            model_name=emb_cfg.get('model_name', "all-MiniLM-L6-v2"),
            device=emb_cfg.get('device', 'cpu')
        )
        
        # 3. Vector Store
        dim = self.embedding_generator.get_embedding_dimension()
        self.vector_store = VectorStore(
            dimension=dim,
            index_type="cosine" 
        )
        
        # 4. Retriever
        ret_cfg = self.config.get('retrieval', {})
        self.retriever = SemanticRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            top_k=ret_cfg.get('top_k', 3),
            similarity_threshold=ret_cfg.get('similarity_threshold', 0.0)
        )
        
        # 5. Context Assembler
        self.assembler = ContextAssembler()
        
        # 6. LLM Inference
        llm_cfg = self.config.get('llm', {})
        provider = llm_cfg.get('provider', 'ollama')
        self.llm = LLMInference(
            provider=provider,
            model=llm_cfg.get(provider, {}).get('model'),
            config=llm_cfg.get(provider, {})
        )
    
    def ingest_document(self, text: str, source: str = None, 
                        file_type: str = "text", extension: str = "txt") -> dict:
        """
        Process and store a document in external memory.
        """
        # 1. Chunk
        chunks = self.chunker.chunk_by_words(text)
        metadata = self.chunker.get_chunk_metadata(chunks)
        for m in metadata:
            if source: m['source'] = source
            m['file_type'] = file_type
            m['extension'] = extension
                
        # 2. Embed
        embeddings = self.embedding_generator.embed_batch(chunks)
        
        # 3. Store
        self.vector_store.add(embeddings, chunks, metadata)
        
        # 4. Auto-save
        self.save_memory(self.persistence_path)
        
        return {
            'chunks_created': len(chunks),
            'embeddings_stored': len(embeddings),
            'source': source,
            'file_type': file_type
        }
    
    def ingest_file(self, file_path: str) -> dict:
        """
        Read and ingest a text file with metadata.
        """
        path = Path(file_path)
        ext = path.suffix[1:] if path.suffix else "txt"
        
        # Determine basic file type
        file_type_map = {
            'md': 'markdown',
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'tsx': 'react-typescript',
            'jsx': 'react-javascript',
            'html': 'html',
            'css': 'css',
            'scss': 'sass',
            'txt': 'text',
            'yaml': 'config',
            'yml': 'config',
            'json': 'data',
            'csv': 'data',
            'sql': 'database',
            'sh': 'shell',
            'bash': 'shell',
            'go': 'go',
            'rs': 'rust',
            'c': 'c',
            'cpp': 'cpp',
            'h': 'header',
            'hpp': 'header',
            'java': 'java',
            'rb': 'ruby',
            'php': 'php',
            'env': 'config'
        }
        f_type = file_type_map.get(ext, 'text')
        
        with open(path, 'r') as f:
            text = f.read()
            
        return self.ingest_document(
            text, 
            source=path.name, 
            file_type=f_type, 
            extension=ext
        )
    
    def query(self, question: str, k: int = None, 
              return_context: bool = False,
              template: str = "default") -> dict:
        """
        Query the system and get LLM response.
        """
        # 1. Retrieve
        retrieved_chunks = self.retriever.retrieve(question, k=k)
        
        # 2. Assemble Prompt
        prompt = self.assembler.assemble_prompt(question, retrieved_chunks, template_name=template)
        
        # 3. Inference
        llm_response = self.llm.generate(prompt)
        
        result = {
            'answer': llm_response['response'],
            'sources': retrieved_chunks,
            'tokens_used': llm_response.get('tokens_used', {}),
            'latency_ms': llm_response.get('latency_ms', 0)
        }
        
        if return_context:
            result['full_prompt'] = prompt
            
        return result
    
    def save_memory(self, path: str):
        """Save vector store to disk."""
        self.vector_store.save(path)
    
    def load_memory(self, path: str):
        """Load vector store from disk."""
        self.vector_store.load(path)
    
    def get_statistics(self) -> dict:
        """
        Get system statistics.
        """
        stats = self.vector_store.get_statistics()
        stats.update({
            'embedding_model': self.embedding_generator.model_name,
            'llm_provider': self.llm.provider,
            'llm_model': self.llm.model
        })
        return stats
