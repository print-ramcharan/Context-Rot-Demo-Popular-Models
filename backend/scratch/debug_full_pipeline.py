import sys
import yaml
import os
import json
from src.embedding import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retrieval import SemanticRetriever
from src.context_assembly import ContextAssembler

def debug_full_prompt():
    query = "What does Alice find on the table in the hall?"
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    gen = EmbeddingGenerator(
        model_name=config['embedding']['model_name']
    )
    
    store = VectorStore(
        dimension=gen.get_embedding_dimension(),
        index_type=config['storage']['index_type']
    )
    store.load(config['storage']['index_path'])
    
    retriever = SemanticRetriever(
        store,
        gen,
        top_k=15,
        alpha=0.4 # Use the tuned alpha
    )
    
    assembler = ContextAssembler(
        max_context_length=6000
    )
    
    results = retriever.retrieve(query)
    compact_chunks = assembler.compress_context(
        query, 
        results,
        max_context_chars=6000,
        max_sentences=8,
        max_chunks_after_compaction=6
    )
    
    context = "\n\n".join([c.get('text', '') for c in compact_chunks])
    
    print("--- ASSEMBLED CONTEXT ---")
    print(context)
    print("--- END CONTEXT ---")
    
    if "golden key" in context.lower():
        print("\nSUCCESS: 'golden key' found in context!")
    else:
        print("\nFAILURE: 'golden key' NOT found in context.")

if __name__ == "__main__":
    debug_full_prompt()
