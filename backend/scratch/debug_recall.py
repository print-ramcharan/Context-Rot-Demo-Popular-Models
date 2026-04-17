import sys
import yaml
import os
from src.embedding import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retrieval import SemanticRetriever

def debug():
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
        top_k=20, # Higher for debugging
        alpha=config['retrieval']['alpha']
    )
    
    results = retriever.retrieve(query)
    print(f"Retrieved {len(results)} chunks.")
    for i, res in enumerate(results):
        print(f"\n--- Chunk {i+1} (Score: {res['score']:.4f}) ---")
        print(res['text'])

if __name__ == "__main__":
    debug()
