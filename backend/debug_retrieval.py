from main import ExternalMemorySystem
import json

system = ExternalMemorySystem()
query = "Tell me about the AMS subsystem"
results = system.retriever.vector_store.search(system.retriever.embedding_generator.embed_text(query), k=5)

print(f"Query: {query}")
print("Raw search results (scores/distances):")
for i in range(len(results['chunks'])):
    print(f"--- Result {i} (Score: {results['distances'][i]}) ---")
    print(results['chunks'][i])
    print("-" * 20)

retrieved = system.retriever.retrieve(query)
print(f"\nRetrieved with threshold {system.retriever.similarity_threshold}: {len(retrieved)} chunks")

if len(retrieved) == 0:
    print("\n[ACTION] Clearing memory and re-ingesting document once.")
    system.vector_store.clear()
    system.ingest_file("data/mars_technical_spec.md")
    print("Re-checking with raw search...")
    new_raw = system.vector_store.search(system.retriever.embedding_generator.embed_text(query), k=5)
    for i in range(len(new_raw['chunks'])):
        print(f"--- New Result {i} (Score: {new_raw['distances'][i]}) ---")
        print(new_raw['chunks'][i][:100] + "...")
    
    new_retrieved = system.retriever.retrieve(question=query) # Use question=query if that's the arg name
    print(f"Retrieved after re-ingest: {len(new_retrieved)} chunks")
    for i, r in enumerate(new_retrieved):
        print(f"Chunk {i} Score: {r['score']}")
