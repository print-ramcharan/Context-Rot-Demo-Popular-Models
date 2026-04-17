import json
from src.vector_store import VectorStore
store = VectorStore(dimension=384, index_type="L2")
store.load("memory_index")

chunks = store.chunks
from collections import Counter
c = Counter(chunks)
dups = {k:v for k,v in c.items() if v > 1}
print(f"Total chunks: {len(chunks)}")
print(f"Number of unique chunks with duplicates: {len(dups)}")
for k, v in list(dups.items())[:3]:
    print(f"Chunk starting with: {k[:50]}... appears {v} times")
