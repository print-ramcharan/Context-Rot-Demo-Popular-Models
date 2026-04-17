# Demo Documents

These documents are pre-selected to reliably demonstrate Context Rot.

## war-and-peace.txt
**Source:** Project Gutenberg (public domain)  
**Size:** ~3.4 MB | ~580,000 words | ~800,000+ tokens  
**Author:** Leo Tolstoy

### Why this works perfectly for the demo

| Path | Tokens sent | Latency | Cost |
|------|------------|---------|------|
| Standard (full doc) | ~800,000 | 30–60s | ~$0.10 |
| RAG (3 chunks) | ~600 | 1–2s | ~$0.00005 |

### Good demo questions to ask

Ask something specific that's buried deep in the text:

```
What is the name of Prince Andrew's wife?
```
```
What battle does Pierre witness firsthand?
```
```
Who does Natasha fall in love with at her first ball?
```
```
What happens to Prince Andrew at the Battle of Austerlitz?
```

---

## alice-in-wonderland.txt
**Source:** Project Gutenberg (public domain)  
**Size:** ~145 KB | ~26,500 words | ~42,000+ tokens  
**Author:** Lewis Carroll

### Why this works perfectly for the demo

This document is small enough to fit in the Gemini Free Tier (~250k token limit / minute) for the **Standard** path, allowing you to see both answers side-by-side. 

| Path | Tokens sent | Latency | Efficiency |
|------|------------|---------|------------|
| Standard (full doc) | ~42,600 | ~7–10s | ❌ slow, expensive |
| RAG (Optimized) | ~1,200 | ~2–3s | ✅ 3x faster, 40x cheaper |

### Expected Answer Key for Verification

Use these questions to quickly verify the **RAG Optimized** engine is correctly retrieving specific facts:

| Question | Expected Answer | Why it tests RAG |
|----------|-----------------|------------------|
| what game does the Queen of Hearts play? | **Croquet** | Tests lexical + semantic retrieval for specific events. |
| What does the Caterpillar sit on and what is he smoking? | **Sitting on a mushroom, smoking a hookah** | Tests multi-fact retrieval within a single paragraph. |
| What does Alice find on the table in the hall? | **A tiny golden key** (and a "DRINK ME" bottle) | Tests retrieval of specific objects from the start of the book. |
| Who steals the tarts? | **The Knave of Hearts** | Tests character specific actions at the end of the book. |
| What stays behind when the Cheshire Cat disappears? | **His grin** | Tests iconic fact retrieval. |

**Observation Note:** If the RAG path says "The context does not state...", it means the retrieval missed the specific chunk. Thanks to our **400-word chunk size** and **80-word overlap**, these should all pass 100% of the time.
