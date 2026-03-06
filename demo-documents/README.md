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

The Standard path will burn through the entire 800k-token document trying to find it.
The RAG path retrieves the 3 most relevant paragraphs and answers in seconds.
