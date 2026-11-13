# Context Rot Demo Presentation Script

## 1. Setup

- Start the backend and frontend.
- Upload `alice-in-wonderland.txt` with **Clear existing data** enabled.
- Run the same question through **Standard** and **RAG**.
- Point out answer quality, response time, and the amount of context sent.

## 2. Fast Baseline Questions

| Question | Expected answer | What it demonstrates |
| --- | --- | --- |
| What game does the Queen of Hearts play? | Croquet | A precise fact retrieval check |
| What does the Caterpillar sit on and what is he smoking? | A mushroom and a hookah | Multiple facts in one passage |
| Who steals the tarts? | The Knave of Hearts | Retrieval from a later chapter |
| What stays behind when the Cheshire Cat disappears? | His grin | A distinctive phrase and entity |
| What does Alice find on the table in the hall? | A tiny golden key and a bottle marked DRINK ME | Two nearby objects |

## 3. Context-Rot Moment

Use the longest available document or the standard path and ask:

> Who does Alice meet after she falls down the rabbit hole, and what does she notice first?

Then ask the same question with RAG. Explain that the standard path pays for the
whole document even when the answer lives in one small passage, while RAG retrieves
only the relevant chunks.

## 4. Cross-Document Retrieval

Upload `rfc-8446.txt` **without** clearing the existing index.

| Question | Expected answer | What to point out |
| --- | --- | --- |
| What is the main purpose of TLS 1.3? | To prevent eavesdropping, tampering, and message forgery | The technical document is selected instead of Alice |
| Which protocol versions does TLS 1.3 replace? | TLS 1.2 and earlier versions | Exact keyword and version retrieval |
| What game does the Queen of Hearts play? | Croquet | The earlier Alice memory remains available |

## 5. Grounding and Abstention

Ask a question whose answer is not in either document:

> What is the capital of Australia?

Expected behavior: the assistant should say that the information is not present in
the provided context. If it answers confidently, use that as a brief discussion of
why retrieval grounding and an explicit "I don't know" policy matter.

## 6. Closing Questions

- How many chunks were retrieved for each answer?
- Which answer would become expensive if the source document were 800,000 tokens?
- What happens when the same fact is phrased with different words?
- What failure mode do we see when a precise identifier is present but dense retrieval misses it?
- Which improvements would you add next: citations, reranking, hybrid BM25 search, or caching?