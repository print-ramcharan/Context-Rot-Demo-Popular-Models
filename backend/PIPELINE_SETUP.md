# Evidence-grounded RAG implementation

This branch upgrades the active document upload and `/query/rag` paths. It does not run or add test cases.

## Backend setup

From the repository root:

```bash
git fetch origin
git checkout implementation/evidence-grounded-rag
cd backend

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set your OpenRouter key in `backend/.env`:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Start the API from the backend directory:

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

## Existing document index migration

The default embedding model changes from MiniLM (384 dimensions) to BGE-base (768 dimensions). An old FAISS index cannot be queried with vectors from a different embedding model.

Previously uploaded source files remain in `backend/data`. Once the API starts with the new dependencies and model, rebuild the index from those source files:

```bash
curl -X POST http://127.0.0.1:8000/reindex
```

This rebuild uses a temporary in-memory index and only replaces the active index after at least one document has been indexed successfully. If no source files exist, upload the documents again through the application.

The first Docling conversion may download its layout/OCR model artifacts. Later conversions reuse the local cache.

## Ask questions

Use the existing application UI and its RAG query mode. The endpoint remains:

```http
POST /query/rag
Content-Type: application/json

{"user_query":"Your question about the uploaded documents"}
```

The response remains newline-delimited JSON for the existing UI:
- `metadata`: selected context and retrieval timing
- `text`: final answer or explicit abstention
- `final`: verification state, source citations and latency

The RAG answer path now performs structured claim generation followed by a separate evidence-verification pass. It fails closed to:

`I don't know based on the provided documents.`

when retrieval returns no evidence, the model output is malformed, citations are invalid, or the verifier does not affirm support.

## Model configuration

Default practical local embedding model:

`BAAI/bge-base-en-v1.5`

Default lightweight reranker:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

For a higher-memory machine, Qwen models can be selected in `config.yaml`:
- Embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Reranking: `Qwen/Qwen3-Reranker-0.6B`

When switching to Qwen embeddings, set `query_instruction: ""` and `query_prompt_name: "query"`. Any embedding-model change requires `POST /reindex`.

## Important limitation

A separate verification pass reduces unsupported answers but cannot mathematically guarantee correctness. The verifier is an LLM-based judge, and retrieval or extraction can still miss source evidence. The application therefore abstains on uncertain or malformed verification outcomes rather than claiming certainty.
