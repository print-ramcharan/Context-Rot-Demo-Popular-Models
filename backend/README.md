# LLM Memory System

This project implements an external memory system for Large Language Models to mitigate context degradation.

## Project Structure

- `backend/`: Python-based external memory system using FAISS and Ollama/Gemini.
  - `api.py`: FastAPI entrypoint.
  - `chunker.py`: Document chunking utilities.
  - `vector_store.py`: Vector store helpers (updated with new method).
- `frontend/`: Web interface for interacting with the memory system (updated UI).

## Getting Started

### Backend

```bash
cd backend
# install dependencies per backend/README.md
fastapi dev api.py
```

### Environment Variables

Make sure required API keys are set in your environment (see backend/README.md for exact names).
