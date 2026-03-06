from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.inference import LLMInference 
from src.retrieval import SemanticRetriever 
from src.vector_store import VectorStore
from src.embedding import EmbeddingGenerator
from src.chunker import TextChunker  # You'll need to create this
import os
import random
import shutil
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="Context Rot Demo API",
    description="API for testing LLM context degradation",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    user_query: str

class UploadResponse(BaseModel):
    status: str
    message: str
    chunks_created: int = 0
    embeddings_stored: int = 0

import yaml

def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
llm_cfg = config.get('llm', {})
gemini_cfg = llm_cfg.get('gemini', {})

# ── Initialize AI Components ──────────────────────────────────────────────────
store = VectorStore(dimension=384)
gen = EmbeddingGenerator()

# Single model used for BOTH paths — the only variable is the context strategy:
#   Standard: dumps the entire document (naive full-context stuffing)
#   RAG:      retrieves only the 3 most relevant chunks via semantic search
# This is a pure, honest comparison of retrieval strategy vs brute-force context.
llm = LLMInference(
    provider="gemini",
    model="gemini-2.5-flash",
    config=gemini_cfg
)

retriever = SemanticRetriever(store, gen, top_k=5)
chunker = TextChunker(chunk_size=1024, overlap=150)

# Load existing index if it exists
INDEX_PATH = "memory_index"
if os.path.exists(INDEX_PATH) and os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
    try:
        store.load(INDEX_PATH)
        logger.info(f"Loaded existing memory index from {INDEX_PATH} with {len(store.chunks)} chunks")
    except Exception as e:
        logger.error(f"Failed to load existing index: {e}")

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# ROOT & HEALTH ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Root endpoint with available endpoints."""
    return {
        "message": "Context Rot Demo API is running! 🚀",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "upload": "POST /upload",
            "query": "POST /query",
            "stats": "GET /stats"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store": {
            "total_chunks": store.index.ntotal if store.index else 0
        }
    }

# ============================================================================
# UPLOAD ENDPOINT - Automated Ingestion with Live FAISS Update
# ============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and ingest a file into the vector store.
    Automatically chunks, embeds, and updates FAISS index in memory.
    """
    try:
        logger.info(f"Received file: {file.filename}")
        
        # 1. Save file to disk
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved file to {file_path}")
        
        # 2. Extract text from file
        text_content = _extract_text_from_file(file_path, file.filename)
        if not text_content.strip():
            raise ValueError(f"No text content extracted from {file.filename}")
        logger.info(f"Extracted {len(text_content)} characters")
        
        # 3. Chunk the text
        chunks = chunker.chunk(text_content)
        logger.info(f"Created {len(chunks)} chunks")
        
        # 4. Generate embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = gen.embed_batch(chunk_texts, batch_size=32, show_progress=False)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # 5. Add to vector store (LIVE UPDATE)
        metadata_list = [
            {
                "source": file.filename,
                "chunk_index": i,
                "char_offset": chunk.get('offset', 0),
                "chunk_size": len(chunk['text'])
            }
            for i, chunk in enumerate(chunks)
        ]
        
        store.add(embeddings, chunk_texts, metadata=metadata_list)
        logger.info(f"Added {len(chunks)} chunks to vector store")
        
        # Save to disk
        try:
            store.save(INDEX_PATH)
            logger.info("Saved updated memory index to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
        
        return UploadResponse(
            status="success",
            message=f"File '{file.filename}' ingested successfully",
            chunks_created=len(chunks),
            embeddings_stored=len(embeddings)
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error uploading file: {str(e)}"
        )

# ============================================================================
# QUERY ENDPOINT - Dual Path (Standard vs RAG)
# ============================================================================

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Dual-path query — honest Context Rot demonstration.

    Both paths use the SAME model (gemini-2.5-flash).
    The ONLY variable is how much context is given:

      Standard (Context Rot):
        - Entire document dumped into the prompt verbatim
        - Tokens: 50k–500k+, latency 5–30s, cost ~$0.01+
        - With large docs: model gets 'lost in the middle'

      RAG (Optimized):
        - Only top-3 semantically retrieved chunks sent
        - Tokens: ~600, latency ~1s, cost ~$0.00005
        - Always precise, regardless of document size
    """
    try:
        user_query = request.user_query.strip()
        if not user_query:
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing query: {user_query[:100]}...")

        # ── PATH A: STANDARD (Context Rot) ────────────────────────────────────
        # Naive approach: concatenate every chunk and send the whole document.
        all_chunks = store.get_all_texts()
        entire_document = "\n\n".join(all_chunks) if all_chunks else "(No document uploaded)"

        # 🚨 FREE TIER LIMIT: Gemini free tier has a 250,000 token-per-minute limit.
        # War & Peace is ~800,000 tokens, which instantly triggers a 429 Quota Exceeded error.
        # We cap the naive document dump to ~150,000 tokens (approx 600,000 characters)
        # to ensure it successfully runs and demonstrates the performance difference.
        _MAX_CHARS_FREE_TIER = 600000
        if len(entire_document) > _MAX_CHARS_FREE_TIER:
            entire_document = entire_document[:_MAX_CHARS_FREE_TIER] + "\n\n...[DOCUMENT TRUNCATED DUE TO 250K API TOKEN LIMIT]"

        standard_prompt = f"""You are given the complete text of a document. Read it and answer the question below.

{entire_document}

Question: {user_query}

Answer:"""

        standard_result = llm.generate(standard_prompt, max_tokens=2000, temperature=0.7)
        logger.info(
            f"Standard: {len(all_chunks)} chunks, "
            f"{len(entire_document)} chars, model=gemini-2.5-flash"
        )

        # ── PATH B: RAG (Optimized) ────────────────────────────────────────────
        # Retrieve only the 5 most semantically relevant chunks.
        retrieved_chunks = retriever.retrieve(user_query, k=5)
        context = (
            "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
            if retrieved_chunks else "(No context retrieved)"
        )

        rag_prompt = f"""Use ONLY the following retrieved context to answer the question precisely.

<context>
{context}
</context>

Question: {user_query}

Answer:"""

        rag_result = llm.generate(rag_prompt, max_tokens=2000, temperature=0.7)
        logger.info(f"RAG: {len(retrieved_chunks)} chunks retrieved, model=gemini-2.5-flash")

        return {
            "status": "success",
            "query": user_query,
            "total_chunks": len(all_chunks),
            "retrieved_chunks_count": len(retrieved_chunks),
            "responses": {
                "standard": {
                    "text": standard_result.get('response', ''),
                    "model": standard_result.get('model', 'gemini-2.5-flash'),
                    "latency_ms": standard_result.get('latency_ms', 0),
                    "tokens_used": standard_result.get('tokens_used', {})
                },
                "rag": {
                    "text": rag_result.get('response', ''),
                    "model": rag_result.get('model', 'gemini-2.5-flash'),
                    "latency_ms": rag_result.get('latency_ms', 0),
                    "tokens_used": rag_result.get('tokens_used', {}),
                    "context_used": context[:500] + "..." if len(context) > 500 else context
                }
            },
            "sources": [chunk.get('text', '')[:100] + "..." for chunk in retrieved_chunks[:3]]
        }

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ============================================================================
# STATS ENDPOINT
# ============================================================================

@app.get("/stats")
def get_stats():
    """Get system statistics."""
    return {
        "status": "success",
        "vector_store": {
            "total_chunks": store.index.ntotal if store.index else 0,
            "embedding_dimension": 384,
            "index_type": store.index_type
        },
        "files_ingested": len(list(DATA_DIR.glob("*"))),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_provider": "gemini"
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_text_from_file(file_path: Path, filename: str) -> str:
    """Extract text from various file formats."""
    try:
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif filename.endswith('.pdf'):
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text
            except ImportError:
                logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
                return ""
        
        elif filename.endswith('.docx'):
            try:
                from docx import Document
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                logger.warning("python-docx not installed. Install with: pip install python-docx")
                return ""
        
        elif filename.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            # Default: treat as plain text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        return ""
