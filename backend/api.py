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

# Models for the comparison:
#   Both use 2.5-flash to ensure quota availability and fair comparison.
#   The difference is strictly the context management strategy (RAG vs Naive Full Context).
llm_standard = LLMInference(
    provider="gemini",
    model="models/gemini-2.5-flash",
    config=gemini_cfg
)

llm_rag = LLMInference(
    provider="gemini",
    model="models/gemini-2.5-flash",
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

@app.post("/query/standard")
async def handle_query_standard(request: QueryRequest):
    """
    Standard path (Context Rot) endpoint.
    Retrieves the entire document and injects noise to demonstrate 'Lost-in-the-Middle'.
    """
    try:
        user_query = request.user_query.strip()
        if not user_query:
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing Standard query: {user_query[:100]}...")

        # 1. Get all text chunks
        all_chunks = store.get_all_texts()
        if not all_chunks:
            return {"status": "error", "message": "No document uploaded"}

        # 2. Inject noise (irrelevant filler text) to demonstrate "Context Rot"
        # We interleave real chunks with garbage chunks to simulate background noise.
        NOISE_FILLERS = [
            "LOREM IPSUM: Irrelevant filler text about weather in an unrelated city.",
            "SYSTEM LOG: Error 404 in a coffee machine at sector 7G. Maintenance required.",
            "HISTORICAL FACT: The first computer bug was an actual moth found in a relay.",
            "RANDOM CLIP: 'I think there is a world market for about five computers.' - Thomas Watson.",
            "RECIPE: To make a perfect cup of tea, boil water and steep for 3 minutes."
        ]
        
        noisy_document = []
        for i, chunk in enumerate(all_chunks):
            noisy_document.append(chunk)
            if i % 3 == 0: # Add noise every 3 chunks
                noisy_document.append(f"[SYSTEM NOISE: {random.choice(NOISE_FILLERS)}]")
        
        entire_document = "\n\n".join(noisy_document)

        # 🚨 FREE TIER LIMIT
        _MAX_CHARS_FREE_TIER = 500000
        if len(entire_document) > _MAX_CHARS_FREE_TIER:
            entire_document = entire_document[:_MAX_CHARS_FREE_TIER] + "\n\n...[TRUNCATED]"

        standard_prompt = f"""You are an assistant. Below is a document that may contain irrelevant noise. 
Answer the question below accurately, even if the information is buried in noise.

<DOCUMENT>
{entire_document}
</DOCUMENT>

Question: {user_query}

Answer:"""

        standard_result = llm_standard.generate(standard_prompt, max_tokens=1000, temperature=0.7)
        
        return {
            "status": "success",
            "query": user_query,
            "total_chunks": len(all_chunks),
            "response": {
                "text": standard_result.get('response', ''),
                "model": standard_result.get('model', 'gemini-2.5-flash'),
                "latency_ms": standard_result.get('latency_ms', 0),
                "tokens_used": standard_result.get('tokens_used', {})
            }
        }

    except Exception as e:
        logger.error(f"Standard Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/rag")
async def handle_query_rag(request: QueryRequest):
    """
    RAG path (Optimized) endpoint.
    Retrieves only the top-5 semantically relevant chunks.
    """
    try:
        user_query = request.user_query.strip()
        if not user_query:
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing RAG query: {user_query[:100]}...")

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

        rag_result = llm_rag.generate(rag_prompt, max_tokens=1000, temperature=0.7)
        
        return {
            "status": "success",
            "query": user_query,
            "retrieved_chunks_count": len(retrieved_chunks),
            "response": {
                "text": rag_result.get('response', ''),
                "model": rag_result.get('model', 'gemini-2.5-flash'),
                "latency_ms": rag_result.get('latency_ms', 0),
                "tokens_used": rag_result.get('tokens_used', {}),
                "context_used": context[:500] + "..." if len(context) > 500 else context
            },
            "sources": [chunk.get('text', '')[:100] + "..." for chunk in retrieved_chunks[:3]]
        }

    except Exception as e:
        logger.error(f"RAG Query error: {str(e)}")
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
