from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.inference import LLMInference 
from src.retrieval import SemanticRetriever 
from src.vector_store import VectorStore
from src.embedding import EmbeddingGenerator
from src.chunking import TextChunker
from src.context_assembly import ContextAssembler
from src.reranker import CrossEncoderReranker
from src.utils.metrics import MetricsCollector
import os
import random
import shutil
from pathlib import Path
from dotenv import load_dotenv
import logging
import time
from typing import Optional
from conversation_store import ConversationStore

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
    timings_ms: Optional[dict] = None

class StoreConversationRequest(BaseModel):
    platform: str
    session_id: str
    prompt: str
    response: str


class RetrieveContextRequest(BaseModel):
    query: str
    top_k: int = 5
    platform_filter: Optional[str] = None
    similarity_threshold: float = 0.0

import yaml

def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
llm_cfg = config.get('llm', {})
gemini_cfg = llm_cfg.get('gemini', {})
gemini_with_cache_cfg = {**gemini_cfg, "semantic_cache": llm_cfg.get("semantic_cache", {})}
chunk_cfg = config.get('chunking', {})
embedding_cfg = config.get('embedding', {})
retrieval_cfg = config.get('retrieval', {})
storage_cfg = config.get('storage', {})
context_cfg = config.get('context', {})

metrics = MetricsCollector()

# ── Initialize AI Components ──────────────────────────────────────────────────
conv_store = ConversationStore(storage_dir="conversations")

# Initialize components with optimized performance settings
# ----------------------------------------------------------------------------
gen = EmbeddingGenerator(
    model_name=embedding_cfg.get('model_name', "all-MiniLM-L6-v2"),
    device=embedding_cfg.get('device', 'cpu'),
    cache_path=embedding_cfg.get('cache_path', "cache/embeddings.sqlite"),
    cache_max_items=embedding_cfg.get('cache_max_items', 200000),
    cache_enabled=embedding_cfg.get('cache_enabled', True)
)
store = VectorStore(
    dimension=gen.get_embedding_dimension(),
    index_type=storage_cfg.get('index_type', "cosine")
)
retriever = SemanticRetriever(
    store,
    gen,
    top_k=retrieval_cfg.get('top_k', 10),
    similarity_threshold=retrieval_cfg.get('similarity_threshold', 0.0),
    mode=retrieval_cfg.get('mode', "semantic"),
    dense_k=retrieval_cfg.get('dense_k', 30),
    bm25_k=retrieval_cfg.get('bm25_k', 30),
    alpha=retrieval_cfg.get('alpha', 0.65),
    enable_rerank=retrieval_cfg.get('rerank_enabled', False),
    rerank_top_n=retrieval_cfg.get('rerank_top_n', 6),
    rerank_candidate_pool=retrieval_cfg.get('rerank_candidate_pool', 30),
    query_cache_ttl_s=retrieval_cfg.get('query_cache_ttl_s', 300),
    query_cache_max_items=retrieval_cfg.get('query_cache_max_items', 10000)
)
if retrieval_cfg.get('rerank_enabled', False):
    retriever.set_reranker(
        CrossEncoderReranker(model_name=retrieval_cfg.get('rerank_model', "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    )
chunker = TextChunker(
    chunk_size=chunk_cfg.get('chunk_size', 220),
    overlap=chunk_cfg.get('overlap', 30)
)
assembler = ContextAssembler(
    max_context_length=context_cfg.get('max_context_chars', 4000)
)

# Models for the comparison:
#   Both use 2.5-flash to ensure quota availability and fair comparison.
#   The difference is strictly the context management strategy (RAG vs Naive Full Context).
llm_standard = LLMInference(
    provider="gemini",
    model="models/gemini-2.5-flash",
    config=gemini_with_cache_cfg
)

llm_rag = LLMInference(
    provider="gemini",
    model="models/gemini-2.5-flash",
    config=gemini_with_cache_cfg
)

# Define index storage path
INDEX_PATH = storage_cfg.get('index_path', "memory_index")
AUTO_LOAD_INDEX = storage_cfg.get('auto_load', True)

# Load existing index if it exists
if AUTO_LOAD_INDEX and os.path.exists(INDEX_PATH) and os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
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
            "store_conversation": "POST /store-conversation",
            "retrieve_context": "POST /retrieve-context",
            "list_sessions": "GET /list-sessions",
            "delete_session": "DELETE /delete-session/{session_id}",
            "conv_stats": "GET /conv-stats",
            "stats": "GET /stats",
            "metrics": "GET /metrics"
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
async def upload_file(
    file: UploadFile = File(...),
    clear_existing_data: bool = Query(False, alias="clear_existing")
):
    """
    Upload and ingest a file into the vector store.
    Automatically chunks, embeds, and updates FAISS index in memory.
    """
    try:
        logger.info(f"Received file: {file.filename}")
        
        # 0. Optionally clear previous store data to ensure clean comparison
        if clear_existing_data:
            store.clear()
            retriever.invalidate_cache()
            logger.info("Cleared previous vector store data")
        
        # 1. Save file to disk
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved file to {file_path}")
        
        # 2. Extract text from file
        extract_start = time.perf_counter()
        text_content = _extract_text_from_file(file_path, file.filename)
        extract_ms = (time.perf_counter() - extract_start) * 1000
        if not text_content.strip():
            raise ValueError(f"No text content extracted from {file.filename}")
        logger.info(f"Extracted {len(text_content)} characters")
        
        # 3. Chunk the text
        chunk_start = time.perf_counter()
        chunk_dicts = chunker.chunk_document(text_content)
        chunk_ms = (time.perf_counter() - chunk_start) * 1000
        logger.info(f"Created {len(chunk_dicts)} chunks")
        
        # 4. Generate embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunk_dicts]
        embed_start = time.perf_counter()
        embeddings = gen.embed_batch(chunk_texts, batch_size=32, show_progress=False)
        embed_ms = (time.perf_counter() - embed_start) * 1000
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # 5. Add to vector store (LIVE UPDATE)
        metadata_list = [
            {
                "source": file.filename,
                "chunk_index": i,
                "char_offset": chunk.get('offset', 0),
                "chunk_size": chunk.get('length', len(chunk['text'])),
                "word_count": chunk.get('word_count', len(chunk['text'].split()))
            }
            for i, chunk in enumerate(chunk_dicts)
        ]
        
        index_start = time.perf_counter()
        store.add(embeddings, chunk_texts, metadata=metadata_list)
        retriever.invalidate_cache()
        index_ms = (time.perf_counter() - index_start) * 1000
        logger.info(f"Added {len(chunk_dicts)} chunks to vector store")
        
        # Save to disk
        save_start = time.perf_counter()
        try:
            store.save(INDEX_PATH)
            logger.info("Saved updated memory index to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
        save_ms = (time.perf_counter() - save_start) * 1000
        
        timings = {
            "extract_ms": extract_ms,
            "chunk_ms": chunk_ms,
            "embed_ms": embed_ms,
            "index_ms": index_ms,
            "save_ms": save_ms
        }
        metrics.increment("upload_requests")
        for name, value in timings.items():
            metrics.observe(f"upload_{name}", value)
        return UploadResponse(
            status="success",
            message=f"File '{file.filename}' ingested successfully",
            chunks_created=len(chunk_dicts),
            embeddings_stored=len(embeddings),
            timings_ms=timings
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
        total_start = time.perf_counter()
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
        total_ms = (time.perf_counter() - total_start) * 1000
        metrics.increment("query_standard_requests")
        metrics.observe("query_standard_total_ms", total_ms)
        
        return {
            "status": "success",
            "query": user_query,
            "total_chunks": len(all_chunks),
            "response": {
                "text": standard_result.get('response', ''),
                "model": standard_result.get('model', 'gemini-2.5-flash'),
                "latency_ms": standard_result.get('latency_ms', 0),
                "tokens_used": standard_result.get('tokens_used', {}),
                "cached": standard_result.get('cached', False)
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
        total_start = time.perf_counter()
        user_query = request.user_query.strip()
        if not user_query:
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing RAG query: {user_query[:100]}...")

        # Retrieve top chunks
        retrieve_start = time.perf_counter()
        retrieved_chunks = retriever.retrieve(user_query, k=retrieval_cfg.get('top_k', 10))
        retrieve_ms = (time.perf_counter() - retrieve_start) * 1000

        compact_start = time.perf_counter()
        if context_cfg.get('compact', True):
            compact_chunks = assembler.compress_context(
                user_query,
                retrieved_chunks,
                max_context_chars=context_cfg.get('max_context_chars', 2600),
                max_sentences=context_cfg.get('max_sentences', 2),
                max_chunks_after_compaction=context_cfg.get('max_chunks', 4)
            )
        else:
            compact_chunks = retrieved_chunks
        compact_ms = (time.perf_counter() - compact_start) * 1000

        context = (
            "\n\n".join([chunk['text'] for chunk in compact_chunks])
            if compact_chunks else "(No context retrieved)"
        )

        rag_prompt = f"""Use ONLY the following retrieved context to answer the question precisely.

<context>
{context}
</context>

Question: {user_query}

Answer:"""

        llm_start = time.perf_counter()
        rag_result = llm_rag.generate(rag_prompt, max_tokens=1000, temperature=0.7)
        llm_ms = (time.perf_counter() - llm_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        prompt_chars = len(rag_prompt)
        timings = {
            "retrieve_ms": retrieve_ms,
            "compact_ms": compact_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
            "prompt_chars": prompt_chars
        }
        metrics.increment("query_rag_requests")
        for name, value in timings.items():
            if name.endswith("_ms"):
                metrics.observe(f"query_rag_{name}", value)
        
        return {
            "status": "success",
            "query": user_query,
            "retrieved_chunks_count": len(retrieved_chunks),
            "compact_chunks_count": len(compact_chunks),
            "response": {
                "text": rag_result.get('response', ''),
                "model": rag_result.get('model', 'gemini-2.5-flash'),
                "latency_ms": rag_result.get('latency_ms', 0),
                "tokens_used": rag_result.get('tokens_used', {}),
                "cached": rag_result.get('cached', False),
                "context_used": context[:500] + "..." if len(context) > 500 else context
            },
            "sources": [chunk.get('text', '')[:100] + "..." for chunk in retrieved_chunks[:3]],
            "timings_ms": timings
        }

    except Exception as e:
        logger.error(f"RAG Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_store():
    """Clear the vector store memory."""
    try:
        store.clear()
        retriever.invalidate_cache()
        logger.info("Vector store manually cleared")
        return {"status": "success", "message": "Memory cleared"}
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
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

@app.get("/metrics")
def get_metrics():
    """Get in-memory performance metrics."""
    return metrics.snapshot()

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


 
@app.post("/store-conversation")
async def store_conversation(request: StoreConversationRequest):
    try:
        if not request.prompt.strip() and not request.response.strip():
            raise ValueError("Both prompt and response are empty")
 
        result = conv_store.store_conversation(
            platform=request.platform,
            session_id=request.session_id,
            prompt=request.prompt,
            response=request.response,
            embedding_generator=gen,
        )
 
        logger.info(
            f"Stored conversation: platform={request.platform} "
            f"session={request.session_id} chunks={result['chunk_count']}"
        )
        return {"status": "success", **result}
 
    except Exception as e:
        logger.error(f"store-conversation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
 
 
@app.post("/retrieve-context")
async def retrieve_context(request: RetrieveContextRequest):
    try:
        if not request.query.strip():
            raise ValueError("Query cannot be empty")
 
        chunks = conv_store.retrieve_context(
            query=request.query,
            embedding_generator=gen,
            top_k=request.top_k,
            platform_filter=request.platform_filter,
            similarity_threshold=request.similarity_threshold,
        )
 
        return {
            "status": "success",
            "query": request.query,
            "results": chunks,
            "count": len(chunks),
        }
 
    except Exception as e:
        logger.error(f"retrieve-context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/list-sessions")
def list_sessions():
    return {
        "status": "success",
        "sessions": conv_store.list_sessions(),
        "total": len(conv_store.list_sessions()),
    }
 
 
@app.delete("/delete-session/{session_id}")
def delete_session(session_id: str):
    try:
        result = conv_store.delete_session(session_id)
 
        if not result["deleted"]:
            raise HTTPException(
                status_code=404,
                detail=result.get("reason", "session not found")
            )
 
        logger.info(f"Deleted session: {session_id}")
        return {"status": "success", **result}
 
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete-session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/conv-stats")
def conv_stats():
    return {"status": "success", **conv_store.get_stats()}
 
 
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
 
def _extract_text_from_file(file_path: Path, filename: str) -> str:
    try:
        if filename.endswith('.txt') or filename.endswith('.md'):
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
                logger.warning("PyPDF2 not installed. pip install PyPDF2")
                return ""
 
        elif filename.endswith('.docx'):
            try:
                from docx import Document
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                logger.warning("python-docx not installed. pip install python-docx")
                return ""
 
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
 
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        return ""
