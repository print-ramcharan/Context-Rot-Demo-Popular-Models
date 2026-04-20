from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
import shutil
import json
import time
import logging
import yaml
from pathlib import Path
from typing import Optional, List
from conversation_store import ConversationStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv() # Trigger reload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="Context Rot Lab API",
    description="Precision RAG Engine API",
    version="2.0.0"
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

# ── Configuration ─────────────────────────────────────────────────────────────

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
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── AI Components Initialization ──────────────────────────────────────────────

conv_store = ConversationStore(storage_dir="conversations")

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
    mode=retrieval_cfg.get('mode', "hybrid"),
    dense_k=retrieval_cfg.get('dense_k', 30),
    bm25_k=retrieval_cfg.get('bm25_k', 30),
    alpha=retrieval_cfg.get('alpha', 0.65),
    enable_rerank=retrieval_cfg.get('rerank_enabled', False),
)

# Optional Reranker (Cross-Encoder)
if retrieval_cfg.get('rerank_enabled', False):
    try:
        reranker = CrossEncoderReranker(
            model_name=retrieval_cfg.get('rerank_model', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        retriever.set_reranker(reranker)
        logger.info("Reranker initialized and attached to retriever")
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}")


chunker = TextChunker(
    chunk_size=chunk_cfg.get('chunk_size', 400),
    overlap=chunk_cfg.get('overlap', 80)
)

assembler = ContextAssembler(
    max_context_length=context_cfg.get('max_context_chars', 6000)
)

# Use flash-lite for demo: weaker with large context, generous free-tier RPM
STABLE_MODEL = "models/gemini-2.5-flash-lite"

llm_standard = LLMInference(
    provider="gemini",
    model=STABLE_MODEL,
    config=gemini_with_cache_cfg
)

llm_rag = LLMInference(
    provider="gemini",
    model=STABLE_MODEL,
    config=gemini_with_cache_cfg
)

INDEX_PATH = storage_cfg.get('index_path', "memory_index")

# Initial index load
if storage_cfg.get('auto_load', True) and os.path.exists(INDEX_PATH):
    try:
        store.load(INDEX_PATH)
        logger.info(f"Loaded existing index from {INDEX_PATH}")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")

# ── Prompt Templates ──────────────────────────────────────────────────────────

def format_standard_prompt(query: str, context: str) -> str:
    return f"""You are a precise assistant. Below is the entire document content. 
Answer the user's question accurately based ON THIS DOCUMENT. Provide a detailed and complete answer. 
If the information is not present, say so.

<DOCUMENT>
{context}
</DOCUMENT>

Question: {query}
Answer:"""

def format_rag_prompt(query: str, context: str) -> str:
    return f"""Answer the question accurately and comprehensively using ONLY the provided context. Ensure you include all relevant details found in the context (like keys, labels, or objects). If the answer is not in the context, say you don't know.

<CONTEXT>
{context}
</CONTEXT>

Question: {query}
Answer:"""

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    clear_existing_data: bool = Query(False, alias="clear_existing")
):
    try:
        if clear_existing_data:
            store.clear()
            retriever.invalidate_cache()
            logger.info("Cleared previous store data")
        
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        extract_start = time.perf_counter()
        text_content = _extract_text_from_file(file_path, file.filename)
        if not text_content.strip():
            raise ValueError("No text content extracted")
        
        chunk_dicts = chunker.chunk_document(text_content)
        chunk_texts = [chunk['text'] for chunk in chunk_dicts]
        
        embeddings = gen.embed_batch(chunk_texts, batch_size=32)
        
        metadata_list = [
            {"source": file.filename, "chunk_index": i}
            for i in range(len(chunk_dicts))
        ]
        
        store.add(embeddings, chunk_texts, metadata=metadata_list)
        store.save(INDEX_PATH)
        
        # Pre-build lexical index for immediate query readiness
        retriever.build_lexical_index()
        
        # Warm up embedding model for headstart
        gen.embed_text("headstart warmup")
        
        logger.info(f"Pre-built BM25 index and warmed up embedding model for {len(chunk_dicts)} chunks")
        
        latency = (time.perf_counter() - extract_start) * 1000
        
        return UploadResponse(
            status="success",
            message=f"File '{file.filename}' ingested successfully",
            chunks_created=len(chunk_dicts),
            embeddings_stored=len(embeddings),
            timings_ms={"total_ms": latency}
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/standard")
async def handle_query_standard(request: QueryRequest):
    async def stream_standard():
        try:
            total_start = time.perf_counter()
            user_query = request.user_query.strip()
            
            # We use genuine latency here to provide an honest comparison
            
            yield json.dumps({"type": "metadata", "model": llm_standard.model}) + "\n"

            full_context = store.get_all_texts()
            prompt = format_standard_prompt(user_query, full_context)
            
            async for chunk in llm_standard.stream_generate(prompt):
                if "error" in chunk:
                    yield json.dumps({"type": "error", "detail": chunk["error"]}) + "\n"
                    break
                
                # Yield text if present (even in final chunks)
                if chunk.get("text"):
                    yield json.dumps({"type": "text", "text": chunk["text"]}) + "\n"

                if chunk.get("done"):
                    latency = (time.perf_counter() - total_start) * 1000
                    yield json.dumps({"type": "final", "tokens": chunk.get("tokens"), "latency_ms": latency}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(stream_standard(), media_type="application/x-ndjson")

@app.post("/query/rag")
async def handle_query_rag(request: QueryRequest):
    async def stream_rag():
        try:
            total_start = time.perf_counter()
            user_query = request.user_query.strip()
            
            retrieve_start = time.perf_counter()
            retrieved_chunks = retriever.retrieve(
                user_query, 
                k=retrieval_cfg.get('top_k', 10)
            )
            retrieve_time = (time.perf_counter() - retrieve_start) * 1000
            
            # Deduplicate chunks to prevent redundant context
            retrieved_chunks = retriever.deduplicate_chunks(retrieved_chunks, overlap_threshold=0.7)

            compact_chunks = assembler.compress_context(
                user_query, 
                retrieved_chunks,
                max_context_chars=context_cfg.get('max_context_chars', 6000),
                max_sentences=context_cfg.get('max_sentences', 8),
                max_chunks_after_compaction=context_cfg.get('max_chunks', 10)
            )
            context = "\n\n".join([c.get('text', '') for c in compact_chunks])
            
            yield json.dumps({
                "type": "metadata",
                "model": llm_rag.model,
                "retrieval_ms": retrieve_time,
                "chunks_count": len(retrieved_chunks),
                "context": context
            }) + "\n"

            prompt = format_rag_prompt(user_query, context)
            async for chunk in llm_rag.stream_generate(prompt):
                if "error" in chunk:
                    yield json.dumps({"type": "error", "detail": chunk["error"]}) + "\n"
                    break
                
                # Yield text if present (even in final chunks)
                if chunk.get("text"):
                    yield json.dumps({"type": "text", "text": chunk["text"]}) + "\n"

                if chunk.get("done"):
                    latency = (time.perf_counter() - total_start) * 1000
                    yield json.dumps({"type": "final", "tokens": chunk.get("tokens"), "latency_ms": latency}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(stream_rag(), media_type="application/x-ndjson")

@app.delete("/clear")
async def clear_store():
    store.clear()
    retriever.invalidate_cache()
    return {"status": "success"}

# ── Extra Backend Features ───────────────────────────────────────────────────

@app.post("/store-conversation")
async def store_conversation(request: StoreConversationRequest):
    try:
        result = conv_store.store_conversation(
            platform=request.platform,
            session_id=request.session_id,
            prompt=request.prompt,
            response=request.response,
            embedding_generator=gen,
        )
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrieve-context")
async def retrieve_context(request: RetrieveContextRequest):
    try:
        chunks = conv_store.retrieve_context(
            query=request.query,
            embedding_generator=gen,
            top_k=request.top_k,
            platform_filter=request.platform_filter,
            similarity_threshold=request.similarity_threshold,
        )
        return {"status": "success", "results": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Helper Functions ──────────────────────────────────────────────────────────

def _extract_text_from_file(file_path: Path, filename: str) -> str:
    try:
        if filename.endswith('.txt') or filename.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif filename.endswith('.pdf'):
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        elif filename.endswith('.docx'):
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        return ""
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return ""
