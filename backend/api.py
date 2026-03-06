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
store = VectorStore(dimension=384)  # Dimension for MiniLM
gen = EmbeddingGenerator()

# Standard path: weaker model (gemini-1.5-flash-8b) deliberately chosen to
# show context rot — it has limited reasoning capacity over noisy long contexts.
standard_llm = LLMInference(
    provider="gemini",
    model="gemini-1.5-flash-8b",
    config=gemini_cfg
)

# RAG path: stronger model (gemini-2.5-flash) with clean, retrieved context.
rag_llm = LLMInference(
    provider="gemini",
    model="gemini-2.5-flash",
    config=gemini_cfg
)

retriever = SemanticRetriever(store, gen, top_k=3)
chunker = TextChunker(chunk_size=512, overlap=50)

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
# QUERY ENDPOINT - Dual Path (RAG + Standard)
# ============================================================================

# ── Context Rot: Noise Injection Helper ───────────────────────────────────────

# Irrelevant filler paragraphs injected into the middle of the Standard path's
# context. This simulates the "Lost in the Middle" effect documented in
# Liu et al. (2023), causing the weaker model to lose track of the answer.
_NOISE_PARAGRAPHS = [
    "The annual migration of monarch butterflies spans thousands of miles across North America, following routes passed down through generations encoded in their genetic memory.",
    "In classical thermodynamics, entropy is a measure of the number of microscopic configurations that correspond to a thermodynamic system's macroscopic state.",
    "The construction of the Great Wall of China took place over many centuries, involving millions of workers during various dynasties, primarily as a military fortification.",
    "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose through chlorophyll.",
    "The Silk Road was an ancient network of trade routes connecting the East and West, facilitating the exchange of goods, culture, and knowledge for centuries.",
    "In computer science, a binary search tree is a rooted binary tree in which any node's value is greater than all values in its left subtree and less than all in its right.",
    "Marine biologists have discovered that dolphins use a sophisticated system of clicks and whistles to communicate, with some researchers believing they possess individual names.",
    "The Renaissance period marked a profound cultural and intellectual transformation in Europe, with artists and scholars drawing inspiration from classical Greek and Roman ideals.",
    "Quantum entanglement occurs when two particles become correlated such that the quantum state of each particle cannot be described independently of the other, even when separated.",
    "The Amazon rainforest produces approximately 20% of the world's oxygen and is home to an estimated 10% of all species living on Earth, making it critical to global biodiversity.",
    "During the Industrial Revolution, the invention of the steam engine transformed manufacturing, transportation, and agriculture, fundamentally reshaping the global economy.",
    "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections, creating a network more complex than the entire known universe.",
    "Plate tectonics theory explains how Earth's lithosphere is divided into large rigid plates that move relative to each other, causing earthquakes, volcanoes, and mountain formation.",
    "Jazz music emerged in the early 20th century in New Orleans, blending African-American musical traditions with European harmonic structures to create a uniquely American art form.",
    "The speed of light in a vacuum is exactly 299,792,458 metres per second, serving as a fundamental constant in physics and the universe's ultimate speed limit.",
]

def _inject_noise_into_context(chunks: list[str], noise_ratio: float = 0.6) -> str:
    """
    Builds a noisy context by interleaving real document chunks with irrelevant
    noise paragraphs, then shuffling. This simulates context rot — the model
    must find the answer buried in irrelevant text, degrading its performance.

    noise_ratio: proportion of noise paragraphs relative to real chunks.
    """
    if not chunks:
        return "(No document uploaded)"

    n_noise = max(int(len(chunks) * noise_ratio), 8)
    noise_sample = ((_NOISE_PARAGRAPHS * ((n_noise // len(_NOISE_PARAGRAPHS)) + 1))[:n_noise])

    # Interleave: place noise between real chunks and also surround the answer area
    combined = []
    for i, chunk in enumerate(chunks):
        # Inject 1–2 noise paragraphs before each real chunk
        n_inject = random.randint(1, 2)
        combined.extend(random.sample(noise_sample, min(n_inject, len(noise_sample))))
        combined.append(chunk)

    random.shuffle(noise_sample)  # Add trailing noise
    combined.extend(noise_sample[:4])

    return "\n\n".join(combined)


# ── Query Endpoint ─────────────────────────────────────────────────────────────

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Dual-path query for Context Rot demonstration.

    Standard path (Context Rot):
      - Uses gemini-1.5-flash-8b (weaker, lower reasoning capacity)
      - Receives ALL document chunks PLUS injected noise paragraphs
      - Simulates 'Lost in the Middle': the model struggles to find the
        answer buried in irrelevant text

    RAG path (Optimized):
      - Uses gemini-2.5-flash (stronger model)
      - Receives ONLY the 3 most semantically relevant chunks
      - Clean, precise context → accurate, fast answer
    """
    try:
        user_query = request.user_query.strip()
        if not user_query:
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing query: {user_query[:100]}...")

        # ── PATH A: STANDARD (Context Rot) ────────────────────────────────────
        all_chunks = store.get_all_texts()

        # Build a noisy context: real chunks interleaved with irrelevant filler
        noisy_context = _inject_noise_into_context(all_chunks, noise_ratio=0.8)

        standard_prompt = f"""You are given a large document. Read it carefully and answer the question.

--- DOCUMENT START ---
{noisy_context}
--- DOCUMENT END ---

Question: {user_query}

Answer based only on the document above:"""

        standard_result = standard_llm.generate(standard_prompt, max_tokens=500, temperature=0.7)
        logger.info(
            f"Standard: {len(all_chunks)} real chunks + noise, "
            f"{len(noisy_context)} chars, model=gemini-1.5-flash-8b"
        )

        # ── PATH B: RAG (Optimized) ───────────────────────────────────────────
        retrieved_chunks = retriever.retrieve(user_query, k=3)
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

        rag_result = rag_llm.generate(rag_prompt, max_tokens=500, temperature=0.3)
        logger.info(f"RAG: {len(retrieved_chunks)} chunks, model=gemini-2.5-flash")

        return {
            "status": "success",
            "query": user_query,
            "total_chunks": len(all_chunks),
            "retrieved_chunks_count": len(retrieved_chunks),
            "responses": {
                "standard": {
                    "text": standard_result.get('response', ''),
                    "model": standard_result.get('model', 'gemini-1.5-flash-8b'),
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
