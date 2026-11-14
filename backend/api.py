from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.inference import LLMInference 
from src.retrieval import SemanticRetriever 
from src.vector_store import VectorStore
from src.embedding import EmbeddingGenerator
from src.chunking import TextChunker
from src.document_identity import document_id_for, chunk_id_for
from src.context_assembly import ContextAssembler
from src.reranker import CrossEncoderReranker
from src.document_extraction import extract_document
from src.grounded_answer import answer_with_evidence, ABSTENTION
from src.utils.metrics import (
    MetricsCollector,
    advanced_ranking_metrics,
    context_quality_metrics,
    rank_metrics,
    summarize_metric,
    text_metrics,
)
import os
import shutil
import json
import time
import logging
import yaml
from pathlib import Path
from typing import Optional, List, Any
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
    document_id: Optional[str] = None
    duplicate: bool = False
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

class EvaluationCase(BaseModel):
    query: str
    expected_answer: str = ""
    relevant_sources: List[str] = []
    generate_answer: bool = False
    context_budget_tokens: Optional[int] = None
    unanswerable: bool = False
    context_variant: Optional[str] = None
    evidence_position: Optional[str] = None
    distractor_count: int = 0
    context_length_tokens: Optional[int] = None

class EvaluationRequest(BaseModel):
    cases: List[EvaluationCase]
    top_k: int = 5
    evaluation_k: List[int] = [5, 10, 20]

class ComparisonRequest(EvaluationRequest):
    pass
# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
llm_cfg = config.get('llm', {})
openrouter_cfg = llm_cfg.get('openrouter', {})
openrouter_with_cache_cfg = {
    **openrouter_cfg,
    "semantic_cache": llm_cfg.get("semantic_cache", {}),
}
chunk_cfg = config.get('chunking', {})
embedding_cfg = config.get('embedding', {})
retrieval_cfg = config.get('retrieval', {})
storage_cfg = config.get('storage', {})
context_cfg = config.get('context', {})

metrics = MetricsCollector()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

METRIC_CATALOG = {
    "retrieval": [
        "precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr_at_k",
        "average_precision_at_k", "map_at_k", "ndcg_at_k",
        "unique_document_coverage",
    ],
    "hybrid_and_reranking": [
        "absolute_recall_improvement", "relative_recall_improvement",
        "ndcg_improvement", "reranking_gain", "evidence_loss",
        "candidate_utilization",
    ],
    "context": [
        "context_precision", "context_recall", "source_retention",
        "chunk_retention", "textual_evidence_retention",
        "evidence_loss_rate", "word_count_reduction",
        "word_compression_ratio", "evidence_density_before_per_word",
        "evidence_density_after_per_word", "evidence_density_gain",
        "model_token_reduction_unavailable_without_model_tokenizer",
    ],
    "answer": [
        "exact_match", "answer_token_precision", "answer_token_recall",
        "answer_token_f1", "semantic_answer_similarity",
        "faithfulness", "hallucination_rate", "citation_precision",
        "citation_recall", "citation_f1", "answer_completeness",
        "abstention_accuracy", "answer_accuracy",
    ],
    "context_degradation": [
        "accuracy_at_context_length", "context_degradation",
        "context_degradation_rate", "robustness_score",
        "position_sensitivity", "distractor_sensitivity",
        "noise_robustness", "context_utilization",
        "mitigation_gain", "context_accuracy_auc",
    ],
    "external_memory": [
        "memory_precision", "memory_recall", "memory_f1",
        "retention_accuracy", "update_accuracy", "staleness_rate",
        "conflict_rate", "memory_consistency", "memory_token_efficiency",
        "memory_gain",
    ],
    "system": [
        "ingestion_ms", "retrieval_ms", "reranking_ms", "compaction_ms",
        "generation_ms", "end_to_end_ms", "mean_latency", "median_latency",
        "p95_latency", "p99_latency", "throughput", "failure_rate",
        "input_tokens", "context_tokens", "output_tokens", "total_tokens",
        "token_savings", "cost_per_query", "cost_reduction",
    ],
    "statistical_reliability": [
        "sample_size", "standard_deviation", "standard_error",
        "confidence_interval_95", "paired_improvement",
        "paired_t_test", "wilcoxon_signed_rank", "effect_size",
    ],
}

# ── AI Components Initialization ──────────────────────────────────────────────

conv_store = ConversationStore(storage_dir="conversations")

gen = EmbeddingGenerator(
    model_name=embedding_cfg.get('model_name', "all-MiniLM-L6-v2"),
    device=embedding_cfg.get('device', 'cpu'),
    cache_path=embedding_cfg.get('cache_path', "cache/embeddings.sqlite"),
    cache_max_items=embedding_cfg.get('cache_max_items', 200000),
    cache_enabled=embedding_cfg.get('cache_enabled', True),
    query_instruction=embedding_cfg.get("query_instruction", ""),
    query_prompt_name=embedding_cfg.get("query_prompt_name")
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
    rerank_top_n=retrieval_cfg.get('rerank_top_n', 8),
    rerank_candidate_pool=retrieval_cfg.get('rerank_candidate_pool', 20),
    query_cache_ttl_s=retrieval_cfg.get('query_cache_ttl_s', 300),
    query_cache_max_items=retrieval_cfg.get('query_cache_max_items', 10000),
)

# Optional Reranker (Cross-Encoder)
if retrieval_cfg.get('rerank_enabled', False):
    try:
        reranker = CrossEncoderReranker(
            model_name=retrieval_cfg.get('rerank_model', "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=embedding_cfg.get("device")
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

# Use a listed OpenRouter free model by default; override it in config.yaml.
STABLE_MODEL = openrouter_cfg.get(
    "model",
    "openai/gpt-4o-mini",
)

llm_standard = LLMInference(
    provider="openrouter",
    model=STABLE_MODEL,
    config=openrouter_with_cache_cfg
)

llm_rag = LLMInference(
    provider="openrouter",
    model=STABLE_MODEL,
    config=openrouter_with_cache_cfg
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
    return f"""Answer the user's question using ONLY the provided context.
Return one concise sentence that directly answers what the question asks (person, object, place, or event).
Prefer a direct statement from the context over nearby narrative details. Do not answer a different
question, repeat unrelated events, or add unsupported details. If the context does not contain the
answer, say "I don't know."

<CONTEXT>
{context}
</CONTEXT>

Question: {query}
Answer:"""

# ── API Endpoints ─────────────────────────────────────────────────────────────

RESEARCH_CATALOG = {
    "paper_readiness": {
        "minimum_independent_queries": 30,
        "required_relevance_labels": True,
        "required_repeated_latency_runs": 3,
        "confidence_intervals_require_independent_queries": True,
        "status": "incomplete_until_benchmarks_are_run",
    },
    "experiments": [
        {
            "id": "retrieval_comparison",
            "title": "Retrieval comparison",
            "configurations": ["BM25", "Dense FAISS", "Hybrid RRF", "Hybrid + Cross-Encoder"],
            "metrics": ["Precision@K", "Recall@K", "MRR@K", "nDCG@K", "MAP@K"],
            "datasets": ["SciFact", "NFCorpus"],
            "runner": "evaluation/runners/run_retrieval.py",
            "status": "runner_available",
        },
        {
            "id": "context_rot",
            "title": "Context degradation",
            "configurations": ["question-only", "recent-history", "long-context", "external-memory"],
            "context_lengths_tokens": [4000, 8000, 16000, 32000, 64000],
            "evidence_positions": ["beginning", "middle", "end"],
            "metrics": ["accuracy", "evidence_recall", "context_utilization", "degradation_rate"],
            "runner": "evaluation/runners/run_context_degradation.py",
            "status": "runner_available",
        },
        {
            "id": "answer_quality",
            "title": "Answer quality",
            "metrics": [
                "exact_match", "token_f1", "semantic_similarity",
                "answer_relevance", "claim_level_faithfulness",
                "answer_correctness", "abstention_accuracy",
            ],
            "status": "lexical_proxies_available_judge_required_for_semantic_metrics",
        },
        {
            "id": "efficiency",
            "title": "Efficiency and scalability",
            "metrics": [
                "input_tokens", "output_tokens", "token_reduction",
                "cost_per_query", "cold_latency", "warm_latency",
                "p50", "p95", "p99", "throughput", "memory_overhead",
            ],
            "status": "partial_runtime_telemetry",
        },
        {
            "id": "external_memory",
            "title": "External-memory ablation",
            "configurations": ["without memory", "with memory", "cross-platform memory"],
            "metrics": ["Recall@K", "MRR@K", "answer_correctness", "staleness", "isolation"],
            "runner": "evaluation/runners/run_memory_benchmark.py",
            "status": "smoke_runner_available",
        },
        {
            "id": "statistical_validation",
            "title": "Statistical validation",
            "methods": ["paired bootstrap CI", "Wilcoxon signed-rank", "Holm-Bonferroni"],
            "module": "evaluation/metrics/statistical_tests.py",
            "status": "available",
        },
    ],
    "baselines": [
        {"id": "B1", "title": "Full document to LLM", "status": "generation_experiment_required"},
        {"id": "B2", "title": "Dense RAG", "status": "retrieval_runner_available"},
        {"id": "B3", "title": "BM25 RAG", "status": "retrieval_runner_available"},
        {"id": "B4", "title": "Hybrid RRF", "status": "retrieval_runner_available"},
        {"id": "B5", "title": "Hybrid + cross-encoder", "status": "retrieval_runner_available"},
        {"id": "B6", "title": "Proposed hybrid + memory + compaction", "status": "end_to_end_experiment_required"},
    ],
}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
def get_metrics():
    """Return online pipeline telemetry and the metric definitions used by evaluation."""
    return {
        "status": "success",
        "runtime": metrics.snapshot(),
        "index": {
            "documents": store.index.ntotal if store.index else 0,
            "embedding_dimension": gen.get_embedding_dimension(),
            "retrieval_mode": retrieval_cfg.get("mode", "hybrid"),
            "reranking_enabled": retrieval_cfg.get("rerank_enabled", False),
        },
        "available_metrics": METRIC_CATALOG,
    }

@app.get("/metrics/catalog")
def metrics_catalog():
    """Describe the evaluation scorecard and which metrics need ground truth."""
    return {"status": "success", "categories": METRIC_CATALOG}

@app.get("/research/catalog")
def research_catalog():
    """Return the reproducibility scorecard and available paper experiments."""
    return {"status": "success", **RESEARCH_CATALOG}

@app.get("/documents")
def list_indexed_documents():
    """List the source files currently represented in the active vector index."""
    sources = sorted({
        str(metadata.get("source"))
        for metadata in store.metadata
        if metadata.get("source")
    })
    return {
        "status": "success",
        "documents": [
            {
                "name": source,
                "chunks": sum(
                    1 for metadata in store.metadata
                    if metadata.get("source") == source
                ),
            }
            for source in sources
        ],
        "total": len(sources),
    }

@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    """Run reproducible retrieval/evaluation cases against the current index."""
    if not request.cases:
        raise HTTPException(status_code=400, detail="At least one evaluation case is required")
    evaluation_k = tuple(sorted(set(request.evaluation_k)))
    if not evaluation_k or any(cutoff <= 0 for cutoff in evaluation_k):
        raise HTTPException(status_code=400, detail="evaluation_k must contain positive cutoffs")
    available_sources = {
        str(metadata.get("source"))
        for metadata in store.metadata
        if metadata.get("source")
    }
    valid_cases = []
    invalid_cases = []
    for case_index, case in enumerate(request.cases):
        errors = []
        if not case.query.strip():
            errors.append("query is empty")
        unknown_sources = sorted({
            source.split(":", 1)[0]
            for source in case.relevant_sources
            if source.split(":", 1)[0] not in available_sources
        })
        if unknown_sources:
            errors.append(
                f"unknown indexed sources: {', '.join(unknown_sources)}"
            )
        if errors:
            invalid_cases.append({
                "case_index": case_index,
                "query": case.query,
                "errors": errors,
            })
        else:
            valid_cases.append((case_index, case))
    if not valid_cases:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No valid evaluation cases were supplied.",
                "invalid_cases": invalid_cases,
            },
        )

    results: list[dict[str, Any]] = []
    for case_index, case in valid_cases:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        retrieved, retrieval_trace = retriever.retrieve_with_trace(
            case.query,
            evaluation_k=evaluation_k,
            threshold=retrieval_cfg.get("similarity_threshold", 0.0),
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        relevant = set(case.relevant_sources)
        ranking = advanced_ranking_metrics(
            retrieved,
            relevant,
            max(evaluation_k),
        )
        ranking_by_k = {
            str(cutoff): advanced_ranking_metrics(retrieved, relevant, cutoff)
            for cutoff in evaluation_k
        }
        compaction_started = time.perf_counter()
        compacted = assembler.compress_context(
            case.query,
            retrieved,
            max_context_chars=context_cfg.get("max_context_chars", 6000),
            max_sentences=context_cfg.get("max_sentences", 8),
            max_chunks_after_compaction=context_cfg.get("max_chunks", 10),
        )
        compaction_ms = (time.perf_counter() - compaction_started) * 1000
        context = "\n\n".join(item.get("text", "") for item in compacted)
        def chunk_id(item: dict) -> str:
            metadata = item.get("metadata", {})
            return f"{metadata.get('source', 'index')}:{metadata.get('chunk_index', '')}"
        input_chunk_ids = [chunk_id(item) for item in retrieved]
        retained_chunk_ids = [chunk_id(item) for item in compacted]
        retrieval_trace["stages"]["compaction"] = {
            "input_chunk_ids": input_chunk_ids,
            "retained_chunk_ids": retained_chunk_ids,
            "removed_chunk_ids": [
                chunk_id for chunk_id in input_chunk_ids
                if chunk_id not in retained_chunk_ids
            ],
            # These are whitespace-delimited word counts, not model tokens.
            "input_words": len(
                " ".join(item.get("text", "") for item in retrieved).split()
            ),
            "output_words": len(context.split()),
            "token_counts_measured": False,
            "latency_ms": compaction_ms,
        }
        retrieved_sources = {
            str(item.get("source") or item.get("metadata", {}).get("source", ""))
            for item in retrieved
        } - {""}
        context_metrics = context_quality_metrics(
            retrieved,
            compacted,
            relevant,
            case.context_budget_tokens,
        )
        context_metrics.update({
            "chunks_before": len(retrieved),
            "chunks_after": len(compacted),
        })
        deduplicated = []
        seen_context_ids = set()
        for item in retrieved:
            metadata = item.get("metadata", {})
            identifier = (
                item.get("id")
                or item.get("chunk_id")
                or f"{metadata.get('source', '')}:{metadata.get('chunk_index', '')}"
            )
            if identifier not in seen_context_ids:
                seen_context_ids.add(identifier)
                deduplicated.append(item)
        context_variants = {
            "retrieved context": context_quality_metrics(
                retrieved, retrieved, relevant, case.context_budget_tokens
            ),
            "deduplicated context": context_quality_metrics(
                retrieved, deduplicated, relevant, case.context_budget_tokens
            ),
            "compacted context": context_metrics,
        }
        answer = ""
        generation_ms = 0.0
        tokens = {}
        generated = {}
        generation_status = "skipped"
        generation_error = None
        if case.generate_answer:
            generation_started = time.perf_counter()
            try:
                last_generation_error = None
                generation_limit = context_cfg.get("max_generation_chars", 3000)
                for context_chars in (
                    generation_limit,
                    max(1500, generation_limit // 2),
                    1000,
                ):
                    prompt = format_rag_prompt(
                        case.query,
                        context[:context_chars],
                    )
                    for budget in (256, 128, 64):
                        try:
                            generated = llm_rag.generate(
                                prompt,
                                max_tokens=budget,
                            )
                            break
                        except Exception as exc:
                            last_generation_error = exc
                            if "402" not in str(exc):
                                raise
                    else:
                        continue
                    break
                else:
                    raise last_generation_error
            except Exception as exc:
                message = str(exc)
                provider_detail = message.split("https://", 1)[0].strip()
                if "prompt tokens limit exceeded" in message.lower():
                    message = (
                        "OpenRouter rejected the generation prompt because the "
                        "key prompt-token limit is too small. The generation "
                        "context is capped separately from retrieval metrics. "
                        f"Provider detail: {provider_detail[:500]}"
                    )
                elif "402" in message or "credits" in message.lower():
                    logger.warning("Evaluation answer generation skipped: OpenRouter credits are insufficient")
                    message = (
                        "OpenRouter rejected the request after trying 256, 128, "
                        "and 64 output tokens. This usually means the API key's "
                        "total limit is below the remaining account credits. "
                        "Increase that key limit in OpenRouter or use another key. "
                        f"Provider detail: {provider_detail[:500]}"
                    )
                else:
                    logger.exception("Evaluation answer generation failed")
                generation_status = "failed"
                generation_error = message
            else:
                generation_ms = (time.perf_counter() - generation_started) * 1000
                answer = generated.get("response", "")
                tokens = generated.get("tokens_used", {})
                generation_status = "completed"
        result = {
            "case_index": case_index,
            "query": case.query,
            "expected_answer": case.expected_answer,
            "generated_answer": answer if generation_status == "completed" else None,
            "retrieved_sources": sorted(retrieved_sources),
            "retrieval": ranking,
            "retrieval_by_k": ranking_by_k,
            "trace": retrieval_trace,
            "context": context_metrics,
            "context_variants": context_variants,
            "answer": (
                text_metrics(answer, case.expected_answer, context)
                if case.expected_answer and generation_status == "completed" else None
            ),
            "generation": {
                "status": generation_status,
                "error": generation_error,
                "answer_length": len(answer),
            },
            "experiment": {
                "context_variant": case.context_variant,
                "evidence_position": case.evidence_position,
                "distractor_count": case.distractor_count,
                "context_length_tokens": case.context_length_tokens,
                "unanswerable": case.unanswerable,
            },
            "system": {
                "retrieval_ms": retrieval_ms,
                "embedding_ms": retrieval_trace["stages"]["embedding"]["latency_ms"],
                "bm25_ms": retrieval_trace["stages"]["bm25"]["latency_ms"],
                "faiss_ms": retrieval_trace["stages"]["dense"]["latency_ms"],
                "rrf_ms": retrieval_trace["stages"]["rrf"]["latency_ms"],
                "reranking_ms": retrieval_trace["stages"]["reranker"]["latency_ms"],
                "compaction_ms": compaction_ms,
                "generation_ms": generation_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
                "prompt_tokens": tokens.get("prompt", 0),
                "completion_tokens": tokens.get("completion", 0),
                "cache_hit": bool(generated.get("cache_hit", False))
                if case.generate_answer else False,
            },
        }
        metrics.increment("evaluation_cases")
        metrics.observe("evaluation_retrieval_ms", retrieval_ms)
        results.append(result)

    metric_names = (
        ("precision_at_k", "precision_at_k"),
        ("recall_at_k", "recall_at_k"),
        ("hit_rate_at_k", "hit_rate_at_k"),
        ("mrr_at_k", "mrr_at_k"),
        ("average_precision_at_k", "map_at_k"),
        ("ndcg_at_k", "ndcg_at_k"),
        ("evidence_retention", "evidence_retention"),
        ("token_reduction", "token_reduction"),
        ("answer_token_f1", "answer_token_f1"),
        ("faithfulness_proxy", "faithfulness_proxy"),
    )
    summary_metrics = {}
    for source_name, output_name in metric_names:
        values = []
        for result in results:
            source = result["retrieval"] if source_name in result["retrieval"] else (
                result["context"] if source_name in result["context"]
                else result.get("answer") or {}
            )
            value = source.get(source_name)
            if value is not None:
                values.append(value)
        summary_metrics[output_name] = summarize_metric(values)
    latency_values = [result["system"]["total_ms"] for result in results]
    return {
        "status": "success",
        "cases": results,
        "invalid_cases": invalid_cases,
        "summary": {
            "case_count": len(results),
            "metrics": summary_metrics,
            "latency": summarize_metric(latency_values),
        },
    }

@app.post("/evaluate/compare")
def compare_pipeline_stages(request: ComparisonRequest):
    """Compare retrieval stages on the exact same query and label set."""
    if not request.cases:
        raise HTTPException(status_code=400, detail="At least one comparison case is required")

    baseline = evaluate(EvaluationRequest(
        cases=[case.copy(update={"generate_answer": False}) for case in request.cases],
        evaluation_k=request.evaluation_k,
    ))
    cutoffs = tuple(sorted(set(request.evaluation_k)))
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise HTTPException(status_code=400, detail="evaluation_k must contain positive cutoffs")
    stage_names = ("bm25", "dense", "rrf", "reranked")
    stage_values = {
        name: {cutoff: [] for cutoff in cutoffs}
        for name in stage_names
    }
    latency_values = {name: [] for name in (
        "embedding", "bm25", "faiss", "rrf", "reranker", "compaction", "end_to_end"
    )}

    for result in baseline["cases"]:
        case = request.cases[result["case_index"]]
        relevant = set(case.relevant_sources)
        trace = result["trace"]
        stage_ids = {
            "bm25": trace["stages"]["bm25"]["retrieved_chunk_ids"],
            "dense": trace["stages"]["dense"]["retrieved_chunk_ids"],
            "rrf": trace["stages"]["rrf"]["candidate_chunk_ids"],
            "reranked": trace["stages"]["reranker"]["reranked_chunk_ids"],
        }
        for name, ids in stage_ids.items():
            for cutoff in cutoffs:
                stage_values[name][cutoff].append(
                    advanced_ranking_metrics(
                        [{"chunk_id": chunk_id} for chunk_id in ids],
                        relevant,
                        cutoff,
                    )
                )
        system = result["system"]
        trace_stages = trace["stages"]
        for name, value in {
            "embedding": trace_stages["embedding"]["latency_ms"],
            "bm25": trace_stages["bm25"]["latency_ms"],
            "faiss": trace_stages["dense"]["latency_ms"],
            "rrf": trace_stages["rrf"]["latency_ms"],
            "reranker": trace_stages["reranker"]["latency_ms"],
            "compaction": trace_stages["compaction"]["latency_ms"],
            "end_to_end": system["total_ms"],
        }.items():
            latency_values[name].append(value)

    def average(stage: str, cutoff: int) -> dict:
        values = stage_values[stage][cutoff]
        fields = ("precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")
        return {
            field: summarize_metric([
                item[field] for item in values if item[field] is not None
            ])
            for field in fields
        }

    context_variants = {
        "retrieved context": [],
        "deduplicated context": [],
        "compacted context": [],
    }
    for result in baseline["cases"]:
        for name in context_variants:
            context_variants[name].append(result["context_variants"][name])

    def context_average(name: str) -> dict:
        values = context_variants[name]
        return {
            "evidence_recall": summarize_metric([
                item["context_recall"]
                for item in values if item["context_recall"] is not None
            ]),
            "word_count": summarize_metric([
                item["word_count_after"] for item in values
            ]),
            "retention": summarize_metric([
                item["evidence_retention"]
                for item in values if item["evidence_retention"] is not None
            ]),
        }

    return {
        "status": "success",
        "query_count": len(request.cases),
        "valid_query_count": len(baseline["cases"]),
        "invalid_cases": baseline.get("invalid_cases", []),
        "configuration": {
            "evaluation_k": list(cutoffs),
            "same_queries_and_labels": True,
            "relevance_label_count": sum(
                len(case.relevant_sources) for case in request.cases
            ),
            "generation": "not_run",
        },
        "retrieval_comparison": [
            {
                "method": name,
                "metrics_by_k": {
                    str(cutoff): average(name, cutoff)
                    for cutoff in cutoffs
                },
                "metrics": average(name, max(cutoffs)),
            }
            for name in stage_names
        ],
        "context_comparison": [
            {"configuration": name, "metrics": context_average(name)}
            for name in context_variants
        ],
        "generation_comparison": {
            "status": "not_run",
            "reason": "Run generation separately when the provider has sufficient credits.",
            "configurations": [
                "full document",
                "dense RAG",
                "hybrid RAG",
                "hybrid plus reranker",
                "complete pipeline",
            ],
        },
        "latency_comparison": {
            name: summarize_metric(values)
            for name, values in latency_values.items()
        },
        "per_query": [
            {
                "case_index": result["case_index"],
                "query": result["query"],
                "trace": result["trace"],
                "retrieval": result["retrieval"],
                "relevant_sources": request.cases[result["case_index"]].relevant_sources,
            }
            for result in baseline["cases"]
        ],
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    clear_existing_data: bool = Query(False, alias="clear_existing")
):
    temp_path = None
    try:
        if clear_existing_data:
            store.clear()
            retriever.invalidate_cache()

        safe_filename = Path(file.filename or "upload").name
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt", ".md"}:
            raise HTTPException(status_code=400, detail="Supported files: PDF, DOCX, TXT and Markdown")

        temp_path = DATA_DIR / f".upload-{time.time_ns()}{suffix}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_start = time.perf_counter()
        text_content = extract_document(temp_path)
        if not text_content.strip():
            raise ValueError("No text content extracted")

        document_id = document_id_for(text_content)
        if store.contains_document(document_id):
            temp_path.unlink(missing_ok=True)
            return UploadResponse(
                status="success",
                message="Document content is already indexed",
                chunks_created=0,
                embeddings_stored=0,
                document_id=document_id,
                duplicate=True,
                timings_ms={"total_ms": (time.perf_counter() - extract_start) * 1000}
            )

        permanent_path = DATA_DIR / f"{document_id[:16]}_{safe_filename}"
        temp_path.replace(permanent_path)
        temp_path = None

        chunk_dicts = chunker.chunk_document(text_content)
        if not chunk_dicts:
            raise ValueError("Document extraction produced no indexable sections")

        # Section headings are included in the retrieval representation, while
        # the stored evidence remains the original extracted child passage.
        embedding_texts = [
            ("Section: " + " > ".join(chunk.get("section_path", [])) + "\n" + chunk["text"]).strip()
            for chunk in chunk_dicts
        ]
        embeddings = gen.embed_batch(embedding_texts, batch_size=16)

        metadata_list = []
        for i, chunk in enumerate(chunk_dicts):
            metadata_list.append({
                "document_id": document_id,
                "chunk_id": chunk_id_for(document_id, i, chunk["text"]),
                "source": safe_filename,
                "chunk_index": i,
                "parent_id": chunk.get("parent_id"),
                "section_path": chunk.get("section_path", []),
                "section_index": chunk.get("section_index"),
                "char_offset": chunk["offset"],
                "char_length": chunk["length"],
                "word_count": chunk["word_count"],
                "embedding_model": gen.model_name,
            })

        store.add(embeddings, [chunk["text"] for chunk in chunk_dicts], metadata=metadata_list)
        retriever.invalidate_cache()
        store.save(INDEX_PATH)
        retriever.build_lexical_index()
        gen.embed_query("warmup query")

        latency = (time.perf_counter() - extract_start) * 1000
        logger.info("Indexed %s chunks from %s", len(chunk_dicts), safe_filename)
        return UploadResponse(
            status="success",
            message=f"File '{safe_filename}' ingested successfully",
            chunks_created=len(chunk_dicts),
            embeddings_stored=len(embeddings),
            document_id=document_id,
            duplicate=False,
            timings_ms={"total_ms": latency}
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

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
        total_start = time.perf_counter()
        user_query = request.user_query.strip()
        if not user_query:
            yield json.dumps({"type": "text", "text": ABSTENTION}) + "\n"
            yield json.dumps({"type": "final", "abstained": True, "reason": "empty_query"}) + "\n"
            return
        try:
            retrieve_start = time.perf_counter()
            retrieved_chunks = retriever.retrieve(
                user_query,
                k=retrieval_cfg.get("top_k", 12),
                use_cache=False,
            )
            retrieve_time = (time.perf_counter() - retrieve_start) * 1000
            retrieved_chunks = retriever.deduplicate_chunks(
                retrieved_chunks, overlap_threshold=0.82
            )
            context_chunks = assembler.truncate_to_fit(
                retrieved_chunks,
                context_cfg.get("max_context_chars", 18000),
            )
            context = "\n\n".join(
                "[Source: " + str(item.get("metadata", {}).get("source", "Unknown"))
                + " | Section: " + " > ".join(item.get("metadata", {}).get("section_path", []))
                + "]\n" + item.get("text", "")
                for item in context_chunks if item.get("text")
            )

            yield json.dumps({
                "type": "metadata",
                "model": llm_rag.model,
                "retrieval_ms": retrieve_time,
                "chunks_count": len(context_chunks),
                "context": context,
            }) + "\n"

            if not context_chunks:
                result = {
                    "answer": ABSTENTION,
                    "abstained": True,
                    "citations": [],
                    "reason": "no_retrieved_evidence",
                }
            else:
                result = answer_with_evidence(
                    llm_rag,
                    user_query,
                    context_chunks,
                    max_tokens=llm_cfg.get("max_tokens", 700),
                )

            yield json.dumps({"type": "text", "text": result["answer"]}) + "\n"
            latency = (time.perf_counter() - total_start) * 1000
            yield json.dumps({
                "type": "final",
                "latency_ms": latency,
                "abstained": result.get("abstained", True),
                "reason": result.get("reason"),
                "citations": result.get("citations", []),
            }) + "\n"
        except Exception as exc:
            logger.exception("Grounded RAG query failed")
            # A generation/verification failure must never turn into a guessed answer.
            yield json.dumps({"type": "text", "text": ABSTENTION}) + "\n"
            yield json.dumps({
                "type": "final",
                "abstained": True,
                "reason": "pipeline_error",
                "detail": str(exc),
                "latency_ms": (time.perf_counter() - total_start) * 1000,
            }) + "\n"

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
        if not request.prompt.strip() and not request.response.strip():
            raise ValueError("Both prompt and response are empty — nothing to store")

        result = conv_store.store_conversation(
            platform=request.platform,
            session_id=request.session_id,
            prompt=request.prompt,
            response=request.response,
            embedding_generator=gen,
        )
        logger.info(f"Stored conversation: platform={request.platform} session={request.session_id} chunks={result['chunk_count']}")
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
            "count": len(chunks)
        }
    except Exception as e:
        logger.error(f"retrieve-context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-sessions")
def list_sessions():
    sessions = conv_store.list_sessions()
    return {
        "status": "success",
        "sessions": sessions,
        "total": len(sessions)
    }

@app.delete("/delete-session/{session_id}")
def delete_session(session_id: str):
    try:
        result = conv_store.delete_session(session_id)
        if not result.get("deleted", False):
            raise HTTPException(status_code=404, detail=result.get("reason", "Session not found"))
        
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

# ── Helper Functions ──────────────────────────────────────────────────────────

def _extract_text_from_file(file_path: Path, filename: str) -> str:
    """Backward-compatible wrapper around the layout-aware extractor."""
    return extract_document(file_path)
