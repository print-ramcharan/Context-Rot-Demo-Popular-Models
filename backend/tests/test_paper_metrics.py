import json

from evaluation.build_paper_metrics import summarize_longmemeval


def test_paper_bundle_marks_official_judge_missing_instead_of_fabricating():
    run = {
        "run_id_utc": "test",
        "dataset_sha256": "abc",
        "n_questions": 1,
        "provider": "test-provider",
        "model": "test-model",
        "per_question": [{
            "question_id": "q1",
            "arm": "external_memory",
            "exact_match": 0.0,
            "token_f1": 0.5,
            "prompt_tokens": 100,
            "generation_latency_ms": 20.0,
            "memory_retrieval_latency_ms": 3.0,
            "tokens_used": {"prompt": 100, "completion": 10, "total": 110},
            "retrieval_recall_at_k": 1.0,
            "retrieval_precision_at_k": 0.2,
            "retrieval_mrr_at_k": 1.0,
            "retrieval_ndcg_at_k": 1.0,
            "retrieval_map_at_k": 1.0,
            "retrieval_hit_at_k": 1,
            "retrieval_evidence_completeness_at_k": 1,
            "context_session_coverage": None,
            "memory_indexing_latency_ms": 10.0,
            "memory_index_size_bytes": 1024,
        }],
    }
    result = summarize_longmemeval(run, None)
    assert result["automatic_diagnostics_not_official_score"]["external_memory"]["automatic_token_f1"] == 0.5
    assert result["official_judge_evaluation"]["status"] == "not_provided"
    assert result["automatic_diagnostics_not_official_score"]["external_memory"]["token_usage"]["total"]["sum"] == 110.0
