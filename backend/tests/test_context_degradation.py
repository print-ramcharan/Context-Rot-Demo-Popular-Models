import math

import pytest

from evaluation.runners.run_context_degradation import (
    build_prompt,
    exact_match,
    fit_context,
    normalize_answer,
    retrieval_metrics,
    extended_retrieval_metrics,
    select_history,
    token_f1,
)


class CharacterCounter:
    """Deterministic, offline approximation; production uses the model tokenizer."""
    def count(self, text: str) -> int:
        return math.ceil(len(text) / 3)


def _history(count=8):
    sessions = [[{"role": "user", "content": f"message-{i}"}] for i in range(count)]
    ids = [f"session-{i}" for i in range(count)]
    dates = [f"2026-01-{i + 1:02d}" for i in range(count)]
    return sessions, ids, dates


def test_normalize_answer():
    assert normalize_answer("The QUICK, brown fox!") == "quick brown fox"


def test_exact_match_ignores_case_punctuation_and_articles():
    assert exact_match("The answer is Paris.", "answer is PARIS") == 1.0


def test_token_f1_partial_overlap():
    assert token_f1("Paris is the capital", "Paris capital city") == pytest.approx(2 / 3)


def test_select_history_uses_most_recent_sessions_in_original_order():
    sessions, ids, dates = _history()
    selected = select_history(sessions, ids, dates, "5")
    assert [row[0] for row in selected] == [
        "session-3", "session-4", "session-5", "session-6", "session-7"
    ]


def test_select_history_all_returns_full_history():
    sessions, ids, dates = _history()
    selected = select_history(sessions, ids, dates, "all")
    assert len(selected) == 8
    assert selected[0][0] == "session-0"
    assert selected[-1][0] == "session-7"


def test_select_history_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="must align"):
        select_history([[]], [], [], "all")


def test_fit_context_truncates_large_newest_session_and_respects_budget():
    question = "What was discussed?"
    units = [
        "[session_id:old] [source_session_id:old]\nold history",
        "[session_id:new] [source_session_id:new]\n" + ("recent detail " * 500),
    ]
    context, count, chosen = fit_context(
        question, units, 128, CharacterCounter()
    )
    assert count <= 128
    assert "[source_session_id:new]" in context
    assert chosen


def test_fit_context_preserves_chronological_order_for_fitting_sessions():
    question = "What happened?"
    units = [
        "[session_id:old] [source_session_id:old]\nfirst",
        "[session_id:new] [source_session_id:new]\nsecond",
    ]
    context, count, chosen = fit_context(question, units, 512, CharacterCounter())
    assert count <= 512
    assert context.index("first") < context.index("second")
    assert len(chosen) == 2


def test_fit_context_rejects_budget_smaller_than_question_prompt():
    with pytest.raises(ValueError, match="exceed"):
        fit_context("What city?", [], 1, CharacterCounter())



def test_retrieval_precision_uses_actual_unique_sessions_returned():
    recall, precision, mrr = retrieval_metrics(
        ["relevant-a", "irrelevant-b"],
        {"relevant-a"},
    )
    assert recall == 1.0
    assert precision == 0.5
    assert mrr == 1.0


def test_retrieval_metrics_return_zero_when_nothing_retrieved():
    assert retrieval_metrics([], {"gold"}) == (0.0, 0.0, 0.0)


def test_retrieval_metrics_are_unavailable_without_gold_evidence():
    assert retrieval_metrics(["session"], set()) == (None, None, None)

def test_fit_context_keeps_multiple_sessions_when_budget_allows():
    units = [
        "[session_id:old] [source_session_id:old]\n" + ("older detail " * 100),
        "[session_id:new] [source_session_id:new]\n" + ("newer detail " * 100),
    ]
    context, count, chosen = fit_context(
        "What happened?", units, 256, CharacterCounter()
    )
    assert count <= 256
    assert "[source_session_id:old]" in context
    assert "[source_session_id:new]" in context
    assert len(chosen) == 2


def test_extended_retrieval_metrics_include_ranking_and_evidence_completeness():
    metrics = extended_retrieval_metrics(
        ["irrelevant", "gold-a", "gold-b"],
        {"gold-a", "gold-b"},
    )
    assert metrics["hit_at_k"] == 1
    assert metrics["ndcg_at_k"] > 0
    assert metrics["average_precision_at_k"] > 0
    assert metrics["evidence_completeness_at_k"] == 1


def test_extended_retrieval_metrics_are_unavailable_without_gold():
    metrics = extended_retrieval_metrics(["session"], set())
    assert metrics["hit_at_k"] is None
    assert metrics["ndcg_at_k"] is None



def test_precision_at_k_uses_requested_cutoff_when_fewer_results_returned():
    recall, precision, mrr = retrieval_metrics(["gold"], {"gold"}, k=5)
    assert recall == 1.0
    assert precision == 0.2
    assert mrr == 1.0
