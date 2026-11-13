from argparse import Namespace

import pytest

from evaluation.runners.run_context_degradation import (
    PromptTokenCounter,
    build_prompt,
    exact_match,
    fit_context,
    normalize_answer,
    select_history,
    token_f1,
)


class CharacterCounter:
    """Deterministic test double; production uses the selected model tokenizer."""
    def count(self, text: str) -> int:
        return len(text)


def test_normalize_answer():
    assert normalize_answer("The QUICK, brown fox!") == "quick brown fox"


def test_exact_match_ignores_case_punctuation_and_articles():
    assert exact_match("The answer is Paris.", "answer is PARIS") == 1.0


def test_token_f1_partial_overlap():
    assert token_f1("Paris is the capital", "Paris capital city") == pytest.approx(4 / 7)


def test_select_history_uses_most_recent_sessions():
    sessions = [[{"role": "user", "content": str(i)}] for i in range(4)]
    selected = select_history(sessions, ["s0", "s1", "s2", "s3"], "2")
    assert [sid for sid, _ in selected] == ["s2", "s3"]


def test_select_history_all():
    sessions = [[{"role": "user", "content": "hello"}]]
    assert len(select_history(sessions, ["s1"], "all")) == 1


def test_select_history_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="length mismatch"):
        select_history([[]], [], "all")


def test_fit_context_respects_prompt_budget_and_returns_selected_units():
    question = "What city?"
    units = ["[session_id:s1] Paris", "[session_id:s2] London"]
    budget = len(build_prompt(question, units[0]))
    context, count, chosen = fit_context(question, units, budget, CharacterCounter())
    assert chosen == [units[0]]
    assert context == units[0]
    assert count <= budget


def test_fit_context_rejects_budget_smaller_than_question_prompt():
    with pytest.raises(ValueError, match="exceeding budget"):
        fit_context("What city?", [], 1, CharacterCounter())
