from src.utils.metrics import context_quality_metrics


def test_context_metrics_do_not_mislabel_words_as_model_tokens():
    before = [
        {
            "text": "Alice joined the company in 2020.",
            "metadata": {"source": "profile.txt", "chunk_index": 0},
        },
        {
            "text": "Bob joined the company in 2021.",
            "metadata": {"source": "profile.txt", "chunk_index": 1},
        },
    ]
    after = [
        {
            "text": "Alice joined in 2020.",
            "metadata": {"source": "profile.txt", "chunk_index": 0},
        },
    ]

    result = context_quality_metrics(
        before, after, relevant={"profile.txt"}, budget_tokens=100
    )

    assert result["word_count_before"] == 12
    assert result["word_count_after"] == 5
    assert result["word_count_reduction"] == 7 / 12
    assert result["source_retention"] == 1.0
    assert result["chunk_retention"] == 0.5
    assert result["textual_evidence_retention"] is None
    assert result["tokens_before"] is None
    assert result["tokens_after"] is None
    assert result["token_reduction"] is None
    assert result["budget_utilization"] is None
