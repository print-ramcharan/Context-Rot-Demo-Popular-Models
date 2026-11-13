from evaluation.runners.run_retrieval import _collapse_parent_candidates, _map_to_parent_records


def _chunk(parent_id, score):
    return {"text": parent_id, "score": score, "metadata": {"parent_id": parent_id}}


def test_parent_chunks_are_deduplicated_by_best_score():
    chunks = [_chunk("doc-a", 0.4), _chunk("doc-a", 0.9), _chunk("doc-b", 0.8)]
    parents = _collapse_parent_candidates(chunks)
    assert [item["parent_id"] for item in parents] == ["doc-a", "doc-b"]
    assert parents[0]["score"] == 0.9


def test_mapping_returns_exactly_requested_depth_and_unique_ids():
    chunks = [_chunk(f"doc-{i}", 1.0 - i / 100) for i in range(25)]
    ids, scores, grades = _map_to_parent_records(
        chunks, {"q": {"doc-0": 1, "doc-21": 1}}, "q", evaluation_depth=20
    )
    assert len(ids) == len(set(ids)) == 20
    assert len(scores) == len(grades) == 20
    assert grades[0] == 1
    assert "doc-21" not in ids


def test_mapping_reports_honest_shortfall():
    chunks = [_chunk("doc-a", 1.0), _chunk("doc-a", 0.9), _chunk("doc-b", 0.8)]
    ids, _, _, diagnostics = _map_to_parent_records(
        chunks, {"q": {"doc-a": 1}}, "q", evaluation_depth=20,
        include_diagnostics=True,
    )
    assert ids == ["doc-a", "doc-b"]
    assert diagnostics["unique_parent_count_before_reranking"] == 2
    assert diagnostics["parent_deduplication_count"] == 1
