"""Document-id mapping helpers shared across runners.

BEIR relevance judgments are keyed by original document ID. The retrieval
pipeline returns sub-chunks, so we must collapse retrieved chunks to their
parent document IDs and deduplicate before computing document-level metrics.
"""

from typing import Dict, List, Sequence


def parent_ids_from_chunks(retrieved_chunks: Sequence[dict]) -> List[str]:
    """Return a deduplicated, rank-ordered list of parent document IDs.

    Chunks may map to a parent document via metadata ``parent_id`` or
    ``doc_id``. If neither is present, fall back to the chunk text.
    """
    seen = set()
    ordered = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        parent = (
            meta.get("parent_id")
            or meta.get("doc_id")
            or chunk.get("text", "")
        )
        if parent is None:
            continue
        parent = str(parent)
        if parent not in seen:
            seen.add(parent)
            ordered.append(parent)
    return ordered


def parent_scores_for_query(
    retrieved_chunks: Sequence[dict], qrels: Dict[str, Dict[str, int]], query_id: str
) -> List[int]:
    """Map retrieved chunks to gold relevance grades for a single query."""
    parent_ids = parent_ids_from_chunks(retrieved_chunks)
    rel_by_doc = qrels.get(query_id, {})
    return [int(rel_by_doc.get(pid, 0)) for pid in parent_ids]
