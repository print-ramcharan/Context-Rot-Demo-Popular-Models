import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank first-stage candidates with a cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.model_name = model_name
        self.model = None
        self.last_trace = {}
        try:
            self.model = CrossEncoder(model_name, device=device) if device else CrossEncoder(model_name)
        except Exception as exc:
            logger.exception("Failed to load reranker %s", model_name)
            self.load_error = str(exc)
            self.model = None

    def rerank(self, query: str, candidates: list[dict], top_n: int = 8) -> list[dict]:
        self.last_trace = {"scored_chunk_ids": [], "raw_scores": []}
        if not candidates:
            return []
        if self.model is None:
            raise RuntimeError(
                f"Cross-encoder '{self.model_name}' is unavailable: "
                f"{getattr(self, 'load_error', 'model not loaded')}"
            )

        pairs = [
            [query, candidate.get("rerank_text") or (
                "Section: " + " > ".join(candidate.get("metadata", {}).get("section_path", []))
                + "\n" + candidate.get("text", "")
            )]
            for candidate in candidates
        ]
        scores = self.model.predict(pairs, show_progress_bar=False)
        def chunk_id(candidate: dict, index: int) -> str:
            metadata = candidate.get("metadata", {})
            explicit_id = metadata.get("chunk_id")
            if explicit_id is not None:
                return str(explicit_id)
            source = metadata.get("source") or "index"
            chunk_index = metadata.get("chunk_index")
            if chunk_index is None:
                chunk_index = index
            return f"{source}:{chunk_index}"

        self.last_trace = {
            "scored_chunk_ids": [
                chunk_id(candidate, index)
                for index, candidate in enumerate(candidates)
            ],
            "raw_scores": [float(score) for score in scores],
        }

        ranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["original_score"] = candidate.get("score")
            item["rerank_score"] = float(score)
            item["original_rank"] = candidate.get("rank")
            item["score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        selected = ranked[:top_n]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected
