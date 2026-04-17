import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidate chunks using a cross-encoder.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load reranker model {self.model_name}: {e}")
            self.model = None

    def rerank(self, query: str, candidates: list[dict], top_n: int = 6) -> list[dict]:
        if not candidates or self.model is None:
            return candidates

        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores):
            updated = dict(candidate)
            updated["rerank_score"] = float(score)
            updated["score"] = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
