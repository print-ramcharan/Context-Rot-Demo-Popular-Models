"""Load a BEIR dataset and chunk documents into parent-document-tracked chunks.

The returned chunk list carries a ``parent_id`` in metadata so that the
retrieval results can be mapped back to the original BEIR document IDs used
by the relevance judgments (``qrels``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

from beir.datasets.data_loader import GenericDataLoader

from src.chunker import TextChunker

logger = logging.getLogger(__name__)


def load_beir_dataset(data_path: str | Path, dataset_name: str):
    """Load ``corpus``, ``queries``, and ``qrels`` from a BEIR dataset directory."""
    data_path = Path(data_path)
    # BEIR extracts into a nested directory: {data_path}/{dataset_name}/
    nested = data_path / dataset_name
    if nested.exists():
        data_path = nested
    loader = GenericDataLoader(str(data_path))
    corpus, queries, qrels = loader.load(split="test")
    # Normalise qrels: {qid: {doc_id: grade}}
    qrels = {str(qid): {str(doc_id): int(grade) for doc_id, grade in qrels[qid].items()} for qid in qrels}
    return corpus, queries, qrels


def chunk_corpus(corpus: Dict[str, dict], chunk_size: int = 400, overlap: int = 80):
    """Chunk BEIR documents and return a list of chunk dicts with parent metadata."""
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = []
    for doc_id, doc in corpus.items():
        text = doc.get("text", "")
        title = doc.get("title", doc_id)
        doc_chunks = chunker.chunk(text)
        for chunk in doc_chunks:
            chunks.append(
                {
                    "text": chunk["text"],
                    "metadata": {
                        "doc_id": str(doc_id),
                        "parent_id": str(doc_id),
                        "source": title,
                        "offset": chunk.get("offset", 0),
                        "length": chunk.get("length", len(chunk["text"])),
                    },
                }
            )
    return chunks
