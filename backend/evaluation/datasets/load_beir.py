"""Load official BEIR datasets and create document-linked retrieval chunks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from beir.datasets.data_loader import GenericDataLoader
from src.chunking import TextChunker

logger = logging.getLogger(__name__)


def _is_dataset_root(path: Path) -> bool:
    return (
        (path / "corpus.jsonl").is_file()
        and (path / "queries.jsonl").is_file()
        and (path / "qrels" / "test.tsv").is_file()
    )


def load_beir_dataset(data_path: str | Path, dataset_name: str):
    """Load the official BEIR test split, downloading it if it is absent."""
    data_path = Path(data_path)
    candidates = [data_path, data_path / dataset_name]
    dataset_root = next((candidate for candidate in candidates if _is_dataset_root(candidate)), None)

    if dataset_root is None:
        from beir import util

        data_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        logger.info("Downloading official BEIR dataset %s from %s", dataset_name, url)
        extracted = Path(util.download_and_unzip(url, str(data_path.parent)))
        candidates = [extracted, data_path, data_path / dataset_name, data_path.parent / dataset_name]
        dataset_root = next((candidate for candidate in candidates if _is_dataset_root(candidate)), None)

    if dataset_root is None:
        raise FileNotFoundError(
            f"BEIR dataset '{dataset_name}' is missing corpus.jsonl, queries.jsonl, "
            "or qrels/test.tsv after download. Check the dataset archive and data directory."
        )

    loader = GenericDataLoader(data_folder=str(dataset_root))
    corpus, queries, qrels = loader.load(split="test")
    qrels = {
        str(qid): {str(doc_id): int(grade) for doc_id, grade in qrels[qid].items()}
        for qid in qrels
    }
    logger.info(
        "Loaded %s test split: %d documents, %d queries, %d qrel queries",
        dataset_name, len(corpus), len(queries), len(qrels)
    )
    return corpus, queries, qrels


def chunk_corpus(
    corpus: Dict[str, dict],
    chunk_size: int = 350,
    overlap: int = 50,
):
    """Chunk BEIR documents with the production structure-aware word chunker."""
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = []
    for doc_id, doc in corpus.items():
        text = doc.get("text", "")
        title = doc.get("title", doc_id)
        for child_index, child in enumerate(chunker.chunk_document(text)):
            chunks.append({
                "text": child["text"],
                "metadata": {
                    "doc_id": str(doc_id),
                    "parent_id": str(doc_id),
                    "chunk_index": child_index,
                    "chunk_id": f"{doc_id}:{child_index}",
                    "section_parent_id": child.get("parent_id"),
                    "section_path": child.get("section_path", []),
                    "source": title,
                    "title": title,
                    "offset": child.get("offset", 0),
                    "length": child.get("length", len(child["text"])),
                },
            })
    return chunks
