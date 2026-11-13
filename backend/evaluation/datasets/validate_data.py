"""Validate BEIR dataset loading before running the benchmark.

Checks:
- corpus / queries / qrels counts
- document-ID preservation (no missing or duplicate qrels IDs)
- chunk-to-parent mapping correctness
- qrels grade distribution
- parent-id coverage of retrieved chunks
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict

from evaluation.datasets.load_beir import load_beir_dataset

from evaluation.datasets.load_beir import chunk_corpus

DATA_ROOT = Path("evaluation/data")


def load_and_report(dataset: str, data_root: Path):
    data_path = data_root / dataset
    corpus, queries, qrels = load_beir_dataset(data_path, dataset)
    print(f"\n===== {dataset} =====")
    print(f"corpus documents : {len(corpus)}")
    print(f"queries          : {len(queries)}")
    qrel_total = sum(len(v) for v in qrels.values())
    print(f"qrels total      : {qrel_total}")

    # 1. Missing / duplicate qrels IDs
    all_doc_ids = set(corpus.keys())
    all_qrels_ids = set()
    duplicate_qrels = []
    for qid, rels in qrels.items():
        for doc_id in rels:
            if doc_id in all_qrels_ids:
                duplicate_qrels.append((qid, doc_id))
            all_qrels_ids.add(doc_id)
    missing_qrels = all_qrels_ids - all_doc_ids
    print(f"qrels IDs missing from corpus  : {len(missing_qrels)} {sorted(missing_qrels)[:10]}")
    print(f"qrels IDs duplicated globally  : {len(duplicate_qrels)} {duplicate_qrels[:10]}")

    # 2. Grade distribution
    grades = Counter()
    for qid, rels in qrels.items():
        for doc_id, g in rels.items():
            grades[int(g)] += 1
    print(f"grade distribution             : {dict(sorted(grades.items()))}")

    # 3. Chunk -> parent mapping
    chunk_size = 400
    overlap = 80
    chunks = chunk_corpus(corpus, chunk_size=chunk_size, overlap=overlap)
    print(f"chunked -> {len(chunks)} chunks ({chunk_size} chars, {overlap} overlap)")
    retained = 0
    lost_parents = set()
    for c in chunks:
        meta = c.get("metadata", {})
        if meta.get("parent_id"):
            retained += 1
        else:
            lost_parents.add(c.get("text", "")[:40])
    print(f"chunks with parent_id           : {retained}/{len(chunks)}")
    print(f"chunks WITHOUT parent_id        : {len(lost_parents)} {list(lost_parents)[:5]}")

    # 4. Parent coverage: do retrieved chunk parents map back to real BEIR docs?
    parent_ids = {str(c["metadata"]["parent_id"]) for c in chunks}
    print(f"distinct parent IDs consumed    : {len(parent_ids)}")
    print(f"parent IDs that exist in corpus : {len(parent_ids & all_doc_ids)}")
    print(f"parent IDs NOT in corpus        : {len(parent_ids - all_doc_ids)}")

    return corpus, queries, qrels, chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    args = parser.parse_args()

    for name in args.datasets:
        load_and_report(name, DATA_ROOT)


if __name__ == "__main__":
    main()
