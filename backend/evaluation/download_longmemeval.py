"""Download and validate the cleaned LongMemEval-S release.

Run from the backend directory:
    python evaluation/download_longmemeval.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
REQUIRED = {
    "question_id", "question_type", "question", "answer", "question_date",
    "haystack_session_ids", "haystack_dates", "haystack_sessions",
    "answer_session_ids",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 500:
        raise ValueError(f"Expected the cleaned LongMemEval-S array with 500 records; got {len(data) if isinstance(data, list) else type(data).__name__}")
    seen = set()
    sessions = 0
    for index, item in enumerate(data):
        missing = REQUIRED - set(item)
        if missing:
            raise ValueError(f"Record {index} is missing fields: {sorted(missing)}")
        qid = item["question_id"]
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError(f"Invalid or duplicate question_id at record {index}: {qid!r}")
        seen.add(qid)
        lengths = (
            len(item["haystack_session_ids"]),
            len(item["haystack_dates"]),
            len(item["haystack_sessions"]),
        )
        if len(set(lengths)) != 1:
            raise ValueError(f"History arrays are misaligned for {qid}: {lengths}")
        sessions += lengths[0]
        for session in item["haystack_sessions"]:
            if not isinstance(session, list):
                raise ValueError(f"Session is not a turn list for {qid}")
            for turn in session:
                if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                    raise ValueError(f"Malformed turn in {qid}")
    return len(data), sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/data/longmemeval_s_cleaned.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        raise SystemExit(f"File already exists: {output}. Use --force to replace it.")

    fd, temp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".part", dir=output.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "context-memory-paper-eval/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as target:
            total = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                total += len(block)
                print(f"Downloaded {total / (1024 * 1024):.1f} MiB", end="\r", flush=True)
        count, session_count = validate(temp_path)
        digest = sha256(temp_path)
        temp_path.replace(output)
        print(f"Validated questions: {count}")
        print(f"Aligned history sessions: {session_count}")
        print(f"File size: {output.stat().st_size / (1024 * 1024):.1f} MiB")
        print(f"SHA256: {digest}")
        print(f"DATASET={output}")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
