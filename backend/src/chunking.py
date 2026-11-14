import re
import hashlib


_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+.+|\d+(?:\.\d+)+[.)]?\s+.{2,100}|\d+[.)]\s+.{2,100}|[A-Z][A-Z0-9 ,:/()&-]{3,80})\s*$"
)


class TextChunker:
    """
    Structure-aware chunker.

    Headings and paragraph boundaries are preserved. Child chunks are sized by
    approximate word count (portable without a model-specific tokenizer), carry
    their section path and stable parent identifier, and retain source offsets.
    """

    def __init__(self, chunk_size=350, overlap=50):
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be an integer in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _sections(self, text: str) -> list[dict]:
        lines = text.splitlines(keepends=True)
        sections = []
        headings = []
        current_blocks = []
        offset = 0

        def flush():
            nonlocal current_blocks
            if current_blocks:
                body_start = current_blocks[0][1]
                body_end = current_blocks[-1][2]
                sections.append({
                    "section_path": list(headings),
                    "blocks": current_blocks,
                    "start": body_start,
                    "end": body_end,
                })
                current_blocks = []

        paragraph_lines = []
        paragraph_start = None
        paragraph_end = None

        def flush_paragraph():
            nonlocal paragraph_lines, paragraph_start, paragraph_end
            if paragraph_lines:
                value = "".join(paragraph_lines).strip()
                if value:
                    # Preserve original boundaries while excluding surrounding whitespace.
                    left_trim = len("".join(paragraph_lines)) - len("".join(paragraph_lines).lstrip())
                    right_trimmed = "".join(paragraph_lines).rstrip()
                    start = paragraph_start + left_trim
                    end = paragraph_start + len(right_trimmed)
                    current_blocks.append((value, start, end))
            paragraph_lines = []
            paragraph_start = None
            paragraph_end = None

        for line in lines:
            stripped = line.strip()
            line_start = offset
            line_end = offset + len(line)
            offset = line_end

            if not stripped:
                flush_paragraph()
                continue

            if _HEADING_RE.match(stripped):
                flush_paragraph()
                flush()
                heading = re.sub(r"^#{1,6}\s+", "", stripped).strip()
                if stripped.startswith("#"):
                    level = len(stripped) - len(stripped.lstrip("#"))
                    headings[:] = headings[:max(0, level - 1)]
                elif re.match(r"^\d", stripped):
                    level = max(1, stripped.split()[0].count(".") + 1)
                    headings[:] = headings[:max(0, level - 1)]
                else:
                    headings[:] = headings[:1]
                headings.append(heading)
                continue

            if paragraph_start is None:
                paragraph_start = line_start
            paragraph_lines.append(line)
            paragraph_end = line_end

        flush_paragraph()
        flush()
        if not sections and text.strip():
            start = len(text) - len(text.lstrip())
            end = len(text.rstrip())
            sections = [{
                "section_path": [],
                "blocks": [(text[start:end], start, end)],
                "start": start,
                "end": end,
            }]
        return sections

    def chunk_document(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []

        chunks = []
        chunk_index = 0
        step = self.chunk_size - self.overlap

        for section_index, section in enumerate(self._sections(text)):
            blocks = section["blocks"]
            if not blocks:
                continue

            parent_material = "\n".join(block[0] for block in blocks)
            parent_id = hashlib.sha256(
                ("/".join(section["section_path"]) + parent_material).encode("utf-8")
            ).hexdigest()[:20]

            def emit(selected):
                nonlocal chunk_index
                if not selected:
                    return
                source_start = selected[0][1]
                source_end = selected[-1][2]
                source_text = text[source_start:source_end].strip()
                if source_text:
                    chunks.append({
                        "text": source_text,
                        "offset": source_start,
                        "length": source_end - source_start,
                        "word_count": len(selected),
                        "chunk_index": chunk_index,
                        "parent_id": parent_id,
                        "section_path": list(section["section_path"]),
                        "section_index": section_index,
                    })
                    chunk_index += 1

            current = []
            for block_text, block_start, _ in blocks:
                block_words = [
                    (match.group(0), block_start + match.start(), block_start + match.end())
                    for match in re.finditer(r"\S+", block_text)
                ]
                if not block_words:
                    continue

                # Keep paragraphs intact when they fit. Split only exceptionally
                # long paragraphs, using overlap to retain local continuity.
                if len(block_words) > self.chunk_size:
                    if current:
                        emit(current)
                        current = current[-self.overlap:] if self.overlap else []
                    start = 0
                    while start < len(block_words):
                        end = min(start + self.chunk_size, len(block_words))
                        emit(block_words[start:end])
                        if end >= len(block_words):
                            break
                        start += step
                    current = []
                    continue

                if current and len(current) + len(block_words) > self.chunk_size:
                    emit(current)
                    overlap_words = current[-self.overlap:] if self.overlap else []
                    current = (
                        overlap_words
                        if len(overlap_words) + len(block_words) <= self.chunk_size
                        else []
                    )
                current.extend(block_words)

            if current:
                emit(current)

        return chunks

    def chunk_by_words(self, text: str) -> list[str]:
        return [item["text"] for item in self.chunk_document(text)]

    def chunk_by_sentences(self, text: str, max_words=300) -> list[str]:
        if not text or not text.strip():
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks, current, count = [], [], 0
        for sentence in sentences:
            size = len(sentence.split())
            if current and count + size > max_words:
                chunks.append(" ".join(current))
                current, count = [], 0
            current.append(sentence)
            count += size
        if current:
            chunks.append(" ".join(current))
        return chunks

    def get_chunk_metadata(self, chunks: list[str]) -> list[dict]:
        return [{
            "chunk_id": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            "position": index,
            "word_count": len(chunk.split()),
            "char_count": len(chunk),
        } for index, chunk in enumerate(chunks)]
