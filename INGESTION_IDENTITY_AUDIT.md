# Stage 2 — Ingestion and Document Identity Audit

Branch: `research/ieee-evaluation`

## Implemented

- Content-addressed document IDs: SHA-256 of the extracted document text.
- Deterministic chunk IDs: SHA-256 over document ID, chunk index, and chunk text.
- Duplicate-content detection in both `ExternalMemorySystem.ingest_document` and the FastAPI upload endpoint.
- Upload responses expose `document_id` and a `duplicate` flag.
- Chunk metadata includes document ID, chunk ID, chunk index, source, and original-source character span.
- Chunk offsets are computed from token spans in the original input, while stored chunk text remains whitespace-normalized.
- Vector-store insertion validates matrix shape, embedding dimension, finite values, chunk types, and metadata alignment.
- Vector-store search validates query shape, dimension, finite values, and positive K; empty stores return empty results.
- Vector-store save and load validate FAISS/chunk/metadata row counts. Load validates into temporary values before replacing the active in-memory store.
- Regression tests cover identity determinism, original-source offsets, invalid vectors, duplicate identity lookup, and corrupted persistence alignment.

## Compatibility and limitations

- Existing index entries created before this change may not have `document_id` metadata. They cannot be reliably deduplicated against new content by this mechanism. Re-ingest into a clean index or migrate old metadata before claiming corpus-wide duplicate elimination.
- Document identity is based on exact extracted text bytes after UTF-8 encoding. Whitespace or extraction differences produce different document IDs, even if the rendered document appears equivalent.
- Chunk IDs are deterministic for a given document content, chunking configuration, chunk position, and chunk text. Changing chunk size or overlap can change chunk IDs.
- Original character offsets refer to the extracted text string supplied to the chunker, not byte offsets in the original PDF/DOCX file. PDF extraction does not preserve page-level source coordinates.
- Save files are still written as separate files. Count validation detects inconsistent snapshots at load/save boundaries but does not provide a transactional multi-file commit or crash-safe atomic snapshot.
- The conversation-memory store is separate from the document vector store. Session isolation and concurrent multi-instance writes remain a separate audit item.

## Verification status

Implementation and regression tests have been added to the research branch. They have **not** been executed in the user's local Python environment yet. Do not treat this stage as verified until the targeted and full test suites pass locally.
