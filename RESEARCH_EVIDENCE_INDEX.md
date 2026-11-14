# Research evidence and executable implementation index

This file points to runnable code and retained experiment artifacts so implementation can be inspected independently of manuscript prose.

## Completed retrieval evidence

- Corrected run: `backend/evaluation/results/runs/depth_audit_20261114_191803`.
- Analysis entry point: `backend/evaluation/analyze_results.py`.
- Depth audit: `backend/evaluation/report_depth_audit.py`.
- Existing-run post-processing wrapper: `backend/evaluation/run_existing_analysis.sh`.
- Analysis outputs: `processed_rank_preserving/` beneath the run's processed-results location; the analysis manifest records the paper-to-raw configuration mapping.
- Test coverage includes reranker trace ordering, retrieval trace identity, and context-quality metric semantics under `backend/tests/`.

To regenerate analysis from the retained completed run without rerunning retrieval or calling an LLM API:

```bash
cd backend
python -m evaluation.analyze_results \
  --run-dir evaluation/results/runs/depth_audit_20261114_191803
python -m evaluation.report_depth_audit \
  --run-dir evaluation/results/runs/depth_audit_20261114_191803
```

## Executable but not yet publication-measured

| Area | Implementation | What it measures | Evidence status |
|---|---|---|---|
| Context degradation | `backend/evaluation/runners/run_context_degradation.py` | Four answer-generation conditions over configurable history sizes; token F1/EM diagnostics, prompt tokens, retrieval metrics, paired tests, official evaluator hypothesis exports | Two-question pilot only; full benchmark and official judge pending |
| LongMemEval evidence retrieval | `backend/evaluation/runners/run_longmemeval_retrieval.py` | Gold evidence-session Recall@K/MRR and resource diagnostics | Runner exists; distinguish from answer-quality evaluation |
| Cross-platform memory | `backend/evaluation/runners/run_memory_benchmark.py` | Memory retrieval metrics, cross-platform recall, latency and local store size | Example JSONL is a smoke fixture, not publication data |
| Resource/scalability | `backend/evaluation/runners/run_resource_scalability.py` | Index/embedding/BM25 build times, disk and memory estimates, process RSS, retrieval latency over corpus fractions | Runner exists; execution required |
| Retrieval trace | `backend/evaluation/` trace instrumentation and `backend/tests/` | Rank provenance, passage identity, reranker failure behavior and context metric semantics | Code and regression tests present; run tests in the project environment |

## Security and deployment controls

The earlier source audit identified wildcard CORS, missing API authentication, pickle persistence, and missing session ownership. The current implementation branch adds:

- Signed, expiring bearer-token authentication for private API routes; `/health` and `/auth/login` remain public.
- PBKDF2-HMAC-SHA256 password verification. The `demo / 123456` account is a development-only default.
- Explicit CORS origin allowlisting and rejection of wildcard origins.
- Principal-scoped conversation-memory retrieval, listing, deletion, and statistics.
- JSON conversation persistence, schema checks, local filesystem permission restrictions, and an in-process store lock.
- A configurable upload-size limit and browser/extension sign-in flows.

Security implementation notes, deployment variables, limitations, and a proposed scale-out architecture are in `SECURITY_AND_SCALABILITY.md`.

**Validation status:** security and conversation-isolation regression tests have been added to `backend/evaluation/run_reviewer_checks.sh`, but they have not yet been executed against this latest commit. The static source audit is not a penetration test.

**Remaining boundaries:** the configured identity is currently a single application account; the document corpus remains global/shared; local conversation files are not encrypted at rest; token revocation, distributed rate limiting, MFA, enterprise identity, malware scanning, and production infrastructure hardening are not implemented. Do not describe this prototype as fully multi-tenant or production-secure.

## Reproducibility boundaries

- The BEIR benchmark encoder is `BAAI/bge-base-en-v1.5`; the application default encoder is `all-MiniLM-L6-v2`.
- Retrieval-stage latency is not end-to-end LLM latency.
- A harness being checked in is not evidence that its benchmark has run.
- The two-question LongMemEval pilot is not statistically powered and cannot establish context-degradation mitigation.
- External model/framework comparisons are not complete unless matched run artifacts exist.
