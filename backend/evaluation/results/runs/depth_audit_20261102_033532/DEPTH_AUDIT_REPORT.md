# Depth audit report

Run directory: `evaluation/results/runs/depth_audit_20261102_033532`

Previous fixed-depth results are preserved separately and are not overwritten.

| Dataset | Config | Queries | Exactly 20 | Shortfall queries | Returned-count distribution |
|---|---:|---:|---:|---:|---|
| scifact | B1 | 300 | 300 | 0 | `{20: 300}` |
| scifact | B2 | 300 | 300 | 0 | `{20: 300}` |
| scifact | B3 | 300 | 300 | 0 | `{20: 300}` |
| scifact | B4 | 300 | 300 | 0 | `{20: 300}` |
| nfcorpus | B1 | 323 | 257 | 66 | `{16: 27, 17: 9, 18: 19, 19: 11, 20: 257}` |
| nfcorpus | B2 | 323 | 323 | 0 | `{20: 323}` |
| nfcorpus | B3 | 323 | 323 | 0 | `{20: 323}` |
| nfcorpus | B4 | 323 | 323 | 0 | `{20: 323}` |

## Diagnostics

Each record contains candidate chunk count, parent counts before and after reranking, filter rejections, parent deduplication count, requested depth, returned count, and depth shortfall.

A shortfall is reported honestly; result lists are never padded.

## Metrics

Run `python -m evaluation.analyze_results --run-dir <run-dir>` before using the aggregate metrics from this run.
