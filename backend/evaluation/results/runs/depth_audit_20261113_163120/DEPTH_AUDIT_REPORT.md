# Depth audit report

Run directory: `evaluation/results/runs/depth_audit_20261113_163120`

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

## Corrected official metrics

```json
{
  "scifact": {
    "B1": {
      "NDCG@10": 0.59361,
      "Recall@10": 0.72722,
      "P@10": 0.079,
      "MRR@10": 0.55698
    },
    "B2": {
      "NDCG@10": 0.67517,
      "Recall@10": 0.81444,
      "P@10": 0.091,
      "MRR@10": 0.63904
    },
    "B3": {
      "NDCG@10": 0.67834,
      "Recall@10": 0.815,
      "P@10": 0.09033,
      "MRR@10": 0.64407
    },
    "B4": {
      "NDCG@10": 0.64913,
      "Recall@10": 0.78711,
      "P@10": 0.088,
      "MRR@10": 0.61385
    }
  },
  "nfcorpus": {
    "B1": {
      "NDCG@10": 0.27369,
      "Recall@10": 0.13834,
      "P@10": 0.1935,
      "MRR@10": 0.46141
    },
    "B2": {
      "NDCG@10": 0.32632,
      "Recall@10": 0.1701,
      "P@10": 0.24737,
      "MRR@10": 0.5058
    },
    "B3": {
      "NDCG@10": 0.3286,
      "Recall@10": 0.17034,
      "P@10": 0.23653,
      "MRR@10": 0.53325
    },
    "B4": {
      "NDCG@10": 0.3403,
      "Recall@10": 0.16544,
      "P@10": 0.24396,
      "MRR@10": 0.55631
    }
  }
}
```

## Corrected statistical results

The complete paired bootstrap and Wilcoxon output is in `evaluation/results/runs/depth_audit_20261113_163120/processed/paired_significance.json`.
