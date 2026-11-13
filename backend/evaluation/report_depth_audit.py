"""Write a depth-audit report for one completed timestamped retrieval run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

def _returned_count(item):
    return int(item.get("retrieved_count", item.get("returned_count", 0)))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.run_dir / "raw"
    records = {}
    for dataset in ("scifact", "nfcorpus"):
        for config in ("B1", "B2", "B3", "B4"):
            paths = sorted((raw / dataset).glob(f"{dataset}_{config}_*.json"))
            items = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            records[(dataset, config)] = items

    lines = [
        "# Depth audit report",
        "",
        f"Run directory: `{args.run_dir}`",
        "",
        "Previous fixed-depth results are preserved separately and are not overwritten.",
        "",
        "| Dataset | Config | Queries | Exactly 20 | Shortfall queries | Returned-count distribution |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for (dataset, config), items in records.items():
        counts = Counter(_returned_count(item) for item in items)

        exact = sum(
            _returned_count(item) == item["requested_depth"]
            for item in items
        )
        shortfall = sum(item["depth_shortfall"] > 0 for item in items)
        lines.append(
            f"| {dataset} | {config} | {len(items)} | {exact} | {shortfall} | "
            f"`{dict(sorted(counts.items()))}` |"
        )

    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "Each record contains candidate chunk count, parent counts before and "
            "after reranking, filter rejections, parent deduplication count, "
            "requested depth, returned count, and depth shortfall.",
            "",
            "A shortfall is reported honestly; result lists are never padded.",
            "",
            "## Metrics",
            "",
            "Run `python -m evaluation.analyze_results --run-dir <run-dir>` "
            "before using the aggregate metrics from this run.",
        ]
    )
    official = args.run_dir / "processed" / "official_beir_metrics.json"
    significance = args.run_dir / "processed" / "paired_significance.json"
    if official.exists():
        lines.extend(
            [
                "",
                "## Corrected official metrics",
                "",
                "```json",
                official.read_text(encoding="utf-8").strip(),
                "```",
            ]
        )
    if significance.exists():
        lines.extend(
            [
                "",
                "## Corrected statistical results",
                "",
                "The complete paired bootstrap and Wilcoxon output is in "
                f"`{significance}`.",
            ]
        )
    output = args.run_dir / "DEPTH_AUDIT_REPORT.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
