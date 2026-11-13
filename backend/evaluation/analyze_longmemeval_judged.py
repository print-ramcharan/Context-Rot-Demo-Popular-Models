"""Analyze official LongMemEval judge outputs across the four experimental arms.

Expected input files are produced by the official LongMemEval evaluate_qa.py:
hypotheses_{history_size}_{arm}_{run_id}.jsonl.eval-results-gpt-4o
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from evaluation.metrics.statistical_tests import (
    paired_bootstrap_ci,
    wilcoxon_signed_rank_test,
    holm_bonferroni,
)

ARMS = ("question_only", "recent_history", "long_context", "external_memory")


def read_judged(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            if qid in rows:
                raise ValueError(f"Duplicate question_id {qid} in {path}")
            label = row.get("autoeval_label", {})
            if label.get("model") != "gpt-4o-2024-08-06":
                raise ValueError(
                    f"{path}: expected official gpt-4o-2024-08-06 judge, "
                    f"got {label.get('model')!r}"
                )
            if not isinstance(label.get("label"), bool):
                raise ValueError(f"{path}:{line_number}: missing boolean autoeval_label")
            rows[qid] = {
                "correct": int(label["label"]),
                "question_type": row.get("question_type"),
            }
    if not rows:
        raise ValueError(f"No judged rows in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--history-size", default="all")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/data/longmemeval_s_cleaned.json"))
    args = parser.parse_args()

    references = json.loads(args.dataset.read_text(encoding="utf-8"))
    expected = {str(row["question_id"]): row for row in references}
    judged = {}
    for arm in ARMS:
        path = args.run_dir / (
            f"hypotheses_{args.history_size}_{arm}_{args.run_id}.jsonl.eval-results-gpt-4o"
        )
        judged[arm] = read_judged(path)
        found = set(judged[arm])
        if found != set(expected):
            raise ValueError(
                f"{arm}: expected {len(expected)} judged questions, got {len(found)}; "
                f"missing={len(set(expected)-found)}, extra={len(found-set(expected))}"
            )

    summary = {
        "benchmark": "LongMemEval-S",
        "judge": "gpt-4o-2024-08-06",
        "history_size": args.history_size,
        "run_id": args.run_id,
        "n_questions": len(expected),
        "arms": {},
        "paired_significance": {},
    }
    for arm in ARMS:
        values = [judged[arm][qid]["correct"] for qid in expected]
        categories = {}
        for qtype in sorted({row.get("question_type", "unknown") for row in expected.values()}):
            qids = [
                qid for qid, row in expected.items()
                if row.get("question_type", "unknown") == qtype
            ]
            cats = [judged[arm][qid]["correct"] for qid in qids]
            categories[qtype] = {
                "correct": sum(cats),
                "n": len(cats),
                "accuracy": statistics.mean(cats) if cats else None,
            }
        abstention_ids = [
            qid for qid, row in expected.items()
            if "abstention" in str(row.get("question_type", "")).lower()
            or "_abs" in qid.lower()
        ]
        abstention_values = [judged[arm][qid]["correct"] for qid in abstention_ids]
        summary["arms"][arm] = {
            "correct": sum(values),
            "n": len(values),
            "accuracy": statistics.mean(values),
            "category_accuracy": categories,
            "abstention_accuracy": (
                statistics.mean(abstention_values) if abstention_values else None
            ),
            "abstention_n": len(abstention_values),
        }

    # Preserve the official per-question judgments in the same analysis JSON.
    # These are the auditable observations used for all aggregate statistics.
    summary["per_question"] = {
        qid: {
            "question_type": expected[qid].get("question_type"),
            **{
                arm: {
                    "correct": judged[arm][qid]["correct"],
                    "judge_model": "gpt-4o-2024-08-06",
                }
                for arm in ARMS
            },
        }
        for qid in sorted(expected)
    }

    all_tests = []
    for baseline in ("question_only", "recent_history", "long_context"):
        ids = sorted(expected)
        baseline_values = [judged[baseline][qid]["correct"] for qid in ids]
        external_values = [judged["external_memory"][qid]["correct"] for qid in ids]
        # Positive values mean external memory improves accuracy over the baseline.
        differences = [external - base for base, external in zip(baseline_values, external_values)]
        test = wilcoxon_signed_rank_test(baseline_values, external_values)
        result = {
            "comparison": f"external_memory_vs_{baseline}",
            "n_questions": len(ids),
            "accuracy_difference": statistics.mean(differences),
            "bootstrap_ci_95": list(paired_bootstrap_ci(differences)),
            "wilcoxon_statistic": test[0] if test else None,
            "wilcoxon_p_value": test[1] if test else None,
        }
        summary["paired_significance"][result["comparison"]] = result
        all_tests.append(result)

    adjusted = holm_bonferroni([row["wilcoxon_p_value"] for row in all_tests])
    for row, p_adjusted in zip(all_tests, adjusted):
        row["wilcoxon_p_value_holm"] = p_adjusted
        row["significant_holm_0_05"] = p_adjusted is not None and p_adjusted < 0.05

    output = args.run_dir / f"longmemeval_official_analysis_{args.run_id}.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"OFFICIAL_ANALYSIS={output}")


if __name__ == "__main__":
    main()
