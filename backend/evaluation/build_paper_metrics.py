"""Build one auditable JSON bundle for standard benchmark paper results.

Inputs are real benchmark outputs only:
- LongMemEval-S runner output JSON
- official LongMemEval GPT-4o judge analysis JSON (optional until judging is run)
- BEIR SciFact/NFCorpus outputs from analyze_results.py

Missing inputs are explicitly marked unavailable; this tool never fabricates scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize_longmemeval(run: dict[str, Any], judged: dict[str, Any] | None) -> dict[str, Any]:
    per_question = run.get("per_question", [])
    if not per_question:
        raise ValueError("LongMemEval runner JSON has no per_question records")
    arms: dict[str, list[dict[str, Any]]] = {}
    for row in per_question:
        arms.setdefault(str(row["arm"]), []).append(row)

    arm_summary = {}
    for arm, rows in arms.items():
        def mean(key: str) -> float | None:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            return sum(values) / len(values) if values else None

        latencies = sorted(float(row["generation_latency_ms"]) for row in rows)
        token_rows = [row.get("tokens_used") or {} for row in rows]
        token_usage = {}
        for token_key in ("prompt", "completion", "total"):
            values = [
                float(item[token_key]) for item in token_rows
                if isinstance(item, dict) and item.get(token_key) is not None
            ]
            token_usage[token_key] = {
                "sum": sum(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
                "observed_calls": len(values),
            }
        arm_summary[arm] = {
            "n": len(rows),
            "automatic_exact_match": mean("exact_match"),
            "automatic_token_f1": mean("token_f1"),
            "mean_prompt_tokens": mean("prompt_tokens"),
            "mean_generation_latency_ms": mean("generation_latency_ms"),
            "mean_retrieval_latency_ms": mean("memory_retrieval_latency_ms"),
            "p50_generation_latency_ms": (
                latencies[int(0.50 * (len(latencies) - 1))] if latencies else None
            ),
            "p95_generation_latency_ms": (
                latencies[int(0.95 * (len(latencies) - 1))] if latencies else None
            ),
            "mean_retrieval_recall_at_k": mean("retrieval_recall_at_k"),
            "mean_retrieval_precision_at_k": mean("retrieval_precision_at_k"),
            "mean_retrieval_mrr_at_k": mean("retrieval_mrr_at_k"),
            "mean_retrieval_ndcg_at_k": mean("retrieval_ndcg_at_k"),
            "mean_retrieval_map_at_k": mean("retrieval_map_at_k"),
            "retrieval_hit_rate_at_k": mean("retrieval_hit_at_k"),
            "evidence_completeness_rate_at_k": mean(
                "retrieval_evidence_completeness_at_k"
            ),
            "mean_context_session_coverage": mean("context_session_coverage"),
            "mean_context_gold_evidence_recall": mean("context_gold_evidence_recall"),
            "context_gold_evidence_complete_rate": mean("context_gold_evidence_complete"),
            "token_usage": token_usage,
            "api_cost": {
                "status": "not_calculated",
                "reason": "No dated provider pricing schedule is supplied to this analysis run.",
            },
            "mean_memory_indexing_latency_ms": mean("memory_indexing_latency_ms"),
            "mean_memory_index_size_bytes": mean("memory_index_size_bytes"),
        }

    judged_summary = None
    if judged is not None:
        judged_summary = {
            "judge": judged.get("judge"),
            "n_questions": judged.get("n_questions"),
            "arms": judged.get("arms"),
            "paired_significance": judged.get("paired_significance"),
            "per_question": judged.get("per_question"),
        }

    return {
        "benchmark": "LongMemEval-S",
        "benchmark_reference": "https://arxiv.org/abs/2410.10813",
        "dataset": run.get("dataset"),
        "dataset_sha256": run.get("dataset_sha256"),
        "n_questions": run.get("n_questions"),
        "expected_questions": 500,
        "coverage_status": (
            "complete" if run.get("n_questions") == 500 else "partial_or_smoke_test"
        ),
        "run_id_utc": run.get("run_id_utc"),
        "reader": {"provider": run.get("provider"), "model": run.get("model")},
        "retrieval_configuration": {
            "method": run.get("retrieval_method"),
            "embedding_model": run.get("embedding_model"),
            "reranker_model": run.get("reranker_model"),
            "candidate_pool_size": run.get("retrieval_candidate_pool_size"),
            "retrieval_k": run.get("retrieval_top_k"),
            "chunk_size_words": run.get("chunk_size_words"),
            "chunk_overlap_words": run.get("chunk_overlap_words"),
        },
        "context_budgets": {
            "standard_prompt_tokens": run.get("max_prompt_tokens"),
            "long_context_prompt_tokens": run.get("long_context_max_prompt_tokens"),
            "history_sizes": run.get("history_sizes"),
            "condition_definitions": run.get("history_condition_definition"),
        },
        "automatic_diagnostics_not_official_score": arm_summary,
        "official_judge_evaluation": (
            judged_summary if judged_summary is not None
            else {
                "status": "not_provided",
                "required_protocol": "Official LongMemEval evaluate_qa.py with GPT-4o judge",
            }
        ),
        "paper_readiness": {
            "full_500_question_run": run.get("n_questions") == 500,
            "official_judge_present": judged is not None,
            "official_judge_full_coverage": (
                judged is not None and judged.get("n_questions") == 500
            ),
        },
        "per_question_raw": per_question,
        "source_file": run.get("_source_file"),
    }


def load_beir(processed_dir: Path | None) -> dict[str, Any]:
    if processed_dir is None:
        return {
            "status": "not_provided",
            "required_datasets": ["BEIR SciFact", "BEIR NFCorpus"],
        }
    metrics_path = processed_dir / "official_beir_metrics.json"
    significance_path = processed_dir / "paired_significance.json"
    per_query_path = processed_dir / "per_query_metrics.json"
    manifest_path = processed_dir / "analysis_manifest.json"
    metrics = load_json(metrics_path, "BEIR aggregate metrics")
    significance = load_json(significance_path, "BEIR paired significance")
    per_query = load_json(per_query_path, "BEIR per-query metrics")
    manifest = load_json(manifest_path, "BEIR analysis manifest")
    expected = {"scifact", "nfcorpus"}
    if set(metrics) != expected or set(per_query) != expected:
        raise ValueError(
            "BEIR bundle must contain exactly the standard SciFact and NFCorpus datasets"
        )
    return {
        "status": "loaded",
        "benchmarks": {
            "scifact": {
                "benchmark": "BEIR SciFact",
                "reference": "https://github.com/beir-cellar/beir",
                "aggregate_metrics": metrics["scifact"],
                "per_query_metrics": per_query["scifact"],
                "paired_significance": significance.get("scifact"),
            },
            "nfcorpus": {
                "benchmark": "BEIR NFCorpus",
                "reference": "https://github.com/beir-cellar/beir",
                "aggregate_metrics": metrics["nfcorpus"],
                "per_query_metrics": per_query["nfcorpus"],
                "paired_significance": significance.get("nfcorpus"),
            },
        },
        "analysis_manifest": manifest,
        "source_files": {
            "aggregate_metrics": str(metrics_path),
            "per_query_metrics": str(per_query_path),
            "paired_significance": str(significance_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longmemeval-json", type=Path, required=True)
    parser.add_argument("--official-longmemeval-analysis", type=Path)
    parser.add_argument(
        "--beir-processed-dir", type=Path,
        help="Directory created by evaluation.analyze_results (processed_rank_preserving)",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/paper_metrics.json"))
    args = parser.parse_args()

    long_path = args.longmemeval_json.resolve()
    long_run = load_json(long_path, "LongMemEval run")
    long_run["_source_file"] = str(long_path)
    judged = (
        load_json(args.official_longmemeval_analysis, "Official LongMemEval judge analysis")
        if args.official_longmemeval_analysis else None
    )
    beir = load_beir(args.beir_processed_dir.resolve() if args.beir_processed_dir else None)

    output = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Auditable metrics bundle for research-paper analysis; unavailable values are never imputed.",
        "provenance": {
            "repository_commit_at_bundle_time": git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "longmemeval_runner_json": str(long_path),
            "longmemeval_runner_json_sha256": sha256_file(long_path),
            "official_judge_analysis_json": (
                str(args.official_longmemeval_analysis.resolve())
                if args.official_longmemeval_analysis else None
            ),
            "beir_processed_dir": (
                str(args.beir_processed_dir.resolve()) if args.beir_processed_dir else None
            ),
        },
        "benchmarks": {
            "longmemeval_s": summarize_longmemeval(long_run, judged),
            "beir": beir,
        },
        "interpretation_guards": [
            "LongMemEval automatic EM/token-F1 are diagnostics; use the official judge score for benchmark answer accuracy.",
            "BEIR SciFact/NFCorpus retrieval metrics are separate from LongMemEval conversational QA metrics.",
            "A benchmark result is reportable only when its full expected query set is present and its provenance is retained.",
            "Smoke tests and partial runs must not be described as full benchmark results.",
            "No API cost is estimated unless exact provider token usage and a dated pricing table are supplied.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "longmemeval_questions": long_run.get("n_questions"),
        "official_judge_included": judged is not None,
        "beir_status": beir.get("status"),
    }, indent=2))
    print(f"PAPER_METRICS_JSON={args.output}")


if __name__ == "__main__":
    main()
