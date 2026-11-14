#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

RUN_DIR="${1:-evaluation/results/runs/depth_audit_20261114_191803}"
if [[ ! -d "$RUN_DIR/raw" ]]; then
  echo "Raw benchmark directory not found: $RUN_DIR/raw" >&2
  exit 1
fi

if [[ -x "$BACKEND_DIR/venv/bin/python" ]]; then
  PYTHON="$BACKEND_DIR/venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "Analyzing existing raw benchmark only: $RUN_DIR"
echo "No retrieval benchmark or LLM generation will be started."

"$PYTHON" -m evaluation.analyze_results --run-dir "$RUN_DIR"
"$PYTHON" -m evaluation.report_depth_audit --run-dir "$RUN_DIR"

echo
echo "Analysis complete."
echo "Metrics: $RUN_DIR/processed_rank_preserving/official_beir_metrics.json"
echo "Statistics: $RUN_DIR/processed_rank_preserving/paired_significance.json"
echo "Depth report: $RUN_DIR/DEPTH_AUDIT_REPORT_PAPER_MAPPING.md"
