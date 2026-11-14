#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Security posture source audit =="
python -m evaluation.security_posture_audit --root . --output evaluation/results/security_posture_audit.json || audit_status=$?
audit_status=${audit_status:-0}

echo
echo "== Targeted regression tests =="
python -m pytest -q tests/test_auth.py tests/test_conversation_store_security.py tests/test_conversation_store.py tests/test_security_posture_audit.py tests/test_reranker_trace.py tests/test_retrieval_trace.py tests/test_context_quality_metrics.py

if [[ "$audit_status" -ne 0 ]]; then
  echo "Security audit reported findings; review the JSON report."
  exit "$audit_status"
fi

echo "Checks completed. Static audit passed; this is not a penetration test."
