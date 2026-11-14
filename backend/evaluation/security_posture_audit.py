"""Static security posture audit for the Context Rot Lab prototype.

This is a source-level guardrail, not a penetration test or proof of security.
Run from the backend directory:
    python -m evaluation.security_posture_audit --root .
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def inspect_sources(root: Path) -> dict[str, Any]:
    api_path = root / "api.py"
    store_path = root / "conversation_store.py"
    if not api_path.is_file() or not store_path.is_file():
        raise FileNotFoundError(
            "Expected api.py and conversation_store.py under the supplied backend root"
        )

    api = api_path.read_text(encoding="utf-8")
    store = store_path.read_text(encoding="utf-8")
    findings: list[dict[str, str]] = []

    def add(code: str, severity: str, path: str, detail: str, remediation: str) -> None:
        findings.append({
            "id": code,
            "severity": severity,
            "file": path,
            "detail": detail,
            "recommended_action": remediation,
        })

    if re.search(r"allow_origins\s*=\s*\[[^]]*['\"]\*['\"]", api):
        add(
            "CORS_WILDCARD", "high", "api.py",
            "CORS accepts every origin.",
            "Use an explicit environment-configured origin allowlist; disable credentials with wildcard origins.",
        )

    auth_markers = (
        "HTTPBearer(", "OAuth2PasswordBearer(", "APIKeyHeader(",
        "Depends(require_authenticated_user)", "Depends(get_current_user)",
        '@app.middleware("http")', "verify_access_token(",
    )
    if not any(marker in api for marker in auth_markers):
        add(
            "NO_VISIBLE_API_AUTH", "critical", "api.py",
            "No recognized FastAPI authentication dependency is visible in the API module.",
            "Add authenticated identity resolution and enforce authorization on every private route.",
        )

    if "pickle.loads" in store or "pickle.loads(" in store:
        add(
            "PICKLE_DESERIALIZATION", "high", "conversation_store.py",
            "Conversation data is deserialized with pickle; loading attacker-controlled pickle can execute code.",
            "Replace pickle persistence with a non-executable, schema-validated format and restrict filesystem access.",
        )
    if "pickle.dumps" in store:
        add(
            "PICKLE_SERIALIZATION", "medium", "conversation_store.py",
            "Conversation chunks and metadata are persisted using pickle.",
            "Migrate persisted records to JSON/SQLite or another non-executable schema-validated format.",
        )

    if "session_id: str" in store and "user_id" not in store:
        add(
            "SESSION_OWNERSHIP_NOT_MODELED", "high", "conversation_store.py",
            "The store API accepts session IDs but has no user/tenant ownership field in its public storage interface.",
            "Bind each session to an authenticated principal and enforce ownership checks on read, write, and delete.",
        )

    if not findings:
        status = "passed_static_checks"
    elif any(f["severity"] == "critical" for f in findings):
        status = "blocking_findings"
    else:
        status = "review_required"

    return {
        "audit_type": "static_source_review",
        "scope": ["api.py", "conversation_store.py"],
        "status": status,
        "finding_count": len(findings),
        "findings": findings,
        "limitations": [
            "This scanner checks source patterns only; it is not a penetration test.",
            "A missing pattern match does not prove that a control is effective.",
            "Runtime configuration, reverse proxies, deployment identity, secrets, and infrastructure are outside this scan.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Backend source root containing api.py and conversation_store.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON report path")
    args = parser.parse_args()

    report = inspect_sources(args.root.resolve())
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
