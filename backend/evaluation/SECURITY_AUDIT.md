# Static security posture audit

## Run

From the repository root:

```bash
cd backend
python -m evaluation.security_posture_audit --root . \
  --output evaluation/results/security_posture_audit.json
```

The command inspects the current API and conversation-store source for a small set of high-risk patterns: wildcard CORS, absence of recognizable FastAPI authentication dependencies, pickle deserialization/serialization, and missing session ownership in the storage interface.

A critical finding produces exit status 1. The JSON report records each finding, severity, file, and recommended action.

## Current source-level findings

The current prototype source contains wildcard CORS, does not show a recognized API authentication dependency, and persists conversation chunks/metadata through Python pickle. Its store interface accepts a session ID without a demonstrated authenticated user/tenant ownership check. These findings are intentionally surfaced by the scanner.

## Scope and limitations

This is an executable static source audit with unit tests, not a penetration test and not a security fix. It does not inspect runtime deployment configuration, reverse proxies, secrets management, TLS termination, database permissions, or actual authorization behavior. A clean scan would not establish that the application is secure.

Before using the prototype with sensitive or multi-user data, implement and test authenticated identity, per-session ownership checks, tenant isolation, explicit CORS allowlists, non-executable schema-validated persistence, encryption, retention/deletion behavior, and prompt-injection/data-exfiltration defenses.
