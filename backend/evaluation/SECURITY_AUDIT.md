# Static security posture audit

## Run

From the repository root:

```bash
cd backend
python -m evaluation.security_posture_audit --root . \
  --output evaluation/results/security_posture_audit.json
```

The scanner checks source patterns for wildcard CORS, recognizable API authentication, pickle persistence, and user ownership in the conversation store. Any finding produces exit status 1.

## Current implementation branch

The earlier audit findings prompted these code changes:

- Private API routes now require a signed, expiring bearer token; `/health` and `/auth/login` are public.
- The development account defaults to `demo / 123456`; the default is disabled outside development.
- CORS uses explicit configured origins and rejects wildcard origins.
- Conversation memory is scoped to the authenticated principal for retrieval, listing, deletion, and statistics.
- Conversation text and metadata use schema-checked JSON rather than pickle.
- Upload size is bounded and conversation-store operations are synchronized within one process.

The static audit and regression tests have been updated to check for these implementation patterns. They have **not yet been executed against the latest branch commit**. Run `bash evaluation/run_reviewer_checks.sh` from `backend/` before treating the implementation as verified.

## Remaining risks and limitations

This is a source-level guardrail, not a penetration test or proof of security. It does not verify runtime authorization behavior, reverse proxies, TLS termination, secret rotation, database permissions, malware scanning, or infrastructure configuration.

The current identity setup is one configured application account, not a user directory. Conversation memory has principal-level filtering, but the uploaded document corpus remains global/shared. Local conversation data is not encrypted at rest. Token revocation, distributed rate limiting, MFA, enterprise identity, full tenant isolation, retention enforcement, and production deployment hardening remain open.

Do not describe this prototype as fully multi-tenant, security-certified, or production-secure. See `SECURITY_AND_SCALABILITY.md` for the proposed scale-out architecture.
