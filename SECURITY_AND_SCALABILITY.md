# Security and Scalability Notes

## Implemented application controls

The implementation branch adds the following baseline controls:

- **Authentication:** private API routes require a signed bearer token. Only `/health` and `/auth/login` are public. Tokens expire after the configured `AUTH_TOKEN_TTL_SECONDS` (default one hour).
- **Credential handling:** passwords are verified using PBKDF2-HMAC-SHA256 with 310,000 iterations and constant-time comparisons. The configured password is not written to the conversation store. The development-only default is `demo / 123456`.
- **Secret management:** production startup requires `AUTH_SECRET_KEY` and `DEMO_PASSWORD`. Supply secrets through the deployment environment or a secret manager; do not commit `.env`.
- **CORS:** browser origins are explicitly allowlisted. Production startup requires `CORS_ALLOWED_ORIGINS`.
- **Conversation ownership:** every conversation record carries a principal ID. Retrieval, listing, deletion, and statistics filter by the authenticated principal. Session identifiers are not treated as authorization.
- **Persistence:** conversation chunks and metadata use JSON rather than Python pickle. JSON is parsed as data and basic record types/counts are validated. Legacy pickle files are intentionally not loaded.
- **Upload limit:** uploads are streamed and rejected above `MAX_UPLOAD_BYTES` (default 10 MiB).

## Current scope and limitations

These controls improve the prototype's baseline posture; they do not establish production-grade security or multi-tenant isolation.

1. Authentication currently uses one configured application account, not a user-registration or enterprise identity system. The user ID is the configured username.
2. The document corpus and document FAISS index remain global and shared by authenticated users. Conversation-memory ownership does not isolate uploaded documents. Do not host mutually untrusted tenants on this configuration.
3. Conversation JSON and FAISS files are not encrypted at rest. Filesystem permissions and encrypted disks/volumes are still required.
4. Tokens are stateless. There is no logout revocation list, refresh-token flow, MFA, password reset, or per-user role/permission model.
5. Login rate limiting, request quotas, audit logging, CSRF protection for cookie auth, malware scanning, and deployment/network hardening are not implemented here.
6. FAISS persistence and in-memory indexes are designed for a single application instance. Multiple workers or replicas must not independently write the same files.
7. Static source checks and unit tests are not a penetration test, threat-model review, or compliance assessment.

## Local development

From `backend/`:

```bash
cp .env.example .env
# Development defaults: demo / 123456
pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8001
```

The frontend login screen uses the same configured credentials. Its bearer token is held in browser `sessionStorage`; this is suitable only as a prototype convenience. A production browser deployment should prefer a same-site, HttpOnly, Secure cookie or a vetted OAuth/OIDC authorization-code flow.

For the browser extension, configure its API base URL and provide an access token after signing in. Do not embed a permanent shared credential in extension source code.

## Production-oriented scaling path

| Layer | Current prototype | Scale-out target |
|---|---|---|
| API | One FastAPI process and in-memory state | Stateless FastAPI replicas behind a load balancer |
| Identity | One configured account and signed bearer token | OIDC provider, per-user identities, roles, token revocation |
| Conversation metadata | Local JSON files | PostgreSQL with user/session foreign keys and transactions |
| Document storage | Local filesystem | Private object storage with tenant-scoped access policies |
| Vector search | Local FAISS index | Managed Qdrant, pgvector, or another shared vector service with tenant filters |
| Ingestion | Synchronous API request | Queue-backed workers, retry policy, idempotent jobs |
| Cache and limits | Process-local cache / no distributed quota | Redis for distributed cache, rate limits, and short-lived coordination |
| Observability | Basic in-process metrics and logs | Centralized structured logs, traces, metrics, alerts, and audit events |

### Recommended request/data flow

1. The API validates identity and derives the principal ID from the verified token; it never trusts a client-supplied user ID.
2. Uploads are validated, stored in private object storage, and queued for extraction/chunking/embedding.
3. Workers write document metadata and vectors to shared stores with tenant/document ACL metadata.
4. Query requests retrieve only vectors and documents authorized for the principal/tenant.
5. Reranking and LLM generation run behind bounded worker pools and provider timeouts, with per-user quotas and cost controls.
6. Results, ownership decisions, failures, and administrative actions are recorded in centralized audit logs.

## Suggested paper wording

The prototype includes bearer-token authentication, password hashing, explicit CORS origins, bounded uploads, non-executable JSON conversation persistence, and principal-scoped conversation-memory operations. These controls were added as application security measures; they were not part of the retrieval-quality benchmark. The document corpus remains shared and the deployment is single-instance, so the implementation should not be described as fully multi-tenant or horizontally scalable. The scale-out architecture above is a proposed deployment design, not a measured result.
