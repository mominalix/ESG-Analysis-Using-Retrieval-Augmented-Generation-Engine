# Deployment guide

## Production prerequisites

1. Choose a persistent vector store (`chroma` or `qdrant`).
2. Configure the corresponding provider extra and service.
3. Store LLM keys and `ADMIN_TOKEN` in a secrets manager, not an image or Compose file.
4. Set `ENVIRONMENT=production`, `DEBUG=false`, `RELOAD=false`, and explicit `ALLOWED_ORIGINS`.
5. Terminate TLS at a trusted reverse proxy and set `TRUSTED_HOSTS`.
6. Decide how documents, metadata, analytics, and generated reports must be retained under your privacy policy.

## Container build

```bash
docker build -t esg-analysis-api:3.0.0 .
docker build -f Dockerfile.streamlit -t esg-analysis-ui:3.0.0 .
```

Both images run as the unprivileged user ID 10001 and include health checks.

## Qdrant deployment

Set:

```dotenv
VECTOR_STORE_TYPE=qdrant
QDRANT_URL=https://your-qdrant-host:6333
QDRANT_API_KEY=...
OPENAI_API_KEY=...
```

Qdrant document enumeration relies on the standard LangChain payload keys `page_content` and `metadata`. Test upload, list, detail, search, and delete operations against the exact Qdrant version used in production.

## Health and readiness

`GET /health` returns:

- `healthy`: vector store and LLM are ready;
- `degraded`: document/search features work but no LLM provider is configured;
- `unhealthy`: the configured vector store is unavailable.

Use the service-level values rather than treating any HTTP 200 as full readiness.

## Scaling notes

- The memory store and analytics collector are process-local. Do not use multiple API workers with either if cross-worker consistency matters.
- Chroma persistence is suitable for a single writer unless your deployment architecture explicitly supports otherwise.
- For horizontally scaled APIs, use Qdrant and replace `AnalyticsService` with a durable telemetry/database adapter.
- Upload indexing is synchronous so a successful response means the document is searchable. Move indexing to a durable queue only when job state, retries, and failure reporting are implemented together.

## Rollback

Pin container image digests, keep vector-store backups, and test schema/client compatibility before upgrades. Rolling back application code does not automatically roll back vector collection configuration or embeddings; embedding model changes should normally use a new collection.
