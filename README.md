# ESG Evidence Studio

An evidence-grounded Retrieval-Augmented Generation (RAG) application for searching ESG documents, inspecting citations, and creating traceable reports. The project combines a FastAPI backend with a Streamlit workspace and supports a zero-infrastructure local retrieval mode plus optional Chroma or Qdrant providers.

## What changed in v3

- Replaced eager service initialization with lazy LLM and vector-store registries.
- Added a real in-memory lexical store so uploads and search work without Docker or an API key.
- Replaced semantic-search-based document listing/statistics with provider-independent document operations.
- Moved ESG framework/category data to [`config/esg_taxonomy.json`](config/esg_taxonomy.json).
- Replaced the 2,600-line Streamlit prototype with a small API-driven frontend.
- Removed fabricated analytics, compliance scores, backups, logs, and local report fallbacks.
- Added typed configuration, constant-time admin token validation, safe error responses, request IDs, CORS controls, tests, linting, CI, and non-root containers.
- Upgraded the active dependency ranges to current major lines verified on PyPI on 2026-08-12.

## Architecture

```text
Streamlit UI
    │ HTTP + SSE
    ▼
FastAPI routes ─── query/report orchestration ─── lazy LLM provider registry
    │
    ├── document extraction (PDF, DOCX, TXT, MD, CSV, XLSX)
    │
    └── vector store registry
          ├── memory  (local lexical retrieval)
          ├── Chroma  (OpenAI embeddings)
          └── Qdrant (OpenAI embeddings)
```

The memory provider is process-local and is intended for development, tests, and small demonstrations. Use a persistent provider in deployments where indexed data must survive restarts.

## Quick start

Requirements: Python 3.11–3.14.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e ".[dev,ui,openai]"
cp .env.example .env       # Windows: Copy-Item .env.example .env
```

The default `VECTOR_STORE_TYPE=memory` requires no infrastructure. Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to enable generated answers. Create a long, random `ADMIN_TOKEN` to enable authenticated report generation.

Run the services in separate terminals:

```bash
uvicorn src.api.main:app --reload
streamlit run ui/main.py
```

- UI: <http://localhost:8501>
- API: <http://localhost:8000>
- OpenAPI: <http://localhost:8000/docs>

## Docker

```bash
docker compose up --build
```

This starts the API and UI with the memory provider. To run Qdrant as well:

```bash
# Set VECTOR_STORE_TYPE=qdrant and OPENAI_API_KEY in .env first
docker compose --profile qdrant up --build
```

## Configuration

All runtime configuration is defined in [`src/core/config.py`](src/core/config.py) and documented in [`.env.example`](.env.example). Important controls:

| Variable | Purpose | Default |
|---|---|---|
| `VECTOR_STORE_TYPE` | `memory`, `chroma`, or `qdrant` | `memory` |
| `DEFAULT_LLM_PROVIDER` | `openai` or `anthropic` | `openai` |
| `OPENAI_MODEL` | Active OpenAI model ID | `gpt-5.6-terra` |
| `ADMIN_TOKEN` | Enables privileged report endpoints | disabled |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | Streamlit local URL |
| `TAXONOMY_PATH` | Framework/category configuration | `config/esg_taxonomy.json` |

Production mode rejects weak/missing admin tokens, wildcard CORS origins, debug mode, and reload mode.

### Adding an ESG framework

Edit [`config/esg_taxonomy.json`](config/esg_taxonomy.json). The API request contracts and frontend selectors use string IDs from that catalog, so adding a framework does not require modifying Python enums or UI lists.

### Adding a provider

Implement `BaseVectorStore` or `BaseLLMProvider`, then register its factory with `VectorStoreService.register_store()` or `LLMService.register_provider()`. Factories are lazy: provider-specific clients and network connections are not created during module import.

## API highlights

```text
GET    /health
GET    /
POST   /api/v1/upload
POST   /api/v1/query
POST   /api/v1/query/stream
POST   /api/v1/documents/search
GET    /api/v1/documents/list
GET    /api/v1/documents/document/{sha256}
DELETE /api/v1/documents/{sha256}
GET    /api/v1/documents/stats
GET    /api/v1/analytics/usage
POST   /api/v1/admin/generate-report
```

Admin routes require `Authorization: Bearer <ADMIN_TOKEN>`. Errors include an `X-Request-ID` response header; logs contain the same ID for investigation.

## Development

```bash
pytest
ruff check src ui tests
ruff format --check src ui tests
```

The test suite forces the memory store and does not call external LLM APIs.

## Dependency policy

[`pyproject.toml`](pyproject.toml) uses bounded compatible ranges instead of freezing every transitive package. Key current lines include FastAPI 0.141, Pydantic 2.13, LangChain Core 1.5, LangChain OpenAI 1.4, Streamlit 1.61, and Qdrant Client 1.19. Dependabot proposes future updates and CI verifies them.

Model choice remains configurable. The default OpenAI integration uses the Responses API and a low reasoning effort; evaluate model quality, latency, and cost on representative ESG questions before production rollout.

## Public release notes

- Never commit `.env`, uploaded documents, vector data, logs, private ESG reports, or API keys.
- The repository currently does not include a license. Choose and add an appropriate open-source license before accepting external reuse or contributions.
- The generated analysis is decision support, not legal, audit, or regulatory advice. Evidence missing from the repository must not be treated as proof of non-compliance.

See [`SECURITY.md`](SECURITY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and [`DEPLOYMENT.md`](DEPLOYMENT.md) before publishing or deploying.
