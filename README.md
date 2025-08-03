# ESG Analysis Platform - Advanced RAG for Sustainability Reporting

## Overview

A next-generation ESG (Environmental, Social, Governance) analysis platform that leverages cutting-edge Retrieval-Augmented Generation (RAG) technology to revolutionize sustainability reporting and compliance analysis. Built for the 2025 regulatory landscape including CSRD, EU Taxonomy, and other evolving ESG frameworks.

## 🌍 ESG Framework Coverage

### Supported Frameworks

| Framework | Coverage | Status |
|-----------|----------|--------|
| **CSRD** | Complete ESRS standards | ✅ Full |
| **GRI** | Universal Standards 2021 | ✅ Full |
| **SASB** | Industry standards | ✅ Full |
| **TCFD** | 2023 framework | ✅ Full |
| **EU Taxonomy** | Environmental objectives | ✅ Full |
| **SEC Climate** | Proposed rules | ✅ Full |

### ESG Categories

- **Environmental**: Climate, pollution, water, biodiversity, circular economy
- **Social**: Workforce, value chain, communities, consumers
- **Governance**: Business conduct, risk management, ethics
## Key Features

### Modern RAG Architecture
- **Hierarchical Retrieval**: Multi-stage document retrieval with re-ranking
- **Query Decomposition**: Break complex ESG questions into sub-queries
- **Hybrid Search**: Combines semantic and keyword search for optimal results
- **Dynamic Context**: Adaptive context window based on query complexity

### ESG-Specific Intelligence
- **Framework Compliance**: CSRD, GRI, SASB, TCFD, EU Taxonomy, SEC Climate
- **Gap Analysis**: Automated identification of compliance gaps and risks
- **Best Practices**: AI-powered recommendations based on industry standards
- **Double Materiality**: Support for impact and financial materiality assessment

### Enterprise-Ready Infrastructure
- **Microservices Architecture**: Scalable, containerized services
- **Multiple Vector Stores**: ChromaDB, Pinecone, Qdrant support
- **Comprehensive Monitoring**: LangSmith integration with structured logging
- **Production Security**: API authentication, rate limiting, data encryption

### Modern User Experience
- **RESTful API**: Comprehensive API with OpenAPI documentation
- **Streaming Responses**: Real-time response generation
- **Batch Processing**: Efficient bulk document processing
- **Interactive UI**: Modern Streamlit interface with analytics dashboard

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ESG Analysis Platform                   │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Frontend UI   │   FastAPI Core  │    Background Services  │
│                 │                 │                         │
│ • Streamlit UI  │ • RAG Service   │ • Document Processing   │
│ • REST API      │ • Query Engine  │ • Vector Indexing       │
│ • Admin Panel   │ • Auth & Rate   │ • Monitoring & Logging  │
│                 │   Limiting      │                         │
├─────────────────┼─────────────────┼─────────────────────────┤
│   Vector Store  │   LLM Services  │    Data Storage         │
│                 │                 │                         │
│ • ChromaDB      │ • OpenAI GPT-4  │ • PostgreSQL            │
│ • Pinecone      │ • Anthropic     │ • Redis Cache           │
│ • Qdrant        │ • Claude        │ • File Storage          │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/mominalix/ESG-Analysis-Using-Retrieval-Augmented-Generation-Engine.git
cd ESG-Analysis-Using-Retrieval-Augmented-Generation-Engine

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# Start all services
docker-compose up -d

# Access the application
# API: http://localhost:8000
# UI: http://localhost:8501
# Grafana: http://localhost:3000 (admin/admin)
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run the Streamlit UI (optional)
streamlit run ui/main.py --server.port 8501
```

## Project Structure

```
ESG-Analysis-Platform/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # Main API application
│   │   ├── models.py            # Pydantic models
│   │   └── routers/             # API route modules
│   ├── core/                     # Core utilities
│   │   ├── config.py            # Configuration management
│   │   ├── logging.py           # Structured logging
│   │   └── exceptions.py        # Custom exceptions
│   └── services/                 # Business logic services
│       ├── llm_service.py       # LLM provider management
│       ├── vector_store_service.py # Vector database management
│       ├── document_service.py  # Document processing
│       └── rag_service.py       # Advanced RAG implementation
├── sample_data/                  # Sample ESG documents and frameworks
│   ├── frameworks/              # ESG framework documents
│   ├── policies/                # Sample company policies
│   └── reports/                 # Example ESG reports
├── ui/                          # Streamlit user interface
├── monitoring/                  # Monitoring and observability
├── requirements.txt             # Python dependencies
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Main application container
└── README.md                   # This file
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# LLM Configuration
OPENAI_API_KEY="your-openai-api-key"
DEFAULT_MODEL="gpt-4o-mini"
TEMPERATURE=0.1

# Vector Database
VECTOR_DB_PROVIDER="chroma"  # chroma, pinecone, qdrant
CHROMA_PERSIST_DIRECTORY="./data/vector_db"

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENT_SIZE_MB=50

# Monitoring
LANGSMITH_API_KEY="your-langsmith-key"  # Optional
ENABLE_TRACING=true
```

### Vector Database Options

#### ChromaDB (Default)
- **Best for**: Local development, small to medium datasets
- **Setup**: No additional configuration required
- **Storage**: Local file system

#### Pinecone
- **Best for**: Production, large-scale deployments
- **Setup**: Requires Pinecone API key and index creation
- **Storage**: Cloud-managed

#### Qdrant
- **Best for**: Self-hosted production, hybrid deployments
- **Setup**: Included in docker-compose
- **Storage**: Self-managed

## API Reference

### Core Endpoints

#### Query ESG Documents
```bash
POST /api/v1/query
{
  "question": "What are CSRD climate disclosure requirements?",
  "esg_framework": "CSRD",
  "search_strategy": "hybrid",
  "k": 5,
  "use_query_decomposition": false
}
```

#### Upload Documents
```bash
POST /api/v1/upload
Content-Type: multipart/form-data
- file: <ESG document>
- esg_framework: "CSRD"
- document_type: "policy"
```

#### Document Search
```bash
POST /api/v1/documents/search
{
  "query": "climate risk assessment",
  "esg_framework": "TCFD",
  "k": 10
}
```

#### System Health
```bash
GET /health
```

### API Documentation

When running in development mode, access interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage Examples

### Upload Sample Documents

```bash
# Upload CSRD framework document
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_data/frameworks/csrd/csrd_overview.md" \
  -F "esg_framework=CSRD" \
  -F "document_type=standard"

# Batch upload company policies
curl -X POST "http://localhost:8000/api/v1/documents/batch-upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample_data/policies/environmental_management_policy.md" \
  -F "esg_framework=Multi-Framework" \
  -F "document_type=policy"
```

### Query Examples

#### Basic ESG Query
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key climate disclosure requirements under CSRD?",
    "esg_framework": "CSRD",
    "search_strategy": "hybrid",
    "k": 5
  }'
```

#### Complex Query with Decomposition
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do our environmental policies compare against CSRD requirements and what gaps exist in our climate risk reporting?",
    "esg_framework": "CSRD",
    "use_query_decomposition": true,
    "k": 10
  }'
```

#### Streaming Response
```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Provide a comprehensive analysis of TCFD climate risk disclosure requirements",
    "esg_framework": "TCFD"
  }'
```

## Monitoring and Observability

### Built-in Monitoring

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Performance Metrics**: Response times, confidence scores, retrieval accuracy
- **Health Checks**: Service health monitoring and dependency checks
- **Error Tracking**: Comprehensive error logging and alerting

### LangSmith Integration

Configure LangSmith for advanced tracing:

```bash
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_PROJECT="esg-analysis-platform"
export ENABLE_TRACING=true
```

### Grafana Dashboards

Access monitoring dashboards at http://localhost:3000 (admin/admin):
- API Performance Metrics
- Document Processing Statistics
- Query Analysis Trends
- System Resource Usage

## Security Considerations

### Production Deployment

1. **API Keys**: Use environment variables or secret management
2. **Authentication**: Implement proper JWT or OAuth2
3. **Rate Limiting**: Configure appropriate limits for your use case
4. **HTTPS**: Enable TLS encryption for all external communications
5. **Data Privacy**: Ensure compliance with GDPR and other regulations

### Recommended Security Headers

```nginx
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options DENY;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000";
```


## Integration Options

### API Integration

```python
import requests

# Initialize ESG Analysis client
class ESGAnalysisClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def query(self, question, framework=None):
        response = requests.post(
            f"{self.base_url}/api/v1/query",
            json={
                "question": question,
                "esg_framework": framework
            }
        )
        return response.json()

# Usage
client = ESGAnalysisClient()
result = client.query(
    "What are the mandatory climate disclosures?",
    framework="CSRD"
)
```

### Webhook Integration

```python
# Webhook endpoint for document processing
@app.post("/webhooks/document-processed")
async def handle_document_processed(payload: dict):
    # Handle completed document processing
    document_id = payload["document_id"]
    status = payload["status"]
    
    if status == "completed":
        # Trigger downstream processes
        await trigger_compliance_analysis(document_id)
```

## Scalability Options

### Horizontal Scaling

1. **API Scaling**: Multiple FastAPI instances behind load balancer
2. **Vector Database**: Distributed vector stores (Pinecone, Weaviate clusters)
3. **Background Processing**: Celery workers for document processing
4. **Caching**: Redis clusters for response caching

### Performance Optimization

1. **Chunking Strategy**: Optimize chunk size and overlap for your use case
2. **Embedding Caching**: Cache embeddings to reduce computation
3. **Query Caching**: Cache frequent query results
4. **Async Processing**: Leverage async/await for I/O operations


### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Code formatting
black src/
isort src/

# Type checking
mypy src/
```


---

**Built with ❤️ for a sustainable future**
