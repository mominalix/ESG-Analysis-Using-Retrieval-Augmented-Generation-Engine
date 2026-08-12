"""Validated public API contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchStrategy(str, Enum):
    SIMILARITY = "similarity"
    HYBRID = "hybrid"


def _optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("Value cannot be blank")
    return normalized


class RAGQueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    esg_framework: str | None = Field(default=None, max_length=80)
    search_strategy: SearchStrategy = SearchStrategy.HYBRID
    k: int = Field(default=5, ge=1, le=50)
    use_query_decomposition: bool = False
    stream: bool = False

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "question": "What evidence supports the climate disclosures in these documents?",
                "esg_framework": "CSRD",
                "search_strategy": "hybrid",
                "k": 5,
            }
        },
    )

    _normalize_framework = field_validator("esg_framework")(_optional_identifier)


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    esg_framework: str | None = Field(default=None, max_length=80)
    esg_category: str | None = Field(default=None, max_length=80)
    document_type: str | None = Field(default=None, max_length=80)
    company_id: str | None = Field(default=None, max_length=200)
    k: int = Field(default=10, ge=1, le=100)
    search_strategy: SearchStrategy = SearchStrategy.SIMILARITY

    model_config = ConfigDict(str_strip_whitespace=True)

    _normalize_framework = field_validator("esg_framework")(_optional_identifier)
    _normalize_category = field_validator("esg_category")(_optional_identifier)
    _normalize_type = field_validator("document_type")(_optional_identifier)
    _normalize_company = field_validator("company_id")(_optional_identifier)


class DocumentMetadata(BaseModel):
    filename: str
    mime_type: str
    file_size_mb: float
    document_hash: str
    esg_framework: str | None = None
    document_type: str | None = None
    company_id: str | None = None
    esg_category: str | None = None
    processed_at: str
    chunk_id: str


class DocumentResponse(BaseModel):
    content: str
    metadata: DocumentMetadata
    retrieval_score: float | None = None
    retrieval_rank: int | None = None


class RAGResponse(BaseModel):
    answer: str
    source_documents: list[DocumentResponse]
    confidence_score: float = Field(ge=0.0, le=1.0)
    retrieval_time_ms: int = Field(ge=0)
    generation_time_ms: int = Field(ge=0)
    total_time_ms: int = Field(ge=0)
    esg_framework: str | None = None
    esg_categories: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int = Field(ge=1)
    processing_time_ms: int = Field(ge=0)
    metadata: DocumentMetadata


class BatchDocumentUploadResponse(BaseModel):
    total_documents: int
    successful_uploads: int
    failed_uploads: int
    document_responses: list[DocumentUploadResponse]
    errors: list[str]
    total_processing_time_ms: int


class SearchResultResponse(BaseModel):
    documents: list[DocumentResponse]
    total_results: int
    search_time_ms: int
    query: str
    search_strategy: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class DocumentStatsResponse(BaseModel):
    total_documents: int
    documents_by_framework: dict[str, int]
    documents_by_category: dict[str, int]
    documents_by_type: dict[str, int]
    total_chunks: int
    average_chunk_size: int
    documents: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: dict[str, str]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    error_code: str | None = None
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UsageStats(BaseModel):
    total_queries: int
    total_documents: int
    average_response_time_ms: float
    most_used_framework: str | None = None
    most_queried_category: str | None = None


class FrameworkStats(BaseModel):
    framework: str
    query_count: int
    document_count: int
    average_confidence: float


class AnalyticsResponse(BaseModel):
    usage_stats: UsageStats
    framework_stats: list[FrameworkStats]
    period_start: datetime
    period_end: datetime
