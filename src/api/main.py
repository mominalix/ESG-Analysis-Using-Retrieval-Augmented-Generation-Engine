"""FastAPI application factory and core RAG endpoints."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.config import settings
from ..core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    ESGAnalysisException,
    LLMProviderError,
    VectorStoreError,
)
from ..core.logging import configure_logging, get_logger
from ..core.taxonomy import get_taxonomy
from ..services.analytics_service import analytics_service
from ..services.document_service import document_service
from ..services.llm_service import llm_service
from ..services.rag_service import rag_service
from ..services.vector_store_service import vector_store_service
from .models import (
    DocumentMetadata,
    DocumentResponse,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    RAGQueryRequest,
    RAGResponse,
)
from .routers import admin, analytics, documents


configure_logging()
logger = get_logger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    taxonomy = get_taxonomy()
    logger.info(
        "Starting ESG Analysis Platform",
        version=settings.api_version,
        environment=settings.environment,
        vector_store=settings.vector_store_type,
        llm_providers=llm_service.list_providers(),
        frameworks=taxonomy.framework_ids,
    )
    yield
    logger.info("Shutting down ESG Analysis Platform")


def _error_response(
    request: Request,
    *,
    status_code: int,
    error: str,
    detail: str | None = None,
    error_code: str | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=error,
        detail=detail,
        error_code=error_code,
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def _document_response(document, score: float | None = None, rank: int | None = None):
    metadata = document.metadata
    return DocumentResponse(
        content=document.page_content,
        metadata=DocumentMetadata(
            filename=str(metadata.get("filename", "unknown")),
            mime_type=str(metadata.get("mime_type", "application/octet-stream")),
            file_size_mb=float(metadata.get("file_size_mb", 0.0)),
            document_hash=str(metadata.get("document_hash", "")),
            esg_framework=metadata.get("esg_framework"),
            document_type=metadata.get("document_type"),
            company_id=metadata.get("company_id"),
            esg_category=metadata.get("esg_category"),
            processed_at=str(metadata.get("processed_at", "")),
            chunk_id=str(metadata.get("chunk_id", "")),
        ),
        retrieval_score=score,
        retrieval_rank=rank,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
    )

    if settings.enable_cors and settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )
    if settings.trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(DocumentProcessingError)
    async def document_error(request: Request, exc: DocumentProcessingError):
        return _error_response(
            request,
            status_code=422,
            error="Document processing failed",
            detail=exc.message,
            error_code=exc.error_code or "DOCUMENT_PROCESSING_ERROR",
        )

    async def dependency_error(request: Request, exc: ESGAnalysisException):
        return _error_response(
            request,
            status_code=503,
            error="A required service is unavailable",
            detail=exc.message,
            error_code=exc.error_code or "DEPENDENCY_UNAVAILABLE",
        )

    for exception_type in (VectorStoreError, LLMProviderError, ConfigurationError):
        app.add_exception_handler(exception_type, dependency_error)

    @app.exception_handler(ESGAnalysisException)
    async def esg_error(request: Request, exc: ESGAnalysisException):
        return _error_response(
            request,
            status_code=400,
            error=exc.message,
            error_code=exc.error_code,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception(
            "Unhandled API error",
            request_id=request.state.request_id,
            path=request.url.path,
            error=str(exc),
        )
        return _error_response(
            request,
            status_code=500,
            error="Internal server error",
            detail="Use the request ID when reporting this error",
            error_code="INTERNAL_ERROR",
        )

    app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

    @app.get("/")
    async def root() -> dict[str, Any]:
        taxonomy = get_taxonomy()
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "description": settings.api_description,
            "docs_url": "/docs" if settings.docs_enabled else None,
            "health_check": "/health",
            "frameworks": [framework.model_dump() for framework in taxonomy.frameworks],
            "supported_frameworks": taxonomy.framework_ids,
            "supported_categories": taxonomy.category_ids,
            "document_types": taxonomy.document_types,
            "supported_extensions": list(settings.supported_extensions),
        }

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        vector_status = await vector_store_service.health()
        llm_status = "healthy" if llm_service.list_providers() else "not_configured"
        services = {
            "api": "healthy",
            "document_service": "healthy",
            "vector_store": vector_status,
            "llm_service": llm_status,
        }
        overall = "healthy"
        if vector_status != "healthy":
            overall = "unhealthy"
        elif llm_status != "healthy":
            overall = "degraded"
        return HealthResponse(
            status=overall,
            timestamp=datetime.now(UTC),
            version=settings.api_version,
            services=services,
        )

    @app.post("/api/v1/query", response_model=RAGResponse)
    async def query_rag(request: RAGQueryRequest) -> RAGResponse:
        if request.esg_framework and request.esg_framework not in get_taxonomy().framework_ids:
            raise HTTPException(status_code=422, detail="Unknown ESG framework")

        response = await rag_service.query(
            question=request.question,
            esg_framework=request.esg_framework,
            search_strategy=request.search_strategy.value,
            k=request.k,
            use_query_decomposition=request.use_query_decomposition,
        )
        api_response = RAGResponse(
            answer=response.answer,
            source_documents=[
                _document_response(
                    document,
                    document.metadata.get("retrieval_score"),
                    document.metadata.get("retrieval_rank"),
                )
                for document in response.source_documents
            ],
            confidence_score=response.confidence_score,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=response.total_time_ms,
            esg_framework=response.esg_framework,
            esg_categories=response.esg_categories or [],
        )
        analytics_service.record_query(
            question=request.question,
            framework=response.esg_framework,
            categories=response.esg_categories or [],
            duration_ms=response.total_time_ms,
            confidence=response.confidence_score,
        )
        return api_response

    @app.post("/api/v1/query/stream")
    async def stream_query_rag(request: RAGQueryRequest) -> StreamingResponse:
        async def generate_stream():
            try:
                async for chunk in rag_service.stream_query(
                    question=request.question,
                    esg_framework=request.esg_framework,
                    search_strategy=request.search_strategy.value,
                    k=request.k,
                ):
                    yield f"event: chunk\ndata: {json.dumps({'chunk': chunk})}\n\n"
                yield "event: done\ndata: {}\n\n"
            except Exception as exc:
                logger.exception("Streaming query failed", error=str(exc))
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/v1/upload", response_model=DocumentUploadResponse, status_code=201)
    async def upload_document(
        file: UploadFile = File(...),
        esg_framework: str | None = Form(default=None),
        document_type: str | None = Form(default=None),
        company_id: str | None = Form(default=None),
    ) -> DocumentUploadResponse:
        started = datetime.now(UTC)
        chunks = await document_service.process_uploaded_document(
            file_content=await file.read(),
            filename=file.filename or "",
            esg_framework=esg_framework,
            document_type=document_type,
            company_id=company_id,
        )
        await vector_store_service.add_documents(chunks)
        elapsed_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
        first_chunk = chunks[0]
        return DocumentUploadResponse(
            document_id=str(first_chunk.metadata["document_hash"]),
            filename=str(first_chunk.metadata["filename"]),
            chunks_created=len(chunks),
            processing_time_ms=elapsed_ms,
            metadata=_document_response(first_chunk).metadata,
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=1 if settings.reload else settings.workers,
        log_level=settings.log_level.lower(),
    )
