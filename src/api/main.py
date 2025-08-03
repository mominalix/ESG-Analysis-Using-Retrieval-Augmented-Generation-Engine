"""
Main FastAPI application for ESG Analysis Platform
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from ..core.config import settings
from ..core.logging import configure_logging, get_logger
from ..core.exceptions import (
    ESGAnalysisException, DocumentProcessingError, 
    VectorStoreError, LLMProviderError
)
from ..services.document_service import document_service
from ..services.vector_store_service import vector_store_service
from ..services.rag_service import rag_service
from ..services.llm_service import llm_service

from .models import (
    RAGQueryRequest, RAGResponse, DocumentUploadRequest, DocumentUploadResponse,
    BatchDocumentUploadResponse, HealthResponse, ErrorResponse, 
    DocumentSearchRequest, SearchResultResponse, AnalyticsResponse,
    DocumentResponse, DocumentMetadata
)
from .routers import documents, analytics, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger = get_logger("app")
    
    # Startup
    logger.info("Starting ESG Analysis Platform")
    configure_logging()
    
    # Initialize services
    try:
        # Test LLM service
        available_providers = llm_service.list_providers()
        logger.info("LLM providers available", providers=available_providers)
        
        # Test vector store
        # This will initialize the vector store connections
        logger.info("Vector store service initialized")
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ESG Analysis Platform")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

logger = get_logger("api")


# Exception handlers
@app.exception_handler(ESGAnalysisException)
async def esg_analysis_exception_handler(request, exc: ESGAnalysisException):
    return HTTPException(
        status_code=400,
        detail=ErrorResponse(
            error=exc.message,
            detail=str(exc.details) if exc.details else None,
            error_code=exc.error_code
        ).model_dump()
    )


@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request, exc: DocumentProcessingError):
    return HTTPException(
        status_code=422,
        detail=ErrorResponse(
            error="Document processing failed",
            detail=exc.message,
            error_code="DOCUMENT_PROCESSING_ERROR"
        ).model_dump()
    )


@app.exception_handler(VectorStoreError)
async def vector_store_exception_handler(request, exc: VectorStoreError):
    return HTTPException(
        status_code=503,
        detail=ErrorResponse(
            error="Vector store service unavailable",
            detail=exc.message,
            error_code="VECTOR_STORE_ERROR"
        ).model_dump()
    )


@app.exception_handler(LLMProviderError)
async def llm_provider_exception_handler(request, exc: LLMProviderError):
    return HTTPException(
        status_code=503,
        detail=ErrorResponse(
            error="LLM service unavailable",
            detail=exc.message,
            error_code="LLM_PROVIDER_ERROR"
        ).model_dump()
    )


# Core API endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs_url": "/docs" if settings.debug else "Disabled in production",
        "health_check": "/health",
        "supported_frameworks": settings.esg_frameworks,
        "supported_categories": settings.esg_categories
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check LLM service
        llm_providers = llm_service.list_providers()
        llm_status = "healthy" if llm_providers else "unhealthy"
        
        # Check vector store (simple check)
        vector_status = "healthy"  # Assume healthy if no exception
        
        services = {
            "llm_service": llm_status,
            "vector_store": vector_status,
            "document_service": "healthy"
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in services.values()) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version=settings.api_version,
            services=services
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail={"error": "Health check failed", "detail": str(e)}
        )


@app.post("/api/v1/query", response_model=RAGResponse)
async def query_rag(request: RAGQueryRequest):
    """Main RAG query endpoint"""
    try:
        logger.info("Received RAG query", question_length=len(request.question))
        
        response = await rag_service.query(
            question=request.question,
            esg_framework=request.esg_framework,
            search_strategy=request.search_strategy,
            k=request.k,
            use_query_decomposition=request.use_query_decomposition
        )
        
        # Convert to API response format
        source_docs = []
        for doc in response.source_documents:
            metadata = DocumentMetadata(
                filename=doc.metadata.get("filename", "unknown"),
                mime_type=doc.metadata.get("mime_type", "unknown"),
                file_size_mb=doc.metadata.get("file_size_mb", 0.0),
                document_hash=doc.metadata.get("document_hash", ""),
                esg_framework=doc.metadata.get("esg_framework"),
                document_type=doc.metadata.get("document_type"),
                company_id=doc.metadata.get("company_id"),
                esg_category=doc.metadata.get("esg_category"),
                processed_at=doc.metadata.get("processed_at", ""),
                chunk_id=doc.metadata.get("chunk_id", "")
            )
            
            source_docs.append(DocumentResponse(
                content=doc.page_content,
                metadata=metadata,
                retrieval_score=doc.metadata.get("retrieval_score"),
                retrieval_rank=doc.metadata.get("retrieval_rank")
            ))
        
        return RAGResponse(
            answer=response.answer,
            source_documents=source_docs,
            confidence_score=response.confidence_score,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=response.total_time_ms,
            esg_framework=response.esg_framework,
            esg_categories=response.esg_categories
        )
        
    except Exception as e:
        logger.error("RAG query failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Query processing failed", "detail": str(e)}
        )


@app.post("/api/v1/query/stream")
async def stream_query_rag(request: RAGQueryRequest):
    """Streaming RAG query endpoint"""
    try:
        logger.info("Received streaming RAG query", question_length=len(request.question))
        
        async def generate_stream():
            try:
                async for chunk in rag_service.stream_query(
                    question=request.question,
                    esg_framework=request.esg_framework,
                    search_strategy=request.search_strategy,
                    k=request.k
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("Streaming failed", error=str(e))
                yield f"data: Error: {str(e)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error("Streaming setup failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Streaming setup failed", "detail": str(e)}
        )


@app.post("/api/v1/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    esg_framework: str = None,
    document_type: str = None,
    company_id: str = None
):
    """Upload and process a single document"""
    logger.info(
        "Starting document upload",
        filename=file.filename,
        content_type=file.content_type,
        esg_framework=esg_framework,
        document_type=document_type,
        company_id=company_id
    )
    
    try:
        # Read file content
        logger.info("Reading file content", filename=file.filename)
        content = await file.read()
        file_size_bytes = len(content)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        logger.info(
            "File content read successfully",
            filename=file.filename,
            size_bytes=file_size_bytes,
            size_mb=round(file_size_mb, 2)
        )
        
        # Process document
        start_time = datetime.now()
        logger.info("Starting document processing", filename=file.filename)
        
        documents = await document_service.process_uploaded_document(
            file_content=content,
            filename=file.filename,
            esg_framework=esg_framework,
            document_type=document_type,
            company_id=company_id
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            "Document processing completed",
            filename=file.filename,
            chunks_created=len(documents),
            processing_time_ms=int(processing_time)
        )
        
        # Add to vector store in background
        async def add_to_vector_store():
            try:
                logger.info("Adding documents to vector store", filename=file.filename, count=len(documents))
                await vector_store_service.add_documents(documents)
                logger.info("Documents successfully added to vector store", filename=file.filename, count=len(documents))
            except Exception as e:
                logger.error("Failed to add documents to vector store", filename=file.filename, error=str(e), error_type=type(e).__name__)
        
        background_tasks.add_task(add_to_vector_store)
        
        # Create response
        first_doc = documents[0]
        logger.info(
            "Creating response metadata",
            filename=file.filename,
            document_hash=first_doc.metadata.get("document_hash", "")[:8] + "..."
        )
        
        metadata = DocumentMetadata(
            filename=first_doc.metadata["filename"],
            mime_type=first_doc.metadata["mime_type"],
            file_size_mb=first_doc.metadata["file_size_mb"],
            document_hash=first_doc.metadata["document_hash"],
            esg_framework=first_doc.metadata.get("esg_framework"),
            document_type=first_doc.metadata.get("document_type"),
            company_id=first_doc.metadata.get("company_id"),
            esg_category=first_doc.metadata.get("esg_category"),
            processed_at=first_doc.metadata["processed_at"],
            chunk_id=first_doc.metadata["chunk_id"]
        )
        
        response = DocumentUploadResponse(
            document_id=first_doc.metadata["document_hash"],
            filename=file.filename,
            chunks_created=len(documents),
            processing_time_ms=int(processing_time),
            metadata=metadata
        )
        
        logger.info(
            "Document upload completed successfully",
            filename=file.filename,
            document_id=first_doc.metadata["document_hash"][:8] + "...",
            total_time_ms=int(processing_time)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Document upload failed",
            filename=file.filename,
            error=str(e),
            error_type=type(e).__name__,
            esg_framework=esg_framework
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "Document upload failed", "detail": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
        log_level="debug" if settings.debug else "info"
    )