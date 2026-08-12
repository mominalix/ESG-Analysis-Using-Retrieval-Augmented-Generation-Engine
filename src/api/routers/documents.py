"""Document search, indexing, inspection, and deletion routes."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from langchain_core.documents import Document

from ...services.document_service import document_service
from ...services.vector_store_service import vector_store_service
from ..models import (
    BatchDocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentResponse,
    DocumentSearchRequest,
    DocumentStatsResponse,
    DocumentUploadResponse,
    SearchResultResponse,
)


router = APIRouter()


def _metadata(document: Document) -> DocumentMetadata:
    metadata = document.metadata
    return DocumentMetadata(
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
    )


def _document_response(
    document: Document,
    *,
    score: float | None = None,
    rank: int | None = None,
    preview: bool = False,
) -> DocumentResponse:
    content = document.page_content
    if preview and len(content) > 300:
        content = f"{content[:300].rstrip()}…"
    return DocumentResponse(
        content=content,
        metadata=_metadata(document),
        retrieval_score=score,
        retrieval_rank=rank,
    )


@router.post("/search", response_model=SearchResultResponse)
async def search_documents(request: DocumentSearchRequest) -> SearchResultResponse:
    started = datetime.now(UTC)
    filters = {
        key: value
        for key, value in {
            "esg_framework": request.esg_framework,
            "esg_category": request.esg_category,
            "document_type": request.document_type,
            "company_id": request.company_id,
        }.items()
        if value
    }
    results = await vector_store_service.search(
        query=request.query,
        k=request.k,
        search_type=request.search_strategy.value,
        filter_dict=filters or None,
    )
    elapsed_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
    return SearchResultResponse(
        documents=[
            _document_response(result.document, score=result.score, rank=result.rank)
            for result in results
        ],
        total_results=len(results),
        search_time_ms=elapsed_ms,
        query=request.query,
        search_strategy=request.search_strategy.value,
    )


@router.post("/batch-upload", response_model=BatchDocumentUploadResponse)
async def batch_upload_documents(
    files: list[UploadFile] = File(...),
    esg_framework: str | None = Query(default=None),
    document_type: str | None = Query(default=None),
    company_id: str | None = Query(default=None),
) -> BatchDocumentUploadResponse:
    started = datetime.now(UTC)
    file_contents = [(await file.read(), file.filename or "") for file in files]
    chunks, errors = await document_service.batch_process_documents(
        files=file_contents,
        esg_framework=esg_framework,
        document_type=document_type,
        company_id=company_id,
    )

    if chunks:
        await vector_store_service.add_documents(chunks)

    chunks_by_document: dict[str, list[Document]] = {}
    for chunk in chunks:
        document_hash = str(chunk.metadata["document_hash"])
        chunks_by_document.setdefault(document_hash, []).append(chunk)

    elapsed_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
    responses = [
        DocumentUploadResponse(
            document_id=document_hash,
            filename=str(document_chunks[0].metadata["filename"]),
            chunks_created=len(document_chunks),
            processing_time_ms=elapsed_ms,
            metadata=_metadata(document_chunks[0]),
        )
        for document_hash, document_chunks in chunks_by_document.items()
    ]
    return BatchDocumentUploadResponse(
        total_documents=len(files),
        successful_uploads=len(responses),
        failed_uploads=len(files) - len(responses),
        document_responses=responses,
        errors=errors,
        total_processing_time_ms=elapsed_ms,
    )


@router.delete("/{document_hash}", status_code=status.HTTP_200_OK)
async def delete_document(document_hash: str) -> dict[str, str]:
    if not await vector_store_service.delete_document(document_hash):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted", "document_id": document_hash}


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    esg_framework: str | None = Query(default=None),
    document_type: str | None = Query(default=None),
) -> DocumentListResponse:
    filters = {
        key: value
        for key, value in {
            "esg_framework": esg_framework,
            "document_type": document_type,
        }.items()
        if value
    }
    documents, total = await vector_store_service.list_documents(
        limit=limit,
        offset=offset,
        filter_dict=filters or None,
    )
    return DocumentListResponse(
        documents=[_document_response(document, preview=True) for document in documents],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(documents) < total,
    )


@router.get("/document/{document_hash}")
async def get_document_details(document_hash: str) -> dict:
    chunks = await vector_store_service.get_document(document_hash)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "metadata": _metadata(chunks[0]),
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "chunk_index": chunk.metadata.get("chunk_index", index),
                "content": chunk.page_content,
            }
            for index, chunk in enumerate(chunks)
        ],
    }


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats() -> DocumentStatsResponse:
    return DocumentStatsResponse.model_validate(await vector_store_service.stats())
