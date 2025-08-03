"""
Document management API routes
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, BackgroundTasks
from datetime import datetime

from ...core.logging import get_logger
from ...services.document_service import document_service
from ...services.vector_store_service import vector_store_service

from ..models import (
    DocumentSearchRequest, SearchResultResponse, DocumentResponse,
    BatchDocumentUploadResponse, DocumentUploadResponse, DocumentMetadata
)

router = APIRouter()
logger = get_logger("documents_api")


@router.post("/search", response_model=SearchResultResponse)
async def search_documents(request: DocumentSearchRequest):
    """Search documents in the vector store"""
    try:
        logger.info("Document search request", query=request.query)
        
        start_time = datetime.now()
        
        # Build filter
        filter_dict = {}
        if request.esg_framework:
            filter_dict["esg_framework"] = request.esg_framework
        if request.esg_category:
            filter_dict["esg_category"] = request.esg_category
        if request.document_type:
            filter_dict["document_type"] = request.document_type
        if request.company_id:
            filter_dict["company_id"] = request.company_id
        
        # Perform search
        search_results = await vector_store_service.search(
            query=request.query,
            k=request.k,
            search_type=request.search_strategy,
            filter_dict=filter_dict if filter_dict else None
        )
        
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Convert to response format
        documents = []
        for result in search_results:
            metadata = DocumentMetadata(
                filename=result.document.metadata.get("filename", "unknown"),
                mime_type=result.document.metadata.get("mime_type", "unknown"),
                file_size_mb=result.document.metadata.get("file_size_mb", 0.0),
                document_hash=result.document.metadata.get("document_hash", ""),
                esg_framework=result.document.metadata.get("esg_framework"),
                document_type=result.document.metadata.get("document_type"),
                company_id=result.document.metadata.get("company_id"),
                esg_category=result.document.metadata.get("esg_category"),
                processed_at=result.document.metadata.get("processed_at", ""),
                chunk_id=result.document.metadata.get("chunk_id", "")
            )
            
            documents.append(DocumentResponse(
                content=result.document.page_content,
                metadata=metadata,
                retrieval_score=result.score,
                retrieval_rank=result.rank
            ))
        
        return SearchResultResponse(
            documents=documents,
            total_results=len(documents),
            search_time_ms=int(search_time),
            query=request.query,
            search_strategy=request.search_strategy
        )
        
    except Exception as e:
        logger.error("Document search failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Document search failed", "detail": str(e)}
        )


@router.post("/batch-upload", response_model=BatchDocumentUploadResponse)
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    esg_framework: Optional[str] = Query(None),
    document_type: Optional[str] = Query(None),
    company_id: Optional[str] = Query(None)
):
    """Upload multiple documents in batch"""
    try:
        logger.info("Batch document upload", file_count=len(files))
        
        start_time = datetime.now()
        document_responses = []
        errors = []
        successful_uploads = 0
        
        # Process files
        file_contents = []
        for file in files:
            try:
                content = await file.read()
                file_contents.append((content, file.filename))
            except Exception as e:
                logger.error("Failed to read file", filename=file.filename, error=str(e))
                errors.append(f"Failed to read {file.filename}: {str(e)}")
        
        # Process documents in batch
        try:
            all_documents = await document_service.batch_process_documents(
                files=file_contents,
                esg_framework=esg_framework,
                document_type=document_type,
                company_id=company_id
            )
            
            # Group documents by filename for response
            docs_by_file = {}
            for doc in all_documents:
                filename = doc.metadata["filename"]
                if filename not in docs_by_file:
                    docs_by_file[filename] = []
                docs_by_file[filename].append(doc)
            
            # Create responses for each file
            for filename, docs in docs_by_file.items():
                first_doc = docs[0]
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
                
                document_responses.append(DocumentUploadResponse(
                    document_id=first_doc.metadata["document_hash"],
                    filename=filename,
                    chunks_created=len(docs),
                    processing_time_ms=0,  # Individual processing time not tracked in batch
                    metadata=metadata
                ))
                successful_uploads += 1
            
            # Add to vector store in background
            async def add_batch_to_vector_store():
                try:
                    await vector_store_service.add_documents(all_documents)
                    logger.info("Batch documents added to vector store", count=len(all_documents))
                except Exception as e:
                    logger.error("Failed to add batch documents to vector store", error=str(e))
            
            background_tasks.add_task(add_batch_to_vector_store)
            
        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
            errors.append(f"Batch processing failed: {str(e)}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchDocumentUploadResponse(
            total_documents=len(files),
            successful_uploads=successful_uploads,
            failed_uploads=len(files) - successful_uploads,
            document_responses=document_responses,
            errors=errors,
            total_processing_time_ms=int(processing_time)
        )
        
    except Exception as e:
        logger.error("Batch upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Batch upload failed", "detail": str(e)}
        )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the vector store"""
    try:
        logger.info("Document deletion request", document_id=document_id)
        
        success = await vector_store_service.delete_documents([document_id])
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail={"error": "Document not found", "document_id": document_id}
            )
            
    except Exception as e:
        logger.error("Document deletion failed", document_id=document_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Document deletion failed", "detail": str(e)}
        )


@router.get("/list")
async def list_documents(
    limit: int = Query(50, description="Maximum number of documents to return"),
    offset: int = Query(0, description="Number of documents to skip"),
    esg_framework: Optional[str] = Query(None, description="Filter by ESG framework"),
    document_type: Optional[str] = Query(None, description="Filter by document type")
):
    """List all documents with pagination and filtering"""
    try:
        logger.info("Listing documents", limit=limit, offset=offset)
        
        # Build filter
        filter_dict = {}
        if esg_framework:
            filter_dict["esg_framework"] = esg_framework
        if document_type:
            filter_dict["document_type"] = document_type
        
        # Get documents from vector store
        # For now, we'll use a search with a very broad query to get all documents
        search_results = await vector_store_service.search(
            query="document",  # Broad query to match most documents
            k=limit + offset,  # Get more than needed for pagination
            search_type="similarity",
            filter_dict=filter_dict if filter_dict else None
        )
        
        # Apply pagination
        paginated_results = search_results[offset:offset + limit]
        
        # Convert to response format
        documents = []
        seen_documents = set()  # Track unique documents by hash
        
        for result in paginated_results:
            doc_hash = result.document.metadata.get("document_hash", "")
            if doc_hash and doc_hash not in seen_documents:
                seen_documents.add(doc_hash)
                
                metadata = DocumentMetadata(
                    filename=result.document.metadata.get("filename", "unknown"),
                    mime_type=result.document.metadata.get("mime_type", "unknown"),
                    file_size_mb=result.document.metadata.get("file_size_mb", 0.0),
                    document_hash=doc_hash,
                    esg_framework=result.document.metadata.get("esg_framework"),
                    document_type=result.document.metadata.get("document_type"),
                    company_id=result.document.metadata.get("company_id"),
                    esg_category=result.document.metadata.get("esg_category"),
                    processed_at=result.document.metadata.get("processed_at", ""),
                    chunk_id=result.document.metadata.get("chunk_id", "")
                )
                
                documents.append(DocumentResponse(
                    content=result.document.page_content[:200] + "..." if len(result.document.page_content) > 200 else result.document.page_content,
                    metadata=metadata,
                    retrieval_score=result.score,
                    retrieval_rank=result.rank
                ))
        
        return {
            "documents": documents,
            "total": len(search_results),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(search_results)
        }
        
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to list documents", "detail": str(e)}
        )


@router.get("/document/{document_hash}")
async def get_document_details(document_hash: str):
    """Get detailed information about a specific document"""
    try:
        logger.info("Getting document details", document_hash=document_hash)
        
        # Search for all chunks of this document
        search_results = await vector_store_service.search(
            query="document",
            k=100,  # Get many results to find all chunks
            search_type="similarity",
            filter_dict={"document_hash": document_hash}
        )
        
        if not search_results:
            raise HTTPException(
                status_code=404,
                detail={"error": "Document not found", "document_hash": document_hash}
            )
        
        # Get document metadata from first chunk
        first_chunk = search_results[0].document
        
        metadata = DocumentMetadata(
            filename=first_chunk.metadata.get("filename", "unknown"),
            mime_type=first_chunk.metadata.get("mime_type", "unknown"),
            file_size_mb=first_chunk.metadata.get("file_size_mb", 0.0),
            document_hash=document_hash,
            esg_framework=first_chunk.metadata.get("esg_framework"),
            document_type=first_chunk.metadata.get("document_type"),
            company_id=first_chunk.metadata.get("company_id"),
            esg_category=first_chunk.metadata.get("esg_category"),
            processed_at=first_chunk.metadata.get("processed_at", ""),
            chunk_id=first_chunk.metadata.get("chunk_id", "")
        )
        
        # Prepare chunks
        chunks = []
        for result in search_results:
            chunks.append({
                "chunk_id": result.document.metadata.get("chunk_id", ""),
                "content": result.document.page_content,
                "score": result.score
            })
        
        return {
            "metadata": metadata,
            "total_chunks": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        logger.error("Failed to get document details", document_hash=document_hash, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get document details", "detail": str(e)}
        )


@router.get("/stats")
async def get_document_stats():
    """Get real document statistics from vector store"""
    try:
        logger.info("Getting document statistics")
        
        # Get all documents (using broad search)
        all_results = await vector_store_service.search(
            query="document",
            k=1000,  # Large number to get most documents
            search_type="similarity"
        )
        
        # Calculate statistics
        unique_documents = {}
        frameworks = {}
        categories = {}
        doc_types = {}
        total_chunks = len(all_results)
        total_content_length = 0
        
        for result in all_results:
            doc_hash = result.document.metadata.get("document_hash", "")
            if doc_hash:
                if doc_hash not in unique_documents:
                    unique_documents[doc_hash] = {
                        "filename": result.document.metadata.get("filename", "unknown"),
                        "framework": result.document.metadata.get("esg_framework"),
                        "category": result.document.metadata.get("esg_category"),
                        "doc_type": result.document.metadata.get("document_type"),
                        "chunks": 0
                    }
                
                unique_documents[doc_hash]["chunks"] += 1
                
                # Count frameworks
                framework = result.document.metadata.get("esg_framework")
                if framework:
                    frameworks[framework] = frameworks.get(framework, 0) + 1
                
                # Count categories  
                category = result.document.metadata.get("esg_category")
                if category:
                    categories[category] = categories.get(category, 0) + 1
                
                # Count document types
                doc_type = result.document.metadata.get("document_type")
                if doc_type:
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                total_content_length += len(result.document.page_content)
        
        avg_chunk_size = total_content_length // total_chunks if total_chunks > 0 else 0
        
        return {
            "total_documents": len(unique_documents),
            "documents_by_framework": frameworks,
            "documents_by_category": categories,
            "documents_by_type": doc_types,
            "total_chunks": total_chunks,
            "average_chunk_size": avg_chunk_size,
            "documents": list(unique_documents.values())
        }
        
    except Exception as e:
        logger.error("Failed to get document statistics", error=str(e))
        return {
            "total_documents": 0,
            "documents_by_framework": {},
            "documents_by_category": {},
            "documents_by_type": {},
            "total_chunks": 0,
            "average_chunk_size": 0,
            "error": str(e)
        }