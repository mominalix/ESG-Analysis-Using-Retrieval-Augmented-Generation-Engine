"""
Document Processing Service for ESG Analysis Platform
"""
import asyncio
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional, BinaryIO, Union
from datetime import datetime
from pathlib import Path
import tempfile
import os

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from unstructured.partition.auto import partition
from langsmith import traceable

from ..core.config import settings
from ..core.logging import LoggingMixin
from ..core.exceptions import DocumentProcessingError


class DocumentProcessor(LoggingMixin):
    """Advanced document processor with multiple format support"""
    
    def __init__(self):
        self.supported_types = {
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'text/plain': self._process_text,
            'text/markdown': self._process_markdown,
            'text/csv': self._process_csv,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._process_excel,
            'application/vnd.ms-excel': self._process_excel,
        }
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    @traceable(name="process_document")
    async def process_document(
        self, 
        file_content: bytes, 
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process document and return chunks"""
        try:
            # Validate file size
            file_size_bytes = len(file_content)
            file_size_mb = file_size_bytes / (1024 * 1024)
            max_size_mb = settings.max_file_size / (1024 * 1024)
            if file_size_bytes > settings.max_file_size:
                raise DocumentProcessingError(
                    f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb:.1f}MB)"
                )
            
            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                # Fallback to extension-based detection
                ext = Path(filename).suffix.lower()
                mime_type = self._get_mime_from_extension(ext)
            
            self.logger.info(
                "Processing document", 
                filename=filename, 
                mime_type=mime_type,
                size_mb=file_size_mb
            )
            
            # Check if file type is supported
            if mime_type not in self.supported_types:
                raise DocumentProcessingError(f"Unsupported file type: {mime_type}")
            
            # Process document based on type
            documents = await self.supported_types[mime_type](file_content, filename)
            
            # Add metadata to all documents
            base_metadata = {
                "filename": filename,
                "mime_type": mime_type,
                "file_size_mb": file_size_mb,
                "document_hash": hashlib.md5(file_content).hexdigest(),
                **(metadata or {})
            }
            
            for doc in documents:
                doc.metadata.update(base_metadata)
            
            self.logger.info(
                "Document processed successfully", 
                filename=filename,
                chunks_created=len(documents)
            )
            
            return documents
            
        except Exception as e:
            self.logger.error("Document processing failed", filename=filename, error=str(e))
            raise DocumentProcessingError(f"Failed to process {filename}: {str(e)}")
    
    def _get_mime_from_extension(self, ext: str) -> str:
        """Get MIME type from file extension"""
        ext_to_mime = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
        }
        return ext_to_mime.get(ext, 'application/octet-stream')
    
    async def _process_pdf(self, file_content: bytes, filename: str) -> List[Document]:
        """Process PDF files"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            try:
                loader = PyPDFLoader(temp_file.name)
                documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                
                # Split documents into chunks
                chunks = self.text_splitter.split_documents(documents)
                return chunks
            finally:
                os.unlink(temp_file.name)
    
    async def _process_docx(self, file_content: bytes, filename: str) -> List[Document]:
        """Process DOCX files"""
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            try:
                loader = Docx2txtLoader(temp_file.name)
                documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                
                chunks = self.text_splitter.split_documents(documents)
                return chunks
            finally:
                os.unlink(temp_file.name)
    
    async def _process_text(self, file_content: bytes, filename: str) -> List[Document]:
        """Process plain text files"""
        text_content = file_content.decode('utf-8')
        
        document = Document(
            page_content=text_content,
            metadata={"source": filename}
        )
        
        chunks = self.text_splitter.split_documents([document])
        return chunks
    
    async def _process_markdown(self, file_content: bytes, filename: str) -> List[Document]:
        """Process Markdown files"""
        text_content = file_content.decode('utf-8')
        
        document = Document(
            page_content=text_content,
            metadata={"source": filename}
        )
        
        chunks = self.markdown_splitter.split_documents([document])
        return chunks
    
    async def _process_csv(self, file_content: bytes, filename: str) -> List[Document]:
        """Process CSV files"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            try:
                loader = CSVLoader(temp_file.name)
                documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                
                # For CSV, we might want different chunking strategy
                chunks = self.text_splitter.split_documents(documents)
                return chunks
            finally:
                os.unlink(temp_file.name)
    
    async def _process_excel(self, file_content: bytes, filename: str) -> List[Document]:
        """Process Excel files"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            try:
                loader = UnstructuredExcelLoader(temp_file.name)
                documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                
                chunks = self.text_splitter.split_documents(documents)
                return chunks
            finally:
                os.unlink(temp_file.name)


class DocumentService(LoggingMixin):
    """Document service with enhanced processing capabilities"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
    
    @traceable(name="process_uploaded_document")
    async def process_uploaded_document(
        self,
        file_content: bytes,
        filename: str,
        esg_framework: Optional[str] = None,
        document_type: Optional[str] = None,
        company_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process uploaded document with ESG-specific metadata"""
        
        # Enhance metadata with ESG-specific information (filter out None values)
        esg_metadata = {
            "processed_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Only add non-None values to metadata
        if esg_framework:
            esg_metadata["esg_framework"] = esg_framework
        if document_type:
            esg_metadata["document_type"] = document_type
        if company_id:
            esg_metadata["company_id"] = company_id
        
        documents = await self.processor.process_document(
            file_content, filename, esg_metadata
        )
        
        # Add ESG-specific enrichment
        for doc in documents:
            doc.metadata["chunk_id"] = f"{filename}_{hash(doc.page_content)}"
            
            # Detect ESG categories if not specified
            if not doc.metadata.get("esg_category"):
                detected_category = self._detect_esg_category(doc.page_content)
                if detected_category:  # Only add if not None/empty
                    doc.metadata["esg_category"] = detected_category
            
            # Clean metadata to remove None values before vector store
            doc.metadata = self._clean_metadata(doc.metadata)
        
        return documents
    
    def _detect_esg_category(self, content: str) -> str:
        """Simple ESG category detection based on keywords"""
        content_lower = content.lower()
        
        environmental_keywords = [
            "carbon", "emission", "climate", "energy", "waste", "water", 
            "renewable", "sustainability", "environmental", "greenhouse"
        ]
        
        social_keywords = [
            "employee", "diversity", "inclusion", "safety", "community",
            "human rights", "labor", "social", "workforce", "training"
        ]
        
        governance_keywords = [
            "governance", "board", "ethics", "compliance", "risk",
            "audit", "transparency", "management", "executive", "policy"
        ]
        
        env_count = sum(1 for keyword in environmental_keywords if keyword in content_lower)
        social_count = sum(1 for keyword in social_keywords if keyword in content_lower)
        gov_count = sum(1 for keyword in governance_keywords if keyword in content_lower)
        
        if env_count > social_count and env_count > gov_count:
            return "Environmental"
        elif social_count > gov_count:
            return "Social"
        else:
            return "Governance"
    
    async def batch_process_documents(
        self,
        files: List[tuple[bytes, str]],  # List of (content, filename) tuples
        **common_metadata: Any
    ) -> List[Document]:
        """Process multiple documents in batch"""
        self.logger.info("Starting batch document processing", file_count=len(files))
        
        all_documents = []
        
        # Process documents in parallel
        tasks = [
            self.process_uploaded_document(content, filename, **common_metadata)
            for content, filename in files
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "Failed to process document in batch", 
                    filename=files[i][1], 
                    error=str(result)
                )
            else:
                all_documents.extend(result)
        
        self.logger.info(
            "Batch processing completed", 
            total_documents=len(all_documents),
            successful_files=len([r for r in results if not isinstance(r, Exception)])
        )
        
        return all_documents
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """Clean metadata to remove None values and ensure compatibility with vector stores"""
        cleaned = {}
        for key, value in metadata.items():
            if value is not None:
                # Convert to acceptable types for vector stores
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                else:
                    # Convert other types to string
                    cleaned[key] = str(value)
        return cleaned


# Global document service instance
document_service = DocumentService()