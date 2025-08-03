"""
Pydantic models for API requests and responses
"""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class ESGFramework(str, Enum):
    """Supported ESG frameworks"""
    CSRD = "CSRD"
    GRI = "GRI"
    SASB = "SASB"
    TCFD = "TCFD"
    EU_TAXONOMY = "EU_Taxonomy"
    SEC_CLIMATE = "SEC_Climate"


class ESGCategory(str, Enum):
    """ESG categories"""
    ENVIRONMENTAL = "Environmental"
    SOCIAL = "Social"
    GOVERNANCE = "Governance"


class DocumentType(str, Enum):
    """Document types"""
    POLICY = "policy"
    REPORT = "report"
    REGULATION = "regulation"
    STANDARD = "standard"
    GUIDANCE = "guidance"


class SearchStrategy(str, Enum):
    """Search strategies"""
    SIMILARITY = "similarity"
    HYBRID = "hybrid"


# Request Models
class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    esg_framework: Optional[ESGFramework] = None
    document_type: Optional[DocumentType] = None
    company_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries"""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    esg_framework: Optional[ESGFramework] = Field(None, description="Specific ESG framework to focus on")
    search_strategy: SearchStrategy = Field(SearchStrategy.HYBRID, description="Search strategy to use")
    k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    use_query_decomposition: bool = Field(False, description="Whether to use query decomposition for complex questions")
    stream: bool = Field(False, description="Whether to stream the response")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What are the key requirements for carbon emission reporting under CSRD?",
                "esg_framework": "CSRD",
                "search_strategy": "hybrid",
                "k": 5,
                "use_query_decomposition": False,
                "stream": False
            }
        }
    )


class DocumentSearchRequest(BaseModel):
    """Request model for document search"""
    query: str = Field(..., min_length=1, max_length=500)
    esg_framework: Optional[ESGFramework] = None
    esg_category: Optional[ESGCategory] = None
    document_type: Optional[DocumentType] = None
    company_id: Optional[str] = None
    k: int = Field(10, ge=1, le=50)
    search_strategy: SearchStrategy = SearchStrategy.SIMILARITY


# Response Models
class DocumentMetadata(BaseModel):
    """Document metadata"""
    filename: str
    mime_type: str
    file_size_mb: float
    document_hash: str
    esg_framework: Optional[str] = None
    document_type: Optional[str] = None
    company_id: Optional[str] = None
    esg_category: Optional[str] = None
    processed_at: str
    chunk_id: str


class DocumentResponse(BaseModel):
    """Document response model"""
    content: str
    metadata: DocumentMetadata
    retrieval_score: Optional[float] = None
    retrieval_rank: Optional[int] = None


class RAGResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str
    source_documents: List[DocumentResponse]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    esg_framework: Optional[str] = None
    esg_categories: Optional[List[str]] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "According to CSRD requirements, companies must report Scope 1, 2, and 3 carbon emissions...",
                "source_documents": [],
                "confidence_score": 0.89,
                "retrieval_time_ms": 245,
                "generation_time_ms": 1820,
                "total_time_ms": 2065,
                "esg_framework": "CSRD",
                "esg_categories": ["Environmental"]
            }
        }
    )


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    filename: str
    chunks_created: int
    processing_time_ms: int
    metadata: DocumentMetadata


class BatchDocumentUploadResponse(BaseModel):
    """Response model for batch document upload"""
    total_documents: int
    successful_uploads: int
    failed_uploads: int
    document_responses: List[DocumentUploadResponse]
    errors: List[str]
    total_processing_time_ms: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]  # service_name -> status
    

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SearchResultResponse(BaseModel):
    """Search result response"""
    documents: List[DocumentResponse]
    total_results: int
    search_time_ms: int
    query: str
    search_strategy: str


# Analytics Models
class UsageStats(BaseModel):
    """Usage statistics"""
    total_queries: int
    total_documents: int
    average_response_time_ms: float
    most_used_framework: Optional[str] = None
    most_queried_category: Optional[str] = None


class FrameworkStats(BaseModel):
    """Framework-specific statistics"""
    framework: str
    query_count: int
    document_count: int
    average_confidence: float


class AnalyticsResponse(BaseModel):
    """Analytics response"""
    usage_stats: UsageStats
    framework_stats: List[FrameworkStats]
    period_start: datetime
    period_end: datetime