"""
Configuration management for ESG Analysis Platform
"""
from functools import lru_cache
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Configuration
    api_title: str = "ESG Analysis RAG Platform"
    api_version: str = "2.0.0"
    api_description: str = "Advanced ESG reporting analysis using RAG and AI"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # Security
    secret_key: str = Field(
        default="change-this-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API Key")
    default_llm_provider: str = "openai"
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Vector Database Configuration
    vector_store_type: str = "QDRANT"  # CHROMA, PINECONE, QDRANT
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "esg_documents"
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "esg-analysis"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "esg_documents"
    
    # Embedding Configuration
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_document: int = 100
    supported_extensions: str = ".pdf,.docx,.txt,.md,.csv,.xlsx"
    upload_dir: str = "./uploads"
    max_file_size: int = 10485760  # 10MB in bytes
    
    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "esg_analysis"
    postgres_user: str = "esg_user"
    postgres_password: str = "esg_password"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # RAG Configuration
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    enable_query_decomposition: bool = True
    enable_hybrid_search: bool = True
    enable_reranking: bool = False
    
    # Monitoring and Observability
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "esg-analysis"
    langsmith_tracing_v2: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # Development Settings
    enable_cors: bool = True
    allowed_origins: str = "http://localhost:3000,http://localhost:8501"
    
    # Docker Configuration
    postgres_password_docker: str = "esg_password"
    redis_password_docker: str = "redis_password"
    
    # Streamlit UI
    streamlit_port: int = 8501
    streamlit_host: str = "0.0.0.0"
    
    # Production Settings
    environment: str = "development"
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    behind_proxy: bool = False
    trusted_hosts: Optional[str] = None
    
    # ESG Specific Configuration
    # We use a different name with `_env` suffix to avoid Pydantic's restriction on leading underscores
    esg_frameworks_env: str = Field(default="", alias="ESG_FRAMEWORKS")
    esg_categories_env: str = Field(default="", alias="ESG_CATEGORIES")

    @property
    def esg_frameworks(self) -> List[str]:
        if not self.esg_frameworks_env:
            return ["CSRD", "GRI", "SASB", "TCFD", "EU_Taxonomy", "SEC_Climate"]
        return [item.strip() for item in self.esg_frameworks_env.split(',')]

    @property
    def esg_categories(self) -> List[str]:
        if not self.esg_categories_env:
            return ["Environmental", "Social", "Governance"]
        return [item.strip() for item in self.esg_categories_env.split(',')]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Export commonly used settings
settings = get_settings()