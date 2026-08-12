"""
Custom exceptions for ESG Analysis Platform
"""

from typing import Optional, Dict, Any


class ESGAnalysisException(Exception):
    """Base exception for ESG Analysis Platform"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DocumentProcessingError(ESGAnalysisException):
    """Raised when document processing fails"""

    pass


class VectorStoreError(ESGAnalysisException):
    """Raised when vector store operations fail"""

    pass


class LLMProviderError(ESGAnalysisException):
    """Raised when LLM provider operations fail"""

    pass


class ESGFrameworkError(ESGAnalysisException):
    """Raised when ESG framework validation fails"""

    pass


class ConfigurationError(ESGAnalysisException):
    """Raised when configuration is invalid"""

    pass


class AuthenticationError(ESGAnalysisException):
    """Raised when authentication fails"""

    pass


class RateLimitError(ESGAnalysisException):
    """Raised when rate limits are exceeded"""

    pass


class ValidationError(ESGAnalysisException):
    """Raised when data validation fails"""

    pass
