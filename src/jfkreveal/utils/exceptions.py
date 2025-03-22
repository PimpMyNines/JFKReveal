"""
Custom exceptions for the JFKReveal application.

This module defines a hierarchy of exceptions specific to different components
and error cases in the application, enabling more precise error handling.
"""


class JFKRevealError(Exception):
    """Base class for all JFKReveal exceptions."""
    pass


# Credential Exceptions
class CredentialError(JFKRevealError):
    """Base class for credential-related exceptions."""
    pass


class MissingCredentialError(CredentialError):
    """Raised when a required credential is missing."""
    pass


class InvalidCredentialError(CredentialError):
    """Raised when a credential is invalid or expired."""
    pass


class CredentialValidationError(CredentialError):
    """Raised when credential validation fails."""
    pass


# API Exceptions
class APIError(JFKRevealError):
    """Base class for API-related exceptions."""
    pass


class RateLimitError(APIError):
    """Raised when an API rate limit is exceeded."""
    
    def __init__(self, message="API rate limit exceeded", retry_after=None, *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.retry_after = retry_after


class ModelUnavailableError(APIError):
    """Raised when a requested LLM model is unavailable."""
    
    def __init__(self, model_name=None, available_models=None, *args, **kwargs):
        message = f"Model {model_name} is unavailable"
        super().__init__(message, *args, **kwargs)
        self.model_name = model_name
        self.available_models = available_models


class ContentPolicyError(APIError):
    """Raised when content is rejected due to content policy."""
    pass


class ContextLengthError(APIError):
    """Raised when input exceeds model's context length."""
    pass


# File and I/O Exceptions
class FileError(JFKRevealError):
    """Base class for file-related exceptions."""
    pass


class PDFExtractionError(FileError):
    """Raised when text extraction from a PDF fails."""
    
    def __init__(self, filename=None, message=None, *args, **kwargs):
        if not message:
            message = f"Failed to extract text from PDF: {filename}"
        super().__init__(message, *args, **kwargs)
        self.filename = filename


class FileCorruptionError(FileError):
    """Raised when a file is corrupted and cannot be processed."""
    pass


class FilePermissionError(FileError):
    """Raised when file operations fail due to permission issues."""
    pass


class DiskSpaceError(FileError):
    """Raised when disk space is insufficient for an operation."""
    pass


# Processing Exceptions
class ProcessingError(JFKRevealError):
    """Base class for processing-related exceptions."""
    pass


class OCRError(ProcessingError):
    """Raised when OCR processing fails."""
    
    def __init__(self, page_num=None, filename=None, *args, **kwargs):
        message = f"OCR failed for page {page_num} in {filename}"
        super().__init__(message, *args, **kwargs)
        self.page_num = page_num
        self.filename = filename


class TesseractNotFoundError(OCRError):
    """Raised when Tesseract OCR executable is not found."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(message="Tesseract OCR executable not found. Please install Tesseract.", 
                         *args, **kwargs)


class ChunkingError(ProcessingError):
    """Raised when document chunking fails."""
    pass


class TextCleaningError(ProcessingError):
    """Raised when text cleaning fails."""
    pass


# Network Exceptions
class NetworkError(JFKRevealError):
    """Base class for network-related exceptions."""
    pass


class ConnectionTimeoutError(NetworkError):
    """Raised when a network connection times out."""
    pass


class ConnectionFailedError(NetworkError):
    """Raised when a network connection fails."""
    pass


class DownloadError(NetworkError):
    """Raised when a file download fails."""
    
    def __init__(self, url=None, status_code=None, *args, **kwargs):
        message = f"Download failed for {url} with status code {status_code}"
        super().__init__(message, *args, **kwargs)
        self.url = url
        self.status_code = status_code


# Vector Database Exceptions
class VectorDBError(JFKRevealError):
    """Base class for vector database exceptions."""
    pass


class EmbeddingError(VectorDBError):
    """Raised when document embedding fails."""
    pass


class VectorSearchError(VectorDBError):
    """Raised when vector search fails."""
    pass


class DBConnectionError(VectorDBError):
    """Raised when connection to the vector database fails."""
    pass


# Analysis Exceptions
class AnalysisError(JFKRevealError):
    """Base class for analysis-related exceptions."""
    pass


class LLMResponseError(AnalysisError):
    """Raised when an LLM response is invalid or cannot be parsed."""
    pass


class StructuredOutputError(AnalysisError):
    """Raised when structured output parsing fails."""
    pass


class AnalysisTimeoutError(AnalysisError):
    """Raised when analysis times out."""
    pass


# Reporting Exceptions
class ReportingError(JFKRevealError):
    """Base class for reporting-related exceptions."""
    pass


class ReportGenerationError(ReportingError):
    """Raised when report generation fails."""
    pass


class TemplateRenderingError(ReportingError):
    """Raised when template rendering fails."""
    pass


# Pipeline Exceptions
class PipelineError(JFKRevealError):
    """Base class for pipeline-related exceptions."""
    pass


class PipelineConfigError(PipelineError):
    """Raised when pipeline configuration is invalid."""
    pass


class PipelineExecutionError(PipelineError):
    """Raised when pipeline execution fails."""
    
    def __init__(self, step=None, component=None, *args, **kwargs):
        message = f"Pipeline execution failed at step: {step} in component: {component}"
        super().__init__(message, *args, **kwargs)
        self.step = step
        self.component = component


class CircuitBreakerError(PipelineError):
    """Raised when a circuit breaker is triggered due to too many failures."""
    
    def __init__(self, component=None, failure_count=None, *args, **kwargs):
        message = f"Circuit breaker triggered for {component} after {failure_count} failures"
        super().__init__(message, *args, **kwargs)
        self.component = component
        self.failure_count = failure_count