"""Custom exceptions for papra-llm-manager."""


class PapraError(Exception):
    """Base exception for papra-llm-manager errors."""

    pass


class ProcessingError(PapraError):
    """Exception for document processing errors."""

    def __init__(self, message: str, document_id: str | None = None):
        self.document_id = document_id
        super().__init__(message)


class TextExtractionError(ProcessingError):
    """Exception for text extraction errors."""

    def __init__(self, message: str, document_id: str | None = None, reason: str | None = None):
        self.reason = reason
        super().__init__(message, document_id)


class TaggingError(PapraError):
    """Exception for tagging errors."""

    def __init__(self, message: str, document_id: str | None = None):
        self.document_id = document_id
        super().__init__(message)


class LLMProviderError(PapraError):
    """Exception for LLM provider errors."""

    def __init__(self, message: str, provider: str | None = None):
        self.provider = provider
        super().__init__(message)


class ValidationError(PapraError):
    """Exception for validation errors."""

    def __init__(self, message: str, field: str | None = None):
        self.field = field
        super().__init__(message)
