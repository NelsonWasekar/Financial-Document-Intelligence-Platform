"""Domain-specific exceptions shared by the extraction and RAG components."""

from __future__ import annotations


class FinancialSystemError(Exception):
    """Base class for expected, application-level failures."""


# Component 1: document extraction
class DocumentProcessingError(FinancialSystemError):
    """Base class for document ingestion and extraction failures."""


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when an uploaded document format is not supported."""


class DocumentTooLargeError(DocumentProcessingError):
    """Raised when an uploaded document exceeds the configured size limit."""


class CorruptedDocumentError(DocumentProcessingError):
    """Raised when a document cannot be parsed because its contents are invalid."""


class ExtractionFailedError(DocumentProcessingError):
    """Raised when supported content cannot yield usable extracted text."""


# Component 2: RAG pipeline
class IndexingError(FinancialSystemError):
    """Raised when corpus indexing or index persistence fails."""


class RetrievalError(FinancialSystemError):
    """Raised when relevant policy content cannot be retrieved safely."""


class LLMError(FinancialSystemError):
    """Raised when the language-model service is unavailable or invalid."""


class CitationValidationError(FinancialSystemError):
    """Raised when generated citations do not match retrieved policy sources."""
