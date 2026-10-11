from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DocumentType(str, Enum):
    """Coarse document-type classification used by the extraction pipeline
    (Component 1.1). Kept separate from the router's own routing-oriented
    DocumentType in document_router.py, which models a different, wider
    set of pipeline destinations (invoice/purchase_order/receipt/etc.)."""

    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    TRANSACTION_LOG = "transaction_log"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    PDF_TEXT = "pdf_text"
    PDF_OCR = "pdf_ocr"
    EXCEL = "excel"
    PLAIN_TEXT = "plain_text"


class PDFClassification(str, Enum):
    TEXT_NATIVE = "text_native"
    SCANNED = "scanned"
    MIXED = "mixed"


@dataclass
class ExtractionResult:
    """Standardised output of DocumentProcessor.process() (Component 1.1)."""

    document_type: DocumentType
    extraction_method: ExtractionMethod
    raw_text: str
    page_count: int
    confidence: float
    processing_time_ms: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Citation:
    document_name: str
    section: str
    page: int
    excerpt: str
    relevance_score: float


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_name: str
    section: str
    page: int
    text: str
    score: float
    retrieval_method: str = ""


@dataclass
class ParentSection:
    section_id: str
    document_name: str
    section: str
    page_start: int
    page_end: int
    text: str


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    document_name: str
    section: str
    page: int
    paragraph_index: int
    text: str


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    confidence: float
    retrieved_chunks: list[RetrievedChunk]
    sufficiency_check_passed: bool
    faithfulness_score: float
    query_id: str
    processing_time_ms: float


@dataclass
class IndexingReport:
    documents_processed: int
    sections_created: int
    chunks_created: int
    changed_documents: list[str]
    unchanged_documents: list[str]
    indexing_time_ms: float


@dataclass
class AuditEvent:
    query_id: str
    user_id: str
    question: str
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    sufficiency_check_passed: bool
    faithfulness_score: float
    confidence: float
    processing_time_ms: float


@dataclass
class PolicyViolation:
    """A transaction-level breach identified by the research agent."""

    transaction_id: str
    violation_type: str
    severity: str
    description: str
    applicable_policy: str
    policy_section: str
