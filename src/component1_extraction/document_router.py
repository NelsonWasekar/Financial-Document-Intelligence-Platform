from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import joblib


class DocumentType(str, Enum):
    INVOICE = "invoice"
    BANK_STATEMENT = "bank_statement"
    PURCHASE_ORDER = "purchase_order"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


PIPELINE_MAP = {
    DocumentType.INVOICE: "invoice_pipeline",
    DocumentType.BANK_STATEMENT: "bank_statement_pipeline",
    DocumentType.PURCHASE_ORDER: "purchase_order_pipeline",
    DocumentType.RECEIPT: "receipt_pipeline",
}


SUPPORTED_FORMATS = {
    "pdf",
    "docx",
    "doc",
    "txt",
    "png",
    "jpg",
    "jpeg",
    "tiff",
}


@dataclass
class ExtractionResult:
    """
    Output from the document extraction/ingestion component.
    """

    document_id: str
    file_format: str
    text: str
    content: bytes


@dataclass
class RoutingDecision:
    document_type: DocumentType
    pipeline: str
    confidence: float
    requires_human_review: bool
    review_reason: Optional[str]
    routing_metadata: dict = field(default_factory=dict)


class DuplicateDocumentError(Exception):
    """Raised when the same document content has already been processed."""


class ClassificationError(Exception):
    """Raised when a document cannot be classified."""


class DeadLetterQueue:
    """
    Simple persistent JSONL dead-letter queue.

    Each failed document is appended as one JSON object per line.
    """

    def __init__(self, path: str | Path = "dead_letter_queue.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def push(
        self,
        extraction_result: ExtractionResult,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> None:
        entry = {
            "document_id": extraction_result.document_id,
            "file_format": extraction_result.file_format,
            "reason": reason,
            "metadata": metadata or {},
        }

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


class DeduplicationStore:
    """
    Content-hash based deduplication store.

    A production implementation could replace this with Redis,
    PostgreSQL, DynamoDB, etc.
    """

    def __init__(self):
        self._hashes: set[str] = set()

    @staticmethod
    def content_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def exists(self, content: bytes) -> bool:
        return self.content_hash(content) in self._hashes

    def add(self, content: bytes) -> str:
        content_hash = self.content_hash(content)
        self._hashes.add(content_hash)
        return content_hash


class DocumentClassifier:
    """
    Loads a trained sklearn pipeline and performs classification.
    """

    def __init__(self, model_path: str | Path):
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> tuple[DocumentType, float]:
        if not text or not text.strip():
            raise ClassificationError("Document contains no extractable text.")

        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.classes_

        best_index = probabilities.argmax()
        predicted_class = classes[best_index]
        confidence = float(probabilities[best_index])

        try:
            document_type = DocumentType(predicted_class)
        except ValueError as exc:
            raise ClassificationError(
                f"Unknown classifier output: {predicted_class}"
            ) from exc

        return document_type, confidence


class DocumentRouter:
    """
    Routes documents to the appropriate extraction pipeline.

    Routing rules:
        confidence > 0.85
            -> automatic routing

        0.50 <= confidence <= 0.85
            -> routing with human review

        confidence < 0.50
            -> exception / dead-letter queue
    """

    AUTO_ROUTE_THRESHOLD = 0.85
    HUMAN_REVIEW_THRESHOLD = 0.50

    def __init__(
        self,
        classifier: DocumentClassifier,
        dedup_store: Optional[DeduplicationStore] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
    ):
        self.classifier = classifier
        self.dedup_store = dedup_store or DeduplicationStore()
        self.dead_letter_queue = dead_letter_queue or DeadLetterQueue()

    def route(self, extraction_result: ExtractionResult) -> RoutingDecision:
        self._validate_file_format(extraction_result.file_format)

        # ---------------------------------------------------------
        # 1. Deduplication
        # ---------------------------------------------------------
        content_hash = self.dedup_store.content_hash(
            extraction_result.content
        )

        if self.dedup_store.exists(extraction_result.content):
            raise DuplicateDocumentError(
                f"Duplicate document detected: {content_hash}"
            )

        # Register the document before routing.
        self.dedup_store.add(extraction_result.content)

        # ---------------------------------------------------------
        # 2. Classification
        # ---------------------------------------------------------
        try:
            document_type, confidence = self.classifier.predict(
                extraction_result.text
            )
        except ClassificationError as exc:
            self.dead_letter_queue.push(
                extraction_result,
                reason=str(exc),
                metadata={
                    "content_hash": content_hash,
                },
            )
            raise

        pipeline = PIPELINE_MAP.get(document_type)

        if pipeline is None:
            reason = f"No pipeline configured for {document_type.value}"

            self.dead_letter_queue.push(
                extraction_result,
                reason=reason,
                metadata={
                    "content_hash": content_hash,
                    "confidence": confidence,
                },
            )

            raise ClassificationError(reason)

        # ---------------------------------------------------------
        # 3. Confidence-based routing
        # ---------------------------------------------------------
        if confidence > self.AUTO_ROUTE_THRESHOLD:
            return RoutingDecision(
                document_type=document_type,
                pipeline=pipeline,
                confidence=confidence,
                requires_human_review=False,
                review_reason=None,
                routing_metadata={
                    "routing_status": "auto_routed",
                    "content_hash": content_hash,
                    "file_format": extraction_result.file_format,
                },
            )

        if confidence >= self.HUMAN_REVIEW_THRESHOLD:
            return RoutingDecision(
                document_type=document_type,
                pipeline=pipeline,
                confidence=confidence,
                requires_human_review=True,
                review_reason=(
                    "Classification confidence is between "
                    "0.50 and 0.85."
                ),
                routing_metadata={
                    "routing_status": "human_review",
                    "content_hash": content_hash,
                    "file_format": extraction_result.file_format,
                },
            )

        # ---------------------------------------------------------
        # 4. Exception / DLQ routing
        # ---------------------------------------------------------
        reason = (
            f"Classification confidence below threshold: "
            f"{confidence:.4f}"
        )

        self.dead_letter_queue.push(
            extraction_result,
            reason=reason,
            metadata={
                "content_hash": content_hash,
                "confidence": confidence,
            },
        )

        raise ClassificationError(reason)

    @staticmethod
    def _validate_file_format(file_format: str) -> None:
        normalized = file_format.lower().lstrip(".")

        if normalized not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_format}"
            )