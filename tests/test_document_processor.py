import json

import pytest

from src.component1_extraction.document_router import (
    ClassificationError,
    DeadLetterQueue,
    DeduplicationStore,
    DocumentRouter,
    DocumentType,
    DuplicateDocumentError,
    ExtractionResult,
)


class FakeClassifier:
    def __init__(self, document_type, confidence):
        self.document_type = document_type
        self.confidence = confidence

    def predict(self, text):
        if not text:
            raise ClassificationError("No text")

        return self.document_type, self.confidence


def make_document(
    document_id="doc-1",
    text="invoice number 123 total amount 500",
    content=b"unique-content",
):
    return ExtractionResult(
        document_id=document_id,
        file_format="pdf",
        text=text,
        content=content,
    )


def make_router(
    document_type=DocumentType.INVOICE,
    confidence=0.90,
    dlq_path=None,
):
    classifier = FakeClassifier(
        document_type=document_type,
        confidence=confidence,
    )

    dlq = DeadLetterQueue(
        dlq_path or "dead_letter_queue_test.jsonl"
    )

    return DocumentRouter(
        classifier=classifier,
        dedup_store=DeduplicationStore(),
        dead_letter_queue=dlq,
    )


def test_high_confidence_auto_routes(tmp_path):
    router = make_router(
        confidence=0.90,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    decision = router.route(make_document())

    assert decision.document_type == DocumentType.INVOICE
    assert decision.pipeline == "invoice_pipeline"
    assert decision.confidence == 0.90
    assert decision.requires_human_review is False
    assert decision.review_reason is None
    assert decision.routing_metadata["routing_status"] == "auto_routed"


def test_confidence_exactly_above_auto_threshold(tmp_path):
    router = make_router(
        confidence=0.8501,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    decision = router.route(make_document())

    assert decision.requires_human_review is False
    assert decision.pipeline == "invoice_pipeline"


def test_confidence_exactly_085_requires_review(tmp_path):
    router = make_router(
        confidence=0.85,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    decision = router.route(make_document())

    assert decision.requires_human_review is True
    assert decision.pipeline == "invoice_pipeline"
    assert decision.review_reason is not None


def test_medium_confidence_requires_human_review(tmp_path):
    router = make_router(
        confidence=0.70,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    decision = router.route(make_document())

    assert decision.requires_human_review is True
    assert decision.routing_metadata["routing_status"] == "human_review"


def test_confidence_exactly_050_requires_review(tmp_path):
    router = make_router(
        confidence=0.50,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    decision = router.route(make_document())

    assert decision.requires_human_review is True


def test_low_confidence_goes_to_dead_letter_queue(tmp_path):
    dlq_path = tmp_path / "dlq.jsonl"

    router = make_router(
        confidence=0.49,
        dlq_path=dlq_path,
    )

    with pytest.raises(ClassificationError):
        router.route(make_document())

    assert dlq_path.exists()

    entries = [
        json.loads(line)
        for line in dlq_path.read_text().splitlines()
    ]

    assert len(entries) == 1
    assert entries[0]["document_id"] == "doc-1"
    assert "confidence" in entries[0]["metadata"]


def test_empty_document_goes_to_dead_letter_queue(tmp_path):
    dlq_path = tmp_path / "dlq.jsonl"

    router = make_router(
        confidence=0.90,
        dlq_path=dlq_path,
    )

    document = make_document(text="")

    with pytest.raises(ClassificationError):
        router.route(document)

    assert dlq_path.exists()

    entries = [
        json.loads(line)
        for line in dlq_path.read_text().splitlines()
    ]

    assert len(entries) == 1
    assert "no text" in entries[0]["reason"].lower()


def test_duplicate_content_is_rejected(tmp_path):
    router = make_router(
        confidence=0.90,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    document_1 = make_document(
        document_id="doc-1",
        content=b"same-file-content",
    )

    document_2 = make_document(
        document_id="doc-2",
        content=b"same-file-content",
    )

    router.route(document_1)

    with pytest.raises(DuplicateDocumentError):
        router.route(document_2)


def test_different_content_is_not_duplicate(tmp_path):
    router = make_router(
        confidence=0.90,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    document_1 = make_document(
        document_id="doc-1",
        content=b"content-one",
    )

    document_2 = make_document(
        document_id="doc-2",
        content=b"content-two",
    )

    router.route(document_1)
    decision = router.route(document_2)

    assert decision.pipeline == "invoice_pipeline"


@pytest.mark.parametrize(
    "document_type,expected_pipeline",
    [
        (
            DocumentType.INVOICE,
            "invoice_pipeline",
        ),
        (
            DocumentType.BANK_STATEMENT,
            "bank_statement_pipeline",
        ),
        (
            DocumentType.PURCHASE_ORDER,
            "purchase_order_pipeline",
        ),
        (
            DocumentType.RECEIPT,
            "receipt_pipeline",
        ),
    ],
)
def test_document_type_pipeline_mapping(
    tmp_path,
    document_type,
    expected_pipeline,
):
    router = make_router(
        document_type=document_type,
        confidence=0.95,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    decision = router.route(make_document())

    assert decision.document_type == document_type
    assert decision.pipeline == expected_pipeline


def test_unsupported_file_format_is_rejected(tmp_path):
    router = make_router(
        confidence=0.95,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    document = ExtractionResult(
        document_id="doc-1",
        file_format="exe",
        text="invoice data",
        content=b"content",
    )

    with pytest.raises(ValueError, match="Unsupported file format"):
        router.route(document)


def test_content_hash_is_added_to_metadata(tmp_path):
    router = make_router(
        confidence=0.95,
        dlq_path=tmp_path / "dlq.jsonl",
    )

    decision = router.route(make_document())

    assert "content_hash" in decision.routing_metadata
    assert len(decision.routing_metadata["content_hash"]) == 64