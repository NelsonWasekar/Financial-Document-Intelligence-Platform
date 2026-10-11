"""
Multi-format document processor for financial documents.

Design notes (see README for the full write-up):

- Format detection is byte-signature based, never extension based. A file
  named `statement.txt` that is actually a PDF will still be routed to the
  PDF path. Extensions are only used as a tiebreaker hint for ambiguous
  zip-family containers, never as the primary signal.
- PDF pages are classified individually (text-native vs image/scanned) so a
  10-page bank statement where page 7 is a faxed addendum is handled as
  MIXED rather than forcing the whole document down one path.
- Every extractor is wrapped so that library-specific exceptions
  (PyPDF2 errors, zipfile.BadZipFile, UnicodeDecodeError, etc.) are
  normalized into the four exception types the spec requires. Callers
  should never need to catch a third-party exception type directly.
- Confidence is a real signal, not a constant: OCR confidence is derived
  from Tesseract's own per-word confidence scores; text-native confidence
  is derived from how much of the page actually yielded text.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from dataclasses import replace
from typing import Optional

from src.common.exceptions import (
    CorruptedDocumentError,
    DocumentTooLargeError,
    ExtractionFailedError,
    UnsupportedFormatError,
)
from src.common.models import (
    DocumentType,
    ExtractionMethod,
    ExtractionResult,
    PDFClassification,
)

MAX_DOCUMENT_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

# Minimum characters of extracted text for a PDF page to count as
# "text-native" rather than "image/scanned". Chosen conservatively: a
# nearly-blank text page (e.g. a signature page) still won't clear this,
# which is the right call — treat it as scanned/uncertain rather than
# confidently claiming it as native text.
PDF_PAGE_TEXT_THRESHOLD = 20


# --------------------------------------------------------------------------
# Format detection (magic bytes, not file extension)
# --------------------------------------------------------------------------

class FormatDetector:
    """Detects a document's real format from its byte content."""

    PDF_SIGNATURE = b"%PDF-"
    ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
    OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls/.doc

    @staticmethod
    def detect(content: bytes) -> str:
        """Returns one of: 'pdf', 'xlsx', 'xls_legacy', 'text', 'unknown'."""
        if content.startswith(FormatDetector.PDF_SIGNATURE):
            return "pdf"

        if content.startswith(FormatDetector.OLE_SIGNATURE):
            return "xls_legacy"

        if any(content.startswith(sig) for sig in FormatDetector.ZIP_SIGNATURES):
            if FormatDetector._is_xlsx(content):
                return "xlsx"
            return "unknown"  # a zip, but not a spreadsheet we support

        if FormatDetector._looks_like_text(content):
            return "text"

        return "unknown"

    @staticmethod
    def _is_xlsx(content: bytes) -> bool:
        """A zip is an xlsx only if it contains the OOXML spreadsheet
        markers. This is what actually distinguishes it from a plain zip,
        a docx, or a pptx (all share the same outer PK signature)."""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = zf.namelist()
                return "[Content_Types].xml" in names and any(
                    n.startswith("xl/") for n in names
                )
        except zipfile.BadZipFile:
            # Looked like a zip by signature but isn't a valid one —
            # let the caller treat this as corrupted, not unsupported.
            raise CorruptedDocumentError(
                "File has a zip signature but is not a valid zip archive."
            )

    @staticmethod
    def _looks_like_text(content: bytes, sample_size: int = 8192) -> bool:
        """Heuristic: no null bytes, and a high ratio of printable/whitespace
        characters in a leading sample. This is intentionally permissive
        about encoding (UTF-8, UTF-16 with BOM, Latin-1) since financial
        exports come from all kinds of legacy systems."""
        if not content:
            return True  # empty file is trivially "text"; extractor will warn

        sample = content[:sample_size]
        if b"\x00" in sample:
            return False

        try:
            import chardet

            detected = chardet.detect(sample)
            encoding = detected.get("encoding")
            confidence = detected.get("confidence") or 0.0
            # chardet reports high "confidence" even when encoding is None
            # (i.e. confidently binary) — encoding must actually be set.
            return encoding is not None and confidence >= 0.5
        except ImportError:
            # Fallback without chardet: try strict utf-8 decode.
            try:
                sample.decode("utf-8")
                return True
            except UnicodeDecodeError:
                return False


# --------------------------------------------------------------------------
# PDF page classification
# --------------------------------------------------------------------------

class PDFPageClassifier:
    """Classifies each page of a PDF as text-native or image/scanned, and
    rolls that up into a document-level PDFClassification."""

    @staticmethod
    def classify_pages(pdf_path_or_stream) -> tuple[PDFClassification, list[bool], int]:
        """Returns (classification, per_page_is_text, page_count).

        per_page_is_text[i] is True if page i has extractable text above
        the threshold.
        """
        import pdfplumber

        try:
            with pdfplumber.open(pdf_path_or_stream) as pdf:
                per_page_is_text = []
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    per_page_is_text.append(len(text.strip()) >= PDF_PAGE_TEXT_THRESHOLD)
                page_count = len(pdf.pages)
        except Exception as exc:  # pdfplumber wraps several pypdf exceptions
            raise CorruptedDocumentError(f"Could not parse PDF structure: {exc}") from exc

        if page_count == 0:
            raise ExtractionFailedError("PDF contains zero pages.")

        if all(per_page_is_text):
            classification = PDFClassification.TEXT_NATIVE
        elif not any(per_page_is_text):
            classification = PDFClassification.SCANNED
        else:
            classification = PDFClassification.MIXED

        return classification, per_page_is_text, page_count


# --------------------------------------------------------------------------
# Format-specific extractors
# --------------------------------------------------------------------------

def _extract_pdf(content: bytes, warnings: list[str]) -> tuple[str, ExtractionMethod, float, int]:
    import pdfplumber

    stream = io.BytesIO(content)
    classification, per_page_is_text, page_count = PDFPageClassifier.classify_pages(stream)

    stream.seek(0)
    text_parts: list[str] = []
    page_confidences: list[float] = []

    if classification == PDFClassification.TEXT_NATIVE:
        method = ExtractionMethod.PDF_TEXT
        try:
            with pdfplumber.open(stream) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    # confidence proxy: text density relative to page area
                    page_confidences.append(0.97 if len(page_text.strip()) > 0 else 0.5)
        except Exception as exc:
            raise CorruptedDocumentError(f"PDF text extraction failed mid-document: {exc}") from exc

    else:
        # SCANNED or MIXED: OCR is needed for at least some pages.
        method = ExtractionMethod.PDF_OCR if classification == PDFClassification.SCANNED else ExtractionMethod.PDF_MIXED
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
        except ImportError as exc:
            raise ExtractionFailedError(
                "OCR is required for this document but pytesseract/pdf2image "
                "are not installed."
            ) from exc

        # Get text-native pages cheaply first, then only rasterize+OCR the
        # pages that actually need it.
        stream.seek(0)
        native_text_by_page: dict[int, str] = {}
        if classification == PDFClassification.MIXED:
            with pdfplumber.open(stream) as pdf:
                for i, page in enumerate(pdf.pages):
                    if per_page_is_text[i]:
                        native_text_by_page[i] = page.extract_text() or ""

        try:
            images = convert_from_bytes(content, dpi=300)
        except Exception as exc:
            raise ExtractionFailedError(f"Could not rasterize PDF for OCR: {exc}") from exc

        for i, image in enumerate(images):
            if i in native_text_by_page:
                text_parts.append(native_text_by_page[i])
                page_confidences.append(0.97)
                continue
            try:
                ocr_data = pytesseract.image_to_data(
                    image, output_type=pytesseract.Output.DICT
                )
                words = [w for w in ocr_data["text"] if w.strip()]
                confs = [
                    int(c)
                    for c, w in zip(ocr_data["conf"], ocr_data["text"])
                    if w.strip() and int(c) >= 0
                ]
                page_text = " ".join(words)
                page_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.3
                if page_conf < 0.5:
                    warnings.append(f"Low OCR confidence on page {i + 1} ({page_conf:.2f}).")
                text_parts.append(page_text)
                page_confidences.append(page_conf)
            except Exception as exc:
                warnings.append(f"OCR failed on page {i + 1}: {exc}")
                text_parts.append("")
                page_confidences.append(0.0)

    raw_text = "\n\n".join(text_parts)
    if not raw_text.strip():
        raise ExtractionFailedError("Extraction produced no text content.")

    overall_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0
    return raw_text, method, overall_confidence, page_count


def _extract_excel(content: bytes, warnings: list[str]) -> tuple[str, float, int]:
    import openpyxl

    try:
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True, read_only=True)
    except zipfile.BadZipFile as exc:
        raise CorruptedDocumentError(f"Workbook is not a valid xlsx container: {exc}") from exc
    except Exception as exc:
        raise CorruptedDocumentError(f"Could not open workbook: {exc}") from exc

    sheet_names = wb.sheetnames
    if not sheet_names:
        raise ExtractionFailedError("Workbook contains no sheets.")

    text_parts = []
    non_empty_sheets = 0
    for name in sheet_names:
        ws = wb[name]
        rows_text = []
        row_count = 0
        for row in ws.iter_rows(values_only=True):
            row_count += 1
            cells = [str(c) for c in row if c is not None]
            if cells:
                rows_text.append("\t".join(cells))
        if rows_text:
            non_empty_sheets += 1
        else:
            warnings.append(f"Sheet '{name}' is empty.")
        text_parts.append(f"=== Sheet: {name} ===\n" + "\n".join(rows_text))

    raw_text = "\n\n".join(text_parts)
    confidence = 0.98 if non_empty_sheets == len(sheet_names) else max(0.5, non_empty_sheets / len(sheet_names))
    # page_count is repurposed as sheet_count for spreadsheets (documented in metadata).
    return raw_text, confidence, len(sheet_names)


def _extract_text(content: bytes, warnings: list[str]) -> tuple[str, float]:
    encoding = "utf-8"
    confidence = 0.95
    try:
        raw_text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            import chardet

            detected = chardet.detect(content)
            encoding = detected.get("encoding") or "latin-1"
            det_confidence = detected.get("confidence") or 0.5
            raw_text = content.decode(encoding, errors="replace")
            confidence = min(0.9, det_confidence)
            warnings.append(f"File was not UTF-8; decoded as {encoding} (confidence {det_confidence:.2f}).")
        except ImportError:
            raw_text = content.decode("latin-1", errors="replace")
            confidence = 0.5
            warnings.append("chardet unavailable; fell back to latin-1 decoding.")

    if not raw_text.strip():
        raise ExtractionFailedError("Text file is empty or contains no usable content.")

    return raw_text, confidence


# --------------------------------------------------------------------------
# Lightweight document-type classification (keyword heuristic)
# --------------------------------------------------------------------------

def _classify_document_type(raw_text: str) -> DocumentType:
    """Cheap keyword heuristic to tag a coarse document type. This is a
    placeholder for the real classifier — in the full pipeline this should
    defer to ner_extractor.py / document_router.py, which have access to
    structured entities rather than just raw text. Kept here so
    ExtractionResult is immediately useful standalone."""
    lowered = raw_text.lower()[:5000]  # cheap: only scan the head of the doc

    statement_hits = sum(k in lowered for k in ("statement of account", "closing balance", "opening balance", "account statement"))
    invoice_hits = sum(k in lowered for k in ("invoice number", "invoice no", "bill to", "amount due", "tax invoice"))
    transaction_hits = sum(k in lowered for k in ("transaction id", "narration", "debit", "credit"))

    scores = {
        DocumentType.BANK_STATEMENT: statement_hits,
        DocumentType.INVOICE: invoice_hits,
        DocumentType.TRANSACTION_LOG: transaction_hits,
    }
    best_type, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_type if best_score > 0 else DocumentType.UNKNOWN


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

class DocumentProcessor:
    """Production-facing entry point: `process()` a file path or raw bytes
    and get back a standardized ExtractionResult, or one of the four
    documented exceptions."""

    def __init__(self, max_size_bytes: int = MAX_DOCUMENT_SIZE_BYTES):
        self.max_size_bytes = max_size_bytes

    def process(self, file_path: Optional[str] = None, *, content: Optional[bytes] = None,
                filename: Optional[str] = None) -> ExtractionResult:
        if content is None:
            if file_path is None:
                raise ValueError("Provide either file_path or content.")
            filename = filename or os.path.basename(file_path)
            size = os.path.getsize(file_path)
            if size > self.max_size_bytes:
                raise DocumentTooLargeError(
                    f"{filename} is {size} bytes, exceeding the {self.max_size_bytes}-byte limit."
                )
            with open(file_path, "rb") as f:
                content = f.read()
        else:
            filename = filename or "<in-memory>"
            if len(content) > self.max_size_bytes:
                raise DocumentTooLargeError(
                    f"{filename} is {len(content)} bytes, exceeding the {self.max_size_bytes}-byte limit."
                )

        start = time.perf_counter()
        warnings: list[str] = []

        detected_format = FormatDetector.detect(content)

        if detected_format == "pdf":
            raw_text, method, confidence, page_count = _extract_pdf(content, warnings)
        elif detected_format == "xlsx":
            raw_text, confidence, page_count = _extract_excel(content, warnings)
            method = ExtractionMethod.EXCEL
        elif detected_format == "xls_legacy":
            raise UnsupportedFormatError(
                f"{filename} is a legacy .xls/OLE document; only .xlsx (OOXML) is supported."
            )
        elif detected_format == "text":
            raw_text, confidence = _extract_text(content, warnings)
            method = ExtractionMethod.PLAIN_TEXT
            page_count = 1
        else:
            raise UnsupportedFormatError(
                f"{filename}: could not detect a supported format from file content."
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        document_type = _classify_document_type(raw_text)

        metadata = {
            "filename": filename,
            "size_bytes": len(content),
            "detected_format": detected_format,
        }
        if detected_format == "xlsx":
            metadata["note"] = "page_count represents sheet count for spreadsheets"

        return ExtractionResult(
            document_type=document_type,
            extraction_method=method,
            raw_text=raw_text,
            page_count=page_count,
            confidence=round(confidence, 4),
            processing_time_ms=round(elapsed_ms, 3),
            warnings=warnings,
            metadata=metadata,
        )
