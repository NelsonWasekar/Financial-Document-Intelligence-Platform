import hashlib
import json
import pickle
import re
import time
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.common.config import Settings
from src.common.exceptions import (
    CitationValidationError,
    DocumentProcessingError,
    IndexingError,
    LLMError,
    RetrievalError,
)
from src.common.models import (
    ChildChunk,
    Citation,
    IndexingReport,
    ParentSection,
    RAGResponse,
    RetrievedChunk,
)


class FinancialRAGPipeline:
    """
    Production-style Financial RAG pipeline.

    Pipeline:
        PDF
          ↓
        Structure-aware section extraction
          ↓
        Parent / Child chunks
          ↓
        ┌───────────────┬──────────────┐
        │ Vector Search │  BM25 Search │
        └───────────────┴──────────────┘
                  ↓
              RRF Fusion
                  ↓
              Top 10 chunks
                  ↓
             Cross Encoder
                  ↓
              Top 3 chunks
                  ↓
            Ollama LLM
             ↙       ↘
       Sufficiency   Generation
                        ↓
                 Citation Check
                        ↓
                Faithfulness Check
    """

    HEADING_PATTERNS = [
        re.compile(r"^\s*\d+(?:\.\d+)*\s+.+$"),
        re.compile(r"^\s*[A-Z][A-Z\s&\-/]{4,}$"),
    ]

    def __init__(self):
        Settings.create_directories()

        self.embedding_model = SentenceTransformer(
            Settings.EMBEDDING_MODEL
        )

        self.reranker = CrossEncoder(
            Settings.RERANKER_MODEL
        )

        self.ollama_base_url = Settings.OLLAMA_BASE_URL.rstrip("/")
        self.ollama_model = Settings.OLLAMA_MODEL

        self.chunks: list[ChildChunk] = []
        self.parents: list[ParentSection] = []

        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

        self.document_hashes = self._load_document_hashes()

        self._load_index()

    # =========================================================
    # OLLAMA
    # =========================================================

    def _call_ollama(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> str:
        """
        Call local Ollama server.

        Ollama must be running:

            ollama serve

        Example model:

            ollama pull llama3.2:3b
        """

        url = f"{self.ollama_base_url}/api/chat"

        payload = {
            "model": self.ollama_model,
            "messages": [],
            "stream": False,
            "options": {
                "temperature": temperature
            },
        }

        if system_prompt:
            payload["messages"].append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        payload["messages"].append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        # Retry with exponential backoff so this LLM call is never a
        # single-attempt call, per the hard production requirement.
        # This is a local retry independent of Component 3's
        # AgentResilience layer, which may wrap agent-level RAG calls
        # with its own retry/fallback on top of this one.
        max_attempts = 3
        backoff_base = 1.0
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                content = (
                    data.get("message", {})
                    .get("content", "")
                    .strip()
                )

                if not content:
                    raise LLMError(
                        "Ollama returned an empty response."
                    )

                return content

            except requests.exceptions.ConnectionError as exc:
                last_exc = LLMError(
                    f"Cannot connect to Ollama at {self.ollama_base_url}. "
                    f"Make sure Ollama is running with 'ollama serve'."
                )
                last_exc.__cause__ = exc

            except requests.exceptions.Timeout as exc:
                last_exc = LLMError("Ollama request timed out.")
                last_exc.__cause__ = exc

            except requests.exceptions.RequestException as exc:
                last_exc = LLMError(f"Ollama request failed: {exc}")
                last_exc.__cause__ = exc

            except LLMError as exc:
                last_exc = exc

            except Exception as exc:
                last_exc = LLMError(f"Ollama error: {exc}")
                last_exc.__cause__ = exc

            if attempt < max_attempts - 1:
                time.sleep(backoff_base * (2 ** attempt))

        raise last_exc

    # =========================================================
    # DOCUMENT HASHING
    # =========================================================

    def _calculate_file_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:
            while True:
                data = file.read(1024 * 1024)

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    # =========================================================
    # PDF PROCESSING
    # =========================================================

    def _extract_sections(
        self,
        pdf_path: Path,
    ) -> list[ParentSection]:

        sections = []

        try:
            document = fitz.open(pdf_path)

            current_title = "General"
            current_text = []
            current_start_page = 1

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                text = page.get_text("text")

                lines = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                ]

                for line in lines:

                    if self._is_heading(line):

                        if current_text:

                            sections.append(
                                ParentSection(
                                    section_id=str(uuid.uuid4()),
                                    document_name=pdf_path.name,
                                    section=current_title,
                                    page_start=current_start_page,
                                    page_end=page_number,
                                    text="\n".join(current_text).strip(),
                                )
                            )

                        current_title = line
                        current_text = []
                        current_start_page = page_number

                    else:
                        current_text.append(line)

            if current_text:

                sections.append(
                    ParentSection(
                        section_id=str(uuid.uuid4()),
                        document_name=pdf_path.name,
                        section=current_title,
                        page_start=current_start_page,
                        page_end=document.page_count,
                        text="\n".join(current_text).strip(),
                    )
                )

            document.close()

            return sections

        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to process {pdf_path}: {exc}"
            ) from exc

    def _is_heading(self, line: str) -> bool:

        if len(line) > 150:
            return False

        return any(
            pattern.match(line)
            for pattern in self.HEADING_PATTERNS
        )

    # =========================================================
    # CHILD CHUNK CREATION
    # =========================================================

    def _create_child_chunks(
        self,
        parent: ParentSection,
    ) -> list[ChildChunk]:

        paragraphs = re.split(
            r"\n\s*\n|\n(?=\d+\.)",
            parent.text,
        )

        paragraphs = [
            p.strip()
            for p in paragraphs
            if p.strip()
        ]

        children = []

        for index, paragraph in enumerate(paragraphs):

            # If paragraph is extremely long, split by sentences.
            if len(paragraph) > 2500:

                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    paragraph,
                )

                for sentence_index, sentence in enumerate(
                    sentences
                ):

                    if not sentence.strip():
                        continue

                    children.append(
                        ChildChunk(
                            chunk_id=str(uuid.uuid4()),
                            parent_id=parent.section_id,
                            document_name=parent.document_name,
                            section=parent.section,
                            page=parent.page_start,
                            paragraph_index=(
                                index * 1000
                                + sentence_index
                            ),
                            text=sentence.strip(),
                        )
                    )

            else:

                children.append(
                    ChildChunk(
                        chunk_id=str(uuid.uuid4()),
                        parent_id=parent.section_id,
                        document_name=parent.document_name,
                        section=parent.section,
                        page=parent.page_start,
                        paragraph_index=index,
                        text=paragraph,
                    )
                )

        return children

    # =========================================================
    # INDEXING
    # =========================================================

    def index_documents(
        self,
        docs_path: str,
    ) -> IndexingReport:

        start_time = time.perf_counter()

        corpus_path = Path(docs_path)

        if not corpus_path.exists():
            raise IndexingError(
                f"Corpus directory does not exist: {docs_path}"
            )

        pdf_files = sorted(
            corpus_path.glob("*.pdf")
        )

        if not pdf_files:
            raise IndexingError(
                f"No PDF files found in {docs_path}"
            )

        changed_documents = []
        unchanged_documents = []

        new_hashes = {}

        all_parents = []
        all_chunks = []

        for pdf_file in pdf_files:

            current_hash = self._calculate_file_hash(
                pdf_file
            )

            new_hashes[pdf_file.name] = current_hash

            old_hash = self.document_hashes.get(
                pdf_file.name
            )

            if old_hash == current_hash:
                unchanged_documents.append(
                    pdf_file.name
                )
            else:
                changed_documents.append(
                    pdf_file.name
                )

            parents = self._extract_sections(
                pdf_file
            )

            all_parents.extend(parents)

            for parent in parents:

                children = self._create_child_chunks(
                    parent
                )

                all_chunks.extend(children)

        if not all_chunks:
            raise IndexingError(
                "No chunks were created from the corpus."
            )

        # -----------------------------------------------------
        # Embeddings
        # -----------------------------------------------------

        texts = [
            chunk.text
            for chunk in all_chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # BM25
        # -----------------------------------------------------

        tokenized_documents = [
            self._tokenize(text)
            for text in texts
        ]

        bm25 = BM25Okapi(
            tokenized_documents
        )

        self.parents = all_parents
        self.chunks = all_chunks
        self.embeddings = embeddings
        self.bm25 = bm25

        self.document_hashes = new_hashes

        self._save_index()

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return IndexingReport(
            documents_processed=len(pdf_files),
            sections_created=len(all_parents),
            chunks_created=len(all_chunks),
            changed_documents=changed_documents,
            unchanged_documents=unchanged_documents,
            indexing_time_ms=round(
                elapsed_ms,
                2,
            ),
        )

    # =========================================================
    # TOKENIZATION
    # =========================================================

    def _tokenize(self, text: str) -> list[str]:

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    # =========================================================
    # HYBRID RETRIEVAL
    # =========================================================

    def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:

        if self.embeddings is None:
            raise RetrievalError(
                "Vector index is not loaded."
            )

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        scores = (
            self.embeddings @ query_embedding
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        return [
            (
                int(index),
                float(scores[index]),
            )
            for index in top_indices
        ]

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:

        if self.bm25 is None:
            raise RetrievalError(
                "BM25 index is not loaded."
            )

        tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            tokens
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        return [
            (
                int(index),
                float(scores[index]),
            )
            for index in top_indices
        ]

    # =========================================================
    # RECIPROCAL RANK FUSION
    # =========================================================

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:

        rrf_scores: dict[int, float] = {}

        k = Settings.RRF_K

        for rank, (index, _) in enumerate(
            vector_results,
            start=1,
        ):
            rrf_scores[index] = (
                rrf_scores.get(index, 0.0)
                + 1.0 / (k + rank)
            )

        for rank, (index, _) in enumerate(
            bm25_results,
            start=1,
        ):
            rrf_scores[index] = (
                rrf_scores.get(index, 0.0)
                + 1.0 / (k + rank)
            )

        return sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    def _hybrid_retrieve(
        self,
        query: str,
    ) -> list[RetrievedChunk]:

        vector_results = self._vector_search(
            query,
            Settings.TOP_K_RETRIEVAL,
        )

        bm25_results = self._bm25_search(
            query,
            Settings.TOP_K_RETRIEVAL,
        )

        fused_results = (
            self._reciprocal_rank_fusion(
                vector_results,
                bm25_results,
            )
        )

        top_results = fused_results[
            : Settings.TOP_K_RETRIEVAL
        ]

        retrieved = []

        for index, score in top_results:

            chunk = self.chunks[index]

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    section=chunk.section,
                    page=chunk.page,
                    text=chunk.text,
                    score=float(score),
                    retrieval_method="hybrid_rrf",
                )
            )

        return retrieved

    # =========================================================
    # CROSS ENCODER RERANKING
    # =========================================================

    def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:

        if not chunks:
            return []

        pairs = [
            (
                query,
                chunk.text,
            )
            for chunk in chunks
        ]

        scores = self.reranker.predict(
            pairs
        )

        reranked = []

        for chunk, score in zip(
            chunks,
            scores,
        ):
            chunk.score = float(score)
            chunk.retrieval_method = (
                "hybrid_rrf_cross_encoder"
            )

            reranked.append(chunk)

        reranked.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return reranked[
            : Settings.TOP_K_GENERATION
        ]

    # =========================================================
    # CONTEXT FORMATTING
    # =========================================================

    def _build_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> str:

        context_parts = []

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            context_parts.append(
                f"""
SOURCE {index}
Document: {chunk.document_name}
Section: {chunk.section}
Page: {chunk.page}

Content:
{chunk.text}
""".strip()
            )

        return "\n\n".join(
            context_parts
        )

    # =========================================================
    # SUFFICIENCY CHECK
    # =========================================================

    def _check_context_sufficiency(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> bool:

        context = self._build_context(
            chunks
        )

        system_prompt = """
You are a financial policy context evaluator.

Your job is to determine whether the supplied context
contains enough information to answer the user's question.

Do NOT use outside knowledge.

Return ONLY valid JSON:

{
  "sufficient": true,
  "reason": "short explanation"
}

or

{
  "sufficient": false,
  "reason": "short explanation"
}
"""

        prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}

Is the context sufficient to answer the question?
"""

        response = self._call_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )

        try:

            cleaned = response.strip()

            if cleaned.startswith("```"):
                cleaned = re.sub(
                    r"```(?:json)?",
                    "",
                    cleaned,
                ).strip()

            data = json.loads(cleaned)

            return bool(
                data.get(
                    "sufficient",
                    False,
                )
            )

        except json.JSONDecodeError as exc:

            raise LLMError(
                f"Invalid JSON returned by Ollama: {response}"
            ) from exc

    # =========================================================
    # GROUNDED GENERATION
    # =========================================================

    def _generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:

        context = self._build_context(
            chunks
        )

        system_prompt = """
You are a financial policy assistant.

STRICT RULES:

1. Answer ONLY using the supplied context.
2. Do NOT use outside knowledge.
3. Every factual claim MUST contain a citation.
4. Citation format MUST be:

[Document Name, Section Name]

5. If the context does not contain the answer,
   say "Insufficient information."
6. Never invent policy rules.
7. Never invent document names.
8. Never invent section names.
9. Keep the answer concise and professional.
"""

        prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}

Provide a grounded answer.

Every factual statement must have a citation.
"""

        return self._call_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )

    # =========================================================
    # CITATION EXTRACTION
    # =========================================================

    def _extract_citations(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:

        pattern = re.compile(
            r"\[([^,\]]+),\s*([^\]]+)\]"
        )

        matches = pattern.findall(
            answer
        )

        if not matches:
            raise CitationValidationError(
                "Generated answer contains no citations."
            )

        citations = []

        for document_name, section in matches:

            document_name = document_name.strip()
            section = section.strip()

            matching_chunks = [
                chunk
                for chunk in chunks
                if chunk.document_name
                == document_name
                and chunk.section
                == section
            ]

            if not matching_chunks:
                raise CitationValidationError(
                    f"Invalid citation: "
                    f"[{document_name}, {section}]"
                )

            chunk = matching_chunks[0]

            excerpt = chunk.text[
                : Settings.MAX_CITATION_EXCERPT
            ]

            citations.append(
                Citation(
                    document_name=document_name,
                    section=section,
                    page=chunk.page,
                    excerpt=excerpt,
                    relevance_score=chunk.score,
                )
            )

        return citations

    # =========================================================
    # CITATION COVERAGE
    # =========================================================

    def _validate_claim_citations(
        self,
        answer: str,
    ) -> bool:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer.strip(),
        )

        citation_pattern = re.compile(
            r"\[[^,\]]+,\s*[^\]]+\]"
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Skip explicit insufficient-information response
            if sentence.lower() == "insufficient information.":
                continue

            if not citation_pattern.search(
                sentence
            ):
                return False

        return True

    # =========================================================
    # FAITHFULNESS CHECK
    # =========================================================

    def _faithfulness_check(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> float:

        context = self._build_context(
            chunks
        )

        system_prompt = """
You are a faithfulness evaluator.

Determine whether the answer is fully supported
by the supplied financial policy context.

Do NOT use outside knowledge.

Return ONLY JSON:

{
  "faithfulness_score": 0.95,
  "supported_claims": 5,
  "total_claims": 5,
  "unsupported_claims": []
}

The score must be between 0 and 1.
"""

        prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate whether every factual claim in the answer
is supported by the context.
"""

        response = self._call_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )

        try:

            cleaned = response.strip()

            if cleaned.startswith("```"):
                cleaned = re.sub(
                    r"```(?:json)?",
                    "",
                    cleaned,
                ).strip()

            data = json.loads(cleaned)

            score = float(
                data.get(
                    "faithfulness_score",
                    0.0,
                )
            )

            return max(
                0.0,
                min(1.0, score),
            )

        except (
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ) as exc:

            raise LLMError(
                f"Invalid faithfulness response from Ollama: "
                f"{response}"
            ) from exc

    # =========================================================
    # CONFIDENCE
    # =========================================================

    def _calculate_confidence(
        self,
        reranked_chunks: list[RetrievedChunk],
        faithfulness_score: float,
    ) -> float:

        if not reranked_chunks:
            return 0.0

        top_score = reranked_chunks[0].score

        # Convert cross encoder score approximately to 0-1.
        reranker_confidence = (
            1.0 / (1.0 + np.exp(-top_score))
        )

        confidence = (
            0.5 * reranker_confidence
            + 0.5 * faithfulness_score
        )

        return round(
            float(
                max(
                    0.0,
                    min(1.0, confidence),
                )
            ),
            4,
        )

    # =========================================================
    # MAIN QUERY
    # =========================================================

    def query(
        self,
        question: str,
        user_id: str,
    ) -> RAGResponse:

        start_time = time.perf_counter()

        query_id = str(
            uuid.uuid4()
        )

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        try:

            # -------------------------------------------------
            # 1. Hybrid Retrieval
            # -------------------------------------------------

            retrieved_chunks = (
                self._hybrid_retrieve(
                    question
                )
            )

            # -------------------------------------------------
            # 2. Cross Encoder
            # -------------------------------------------------

            reranked_chunks = self._rerank(
                question,
                retrieved_chunks,
            )

            # -------------------------------------------------
            # 3. Sufficiency Check
            # -------------------------------------------------

            sufficient = (
                self._check_context_sufficiency(
                    question,
                    reranked_chunks,
                )
            )

            if not sufficient:

                processing_time_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000

                return RAGResponse(
                    answer="Insufficient information.",
                    citations=[],
                    confidence=0.0,
                    retrieved_chunks=reranked_chunks,
                    sufficiency_check_passed=False,
                    faithfulness_score=1.0,
                    query_id=query_id,
                    processing_time_ms=round(
                        processing_time_ms,
                        2,
                    ),
                )

            # -------------------------------------------------
            # 4. Grounded Generation using Ollama
            # -------------------------------------------------

            answer = self._generate_answer(
                question,
                reranked_chunks,
            )

            # -------------------------------------------------
            # 5. Citation Coverage Check
            # -------------------------------------------------

            if not self._validate_claim_citations(
                answer
            ):

                raise CitationValidationError(
                    "One or more factual claims "
                    "do not contain citations."
                )

            # -------------------------------------------------
            # 6. Citation Validation
            # -------------------------------------------------

            citations = self._extract_citations(
                answer,
                reranked_chunks,
            )

            # -------------------------------------------------
            # 7. Faithfulness
            # -------------------------------------------------

            faithfulness_score = (
                self._faithfulness_check(
                    question,
                    answer,
                    reranked_chunks,
                )
            )

            # -------------------------------------------------
            # 8. Confidence
            # -------------------------------------------------

            confidence = (
                self._calculate_confidence(
                    reranked_chunks,
                    faithfulness_score,
                )
            )

            processing_time_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            return RAGResponse(
                answer=answer,
                citations=citations,
                confidence=confidence,
                retrieved_chunks=reranked_chunks,
                sufficiency_check_passed=True,
                faithfulness_score=faithfulness_score,
                query_id=query_id,
                processing_time_ms=round(
                    processing_time_ms,
                    2,
                ),
            )

        except Exception:

            # Re-raise so Component 3 resilience layer
            # can handle retries/fallbacks.
            raise

    # =========================================================
    # INDEX PERSISTENCE
    # =========================================================

    def _save_index(self):

        Settings.INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            Settings.INDEX_DIR / "chunks.pkl",
            "wb",
        ) as file:

            pickle.dump(
                self.chunks,
                file,
            )

        with open(
            Settings.INDEX_DIR / "parents.pkl",
            "wb",
        ) as file:

            pickle.dump(
                self.parents,
                file,
            )

        np.save(
            Settings.INDEX_DIR / "embeddings.npy",
            self.embeddings,
        )

        with open(
            Settings.INDEX_DIR / "bm25.pkl",
            "wb",
        ) as file:

            pickle.dump(
                self.bm25,
                file,
            )

        with open(
            Settings.INDEX_DIR
            / "document_hashes.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.document_hashes,
                file,
                indent=2,
            )

    def _load_index(self):

        chunks_file = (
            Settings.INDEX_DIR
            / "chunks.pkl"
        )

        parents_file = (
            Settings.INDEX_DIR
            / "parents.pkl"
        )

        embeddings_file = (
            Settings.INDEX_DIR
            / "embeddings.npy"
        )

        bm25_file = (
            Settings.INDEX_DIR
            / "bm25.pkl"
        )

        if not all(
            [
                chunks_file.exists(),
                parents_file.exists(),
                embeddings_file.exists(),
                bm25_file.exists(),
            ]
        ):
            return

        try:

            with open(
                chunks_file,
                "rb",
            ) as file:

                self.chunks = pickle.load(
                    file
                )

            with open(
                parents_file,
                "rb",
            ) as file:

                self.parents = pickle.load(
                    file
                )

            self.embeddings = np.load(
                embeddings_file
            )

            with open(
                bm25_file,
                "rb",
            ) as file:

                self.bm25 = pickle.load(
                    file
                )

        except Exception:
            # If index loading fails, indexing can
            # be performed again.
            self.chunks = []
            self.parents = []
            self.embeddings = None
            self.bm25 = None

    def _load_document_hashes(self) -> dict[str, str]:

        hash_file = (
            Settings.INDEX_DIR
            / "document_hashes.json"
        )

        if not hash_file.exists():
            return {}

        try:

            with open(
                hash_file,
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except Exception:
            return {}