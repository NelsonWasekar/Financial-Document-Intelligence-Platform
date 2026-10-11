# Source architecture

This directory contains the application code for the Financial Document Intelligence System. The project accepts financial documents, retrieves relevant policy text, assesses vendor-payment compliance, and exposes the workflow through a FastAPI service.

## Package layout

| Package | Responsibility |
| --- | --- |
| `common` | Shared configuration, data models, and domain exceptions. |
| `component1_extraction` | Document classification, text extraction, entity extraction, de-duplication, and failed-document logging. |
| `component2_rag` | PDF policy indexing, hybrid vector/BM25 retrieval, reranking, cited answer generation, and evaluation. |
| `component3_agent` | Vendor matching, transaction and policy analysis, risk assessment, resilience handling, and audit logging. |
| `component4_api` | FastAPI endpoints, request context middleware, and in-process metrics. |

## Configuration and runtime data

`common/config.py` reads values from the project-level `.env` file. Important defaults are:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama API endpoint. |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model used for RAG generation and verification. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Vector embedding model. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker. |

The application creates these directories when needed:

- `data/index/` — persisted policy index.
- `data/audit/` — audit records created by agent runs.
- `outputs/` — generated evaluation reports.

These directories are runtime data and are not included in `src/`.

## RAG workflow

`FinancialRAGPipeline` processes policy PDFs through the following sequence:

1. Extract sections and create parent/child chunks.
2. Build embeddings and a BM25 index.
3. Combine vector and keyword search with reciprocal-rank fusion.
4. Rerank the strongest candidates with a cross-encoder.
5. Check whether the retrieved context is sufficient.
6. Generate a grounded response through Ollama, then validate citations and faithfulness.

Before using the pipeline, provide a directory containing one or more PDF policy documents and index it:

```python
from src.component2_rag.rag_pipeline import FinancialRAGPipeline

pipeline = FinancialRAGPipeline()
report = pipeline.index_documents("corpus")
print(report)
```

The evaluator expects a JSON or JSONL dataset. The example entry point in `component2_rag/rag_evaluator.py` uses `data/rag_evaluation.json`; create that file before running it. A JSON dataset must be a list of records containing `question` and `expected_answer`.

```json
[
  {
    "question": "What is the expense approval limit?",
    "expected_answer": "Expenses above the stated limit require approval."
  }
]
```

Run it from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.component2_rag.rag_evaluator
```

On success, the evaluator writes `outputs/rag_eval_report.json`.

## API entry point

The API application is defined in `component4_api/main.py`.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Returns service health and version. |
| `GET` | `/metrics` | Returns in-process request and latency metrics. |
| `POST` | `/api/v1/research` | Runs the vendor research workflow. |
| `GET` | `/api/v1/audits/{query_id}` | Retrieves a stored audit record. |

For installation, test, Docker, and API-start commands, see the [project README](../README.md).
