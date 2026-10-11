# Financial Document Intelligence System

A four-component Python project for extracting financial-document data, retrieving policy information, analysing vendor-payment compliance, and exposing the workflow through an API.

## Components

| Component | Location | Purpose |
| --- | --- | --- |
| 1. Document extraction | `src/component1_extraction` | Routes documents, extracts text, and identifies transaction entities. |
| 2. RAG pipeline | `src/component2_rag` | Indexes financial policies and retrieves cited answers. |
| 3. Research agent | `src/component3_agent` | Matches vendors, checks transactions against policy limits, produces summaries, and writes audit records. |
| 4. API | `src/component4_api` | Provides validated FastAPI endpoints, request IDs, and in-process metrics. |

## Prerequisites

- Python 3.14 (the supplied environment was created with Python 3.14)
- [uv](https://docs.astral.sh/uv/)
- Ollama, only when running a real Component 2/3 research request

The default RAG configuration expects Ollama at `http://localhost:11434` with the `llama3.2:3b` model.

```powershell
ollama serve
ollama pull llama3.2:3b
```

## Setup

Create or activate the local virtual environment, then install the dependencies:

```powershell
uv venv
uv pip install --python .venv -r requirements.txt
```

Configuration is read from `.env`. Do not commit secrets or API keys; use environment variables instead.

## Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.component4_api.main:app --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health and version. |
| `GET` | `/metrics` | In-memory request, error, and latency metrics. |
| `POST` | `/api/v1/research` | Runs the vendor compliance workflow. |
| `GET` | `/api/v1/audits/{query_id}` | Retrieves a saved research audit record. |

Example research request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/research `
  -ContentType 'application/json' `
  -Body '{
    "vendor_name": "Sharma Raj Trading",
    "analyst_id": "analyst-001",
    "start_date": "2026-01-01",
    "end_date": "2026-12-31"
  }'
```

Each response includes an `X-Request-ID` header. Audit records are stored under `data/audit/`.

## Tests

Run the project tests with:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
```

The agent, API, and degradation test suites are isolated from live LLM calls by using fake dependencies. The NER tests also mock the trained model, but importing PyTorch may take longer on first run.

## Outputs

Generated evaluation and example results are kept in `outputs/`:

- `eval_results.json`
- `rag_eval_report.json`
- `sample_agent_run.json`

## Docker

Build and start the API container:

```powershell
docker compose up --build
```

The API is available at `http://127.0.0.1:8000`, and Docker reports the service healthy after `/health` returns a successful response. The Compose service reads runtime configuration from `.env`; keep secrets in that file and do not add them to the image.
