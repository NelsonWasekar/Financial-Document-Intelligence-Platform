"""FastAPI interface for the financial vendor research workflow."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from src.common.config import Settings
from src.component3_agent.audit_logger import AuditLogger
from src.component3_agent.resilience import AgentResilience
from src.component4_api.middleware import RequestContextMiddleware
from src.component4_api.monitoring import monitor


class ResearchRequest(BaseModel):
    vendor_name: str = Field(min_length=1, max_length=200)
    analyst_id: str = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date

    @field_validator("vendor_name", "analyst_id")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("end_date")
    @classmethod
    def validate_date_order(cls, end_date: date, info) -> date:
        start_date = info.data.get("start_date")
        if start_date and end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        return end_date


class HealthResponse(BaseModel):
    status: str
    application: str
    version: str


@lru_cache
def get_agent() -> Any:
    """Create the expensive agent stack once per application process."""
    # Import lazily: loading the RAG dependencies should not delay health checks
    # or OpenAPI schema generation.
    from src.component3_agent.research_agent import FinancialResearchAgent

    return FinancialResearchAgent()


def get_audit_logger() -> AuditLogger:
    return get_agent().audit_logger


app = FastAPI(
    title=Settings.APP_NAME,
    version=Settings.APP_VERSION,
    description="API for financial vendor compliance research.",
)
app.add_middleware(RequestContextMiddleware, monitor=monitor)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        application=Settings.APP_NAME,
        version=Settings.APP_VERSION,
    )


@app.get("/metrics", tags=["system"])
def metrics() -> dict[str, object]:
    return monitor.snapshot()


@app.post("/api/v1/research", tags=["research"])
def research_vendor(
    payload: ResearchRequest,
    request: Request,
    agent: Any = Depends(get_agent),
) -> dict[str, Any]:
    """Run the Component 3 agent and expose its structured result as JSON."""
    try:
        result = agent.research(
            vendor_name=payload.vendor_name,
            analyst_id=payload.analyst_id,
            date_range=(payload.start_date.isoformat(), payload.end_date.isoformat()),
        )
    except Exception as exc:
        fallback = AgentResilience.safe_agent_response(payload.vendor_name, str(exc))
        fallback["request_id"] = request.state.request_id
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=fallback,
        ) from exc

    # Dataclasses in the agent state are serialised correctly by FastAPI.
    result["request_id"] = request.state.request_id
    return result


@app.get("/api/v1/audits/{query_id}", tags=["audit"])
def get_audit_record(
    query_id: str,
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> dict[str, Any]:
    record = audit_logger.read_audit(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit record not found.")
    return record
