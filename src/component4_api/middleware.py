"""HTTP middleware used by the Component 4 FastAPI application."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from src.component4_api.monitoring import APIMonitor

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add a request ID, record timing, and return safe error responses."""

    def __init__(self, app, monitor: APIMonitor) -> None:
        super().__init__(app)
        self.monitor = monitor

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        started_at = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled API error; request_id=%s", request_id)
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "An internal error occurred.",
                    "request_id": request_id,
                },
            )

        duration_ms = (time.perf_counter() - started_at) * 1000
        self.monitor.record_request(response.status_code, duration_ms)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response
