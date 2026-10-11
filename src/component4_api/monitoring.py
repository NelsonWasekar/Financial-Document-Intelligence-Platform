"""Small, dependency-free monitoring primitives for the API service."""

from __future__ import annotations

from collections import Counter
from threading import Lock
from time import perf_counter


class APIMonitor:
    """Store request metrics in memory for the lifetime of the process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._error_count = 0
        self._total_duration_ms = 0.0
        self._status_codes: Counter[str] = Counter()
        self._started_at = perf_counter()

    def record_request(self, status_code: int, duration_ms: float) -> None:
        """Record one completed HTTP request."""
        with self._lock:
            self._request_count += 1
            self._total_duration_ms += duration_ms
            self._status_codes[str(status_code)] += 1
            if status_code >= 500:
                self._error_count += 1

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serialisable copy of current metrics."""
        with self._lock:
            request_count = self._request_count
            return {
                "requests_total": request_count,
                "errors_total": self._error_count,
                "error_rate": (
                    self._error_count / request_count
                    if request_count
                    else 0.0
                ),
                "average_response_time_ms": (
                    self._total_duration_ms / request_count
                    if request_count
                    else 0.0
                ),
                "status_codes": dict(self._status_codes),
                "uptime_seconds": round(perf_counter() - self._started_at, 3),
            }


monitor = APIMonitor()
