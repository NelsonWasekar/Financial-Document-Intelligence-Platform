from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class AgentResilience:
    """
    Retry and graceful-degradation utilities for the
    financial research agent.
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.max_retries = max_retries
        self.retry_delay_seconds = (
            retry_delay_seconds
        )

    def execute_with_retry(
        self,
        operation: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute an operation with retry handling.
        """

        last_error = None

        for attempt in range(
            self.max_retries + 1
        ):
            try:
                return operation(
                    *args,
                    **kwargs,
                )

            except Exception as exc:
                last_error = exc

                if (
                    attempt
                    >= self.max_retries
                ):
                    break

                delay = (
                    self.retry_delay_seconds
                    * (attempt + 1)
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Operation failed after "
            f"{self.max_retries + 1} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def safe_rag_response(
        question: str,
        error: Exception,
    ) -> dict:
        """
        Controlled fallback when RAG is unavailable.

        The agent must not invent policy information.
        """

        return {
            "answer": (
                "Policy information is currently "
                "unavailable. Compliance analysis "
                "must not proceed using assumed policy "
                "information."
            ),
            "citations": [],
            "sufficient": False,
            "error": str(error),
            "question": question,
        }

    @staticmethod
    def safe_agent_response(
        vendor_name: str,
        reason: str,
    ) -> dict:
        """
        Controlled fallback response for agent failures.
        """

        return {
            "vendor_name": vendor_name,
            "risk_level": "unknown",
            "requires_escalation": True,
            "compliance_summary": (
                "Financial research could not be "
                "completed safely. Manual review is "
                "required."
            ),
            "recommendations": [
                "Perform manual vendor verification.",
                "Review transactions manually.",
                "Review applicable financial policies.",
            ],
            "error": reason,
        }
