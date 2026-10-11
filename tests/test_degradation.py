import pytest

from src.component3_agent.resilience import AgentResilience


def test_retry_returns_result_after_transient_failure() -> None:
    attempts = {"count": 0}

    def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError("temporary failure")
        return "success"

    resilience = AgentResilience(max_retries=2, retry_delay_seconds=0)

    assert resilience.execute_with_retry(operation) == "success"
    assert attempts["count"] == 3


def test_retry_raises_after_exhausting_attempts() -> None:
    resilience = AgentResilience(max_retries=1, retry_delay_seconds=0)

    with pytest.raises(RuntimeError, match="2 attempts"):
        resilience.execute_with_retry(lambda: (_ for _ in ()).throw(ValueError("bad input")))


def test_safe_agent_response_requires_manual_review() -> None:
    response = AgentResilience.safe_agent_response("ACME", "RAG unavailable")

    assert response["risk_level"] == "unknown"
    assert response["requires_escalation"] is True
    assert "Manual review" in response["compliance_summary"]
