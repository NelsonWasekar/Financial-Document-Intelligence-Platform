import pytest
from fastapi.testclient import TestClient

from src.component4_api.main import app, get_agent, get_audit_logger


class FakeAgent:
    def research(self, vendor_name: str, analyst_id: str, date_range: tuple[str, str]) -> dict:
        return {
            "query_id": "query-123",
            "vendor_name": vendor_name,
            "analyst_id": analyst_id,
            "date_range": date_range,
            "risk_level": "low",
            "requires_escalation": False,
            "policy_violations": [],
        }


class FakeAuditLogger:
    def read_audit(self, query_id: str):
        if query_id == "query-123":
            return {"query_id": query_id, "risk_level": "low"}
        return None


@pytest.fixture
def client():
    app.dependency_overrides[get_agent] = lambda: FakeAgent()
    app.dependency_overrides[get_audit_logger] = lambda: FakeAuditLogger()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_and_metrics_endpoints(client: TestClient) -> None:
    health = client.get("/health", headers={"X-Request-ID": "health-request"})

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.headers["X-Request-ID"] == "health-request"
    assert client.get("/metrics").status_code == 200


def test_research_endpoint_uses_agent_and_validates_dates(client: TestClient) -> None:
    response = client.post(
        "/api/v1/research",
        json={
            "vendor_name": "Global Office Supplies",
            "analyst_id": "analyst-1",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
        },
    )

    assert response.status_code == 200
    assert response.json()["query_id"] == "query-123"
    assert response.json()["request_id"]

    invalid = client.post(
        "/api/v1/research",
        json={
            "vendor_name": "ACME",
            "analyst_id": "analyst-1",
            "start_date": "2026-12-31",
            "end_date": "2026-01-01",
        },
    )
    assert invalid.status_code == 422


def test_audit_endpoint_returns_record_or_not_found(client: TestClient) -> None:
    assert client.get("/api/v1/audits/query-123").status_code == 200
    assert client.get("/api/v1/audits/missing").status_code == 404
