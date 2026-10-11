from src.common.models import RAGResponse
from src.component3_agent.research_agent import FinancialResearchAgent


class FakeRAG:
    def query(self, question: str, user_id: str) -> RAGResponse:
        return RAGResponse(
            answer="Approval limits: manager 50,000, director 100,000, finance head 500,000.",
            citations=[],
            confidence=1.0,
            retrieved_chunks=[],
            sufficiency_check_passed=True,
            faithfulness_score=1.0,
            query_id="rag-query",
            processing_time_ms=1.0,
        )


class FakeAuditLogger:
    def __init__(self) -> None:
        self.calls = []

    def log_agent_run(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_high_risk_vendor_is_escalated() -> None:
    audit = FakeAuditLogger()
    agent = FinancialResearchAgent(rag_pipeline=FakeRAG(), audit_logger=audit)

    result = agent.research(
        vendor_name="Sharma Raj Trading",
        analyst_id="analyst-1",
        date_range=("2026-01-01", "2026-12-31"),
    )

    assert result["matched_vendor_id"] == "V001"
    assert result["transaction_count"] == 3
    assert result["total_spend"] == 250000.0
    assert result["risk_level"] == "high"
    assert result["requires_escalation"] is True
    assert "FINANCIAL COMPLIANCE ESCALATION" in result["compliance_summary"]
    assert len(audit.calls) == 1


def test_low_risk_vendor_returns_normal_summary() -> None:
    agent = FinancialResearchAgent(rag_pipeline=FakeRAG(), audit_logger=FakeAuditLogger())

    result = agent.research(
        vendor_name="Global Office Supplies",
        analyst_id="analyst-1",
        date_range=("2026-01-01", "2026-12-31"),
    )

    assert result["matched_vendor_id"] == "V003"
    assert result["risk_level"] == "low"
    assert result["requires_escalation"] is False
    assert "VENDOR COMPLIANCE SUMMARY" in result["compliance_summary"]
