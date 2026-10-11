from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional, TypedDict

from langgraph.graph import END, StateGraph
from rapidfuzz import fuzz, process

from src.common.models import PolicyViolation, RAGResponse
from src.component3_agent.audit_logger import AuditLogger

if TYPE_CHECKING:
    from src.component2_rag.rag_pipeline import FinancialRAGPipeline


class VendorResearchState(TypedDict, total=False):
    # Input
    vendor_name: str
    analyst_id: str
    date_range: tuple[str, str]

    # Extracted
    vendor_transactions: list[dict]
    matched_vendor_id: Optional[str]
    fuzzy_match_confidence: float

    # Policy
    applicable_policies: list[str]
    payment_limits: dict
    approval_requirements: dict

    # Analysis
    total_spend: float
    transaction_count: int
    policy_violations: list[PolicyViolation]
    risk_level: str

    # Output
    compliance_summary: str
    recommendations: list[str]
    citations: list[str]
    requires_escalation: bool

    # Internal
    policy_context: str
    clarification_required: bool
    clarification_message: str
    query_id: str


class FinancialResearchAgent:
    """
    LangGraph-based financial document research agent.

    Flow:

        extract_vendor_info
                |
                v
        fetch_transactions
                |
                v
        retrieve_policies
                |
                v
        analyse_compliance
                |
                v
        check_escalation_required
             /       \
            /         \
       escalation    summary
           |            |
           v            v
      create_alert   generate_summary
           |            |
           v            v
          END          END
    """

    # Simulated vendor master.
    # In production this would come from a vendor database.
    VENDOR_MASTER = [
        {
            "vendor_id": "V001",
            "vendor_name": "SHARMA RAJ TRADING PVT LTD",
            "aliases": [
                "SHARMA RAJ TRADING",
                "SHARMA RAJ",
                "SRTPL",
            ],
        },
        {
            "vendor_id": "V002",
            "vendor_name": "ABC TECHNOLOGIES PRIVATE LIMITED",
            "aliases": [
                "ABC TECHNOLOGIES",
                "ABC TECH",
                "ABC TECHNOLOGY",
            ],
        },
        {
            "vendor_id": "V003",
            "vendor_name": "GLOBAL OFFICE SUPPLIES LTD",
            "aliases": [
                "GLOBAL OFFICE",
                "GLOBAL SUPPLIES",
                "GOS LTD",
            ],
        },
        {
            "vendor_id": "V004",
            "vendor_name": "ACME CONSULTING SERVICES",
            "aliases": [
                "ACME CONSULTING",
                "ACME",
            ],
        },
    ]

    # Simulated transaction database.
    TRANSACTIONS_DB = [
        {
            "transaction_id": "TXN001",
            "vendor_id": "V001",
            "vendor_name": "SHARMA RAJ TRADING PVT LTD",
            "amount": 25000.00,
            "date": "2026-01-15",
            "payment_method": "NEFT",
            "approved_by": "manager",
            "invoice_id": "INV-001",
        },
        {
            "transaction_id": "TXN002",
            "vendor_id": "V001",
            "vendor_name": "SHARMA RAJ TRADING PVT LTD",
            "amount": 75000.00,
            "date": "2026-02-10",
            "payment_method": "NEFT",
            "approved_by": "manager",
            "invoice_id": "INV-002",
        },
        {
            "transaction_id": "TXN003",
            "vendor_id": "V001",
            "vendor_name": "SHARMA RAJ TRADING PVT LTD",
            "amount": 150000.00,
            "date": "2026-03-05",
            "payment_method": "RTGS",
            "approved_by": "manager",
            "invoice_id": "INV-003",
        },
        {
            "transaction_id": "TXN004",
            "vendor_id": "V002",
            "vendor_name": "ABC TECHNOLOGIES PRIVATE LIMITED",
            "amount": 45000.00,
            "date": "2026-02-15",
            "payment_method": "NEFT",
            "approved_by": "manager",
            "invoice_id": "INV-004",
        },
        {
            "transaction_id": "TXN005",
            "vendor_id": "V003",
            "vendor_name": "GLOBAL OFFICE SUPPLIES LTD",
            "amount": 12000.00,
            "date": "2026-01-20",
            "payment_method": "UPI",
            "approved_by": "manager",
            "invoice_id": "INV-005",
        },
        {
            "transaction_id": "TXN006",
            "vendor_id": "V004",
            "vendor_name": "ACME CONSULTING SERVICES",
            "amount": 200000.00,
            "date": "2026-03-10",
            "payment_method": "RTGS",
            "approved_by": "director",
            "invoice_id": "INV-006",
        },
    ]

    def __init__(
        self,
        rag_pipeline: Optional[FinancialRAGPipeline] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        if rag_pipeline is None:
            # Avoid loading embedding and reranking dependencies unless the
            # default RAG implementation is actually required.
            from src.component2_rag.rag_pipeline import FinancialRAGPipeline

            self.rag = FinancialRAGPipeline()
        else:
            self.rag = rag_pipeline
        self.audit_logger = audit_logger or AuditLogger()

        self.graph = self._build_graph()

    # =========================================================
    # Graph construction
    # =========================================================

    def _build_graph(self):
        workflow = StateGraph(VendorResearchState)

        workflow.add_node(
            "extract_vendor_info",
            self.extract_vendor_info,
        )

        workflow.add_node(
            "fetch_transactions",
            self.fetch_transactions,
        )

        workflow.add_node(
            "retrieve_policies",
            self.retrieve_policies,
        )

        workflow.add_node(
            "analyse_compliance",
            self.analyse_compliance,
        )

        workflow.add_node(
            "check_escalation_required",
            self.check_escalation_required,
        )

        workflow.add_node(
            "generate_summary",
            self.generate_summary,
        )

        workflow.add_node(
            "create_escalation_alert",
            self.create_escalation_alert,
        )

        workflow.set_entry_point(
            "extract_vendor_info"
        )

        workflow.add_edge(
            "extract_vendor_info",
            "fetch_transactions",
        )

        workflow.add_edge(
            "fetch_transactions",
            "retrieve_policies",
        )

        workflow.add_edge(
            "retrieve_policies",
            "analyse_compliance",
        )

        workflow.add_edge(
            "analyse_compliance",
            "check_escalation_required",
        )

        workflow.add_conditional_edges(
            "check_escalation_required",
            self.route_after_risk_check,
            {
                "escalation": "create_escalation_alert",
                "summary": "generate_summary",
                "clarification": END,
            },
        )

        workflow.add_edge(
            "generate_summary",
            END,
        )

        workflow.add_edge(
            "create_escalation_alert",
            END,
        )

        return workflow.compile()

    # =========================================================
    # NODE 1
    # =========================================================

    def extract_vendor_info(
        self,
        state: VendorResearchState,
    ) -> VendorResearchState:
        """
        Fuzzy match analyst-provided vendor name
        against the vendor master.
        """

        vendor_name = state["vendor_name"]

        if not vendor_name.strip():
            return {
                "matched_vendor_id": None,
                "fuzzy_match_confidence": 0.0,
                "clarification_required": True,
                "clarification_message": (
                    "Vendor name is required."
                ),
            }

        candidates = []

        for vendor in self.VENDOR_MASTER:
            candidates.append(
                (
                    vendor["vendor_name"],
                    vendor["vendor_id"],
                )
            )

            for alias in vendor.get("aliases", []):
                candidates.append(
                    (
                        alias,
                        vendor["vendor_id"],
                    )
                )

        candidate_names = [
            item[0]
            for item in candidates
        ]

        # Normalise case before fuzzy matching.  Vendor master records are
        # uppercase while analysts commonly enter title-case names.
        matches = process.extract(
            vendor_name.upper(),
            candidate_names,
            scorer=fuzz.token_set_ratio,
            limit=5,
        )

        if not matches:
            return {
                "matched_vendor_id": None,
                "fuzzy_match_confidence": 0.0,
                "clarification_required": True,
                "clarification_message": (
                    f"No vendor match found for "
                    f"'{vendor_name}'."
                ),
            }

        best_name, score, _ = matches[0]

        matched_vendor_id = next(
            vendor_id
            for name, vendor_id in candidates
            if name == best_name
        )

        confidence = score / 100.0

        return {
            "matched_vendor_id": matched_vendor_id,
            "fuzzy_match_confidence": confidence,
            "clarification_required": confidence < 0.7,
            "clarification_message": (
                f"Vendor '{vendor_name}' matched to "
                f"'{best_name}' with confidence "
                f"{confidence:.2f}. "
                "Please confirm the vendor."
                if confidence < 0.7
                else ""
            ),
        }

    # =========================================================
    # NODE 2
    # =========================================================

    def fetch_transactions(
        self,
        state: VendorResearchState,
    ) -> VendorResearchState:
        """
        Fetch transactions for matched vendor
        and apply date-range filtering and deduplication.
        """

        vendor_id = state.get(
            "matched_vendor_id"
        )

        confidence = state.get(
            "fuzzy_match_confidence",
            0.0,
        )

        if not vendor_id or confidence < 0.7:
            return {
                "vendor_transactions": [],
                "transaction_count": 0,
                "total_spend": 0.0,
            }

        start_date, end_date = state[
            "date_range"
        ]

        transactions = []

        for transaction in self.TRANSACTIONS_DB:
            if transaction["vendor_id"] != vendor_id:
                continue

            transaction_date = transaction["date"]

            if not (
                start_date
                <= transaction_date
                <= end_date
            ):
                continue

            transactions.append(
                dict(transaction)
            )

        # Deduplicate using transaction ID.
        unique_transactions = {}

        for transaction in transactions:
            unique_transactions[
                transaction["transaction_id"]
            ] = transaction

        transactions = list(
            unique_transactions.values()
        )

        total_spend = sum(
            float(transaction["amount"])
            for transaction in transactions
        )

        return {
            "vendor_transactions": transactions,
            "transaction_count": len(
                transactions
            ),
            "total_spend": total_spend,
        }

    # =========================================================
    # NODE 3
    # =========================================================

    def retrieve_policies(
        self,
        state: VendorResearchState,
    ) -> VendorResearchState:
        """
        Retrieve relevant policies from Component 2 RAG.

        Three independent policy questions are sent to RAG:
        1. Vendor payment policy
        2. Spend/payment limits
        3. Approval requirements
        """

        vendor_name = state["vendor_name"]

        questions = [
            (
                "vendor_payment",
                f"""
                What policies apply to payments made to vendor
                '{vendor_name}'? Include payment controls,
                documentation requirements and restrictions.
                """,
            ),
            (
                "spend_limits",
                f"""
                What are the approved spending and payment limits
                for vendor payments? Include monetary thresholds
                and applicable approval levels.
                """,
            ),
            (
                "approval_requirements",
                f"""
                What approval requirements apply to vendor payments?
                Explain who must approve payments at different
                transaction amounts.
                """,
            ),
        ]

        responses: list[RAGResponse] = []

        for _, question in questions:
            try:
                response = self.rag.query(
                    question=question.strip(),
                    user_id=state["analyst_id"],
                )

                responses.append(response)

            except Exception as exc:
                # Keep the agent alive.
                # Resilience layer can retry externally.
                responses.append(
                    RAGResponse(
                        answer=(
                            "Policy retrieval failed: "
                            f"{exc}"
                        ),
                        citations=[],
                        confidence=0.0,
                        retrieved_chunks=[],
                        sufficiency_check_passed=False,
                        faithfulness_score=0.0,
                        query_id=str(uuid.uuid4()),
                        processing_time_ms=0.0,
                    )
                )

        policy_text = "\n\n".join(
            response.answer
            for response in responses
        )

        citations = []

        for response in responses:
            for citation in response.citations:
                citations.append(
                    (
                        f"[{citation.document_name}, "
                        f"{citation.section}]"
                    )
                )

        # Defaults make the agent usable even when the
        # RAG answer does not expose machine-readable limits.
        payment_limits = {
            "manager_limit": 50000.0,
            "director_limit": 100000.0,
            "finance_head_limit": 500000.0,
        }

        approval_requirements = {
            "up_to_50000": "manager",
            "50000_to_100000": "director",
            "above_100000": "finance_head",
        }

        # Try to extract limits from retrieved RAG text.
        extracted_limits = self._extract_limits(
            policy_text
        )

        if extracted_limits:
            payment_limits.update(
                extracted_limits
            )

        applicable_policies = list(
            dict.fromkeys(citations)
        )

        return {
            "applicable_policies": applicable_policies,
            "payment_limits": payment_limits,
            "approval_requirements": approval_requirements,
            "policy_context": policy_text,
            "citations": applicable_policies,
        }

    # =========================================================
    # NODE 4
    # =========================================================

    def analyse_compliance(
        self,
        state: VendorResearchState,
    ) -> VendorResearchState:
        """
        Cross-check every transaction against
        payment limits and approval requirements.
        """

        transactions = state.get(
            "vendor_transactions",
            [],
        )

        limits = state.get(
            "payment_limits",
            {},
        )

        approval_requirements = state.get(
            "approval_requirements",
            {},
        )

        violations: list[PolicyViolation] = []

        manager_limit = float(
            limits.get(
                "manager_limit",
                50000,
            )
        )

        director_limit = float(
            limits.get(
                "director_limit",
                100000,
            )
        )

        finance_limit = float(
            limits.get(
                "finance_head_limit",
                500000,
            )
        )

        for transaction in transactions:
            amount = float(
                transaction["amount"]
            )

            approved_by = str(
                transaction.get(
                    "approved_by",
                    "",
                )
            ).lower()

            transaction_id = transaction[
                "transaction_id"
            ]

            required_approver = (
                self._required_approver(
                    amount,
                    manager_limit,
                    director_limit,
                )
            )

            # Check approval violation.
            if self._approval_rank(
                approved_by
            ) < self._approval_rank(
                required_approver
            ):
                severity = self._severity_for_amount(
                    amount,
                    director_limit,
                    finance_limit,
                )

                violations.append(
                    PolicyViolation(
                        transaction_id=transaction_id,
                        violation_type=(
                            "APPROVAL_LIMIT_EXCEEDED"
                        ),
                        severity=severity,
                        description=(
                            f"Transaction amount "
                            f"{amount:.2f} requires "
                            f"{required_approver} approval, "
                            f"but was approved by "
                            f"{approved_by or 'unknown'}."
                        ),
                        applicable_policy=(
                            "Expense Approval Matrix"
                        ),
                        policy_section=(
                            "Payment Approval Limits"
                        ),
                    )
                )

            # Check maximum policy limit.
            if amount > finance_limit:
                violations.append(
                    PolicyViolation(
                        transaction_id=transaction_id,
                        violation_type=(
                            "PAYMENT_LIMIT_EXCEEDED"
                        ),
                        severity="critical",
                        description=(
                            f"Transaction amount "
                            f"{amount:.2f} exceeds the "
                            f"maximum approved payment "
                            f"limit of {finance_limit:.2f}."
                        ),
                        applicable_policy=(
                            "Vendor Payment Policy"
                        ),
                        policy_section=(
                            "Maximum Payment Limit"
                        ),
                    )
                )

        risk_level = self._calculate_risk(
            violations
        )

        return {
            "policy_violations": violations,
            "risk_level": risk_level,
        }

    # =========================================================
    # NODE 5
    # =========================================================

    def check_escalation_required(
        self,
        state: VendorResearchState,
    ) -> VendorResearchState:
        """
        Decide whether the research should be escalated.
        """

        confidence = state.get(
            "fuzzy_match_confidence",
            0.0,
        )

        if confidence < 0.7:
            return {
                "requires_escalation": False,
                "clarification_required": True,
                "clarification_message": (
                    state.get(
                        "clarification_message",
                        "Vendor match confidence is below 0.7.",
                    )
                ),
            }

        risk_level = state.get(
            "risk_level",
            "low",
        )

        requires_escalation = (
            risk_level
            in ["high", "critical"]
        )

        return {
            "requires_escalation": (
                requires_escalation
            )
        }

    # =========================================================
    # NODE 6
    # =========================================================

    def generate_summary(
        self,
        state: VendorResearchState,
    ) -> VendorResearchState:
        """
        Generate structured compliance summary
        for low/medium risk cases.
        """

        transactions = state.get(
            "vendor_transactions",
            [],
        )

        violations = state.get(
            "policy_violations",
            [],
        )

        vendor_name = state[
            "vendor_name"
        ]

        total_spend = state.get(
            "total_spend",
            0.0,
        )

        recommendations = self._generate_recommendations(
            violations
        )

        citation_text = state.get(
            "citations",
            [],
        )

        summary = self._format_summary(
            vendor_name=vendor_name,
            state=state,
            recommendations=recommendations,
            citation_text=citation_text,
        )

        return {
            "compliance_summary": summary,
            "recommendations": recommendations,
        }

    # =========================================================
    # NODE 7
    # =========================================================

    def create_escalation_alert(
        self,
        state: VendorResearchState,
    ) -> VendorResearchState:
        """
        Generate a detailed alert for the finance manager.
        """

        violations = state.get(
            "policy_violations",
            [],
        )

        vendor_name = state[
            "vendor_name"
        ]

        alert_lines = [
            "FINANCIAL COMPLIANCE ESCALATION",
            "",
            f"Vendor: {vendor_name}",
            f"Risk Level: {state.get('risk_level', 'unknown').upper()}",
            f"Total Spend: {state.get('total_spend', 0.0):.2f}",
            f"Transaction Count: {state.get('transaction_count', 0)}",
            "",
            "VIOLATIONS:",
        ]

        for violation in violations:
            alert_lines.extend(
                [
                    f"- Transaction: "
                    f"{violation.transaction_id}",
                    f"  Type: "
                    f"{violation.violation_type}",
                    f"  Severity: "
                    f"{violation.severity}",
                    f"  Description: "
                    f"{violation.description}",
                    f"  Policy: "
                    f"{violation.applicable_policy}",
                    f"  Section: "
                    f"{violation.policy_section}",
                    "",
                ]
            )

        alert_lines.append(
            "Recommended action: "
            "Finance manager review required."
        )

        recommendations = [
            "Suspend further payment approval "
            "until the violation is reviewed.",
            "Verify supporting invoices and approvals.",
            "Obtain approval from the required authority.",
            "Record the compliance exception in the audit trail.",
        ]

        summary = "\n".join(
            alert_lines
        )

        summary += (
            "\n\nRECOMMENDATIONS:\n"
            + "\n".join(
                f"- {item}"
                for item in recommendations
            )
        )

        summary += (
            "\n\nPOLICY CITATIONS:\n"
            + "\n".join(
                state.get("citations", [])
            )
        )

        return {
            "compliance_summary": summary,
            "recommendations": recommendations,
            "requires_escalation": True,
        }

    # =========================================================
    # ROUTING
    # =========================================================

    @staticmethod
    def route_after_risk_check(
        state: VendorResearchState,
    ) -> str:
        """
        Conditional routing.

        confidence < 0.7
            -> clarification / END

        high / critical
            -> escalation

        low / medium
            -> summary
        """

        confidence = state.get(
            "fuzzy_match_confidence",
            0.0,
        )

        if confidence < 0.7:
            return "clarification"

        risk_level = state.get(
            "risk_level",
            "low",
        )

        if risk_level in [
            "high",
            "critical",
        ]:
            return "escalation"

        return "summary"

    # =========================================================
    # PUBLIC API
    # =========================================================

    def research(
        self,
        vendor_name: str,
        analyst_id: str,
        date_range: tuple[str, str],
    ) -> VendorResearchState:
        """
        Run end-to-end vendor research.
        """

        query_id = str(
            uuid.uuid4()
        )

        initial_state: VendorResearchState = {
            "vendor_name": vendor_name,
            "analyst_id": analyst_id,
            "date_range": date_range,
            "vendor_transactions": [],
            "matched_vendor_id": None,
            "fuzzy_match_confidence": 0.0,
            "applicable_policies": [],
            "payment_limits": {},
            "approval_requirements": {},
            "total_spend": 0.0,
            "transaction_count": 0,
            "policy_violations": [],
            "risk_level": "low",
            "compliance_summary": "",
            "recommendations": [],
            "citations": [],
            "requires_escalation": False,
            "policy_context": "",
            "clarification_required": False,
            "clarification_message": "",
            "query_id": query_id,
        }

        final_state = self.graph.invoke(
            initial_state
        )

        self.audit_logger.log_agent_run(
            query_id=query_id,
            analyst_id=analyst_id,
            state=final_state,
        )

        return final_state

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _extract_limits(
        text: str,
    ) -> dict:
        """
        Attempt to extract rupee limits from RAG output.
        """

        limits = {}

        patterns = {
            "manager_limit": [
                r"manager.{0,50}?(?:₹|rs\.?|inr)?\s?([\d,]+)",
                r"([\d,]+).{0,50}?manager",
            ],
            "director_limit": [
                r"director.{0,50}?(?:₹|rs\.?|inr)?\s?([\d,]+)",
                r"([\d,]+).{0,50}?director",
            ],
            "finance_head_limit": [
                r"finance.{0,50}?head.{0,50}?(?:₹|rs\.?|inr)?\s?([\d,]+)",
            ],
        }

        for key, regexes in patterns.items():
            for pattern in regexes:
                match = re.search(
                    pattern,
                    text,
                    re.IGNORECASE,
                )

                if match:
                    value = match.group(1).replace(
                        ",",
                        "",
                    )

                    try:
                        limits[key] = float(
                            value
                        )
                        break
                    except ValueError:
                        continue

        return limits

    @staticmethod
    def _required_approver(
        amount: float,
        manager_limit: float,
        director_limit: float,
    ) -> str:

        if amount <= manager_limit:
            return "manager"

        if amount <= director_limit:
            return "director"

        return "finance_head"

    @staticmethod
    def _approval_rank(
        approver: str,
    ) -> int:

        ranks = {
            "unknown": 0,
            "manager": 1,
            "director": 2,
            "finance_head": 3,
            "cfo": 4,
        }

        return ranks.get(
            approver.lower(),
            0,
        )

    @staticmethod
    def _severity_for_amount(
        amount: float,
        director_limit: float,
        finance_limit: float,
    ) -> str:

        if amount > finance_limit:
            return "critical"

        if amount > director_limit:
            return "high"

        return "medium"

    @staticmethod
    def _calculate_risk(
        violations: list[PolicyViolation],
    ) -> str:

        if not violations:
            return "low"

        severities = {
            violation.severity.lower()
            for violation in violations
        }

        if "critical" in severities:
            return "critical"

        if "high" in severities:
            return "high"

        if "medium" in severities:
            return "medium"

        return "low"

    @staticmethod
    def _generate_recommendations(
        violations: list[PolicyViolation],
    ) -> list[str]:

        if not violations:
            return [
                "No policy violations identified.",
                "Continue normal vendor payment monitoring.",
            ]

        recommendations = []

        for violation in violations:
            if (
                violation.violation_type
                == "APPROVAL_LIMIT_EXCEEDED"
            ):
                recommendations.append(
                    "Obtain approval from the required "
                    "approval authority."
                )

            if (
                violation.violation_type
                == "PAYMENT_LIMIT_EXCEEDED"
            ):
                recommendations.append(
                    "Escalate the transaction for finance "
                    "manager review."
                )

        recommendations.append(
            "Verify supporting invoices and "
            "transaction documentation."
        )

        return list(
            dict.fromkeys(
                recommendations
            )
        )

    @staticmethod
    def _format_summary(
        vendor_name: str,
        state: VendorResearchState,
        recommendations: list[str],
        citation_text: list[str],
    ) -> str:

        violations = state.get(
            "policy_violations",
            [],
        )

        lines = [
            "VENDOR COMPLIANCE SUMMARY",
            "",
            f"Vendor: {vendor_name}",
            f"Matched Vendor ID: "
            f"{state.get('matched_vendor_id')}",
            f"Match Confidence: "
            f"{state.get('fuzzy_match_confidence', 0.0):.2f}",
            f"Date Range: "
            f"{state.get('date_range')}",
            "",
            f"Transaction Count: "
            f"{state.get('transaction_count', 0)}",
            f"Total Spend: "
            f"{state.get('total_spend', 0.0):.2f}",
            f"Risk Level: "
            f"{state.get('risk_level', 'low').upper()}",
            "",
            "POLICY VIOLATIONS:",
        ]

        if violations:
            for violation in violations:
                lines.append(
                    f"- {violation.transaction_id}: "
                    f"{violation.description}"
                )
                lines.append(
                    f"  Policy: "
                    f"{violation.applicable_policy}"
                )
                lines.append(
                    f"  Section: "
                    f"{violation.policy_section}"
                )
        else:
            lines.append(
                "No policy violations identified."
            )

        lines.extend(
            [
                "",
                "RECOMMENDATIONS:",
            ]
        )

        for recommendation in recommendations:
            lines.append(
                f"- {recommendation}"
            )

        lines.extend(
            [
                "",
                "POLICY CITATIONS:",
            ]
        )

        if citation_text:
            lines.extend(
                citation_text
            )
        else:
            lines.append(
                "No policy citations returned."
            )

        return "\n".join(lines)


if __name__ == "__main__":
    agent = FinancialResearchAgent()

    result = agent.research(
        vendor_name="Sharma Raj Trading",
        analyst_id="analyst-001",
        date_range=(
            "2026-01-01",
            "2026-12-31",
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )
