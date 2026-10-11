from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.common.config import Settings


class AuditLogger:
    """
    Persistent audit logger for financial research agent runs.

    Every agent execution is stored as a JSON record.
    """

    def __init__(
        self,
        audit_dir: Path | None = None,
    ):
        Settings.create_directories()

        self.audit_dir = (
            audit_dir
            or Settings.AUDIT_DIR
        )

        self.audit_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log_agent_run(
        self,
        query_id: str,
        analyst_id: str,
        state: dict[str, Any],
    ) -> None:
        """
        Save an agent execution to the audit trail.
        """

        record = {
            "query_id": query_id,
            "analyst_id": analyst_id,
            "timestamp": time.time(),
            "vendor_name": state.get(
                "vendor_name"
            ),
            "matched_vendor_id": state.get(
                "matched_vendor_id"
            ),
            "fuzzy_match_confidence": state.get(
                "fuzzy_match_confidence",
                0.0,
            ),
            "date_range": state.get(
                "date_range"
            ),
            "transaction_count": state.get(
                "transaction_count",
                0,
            ),
            "total_spend": state.get(
                "total_spend",
                0.0,
            ),
            "risk_level": state.get(
                "risk_level",
                "low",
            ),
            "requires_escalation": state.get(
                "requires_escalation",
                False,
            ),
            "citations": state.get(
                "citations",
                [],
            ),
            "policy_violations": [
                asdict(violation)
                for violation
                in state.get(
                    "policy_violations",
                    [],
                )
            ],
            "recommendations": state.get(
                "recommendations",
                [],
            ),
            "compliance_summary": state.get(
                "compliance_summary",
                "",
            ),
        }

        output_file = (
            self.audit_dir
            / f"{query_id}.json"
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                record,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    def read_audit(
        self,
        query_id: str,
    ) -> dict[str, Any] | None:
        """
        Read one audit record.
        """

        path = (
            self.audit_dir
            / f"{query_id}.json"
        )

        if not path.exists():
            return None

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
