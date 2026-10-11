import json
import time
from pathlib import Path
from typing import Any

from src.common.config import Settings
from src.component2_rag.rag_pipeline import FinancialRAGPipeline


class RAGEvaluator:
    """
    Evaluates the Financial RAG pipeline.

    Evaluation includes:
    - Answer generation success
    - Expected answer comparison
    - Citation presence
    - Context sufficiency
    - Faithfulness score
    - Confidence score
    - Processing latency
    """

    def __init__(self, pipeline: FinancialRAGPipeline | None = None):
        self.pipeline = pipeline or FinancialRAGPipeline()

        Settings.create_directories()

        self.output_path = (
            Settings.OUTPUT_DIR / "rag_eval_report.json"
        )

    def load_evaluation_data(self, dataset_path: str) -> list[dict[str, Any]]:
        """
        Load evaluation questions from a JSON or JSONL file.

        Supported JSON format:

        [
            {
                "question": "What is the expense approval limit?",
                "expected_answer": "Expenses above ... require approval."
            }
        ]

        Supported JSONL format:

        {"question": "...", "expected_answer": "..."}
        """

        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {dataset_path}"
            )

        if path.suffix.lower() == ".jsonl":
            records = []

            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON on line {line_number}: {exc}"
                        ) from exc

            return records

        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, list):
                raise ValueError(
                    "Evaluation JSON must contain a list of records."
                )

            return data

        raise ValueError(
            "Evaluation dataset must be .json or .jsonl"
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for simple answer comparison."""

        return " ".join(
            text.lower().strip().split()
        )

    def _calculate_answer_match(
        self,
        actual_answer: str,
        expected_answer: str,
    ) -> float:
        """
        Calculate a simple lexical answer match score.

        This is not intended to replace semantic evaluation.
        It provides a deterministic baseline for the assignment.
        """

        actual = self._normalize_text(actual_answer)
        expected = self._normalize_text(expected_answer)

        if not actual or not expected:
            return 0.0

        if actual == expected:
            return 1.0

        expected_words = set(expected.split())
        actual_words = set(actual.split())

        if not expected_words:
            return 0.0

        overlap = expected_words.intersection(actual_words)

        return len(overlap) / len(expected_words)

    def evaluate(
        self,
        dataset_path: str,
        user_id: str = "evaluation_user",
    ) -> dict[str, Any]:
        """
        Run the complete RAG evaluation.
        """

        evaluation_data = self.load_evaluation_data(
            dataset_path
        )

        if not evaluation_data:
            raise ValueError(
                "Evaluation dataset is empty."
            )

        results = []

        total_time = time.perf_counter()

        for index, item in enumerate(
            evaluation_data,
            start=1,
        ):
            question = item.get("question")

            if not question:
                results.append(
                    {
                        "case_id": index,
                        "question": "",
                        "success": False,
                        "error": "Missing question",
                    }
                )
                continue

            expected_answer = item.get(
                "expected_answer",
                "",
            )

            case_start = time.perf_counter()

            try:
                response = self.pipeline.query(
                    question=question,
                    user_id=user_id,
                )

                case_time_ms = (
                    time.perf_counter() - case_start
                ) * 1000

                answer_match = self._calculate_answer_match(
                    response.answer,
                    expected_answer,
                )

                citation_count = len(
                    response.citations
                )

                citations_present = (
                    citation_count > 0
                    and response.sufficiency_check_passed
                )

                result = {
                    "case_id": index,
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": response.answer,
                    "answer_match_score": round(
                        answer_match,
                        4,
                    ),
                    "success": True,
                    "sufficiency_check_passed": (
                        response.sufficiency_check_passed
                    ),
                    "faithfulness_score": round(
                        response.faithfulness_score,
                        4,
                    ),
                    "confidence": round(
                        response.confidence,
                        4,
                    ),
                    "citation_count": citation_count,
                    "citations_present": citations_present,
                    "retrieved_chunks": len(
                        response.retrieved_chunks
                    ),
                    "query_id": response.query_id,
                    "processing_time_ms": round(
                        response.processing_time_ms,
                        2,
                    ),
                    "evaluation_time_ms": round(
                        case_time_ms,
                        2,
                    ),
                }

            except Exception as exc:
                case_time_ms = (
                    time.perf_counter() - case_start
                ) * 1000

                result = {
                    "case_id": index,
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": "",
                    "answer_match_score": 0.0,
                    "success": False,
                    "error": str(exc),
                    "sufficiency_check_passed": False,
                    "faithfulness_score": 0.0,
                    "confidence": 0.0,
                    "citation_count": 0,
                    "citations_present": False,
                    "retrieved_chunks": 0,
                    "evaluation_time_ms": round(
                        case_time_ms,
                        2,
                    ),
                }

            results.append(result)

            print(
                f"[{index}/{len(evaluation_data)}] "
                f"{question}"
            )

        total_time_ms = (
            time.perf_counter() - total_time
        ) * 1000

        summary = self._build_summary(
            results,
            total_time_ms,
        )

        report = {
            "evaluation": {
                "dataset": str(dataset_path),
                "total_cases": len(evaluation_data),
                "total_time_ms": round(
                    total_time_ms,
                    2,
                ),
            },
            "summary": summary,
            "results": results,
        }

        self._save_report(report)

        return report

    def _build_summary(
        self,
        results: list[dict[str, Any]],
        total_time_ms: float,
    ) -> dict[str, Any]:
        """Calculate aggregate evaluation metrics."""

        total = len(results)

        if total == 0:
            return {}

        successful = [
            result
            for result in results
            if result.get("success")
        ]

        successful_count = len(successful)

        if successful_count == 0:
            return {
                "success_rate": 0.0,
                "average_answer_match": 0.0,
                "average_faithfulness": 0.0,
                "average_confidence": 0.0,
                "citation_coverage": 0.0,
                "sufficiency_pass_rate": 0.0,
                "average_latency_ms": 0.0,
            }

        average_answer_match = (
            sum(
                result["answer_match_score"]
                for result in successful
            )
            / successful_count
        )

        average_faithfulness = (
            sum(
                result["faithfulness_score"]
                for result in successful
            )
            / successful_count
        )

        average_confidence = (
            sum(
                result["confidence"]
                for result in successful
            )
            / successful_count
        )

        citation_coverage = (
            sum(
                1
                for result in successful
                if result["citations_present"]
            )
            / successful_count
        )

        sufficiency_pass_rate = (
            sum(
                1
                for result in successful
                if result["sufficiency_check_passed"]
            )
            / successful_count
        )

        average_latency = (
            sum(
                result["processing_time_ms"]
                for result in successful
            )
            / successful_count
        )

        return {
            "success_rate": round(
                successful_count / total,
                4,
            ),
            "average_answer_match": round(
                average_answer_match,
                4,
            ),
            "average_faithfulness": round(
                average_faithfulness,
                4,
            ),
            "average_confidence": round(
                average_confidence,
                4,
            ),
            "citation_coverage": round(
                citation_coverage,
                4,
            ),
            "sufficiency_pass_rate": round(
                sufficiency_pass_rate,
                4,
            ),
            "average_latency_ms": round(
                average_latency,
                2,
            ),
            "total_evaluation_time_ms": round(
                total_time_ms,
                2,
            ),
        }

    def _save_report(
        self,
        report: dict[str, Any],
    ) -> None:
        """Save evaluation report as JSON."""

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\nEvaluation report saved to: "
            f"{self.output_path}"
        )


def main() -> None:
    """
    Example command-line execution.

    Change the dataset path if your evaluation
    dataset is stored somewhere else.
    """

    evaluator = RAGEvaluator()

    report = evaluator.evaluate(
        dataset_path="data/rag_evaluation.json",
        user_id="evaluation_user",
    )

    print("\n========== RAG EVALUATION ==========")

    summary = report["summary"]

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
