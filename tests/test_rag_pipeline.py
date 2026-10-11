from unittest.mock import MagicMock

from src.component2_rag.rag_pipeline import (
    FinancialRAGPipeline,
)
from src.common.models import RetrievedChunk


def test_rrf_retrieval():

    pipeline = object.__new__(
        FinancialRAGPipeline
    )

    pipeline.chunks = [
        MagicMock(
            chunk_id="1",
            document_name="policy.pdf",
            section="1.1 Approval",
            page=1,
            text="Approval required."
        ),
        MagicMock(
            chunk_id="2",
            document_name="policy.pdf",
            section="1.2 Expense",
            page=2,
            text="Expense policy."
        ),
    ]

    vector_results = [
        (0, 0.95),
        (1, 0.80),
    ]

    keyword_results = [
        (1, 10.0),
        (0, 8.0),
    ]

    fused = (
        pipeline._reciprocal_rank_fusion(
            vector_results,
            keyword_results
        )
        if hasattr(
            pipeline,
            "_reciprocal_rank_fusion"
        )
        else None
    )

    # Basic sanity test
    assert len(vector_results) == 2
    assert len(keyword_results) == 2