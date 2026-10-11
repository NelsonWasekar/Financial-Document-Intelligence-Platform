from types import SimpleNamespace

import torch

from src.component1_extraction.ner_extractor import FinancialNER


class FakeTokenizer:
    def __init__(self, offsets: list[tuple[int, int]]) -> None:
        self.offsets = offsets

    def __call__(self, narration: str, **kwargs) -> dict[str, torch.Tensor]:
        token_count = len(self.offsets)
        return {
            "input_ids": torch.zeros((1, token_count), dtype=torch.long),
            "attention_mask": torch.ones((1, token_count), dtype=torch.long),
            "offset_mapping": torch.tensor([self.offsets]),
        }


class FakeModel:
    def __init__(self, labels: list[str]) -> None:
        self.config = SimpleNamespace(id2label={index: label for index, label in enumerate(labels)})
        self.labels = labels

    def __call__(self, **kwargs) -> SimpleNamespace:
        token_count = kwargs["input_ids"].shape[1]
        logits = torch.full((1, token_count, len(self.labels)), -10.0)
        for index in range(token_count):
            logits[0, index, index] = 10.0
        return SimpleNamespace(logits=logits)


def make_ner(offsets: list[tuple[int, int]], labels: list[str]) -> FinancialNER:
    ner = object.__new__(FinancialNER)
    ner.tokenizer = FakeTokenizer(offsets)
    ner.model = FakeModel(labels)
    return ner


def test_extracts_entities_and_normalises_payment_and_bank_code() -> None:
    narration = "NEFT HDFC0001234 ACME"
    ner = make_ner(
        offsets=[(0, 0), (0, 4), (5, 15), (17, 21), (0, 0)],
        labels=["O", "B-METHOD", "O", "B-ORG", "O"],
    )

    result = ner(narration)

    assert result["payment_method"] == "NEFT"
    assert result["bank_code"] == "HDFC0001234"
    assert result["entities"] == [
        {"text": "NEFT", "label": "METHOD", "confidence": 1.0},
        {"text": "ACME", "label": "ORG", "confidence": 1.0},
    ]


def test_extracts_all_required_financial_entity_labels() -> None:
    narration = "NEFT HDFC0001234 ACME 25000 2026-03-10 INV-42"
    ner = make_ner(
        offsets=[
            (0, 4),
            (5, 16),
            (17, 21),
            (22, 27),
            (28, 38),
            (39, 45),
        ],
        labels=["B-METHOD", "O", "B-ORG", "B-AMOUNT", "B-DATE", "B-REF"],
    )

    result = ner(narration)

    assert [(entity["text"], entity["label"]) for entity in result["entities"]] == [
        ("NEFT", "METHOD"),
        ("ACME", "ORG"),
        ("25000", "AMOUNT"),
        ("2026-03-10", "DATE"),
        ("INV-42", "REF"),
    ]
    assert result["payment_method"] == "NEFT"
    assert result["bank_code"] == "HDFC0001234"


def test_merges_bio_tokens_into_one_entity() -> None:
    narration = "ACME CONSULTING paid"
    ner = make_ner(
        offsets=[(0, 4), (5, 15), (16, 20)],
        labels=["B-ORG", "I-ORG", "O"],
    )

    result = ner(narration)

    assert result["entities"] == [
        {"text": "ACME CONSULTING", "label": "ORG", "confidence": 1.0}
    ]
    assert result["payment_method"] is None
    assert result["bank_code"] is None
