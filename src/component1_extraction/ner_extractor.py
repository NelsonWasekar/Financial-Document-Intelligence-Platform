import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class FinancialNER:
    """
    Callable financial transaction NER service.

    The model extracts:
      ORG, AMOUNT, DATE, REF, METHOD

    BANK_CODE is extracted with a deterministic rule because bank/IFSC-like
    codes are highly structured and were not part of the requested BIO labels.
    """

    METHOD_VALUES = {"NEFT", "RTGS", "IMPS", "UPI", "CHEQUE"}

    def __init__(self, model_path="./financial-ner-model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, narration):
        enc = self.tokenizer(
            narration,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=256
        )
        offsets = enc.pop("offset_mapping")[0].tolist()

        logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)

        tags = [
            self.model.config.id2label[int(i)]
            for i in pred_ids
        ]

        token_entities = []
        current = None

        for tag, prob, (start, end) in zip(tags, probs, offsets):
            if start == end:
                continue

            if tag.startswith("B-"):
                if current:
                    token_entities.append(current)
                current = {
                    "label": tag[2:],
                    "start": start,
                    "end": end,
                    "scores": [float(prob.max())]
                }

            elif tag.startswith("I-") and current and current["label"] == tag[2:]:
                current["end"] = end
                current["scores"].append(float(prob.max()))

            else:
                if current:
                    token_entities.append(current)
                    current = None

        if current:
            token_entities.append(current)

        entities = []
        for e in token_entities:
            text = narration[e["start"]:e["end"]].strip()
            if not text:
                continue

            confidence = sum(e["scores"]) / len(e["scores"])
            entities.append({
                "text": text,
                "label": e["label"],
                "confidence": round(confidence, 4)
            })

        # Deterministic METHOD fallback/normalization
        method = next(
            (x["text"].upper() for x in entities if x["label"] == "METHOD"),
            None
        )
        if method not in self.METHOD_VALUES:
            match = re.search(r"\b(NEFT|RTGS|IMPS|UPI|CHEQUE)\b", narration.upper())
            method = match.group(1) if match else method

        # Structured bank-code fallback.
        bank_match = re.search(
            r"\b(?:HDFC|ICIC|SBIN|AXIS|KKBK|YESB|BARB)[A-Z0-9]{5,12}\b",
            narration.upper()
        )

        return {
            "entities": entities,
            "payment_method": method,
            "bank_code": bank_match.group(0) if bank_match else None
        }

if __name__ == "__main__":
    ner = FinancialNER("./financial-ner-model")

    narration = "NEFT/HDFC0001234/SHARMA RAJ TRADING PVT LTD/INV-2024-0891"
    print(ner(narration))
