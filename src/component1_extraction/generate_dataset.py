# Regenerate a synthetic 500-example dataset if needed.
# This is demonstration data, NOT the original assignment dataset.

import json, random
from pathlib import Path

random.seed(42)

ORGS = [
    "SHARMA RAJ TRADING PVT LTD", "ABC INDUSTRIES LTD",
    "TATA SERVICES", "RELIANCE RETAIL", "MAHINDRA LOGISTICS",
    "SUNRISE ENTERPRISES", "GLOBAL TECH SOLUTIONS",
    "KUMAR ELECTRONICS", "GREENFIELD SUPPLIERS",
    "NATIONAL DISTRIBUTORS", "PRIME AUTO PARTS", "ORBITAL SYSTEMS"
]
METHODS = ["NEFT", "RTGS", "IMPS", "UPI", "CHEQUE"]
BANKS = ["HDFC", "ICIC", "SBIN", "AXIS", "KKBK", "YESB", "BARB"]
REF_PREFIXES = ["INV", "REF", "NEFT", "RTGS", "PO", "TXN"]

def make_example():
    method = random.choice(METHODS)
    org = random.choice(ORGS)
    amount = f"{random.randint(500, 250000):,}"
    date = f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.choice([2024,2025,2026])}"
    bank = f"{random.choice(BANKS)}{random.randint(100000,999999)}"
    ref = f"{random.choice(REF_PREFIXES)}-{random.choice([2024,2025,2026])}-{random.randint(1000,9999)}"

    text = random.choice([
        f"{method}/{bank}/{org}/{ref}/AMT {amount}/DATE {date}",
        f"{date} {method} {org} {ref} INR {amount}",
        f"{method} {org} {ref} {amount} {date}",
        f"Payment via {method} to {org}, reference {ref}, amount INR {amount}, date {date}"
    ])

    entities = []
    for value, label in [
        (org, "ORG"), (amount, "AMOUNT"), (date, "DATE"),
        (ref, "REF"), (method, "METHOD")
    ]:
        start = text.find(value)
        if start >= 0:
            entities.append([start, start + len(value), label])

    return {"text": text, "entities": entities}

path = Path("transactions_labeled.jsonl")
with path.open("w", encoding="utf-8") as f:
    for _ in range(500):
        f.write(json.dumps(make_example()) + "\n")

print(f"Generated {path} with 500 examples.")
