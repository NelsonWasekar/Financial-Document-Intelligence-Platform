import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ---------------------------------------------------------
    # Project directories
    # ---------------------------------------------------------
    BASE_DIR = Path(os.getenv("BASE_DIR", "."))

    CORPUS_DIR = BASE_DIR / "corpus"
    INDEX_DIR = BASE_DIR / "data" / "index"
    AUDIT_DIR = BASE_DIR / "data" / "audit"
    OUTPUT_DIR = BASE_DIR / "outputs"

    # ---------------------------------------------------------
    # Embedding model
    # ---------------------------------------------------------
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------------------------------------
    # Cross Encoder / Reranker
    # ---------------------------------------------------------
    RERANKER_MODEL = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ---------------------------------------------------------
    # Ollama LLM configuration
    # ---------------------------------------------------------
    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    OLLAMA_MODEL = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:3b"
    )

    # ---------------------------------------------------------
    # Retrieval configuration
    # ---------------------------------------------------------
    TOP_K_RETRIEVAL = int(
        os.getenv("TOP_K_RETRIEVAL", "10")
    )

    TOP_K_GENERATION = int(
        os.getenv("TOP_K_GENERATION", "3")
    )

    # Reciprocal Rank Fusion constant
    RRF_K = int(
        os.getenv("RRF_K", "60")
    )

    # ---------------------------------------------------------
    # Faithfulness
    # ---------------------------------------------------------
    FAITHFULNESS_THRESHOLD = float(
        os.getenv("FAITHFULNESS_THRESHOLD", "0.85")
    )

    # Maximum citation excerpt length
    MAX_CITATION_EXCERPT = 100

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------
    APP_NAME = os.getenv(
        "APP_NAME",
        "Financial RAG System"
    )

    APP_VERSION = os.getenv(
        "APP_VERSION",
        "1.0.0"
    )

    # ---------------------------------------------------------
    # Create required directories
    # ---------------------------------------------------------
    @classmethod
    def create_directories(cls):
        cls.INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        cls.AUDIT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        cls.OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )