"""Central configuration for the Supply Chain RAG system.

Every tunable value lives here so ingest.py, rag.py, app.py and the FastAPI
backend share one source of truth. Anything can be overridden with an
environment variable (loaded from .env) without touching the code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file sitting next to this file (if present).
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Paths -----------------------------------------------------------------
# Absolute paths, so the store and the documents are found no matter which
# directory the app is launched from.
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Vector store ----------------------------------------------------------
# ONE collection holds both documents. Two collections cannot be searched in a
# single query, which would make every cross-document question impossible.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "supply_chain_docs")

# --- Models ----------------------------------------------------------------
# Provider backend: "openai" (the assignment stack) or "ollama" (local, free).
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
# Stage 7 asks for 0 to 0.2 so the same question gives the same answer.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# --- Models (Ollama, local) ------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def active_embed_model() -> str:
    """Name of the embedding model actually in use, per PROVIDER."""
    return OLLAMA_EMBED_MODEL if PROVIDER == "ollama" else EMBEDDING_MODEL


def active_llm_model() -> str:
    """Name of the answering model actually in use, per PROVIDER."""
    return OLLAMA_LLM_MODEL if PROVIDER == "ollama" else LLM_MODEL


# --- Chunking --------------------------------------------------------------
# 1100 / 200 sits at the top of the range the guide allows (800-1200 size,
# 100-200 overlap). Deliberately large because the handbook's penalty clauses
# are short and numbered: a smaller chunk risks cutting a consequence away
# from the trigger condition that it applies to, which is the single most
# damaging chunking mistake in this assignment (Stage 3 watch-out).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval -------------------------------------------------------------
# 6, not 4. The cross-document questions need a figure from the review AND a
# rule from the handbook, so retrieval has to spend slots on both documents.
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "6"))

# "balanced" reserves a fixed share of the slots for each document type
# (Stage 6, fix 3). "plain" is a single unfiltered similarity search, kept so
# the failure it causes can be demonstrated rather than just described.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "balanced").strip().lower()

# Minimum chunks guaranteed from each document type in balanced mode.
MIN_PER_DOC_TYPE = int(os.getenv("MIN_PER_DOC_TYPE", "2"))

# Document types used for the balanced quota and shown in the source list.
DOC_TYPES = ("review", "policy")

DOC_TYPE_LABELS = {
    "review": "Performance Review",
    "policy": "Procurement Policy",
    "other": "Other document",
}

# --- Request handling ------------------------------------------------------
# Seconds to wait for a single model call before giving up. A stalled request
# with no timeout hangs the UI and the batch test runs indefinitely, so this has
# a value by default rather than relying on the SDK's.
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90"))
# Retries for a timed-out or failed request, handled inside the OpenAI client.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
# Upper bound on the answer length. Without a cap, a small local model can run
# to its full context window repeating itself, which looks exactly like a hang.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "900"))

# --- API key ---------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- FastAPI backend URL (used only when the UI runs in "API mode") --------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def require_api_key() -> str:
    """Return the OpenAI key, or raise a clear error if it is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or set PROVIDER=ollama to run locally without a key."
        )
    return OPENAI_API_KEY
