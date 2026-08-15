"""Shared access to the single persisted ChromaDB collection.

ingest.py (writing) and rag.py (reading) both import get_collection() from
here, so they always talk to the same on-disk store through the same embedding
function. Using one embedding function for indexing AND querying is what makes
similarity search meaningful - vectors from two different models are not
comparable and produce results that look random.

Both documents live in ONE collection. Two collections cannot be searched by a
single query, which would make every cross-document question impossible.
"""
from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

import config


@lru_cache(maxsize=1)
def get_client():
    """Process-wide PersistentClient pointed at CHROMA_DIR.

    PersistentClient writes the index to disk, so chunks indexed in one run are
    still searchable after the app is stopped and started again (Stage 5's
    restart test).
    """
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


@lru_cache(maxsize=1)
def get_embedding_function():
    """Embedding function for every chunk and every query, chosen by PROVIDER."""
    if config.PROVIDER == "ollama":
        return embedding_functions.OllamaEmbeddingFunction(
            url=config.OLLAMA_BASE_URL,
            model_name=config.OLLAMA_EMBED_MODEL,
        )
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.require_api_key(),
        model_name=config.EMBEDDING_MODEL,
    )


def get_collection():
    """Get (or create) the one collection holding both documents.

    Cosine distance is the right default for OpenAI embeddings, which are
    normalised, so cosine and dot-product rank identically.
    """
    return get_client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Delete the collection so the next rebuild starts from empty.

    Used by --reset and by the UI's "Clear index" button. Re-running ingestion
    without this is still safe: chunk ids are deterministic, so a repeat run
    overwrites rather than adds (see ingest._chunk_id).
    """
    client = get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet; that is fine.
        pass
