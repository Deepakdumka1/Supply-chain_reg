"""FastAPI backend (Stage 10 bonus): the brain, separated from the face.

Right now the Streamlit interface can do everything itself - read files, embed,
search, call the model. In a real deployment those live apart: a service holds
the logic and interfaces call it over HTTP. This module is that service.

Three endpoints:

    POST /ingest   multipart PDF upload   -> how many files and chunks handled
    POST /ask      question + top_k       -> answer with its sources
    GET  /stats    -                      -> collection name, chunk count, models

Run it from the project root:

    uvicorn api.main:app --reload

...or from inside the api/ folder:

    uvicorn main:app --reload

Then try every endpoint from the automatic documentation page at
http://localhost:8000/docs before pointing the interface at it.

NOTE ON ACCESS CONTROL: these endpoints are unauthenticated, which is fine for
a local assignment on localhost but is not safe to expose. /ingest accepts file
uploads and /ask spends model credits, so anything reachable from a network
needs an API key or a reverse proxy in front of it first.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

# Make the project root importable whether uvicorn is launched from the root
# (`uvicorn api.main:app`) or from inside api/ (`uvicorn main:app`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
from ingest import collection_stats, ingest_paths  # noqa: E402
from rag import answer_question  # noqa: E402

app = FastAPI(
    title="Supply Chain RAG API",
    description=(
        "Retrieval-Augmented Q&A across a quarterly Supply Chain Performance "
        "Review and a Procurement Policy Handbook. Both documents live in one "
        "Chroma collection so a single search can reach either, and retrieval "
        "reserves a share of its slots for each so cross-document questions get "
        "a figure and the rule it triggers."
    ),
    version="1.0.0",
)


# --- Schemas ---------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        examples=[
            "Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per "
            "million. Which clauses does this trigger, and what does it cost them?"
        ],
    )
    top_k: int = Field(config.DEFAULT_TOP_K, ge=1, le=20)
    mode: str = Field(
        config.RETRIEVAL_MODE,
        description="'balanced' reserves slots per document type; 'plain' is a "
        "single unfiltered similarity search.",
        examples=["balanced"],
    )


class Source(BaseModel):
    file: str
    page: Any
    doc_type: str
    section: str = ""


class Coverage(BaseModel):
    doc_types: dict = Field(default_factory=dict)
    documents: dict = Field(default_factory=dict)
    crossed_documents: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    coverage: Coverage
    retrieval_mode: str
    top_k: int


class PerFile(BaseModel):
    file: str
    doc_type: str
    pages: int
    chunks: int


class IngestResponse(BaseModel):
    files: int
    chunks: int
    per_file: list[PerFile] = []
    skipped: list[str] = []


class StatsResponse(BaseModel):
    provider: str
    collection: str
    total_chunks: int
    chunks_by_document: dict
    chunks_by_type: dict
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_mode: str
    default_top_k: int
    persist_directory: str


# --- Endpoints -------------------------------------------------------------
@app.get("/stats", response_model=StatsResponse, summary="Statistics about the store")
def stats() -> dict:
    """Collection name, chunk counts per document and per type, and the models.

    Use this for the restart test: note total_chunks, stop the service, start it
    again, and call this endpoint. The number must be identical, with no
    re-upload in between.
    """
    return collection_stats()


@app.post("/ingest", response_model=IngestResponse, summary="Upload and index PDFs")
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Accept one or more PDF uploads, index them, and report what was handled.

    Uploads are written to a temporary folder, indexed, and reported per file
    with the document type each was recognised as. Re-uploading the same file
    updates its chunks rather than adding a second copy, because chunk ids are
    derived from the file name, page and content.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    saved: list[str] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="supply_chain_ingest_"))
    for upload in files:
        name = Path(upload.filename or "upload.pdf").name
        dest = tmp_dir / name
        dest.write_bytes(await upload.read())
        saved.append(str(dest))

    result = ingest_paths(saved)
    if not result["files"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No file could be indexed.",
                "skipped": result["skipped"],
            },
        )
    return result


@app.post("/ask", response_model=AskResponse, summary="Ask a question")
def ask(payload: AskRequest) -> dict:
    """Answer a question from the indexed documents, with its sources.

    The response includes `coverage`, which reports how many chunks came from
    each document type and whether retrieval crossed both. On a cross-document
    question, `crossed_documents` must be true - if it is false the answer is
    not trustworthy however well it reads.
    """
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    mode = (payload.mode or config.RETRIEVAL_MODE).strip().lower()
    if mode not in {"balanced", "plain"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{mode}'. Use 'balanced' or 'plain'.",
        )

    result = answer_question(question, top_k=payload.top_k, mode=mode)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "coverage": result["coverage"],
        "retrieval_mode": result["retrieval_mode"],
        "top_k": result["top_k"],
    }


@app.get("/", summary="Service banner")
def root() -> dict:
    return {
        "service": "Supply Chain RAG API",
        "docs": "/docs",
        "endpoints": ["POST /ingest", "POST /ask", "GET /stats"],
    }
