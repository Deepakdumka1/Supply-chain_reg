"""Ingestion pipeline: read PDF -> chunk -> embed -> store in ChromaDB.

Command line:

    python ingest.py                    # index every PDF in data/
    python ingest.py data/review.pdf    # index specific files
    python ingest.py --reset            # wipe the store first, then index data/
    python ingest.py --preview           # extract and chunk only, store nothing

Or import ingest_paths() / ingest_data_dir() from the UI and the API.

Three things make this pipeline different from a naive one, and all three exist
to serve the cross-document questions:

1.  Every chunk carries file name, page number, document type (review/policy)
    and the section heading it sits under. Document type is what lets retrieval
    reserve slots for each document; the page number is what makes an answer
    usable to a buyer who has to open the PDF and confirm the wording.

2.  Chunk boundaries are aligned to the documents' own numbered structure. Each
    numbered clause is exposed to the splitter as a paragraph, so a penalty
    consequence is never cut away from the trigger condition it applies to.

3.  Chunk ids are deterministic, so re-running ingestion overwrites instead of
    adding a second copy of every chunk.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config
from vector_store import get_collection, reset_collection

# A line that opens a numbered section or a numbered clause, e.g.
#   "8. Escalation"                      -> section heading
#   "7.4 Quality: defect rate ..."       -> subsection heading
#   "7.4 Trigger: a supplier's defect .."-> the clause body itself
#   "4.2.2 Where a part is single-sourced, all of the following ..."
# The trailing [A-Z] requirement keeps table cells such as "2.42 lakh per
# container" or "0.25 x total lead time" from being mistaken for headings.
NUMBERED_START = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+[A-Z]")

# A heading is a numbered line short enough to be a title rather than the first
# wrapped line of a paragraph. Body lines in these PDFs wrap at ~100-115 chars.
HEADING_MAX_LEN = 90


# --- Step 1: read the PDF --------------------------------------------------
def load_pdf_pages(path: str | Path) -> list[tuple[int, str]]:
    """Extract text from a PDF, one entry per page.

    Returns (page_number, text) tuples with 1-based page numbers so they match
    what a PDF viewer shows. Pages with no extractable text (blank, or a scanned
    image) are skipped rather than stored as empty chunks.
    """
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((index + 1, text))
    return pages


def strip_repeated_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop running headers and footers that repeat across most pages.

    The document title and "Page 4" appear on every page. Left in, they waste
    room in every chunk and add a line of noise to every embedding. Rather than
    hard-coding this document's header text, any short line appearing on more
    than 60% of pages is treated as furniture and removed.
    """
    if len(pages) < 3:
        return pages

    counts: Counter = Counter()
    for _, text in pages:
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            if len(line) <= 120:
                counts[line] += 1

    threshold = max(2, int(len(pages) * 0.6))
    furniture = {line for line, n in counts.items() if n >= threshold}
    # Page numbers differ per page, so catch them by shape as well.
    page_number_re = re.compile(r"^Page\s+\d+(\s+of\s+\d+)?$", re.IGNORECASE)

    cleaned: list[tuple[int, str]] = []
    for page_number, text in pages:
        kept = [
            ln for ln in text.splitlines()
            if ln.strip() not in furniture and not page_number_re.match(ln.strip())
        ]
        body = "\n".join(kept).strip()
        if body:
            cleaned.append((page_number, body))
    return cleaned


# --- Step 2: work out which document this is ------------------------------
def classify_document(file_name: str, sample_text: str = "") -> str:
    """Return the document type: "review", "policy" or "other".

    The type is the metadata field the balanced retrieval in rag.py filters on,
    so it has to be right even for an arbitrarily named upload. The file name is
    checked first; if it says nothing useful, the first page is sniffed.
    """
    name = file_name.lower()
    if any(word in name for word in ("policy", "policies", "handbook", "manual", "procedure")):
        return "policy"
    if any(word in name for word in ("review", "performance", "scorecard", "report")):
        return "review"

    head = sample_text[:2500].lower()
    policy_hits = sum(
        head.count(word)
        for word in ("policy handbook", "clause", "shall", "handbook", "approval authority")
    )
    review_hits = sum(
        head.count(word)
        for word in ("performance review", "quarter", "scorecard", "spend", "on-time delivery")
    )
    if policy_hits > review_hits:
        return "policy"
    if review_hits > policy_hits:
        return "review"
    return "other"


# --- Step 3: chunk ---------------------------------------------------------
def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Recursive character splitter at the configured size and overlap.

    "\\n\\n" is the first separator it tries. mark_structure() below puts a blank
    line in front of every numbered clause, so the splitter's preferred cut
    points are the document's own clause boundaries. It only falls back to a
    single newline, then a sentence, then a space, when a single clause is
    longer than the chunk size.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def mark_structure(text: str) -> str:
    """Insert a blank line before each numbered section or clause.

    This is the Stage 3 fix. The handbook's consequences are short numbered
    clauses; a chunk that reads "a debit note equal to 2% of the quarterly
    invoice value" with no trigger attached is useless to the model. Making the
    clause boundary the splitter's highest-priority separator means a clause is
    only ever cut if it alone exceeds the chunk size.
    """
    out: list[str] = []
    for line in text.splitlines():
        if NUMBERED_START.match(line.strip()) and out:
            out.append("")
        out.append(line)
    return "\n".join(out)


def find_sections(text: str) -> list[tuple[int, str]]:
    """Locate heading lines in a page, as (character offset, heading text)."""
    sections: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= HEADING_MAX_LEN and NUMBERED_START.match(stripped):
            sections.append((offset, stripped))
        offset += len(line) + 1
    return sections


def _section_for(sections: list[tuple[int, str]], position: int, fallback: str) -> str:
    """The last heading at or before this position in the page."""
    current = fallback
    for offset, heading in sections:
        if offset <= position:
            current = heading
        else:
            break
    return current


def chunk_pages(
    file_name: str,
    doc_type: str,
    pages: list[tuple[int, str]],
) -> list[dict]:
    """Split each page into chunks that carry their own provenance.

    Every chunk's text is prefixed with a source label naming the file, the
    document type, the page and the section. The label is embedded along with
    the content, which helps a query about "the penalty for late delivery" land
    on the handbook rather than on the review's delivery figures, and it lets
    the model cite correctly even from a bare passage.
    """
    splitter = _build_splitter()
    type_label = config.DOC_TYPE_LABELS.get(doc_type, doc_type)
    chunks: list[dict] = []
    carried_section = "Front matter"

    for page_number, raw_text in pages:
        marked = mark_structure(raw_text)
        sections = find_sections(marked)
        cursor = 0

        for piece in splitter.split_text(marked):
            cleaned = piece.strip()
            if not cleaned:
                continue

            found = marked.find(piece, cursor)
            if found == -1:
                found = cursor
            cursor = found + max(1, len(piece) - config.CHUNK_OVERLAP)
            section = _section_for(sections, found, carried_section)

            label = (
                f"[Source: {file_name} | {type_label} | page {page_number} "
                f"| section: {section}]"
            )
            chunks.append(
                {
                    "text": f"{label}\n{cleaned}",
                    "file": file_name,
                    "page": page_number,
                    "doc_type": doc_type,
                    "section": section,
                }
            )

        if sections:
            carried_section = sections[-1][1]

    return chunks


def _chunk_id(file_name: str, page: int, position: int, text: str) -> str:
    """Deterministic id, so re-indexing a file updates instead of duplicating.

    Run ingestion twice without this and every clause is stored twice; the
    duplicates then crowd out the chunk from the other document that the
    cross-document question actually needed.
    """
    raw = f"{file_name}:{page}:{position}:{text[:96]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# --- Step 4: embed and store ----------------------------------------------
def prepare_file(path: str | Path) -> dict:
    """Read and chunk one PDF without touching the vector store.

    Returned dict: {"file", "doc_type", "pages", "chunks", "error"}.
    Used by ingest_paths(), by --preview, and by the chunking checks.
    """
    path = Path(path)
    if not path.exists():
        return {"file": path.name, "error": "not found", "chunks": [], "pages": 0}

    pages = load_pdf_pages(path)
    if not pages:
        return {
            "file": path.name,
            "error": "no extractable text - scanned image?",
            "chunks": [],
            "pages": 0,
        }

    page_count = len(pages)
    pages = strip_repeated_lines(pages)
    doc_type = classify_document(path.name, pages[0][1] if pages else "")
    chunks = chunk_pages(path.name, doc_type, pages)

    return {
        "file": path.name,
        "doc_type": doc_type,
        "pages": page_count,
        "chunks": chunks,
        "error": None if chunks else "produced no chunks",
    }


def ingest_paths(paths: Iterable[str | Path]) -> dict:
    """Load, chunk, embed and store the given PDFs in the one collection.

    Returns {"files", "chunks", "per_file": [...], "skipped": [...]} so the UI
    and the API can report counts per file, as Stage 3 asks.
    """
    collection = get_collection()
    files_processed = 0
    total_chunks = 0
    per_file: list[dict] = []
    skipped: list[str] = []

    for raw_path in paths:
        prepared = prepare_file(raw_path)
        if prepared["error"]:
            skipped.append(f"{prepared['file']} ({prepared['error']})")
            continue

        chunks = prepared["chunks"]
        ids, documents, metadatas = [], [], []
        for position, chunk in enumerate(chunks):
            ids.append(_chunk_id(chunk["file"], chunk["page"], position, chunk["text"]))
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "file": chunk["file"],
                    "page": chunk["page"],
                    "doc_type": chunk["doc_type"],
                    "section": chunk["section"],
                }
            )

        # Embed in batches rather than one call per chunk (Stage 4).
        batch = 100
        for start in range(0, len(ids), batch):
            end = start + batch
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        files_processed += 1
        total_chunks += len(chunks)
        per_file.append(
            {
                "file": prepared["file"],
                "doc_type": prepared["doc_type"],
                "pages": prepared["pages"],
                "chunks": len(chunks),
            }
        )

    return {
        "files": files_processed,
        "chunks": total_chunks,
        "per_file": per_file,
        "skipped": skipped,
    }


def ingest_data_dir() -> dict:
    """Index every PDF currently sitting in the data/ folder."""
    return ingest_paths(sorted(config.DATA_DIR.glob("*.pdf")))


def collection_stats() -> dict:
    """Summary for the /stats endpoint and the UI sidebar.

    Includes the per-document and per-type breakdown, which is how you prove
    both documents actually reached the same collection, and how you compare the
    chunk count before and after a restart.
    """
    collection = get_collection()
    total = collection.count()

    by_file: Counter = Counter()
    by_type: Counter = Counter()
    if total:
        stored = collection.get(include=["metadatas"])
        for meta in stored.get("metadatas", []) or []:
            by_file[meta.get("file", "unknown")] += 1
            by_type[meta.get("doc_type", "other")] += 1

    return {
        "provider": config.PROVIDER,
        "collection": config.COLLECTION_NAME,
        "total_chunks": total,
        "chunks_by_document": dict(by_file),
        "chunks_by_type": dict(by_type),
        "embedding_model": config.active_embed_model(),
        "llm_model": config.active_llm_model(),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "default_top_k": config.DEFAULT_TOP_K,
        "persist_directory": config.CHROMA_DIR,
    }


# --- CLI -------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Index supply chain PDFs into a single ChromaDB collection."
    )
    parser.add_argument("paths", nargs="*", help="PDFs to index (default: everything in data/).")
    parser.add_argument("--reset", action="store_true", help="Delete the index before ingesting.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Extract and chunk only. Prints counts and a sample; stores nothing.",
    )
    args = parser.parse_args()

    paths = args.paths or [str(p) for p in sorted(config.DATA_DIR.glob("*.pdf"))]
    if not paths:
        print(f"No PDFs found in {config.DATA_DIR}.")
        print("Run: python scripts/make_sample_documents.py")
        return

    if args.preview:
        # Stage 2 checkpoint: page count per file, plus a sample to read.
        for path in paths:
            prepared = prepare_file(path)
            if prepared["error"]:
                print(f"{prepared['file']}: SKIPPED ({prepared['error']})")
                continue
            print(f"\n{prepared['file']}")
            print(f"  document type : {prepared['doc_type']}")
            print(f"  pages         : {prepared['pages']}")
            print(f"  chunks        : {len(prepared['chunks'])}")
            sizes = [len(c["text"]) for c in prepared["chunks"]]
            print(f"  chunk chars   : min {min(sizes)}, mean {sum(sizes)//len(sizes)}, max {max(sizes)}")
            print("  first 300 characters of page 1:")
            print("    " + prepared["chunks"][0]["text"][:300].replace("\n", "\n    "))
        return

    if args.reset:
        reset_collection()
        print("Cleared existing collection.")

    result = ingest_paths(paths)
    print(f"{result['files']} files processed, {result['chunks']} chunks stored.")
    for item in result["per_file"]:
        print(
            f"  - {item['file']}: {item['chunks']} chunks "
            f"from {item['pages']} pages (type: {item['doc_type']})"
        )
    for item in result["skipped"]:
        print(f"  ! skipped {item}")

    stats = collection_stats()
    print(f"Collection '{stats['collection']}' now holds {stats['total_chunks']} chunks.")
    print(f"  by document type: {stats['chunks_by_type']}")
    print(f"  persisted at: {stats['persist_directory']}")


if __name__ == "__main__":
    _cli()
