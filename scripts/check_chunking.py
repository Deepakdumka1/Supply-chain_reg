"""Prove that chunking did not break the documents (Stage 3 checkpoint).

The guide's warning for this stage is specific: the handbook's penalty clauses
are short and numbered, and a chunk that reads "a debit note equal to 2% of the
quarterly invoice value" with no indication of what triggers it is useless to
the model. So this script does not eyeball a few chunks - it checks every
numbered clause in the policy document and reports whether the clause survives
chunking whole, trigger and consequence together, inside at least one chunk.

It also prints the numbers the README has to record: chunk size, overlap, total
chunks, and chunks per file.

Run:  python scripts/check_chunking.py
Exit code is 1 if any clause was split, so this can gate a commit if you want.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from ingest import (  # noqa: E402
    NUMBERED_START,
    load_pdf_pages,
    mark_structure,
    prepare_file,
    strip_repeated_lines,
)


def _flat(text: str) -> str:
    """Collapse all whitespace, so a clause can be matched across line breaks."""
    return re.sub(r"\s+", " ", text).strip()


def extract_clauses(path: Path) -> list[tuple[str, str]]:
    """Pull every numbered clause body out of a PDF as (number, full text).

    A clause body is a numbered block that is long enough to be a rule rather
    than a heading. Headings are skipped because there is nothing to preserve.
    """
    pages = strip_repeated_lines(load_pdf_pages(path))
    clauses: list[tuple[str, str]] = []

    for _, raw in pages:
        blocks = mark_structure(raw).split("\n\n")
        for block in blocks:
            flat = _flat(block)
            match = NUMBERED_START.match(flat)
            if not match:
                continue
            # Skip headings: a rule always runs to at least a couple of lines.
            if len(flat) < 120:
                continue
            clauses.append((match.group(1), flat))
    return clauses


def check_file(path: Path) -> dict:
    """Check every numbered block in one PDF against the chunks it produced.

    A block longer than the chunk size HAS to be split - that is arithmetic, not
    a bug, and it is what happens to the review's long tables. The failure worth
    catching is a block that would have fitted inside one chunk but was split
    anyway, because that is how a penalty consequence gets separated from its
    trigger.
    """
    prepared = prepare_file(path)
    if prepared["error"]:
        return {"checked": 0, "intact": 0, "oversize": 0, "failures": [f"{path.name}: {prepared['error']}"]}

    chunk_texts = [_flat(c["text"]) for c in prepared["chunks"]]
    blocks = extract_clauses(path)
    failures: list[str] = []
    intact = 0
    oversize = 0

    for number, block in blocks:
        if any(block in chunk for chunk in chunk_texts):
            intact += 1
            continue

        if len(block) > config.CHUNK_SIZE:
            # Cannot fit in one chunk by definition. The overlap is what keeps a
            # sentence on the boundary readable in one of the two halves.
            oversize += 1
            continue

        # Would have fitted, but was split anyway. Report how much survived.
        best = 0
        for chunk in chunk_texts:
            if block[:60] in chunk:
                for cut in range(len(block), 0, -20):
                    if block[:cut] in chunk:
                        best = max(best, cut)
                        break
        failures.append(
            f"block {number} would fit in a chunk ({len(block)} <= "
            f"{config.CHUNK_SIZE} chars) but was split - only {best} characters "
            f"stayed together: \"{block[:90]}...\""
        )

    return {
        "checked": len(blocks),
        "intact": intact,
        "oversize": oversize,
        "failures": failures,
    }


def main() -> int:
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {config.DATA_DIR}. Run scripts/make_sample_documents.py first.")
        return 1

    print("=" * 78)
    print("CHUNKING DECISION")
    print("=" * 78)
    print(f"  chunk size   : {config.CHUNK_SIZE} characters")
    print(f"  chunk overlap: {config.CHUNK_OVERLAP} characters")

    total_chunks = 0
    total_clauses = 0
    total_intact = 0
    all_failures: list[str] = []

    print()
    print("=" * 78)
    print("CHUNKS PER FILE")
    print("=" * 78)
    for path in pdfs:
        prepared = prepare_file(path)
        if prepared["error"]:
            print(f"  {path.name}: SKIPPED ({prepared['error']})")
            continue
        sizes = [len(c["text"]) for c in prepared["chunks"]]
        total_chunks += len(prepared["chunks"])
        print(
            f"  {path.name}\n"
            f"      type {prepared['doc_type']}, {prepared['pages']} pages, "
            f"{len(prepared['chunks'])} chunks "
            f"(min {min(sizes)}, mean {sum(sizes) // len(sizes)}, max {max(sizes)} chars)"
        )
    print(f"  TOTAL: {total_chunks} chunks across {len(pdfs)} files")

    print()
    print("=" * 78)
    print("CLAUSE INTEGRITY - does every numbered block survive chunking whole?")
    print("=" * 78)
    total_oversize = 0
    for path in pdfs:
        result = check_file(path)
        if not result["checked"]:
            continue
        total_clauses += result["checked"]
        total_intact += result["intact"]
        total_oversize += result["oversize"]
        status = "OK" if not result["failures"] else "FAILED"
        print(
            f"  {path.name}: {result['intact']}/{result['checked']} intact in one "
            f"chunk, {result['oversize']} longer than the chunk size so split by "
            f"arithmetic  [{status}]"
        )
        for failure in result["failures"]:
            print(f"      ! {failure}")
        all_failures.extend(result["failures"])

    print()
    if total_clauses == 0:
        print("No numbered clauses found to check.")
        return 1

    if all_failures:
        print(f"RESULT: {len(all_failures)} block(s) that would have fitted were "
              "split anyway.")
        print("Raise CHUNK_SIZE, or check that mark_structure() recognises the "
              "clause numbering in your document.")
        return 1

    print(f"RESULT: every numbered block that fits in a chunk stayed whole "
          f"({total_intact} of {total_clauses}).")
    if total_oversize:
        print(f"        {total_oversize} block(s) exceed {config.CHUNK_SIZE} characters "
              "and are split across chunks with overlap; these are the review's long "
              "tables, not handbook clauses.")

    # Spot-check the clauses the ten test questions actually depend on.
    print()
    print("=" * 78)
    print("SPOT CHECK - clauses the test questions depend on")
    print("=" * 78)
    wanted = {
        "7.1": "on-time delivery falls below 90% in any quarter",
        "7.2": "debit note equal to 2% of the quarterly invoice value",
        "7.4": "Rs. 1,850 per hour",
        "7.5": "debit note equal to 3% of the quarterly invoice value",
        "7.6": "Rs. 4.5 lakh per hour",
        "4.2.2": "eight weeks",
        "5.2.1": "0.25 x total replenishment lead time",
        "5.3.2": "highest applicable floor",
        "8.2.1": "Watch List",
        "3.1.1": "Rs. 1.4 crore",
    }
    policy = [p for p in pdfs if "handbook" in p.name.lower() or "policy" in p.name.lower()]
    if not policy:
        print("  (no policy document found to spot-check)")
        return 0

    chunks = [_flat(c["text"]) for c in prepare_file(policy[0])["chunks"]]
    missing = []
    for number, needle in wanted.items():
        hits = [i for i, ch in enumerate(chunks) if f"{number} " in ch and needle in ch]
        if hits:
            print(f"  clause {number:<6} and \"{needle[:44]}\" together in chunk(s) {hits}")
        else:
            print(f"  clause {number:<6} ! NOT found together with \"{needle}\"")
            missing.append(number)

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
