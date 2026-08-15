"""Retrieval-Augmented answering: retrieve chunks -> build prompt -> ask GPT-4o.

The public entry point is answer_question(). It returns the answer, the sources
(document name + page number), the chunks that were actually retrieved, and
which documents those chunks came from.

THE PROBLEM THIS FILE EXISTS TO SOLVE
-------------------------------------
A buyer's real question needs a figure from the performance review and a rule
from the policy handbook at the same time. Ask "Kaveri Metals had 88.1% on-time
delivery and 1,150 defects per million - which clauses does this trigger?" and
a plain similarity search returns six chunks that all come from the review,
because the supplier name and both numbers only ever appear there. The handbook
is never consulted, the model receives supplier scorecards and no rules, and it
answers from its own general knowledge of how procurement usually works. The
answer reads beautifully and is entirely invented.

Two of the three fixes the guide offers are applied here:

  * top_k is raised from 4 to 6, so the weaker document has room to appear.
  * "balanced" mode reserves a fixed share of the slots for each document type
    using the doc_type metadata, runs one filtered search per document, and
    merges the results. This is the fix that works every time rather than
    sometimes, which is why it is the default.

"plain" mode is kept so the failure can be demonstrated rather than just
described - scripts/compare_retrieval.py runs the same question both ways.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from functools import lru_cache

from openai import OpenAI

import config
from vector_store import get_collection

# --- The prompt (Stage 7) --------------------------------------------------
# Four parts: who the model is, the retrieved chunks, the user's question, and
# the rules. Rules 1 and 2 are the non-negotiable grounding pair. Rule 3 is the
# one this assignment adds: a buyer needs the figure, the clause it triggers and
# the resulting action stated separately, not summarised into a sentence.
SYSTEM_PROMPT = """You are a procurement analyst at Meridian Industrial Systems. \
You support buyers who act on your answers: they raise debit notes against \
suppliers, set inventory levels, and escalate supplier performance. An answer \
that sounds right but is not in the documents causes real financial damage.

You are given numbered context passages taken from two kinds of internal \
document:
  - a quarterly Supply Chain Performance Review, which contains measured \
figures: supplier scorecards, spend, on-time delivery, defect rates, freight \
costs, inventory, line stoppages and risks.
  - a Procurement Policy Handbook, which contains the rules: supplier \
classification, approval authority, penalty clauses, safety stock calculation \
and escalation paths.

Follow these rules strictly.

1. GROUND EVERY WORD IN THE CONTEXT. Answer only from the context passages \
below. Do not use any outside knowledge of how procurement, penalties, safety \
stock or approval limits usually work. If a figure, a percentage, a rate or a \
clause number does not appear in the context, you may not state it. This \
matters most for penalties: describing a typical industry penalty instead of \
Meridian's actual clause is the worst error you can make.

2. REFUSE WHEN THE ANSWER IS ABSENT - BUT CHECK FIRST. Before you refuse, read \
every passage and look for the specific supplier names, figures, bands, dates \
and clause numbers the question mentions. Refuse only when they are genuinely \
not there. If they are there, answer, even when the passages are terse, in a \
table, or spread across several passages. A wrong refusal is as damaging as a \
wrong answer: it sends the buyer to read the PDF by hand.
   When the context truly does not contain the answer, reply exactly: "The \
information is not available in the uploaded documents." You may then name what \
would be needed to answer. Never guess, and never fill a gap with something \
plausible. A refusal is all-or-nothing: if you refuse, that sentence is your \
entire answer. Never give an answer and then add a refusal, and never refuse and \
then answer anyway - decide one way and say only that.

3. SEPARATE THE FIGURE, THE CLAUSE AND THE ACTION. When an answer requires \
combining a measured figure with a rule, set the three out explicitly and \
label them:
   - Figure: the measured value, and the document it came from.
   - Clause: the clause number and the trigger condition it states.
   - Action: what follows - the consequence, the amount, and who does it.
   If several clauses are triggered, do this for each one, and say plainly that \
more than one applies. Show any arithmetic you do, with the inputs you used.

4. WHEN A RULE HAS A FLOOR, A CAP OR A MINIMUM, APPLY IT. Some rules give a \
formula and also a minimum or maximum that overrides the calculated result. \
Check the context for such a limit before answering, state both values, and say \
which one governs.

5. CITE AS YOU GO. Refer to the source in line, like (Handbook, p. 5) or \
(Review, p. 2), for each figure and each clause.

6. BE PRECISE, NOT FLUENT. Quote figures exactly as written, including units \
such as Rs. lakh or Rs. crore, PPM, days or percent. If the context is \
incomplete or the passages disagree, say what you can support and state what is \
missing.

7. ANSWER ONCE, WITHOUT PADDING. Give each clause and each figure a single \
time. Do not restate the same clause under several headings, and do not compare \
quantities that are not comparable, such as a lead time in days against a \
buffer requirement in weeks. If a clause has lettered sub-requirements, list \
each one once and say whether the context shows it is met."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """OpenAI client, or an OpenAI-compatible client pointed at local Ollama.

    A timeout is not optional here. Without one, a stalled generation request
    hangs the caller forever: the Streamlit spinner spins with nothing behind it,
    and a batch run of the test questions stops dead partway through with an idle
    CPU and no error. Both were observed while building this. The client retries
    a stalled request rather than surfacing the first blip as a failure.
    """
    if config.PROVIDER == "ollama":
        return OpenAI(
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
            timeout=config.REQUEST_TIMEOUT,
            max_retries=config.MAX_RETRIES,
        )
    return OpenAI(
        api_key=config.require_api_key(),
        timeout=config.REQUEST_TIMEOUT,
        max_retries=config.MAX_RETRIES,
    )


# --- Retrieval -------------------------------------------------------------
def _rows(results: dict) -> list[dict]:
    """Flatten one Chroma query result into a list of chunk dicts."""
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    rows: list[dict] = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        meta = meta or {}
        rows.append(
            {
                "id": chunk_id,
                "text": text,
                "file": meta.get("file", "unknown"),
                "page": meta.get("page", "?"),
                "doc_type": meta.get("doc_type", "other"),
                "section": meta.get("section", ""),
                "distance": distance,
            }
        )
    return rows


def _query(question: str, n_results: int, doc_type: str | None = None) -> list[dict]:
    """One similarity search, optionally restricted to a single document type."""
    collection = get_collection()
    if n_results <= 0:
        return []

    kwargs = {
        "query_texts": [question],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if doc_type:
        kwargs["where"] = {"doc_type": doc_type}

    return _rows(collection.query(**kwargs))


def present_doc_types() -> list[str]:
    """Which document types are actually in the collection right now.

    Balanced retrieval only reserves slots for documents that exist, so indexing
    a single file degrades cleanly to an ordinary search instead of returning
    fewer chunks than asked for.
    """
    collection = get_collection()
    present: list[str] = []
    for doc_type in config.DOC_TYPES:
        try:
            found = collection.get(where={"doc_type": doc_type}, limit=1, include=[])
            if (found.get("ids") or []):
                present.append(doc_type)
        except Exception:  # noqa: BLE001 - an empty or missing collection is fine
            continue
    return present


def retrieve_plain(question: str, top_k: int) -> list[dict]:
    """One unfiltered similarity search over the whole collection.

    This is the naive approach. On a cross-document question it usually returns
    every chunk from whichever document happens to share vocabulary with the
    question, and none at all from the other.
    """
    return _query(question, top_k)


def retrieve_balanced(question: str, top_k: int, min_per_type: int | None = None) -> list[dict]:
    """Reserve a fixed share of the slots for each document, then merge.

    One filtered search per document type, each asked for its quota, plus an
    unfiltered search to spend any slots left over on whatever is globally
    closest. Results are merged, de-duplicated by chunk id, and ordered by
    distance so the closest passage is presented to the model first.

    The quota is honoured even when that means returning slightly more than
    top_k: two documents at two chunks each cannot fit in a top_k of 3. With the
    default top_k of 6 and two documents this never happens (3 and 3).
    """
    min_per_type = config.MIN_PER_DOC_TYPE if min_per_type is None else min_per_type
    types = present_doc_types()

    # Only one document indexed: there is nothing to balance.
    if len(types) < 2:
        return retrieve_plain(question, top_k)

    per_type = max(min_per_type, top_k // len(types))
    merged: "OrderedDict[str, dict]" = OrderedDict()

    for doc_type in types:
        for row in _query(question, per_type, doc_type=doc_type):
            merged.setdefault(row["id"], row)

    # Spend any remaining budget on the globally closest chunks.
    remaining = top_k - len(merged)
    if remaining > 0:
        for row in _query(question, top_k):
            if row["id"] not in merged:
                merged[row["id"]] = row
            if len(merged) >= top_k:
                break

    chunks = sorted(merged.values(), key=lambda row: row["distance"])
    keep = max(top_k, min_per_type * len(types))
    return chunks[:keep]


def retrieve(question: str, top_k: int | None = None, mode: str | None = None) -> list[dict]:
    """Retrieve context chunks for a question.

    Always look at what came back before you look at the answer: a wrong answer
    from the right chunks is a prompt problem, and a wrong answer from the wrong
    chunks is a retrieval problem. They are fixed in completely different places.
    """
    top_k = config.DEFAULT_TOP_K if top_k is None else top_k
    mode = (mode or config.RETRIEVAL_MODE).strip().lower()

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    top_k = max(1, min(top_k, count))
    if mode == "plain":
        return retrieve_plain(question, top_k)
    return retrieve_balanced(question, top_k)


# --- Prompt assembly -------------------------------------------------------
def _format_context(chunks: list[dict]) -> str:
    """Number each passage and label it with its document, type and page."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = config.DOC_TYPE_LABELS.get(chunk["doc_type"], chunk["doc_type"])
        header = (
            f"[Passage {i}] document: {chunk['file']} | type: {label} | "
            f"page: {chunk['page']}"
        )
        if chunk.get("section"):
            header += f" | section: {chunk['section']}"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def _dedupe_sources(chunks: list[dict]) -> list[dict]:
    """Collapse to unique (file, page) pairs, keeping retrieval order."""
    seen = set()
    sources: list[dict] = []
    for chunk in chunks:
        key = (chunk["file"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "file": chunk["file"],
                    "page": chunk["page"],
                    "doc_type": chunk["doc_type"],
                    "section": chunk.get("section", ""),
                }
            )
    return sources


def _coverage(chunks: list[dict]) -> dict:
    """Which documents the retrieved chunks came from, and how many from each.

    This is the diagnosis for this assignment. On a cross-document question,
    both document types must appear here. If only one does, retrieval failed,
    however good the answer looks.
    """
    by_type = Counter(chunk["doc_type"] for chunk in chunks)
    by_file = Counter(chunk["file"] for chunk in chunks)
    return {
        "doc_types": dict(by_type),
        "documents": dict(by_file),
        "crossed_documents": len(by_type) > 1,
    }


# --- Answering -------------------------------------------------------------
REFUSAL = "The information is not available in the uploaded documents."


def answer_question(
    question: str,
    top_k: int | None = None,
    mode: str | None = None,
) -> dict:
    """Answer a question from the indexed documents.

    Returns {"answer", "sources", "chunks", "coverage", "retrieval_mode",
    "top_k"}. When nothing is indexed the standard refusal is returned without
    calling the model, so the app can never invent an answer from an empty store.
    """
    question = (question or "").strip()
    top_k = config.DEFAULT_TOP_K if top_k is None else top_k
    mode = (mode or config.RETRIEVAL_MODE).strip().lower()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "chunks": [],
            "coverage": _coverage([]),
            "retrieval_mode": mode,
            "top_k": top_k,
        }

    chunks = retrieve(question, top_k=top_k, mode=mode)
    if not chunks:
        return {
            "answer": REFUSAL,
            "sources": [],
            "chunks": [],
            "coverage": _coverage([]),
            "retrieval_mode": mode,
            "top_k": top_k,
        }

    user_message = (
        f"Context passages:\n\n{_format_context(chunks)}\n\n"
        f"---\nQuestion: {question}\n\n"
        "Answer using only the context passages above, following every rule you "
        "were given. If the answer needs a figure and a rule together, label the "
        "Figure, the Clause and the Action separately."
    )

    response = _client().chat.completions.create(
        model=config.active_llm_model(),
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.MAX_ANSWER_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()

    return {
        "answer": answer,
        "sources": _dedupe_sources(chunks),
        "chunks": chunks,
        "coverage": _coverage(chunks),
        "retrieval_mode": mode,
        "top_k": top_k,
    }


if __name__ == "__main__":
    # Quick manual check:
    #   python rag.py "Which clauses does Kaveri Metals trigger?"
    import sys

    q = " ".join(sys.argv[1:]) or (
        "Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per "
        "million. Which clauses does this trigger and what does it cost them?"
    )
    result = answer_question(q)
    print("Q:", q)
    print()
    print("RETRIEVED FROM:", result["coverage"]["doc_types"])
    for chunk in result["chunks"]:
        print(f"  - {chunk['file']} p.{chunk['page']} "
              f"[{chunk['doc_type']}] distance {chunk['distance']:.4f}")
    print()
    print("A:", result["answer"])
    print()
    print("SOURCES:")
    for src in result["sources"]:
        print(f"  - {src['file']}, page {src['page']}")
