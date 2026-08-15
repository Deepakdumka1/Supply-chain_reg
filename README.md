# Supply Chain RAG — cross-document Q&A over a performance review and a policy handbook

A retrieval-augmented question answering system for a procurement team. It reads two
kinds of internal document into one searchable store and answers questions that need
both at once: a measured figure from the quarterly review, and the rule it triggers from
the policy handbook.

That "both at once" part is the whole point. The review says *"88.1% on-time delivery"*.
The handbook says *"on-time delivery below 90% in any quarter"*. Nowhere do they share a
sentence. Connecting them is what this system is for, and it is what most of the design
decisions below are about.

---

## Important: where the two source documents came from

**The two PDFs this assignment is supposed to ship with were not present in this
repository, and neither was the assignment brief — only the step-by-step guide.** Rather
than build a pipeline with nothing to run it on, `scripts/make_sample_documents.py`
generates a faithful stand-in pair for a fictional company, *Meridian Industrial Systems
Limited*:

| File | Type | Pages | Contents |
|---|---|---|---|
| `data/Meridian_Supply_Chain_Performance_Review_Q2_FY26.pdf` | `review` | 7 | Supplier scorecards, three-quarter trend, freight lanes, inventory, a critical imported part, line stoppages, quality data, risk register |
| `data/Meridian_Procurement_Policy_Handbook_v4.pdf` | `policy` | 8 | Classification categories, approval authority, sourcing rules, safety stock formula and floors, penalty clauses, escalation ladder, definitions |

Every fact the guide names is reproduced exactly, so the ten test questions have real,
checkable answers: Kaveri Metals at 88.1% on-time delivery and 1,150 PPM, Trident
Polymers at 640 PPM, a purchase order of Rs. 1.4 crore, an imported part with a 46-day
lead time, a safety stock formula *plus* minimum floors where the higher value applies,
single-sourced microcontrollers, numbered penalty clauses that each state their own
trigger, and a band-below-B escalation path.

**If you have the real PDFs, drop them into `data/`, delete the generated pair, and run
`python ingest.py --reset`.** Nothing in the pipeline is specific to the generated files:
document type is detected from the file name and, failing that, from the content.

Two deliberate choices in the generated documents:

- **Amounts are written `Rs.`, not `₹`.** The rupee sign is absent from the PDF core fonts
  and extracts as a null byte, which would put junk characters into every chunk. Verified,
  not assumed.
- **Neither document mentions ESG, carbon or sustainability audits.** Test question 10
  asks about an ESG-audit penalty, so the only honest answer is a refusal.

---

## Quick start

Requires Python 3.10+ per the brief. It also runs on 3.9 — every module uses
`from __future__ import annotations` — and 3.9.6 is what it was developed and tested on.

```bash
cd supply-chain

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # then add your OpenAI key
python scripts/make_sample_documents.py   # skip if you have the real PDFs
python ingest.py --reset
streamlit run app.py
```

### Configuring the model provider

`.env` selects the backend. The assignment stack is the default in `.env.example`:

```ini
PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o
```

A local, no-key, no-spend option is also supported, and is what the measured results
further down were produced with:

```ini
PROVIDER=ollama
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

```bash
ollama pull nomic-embed-text && ollama pull llama3.2
```

**Switching provider changes the embedding model, and vectors from two different
embedding models are not comparable.** Always re-index after switching:
`python ingest.py --reset`.

### Everything you can run

| Command | What it does |
|---|---|
| `python scripts/make_sample_documents.py` | Build the two source PDFs |
| `python ingest.py --preview` | Extract and chunk only; prints page counts, chunk counts and a sample. Stores nothing |
| `python ingest.py --reset` | Clear the store, then index everything in `data/` |
| `python scripts/check_chunking.py` | Prove every numbered clause survives chunking with its trigger attached |
| `python scripts/compare_retrieval.py` | Compare retrieval strategies on the cross-document questions. Calls no model |
| `python scripts/run_test_questions.py` | Run all ten test questions and write `test_results.json` |
| `python rag.py "your question"` | Answer one question from the terminal |
| `streamlit run app.py` | The interface |
| `uvicorn api.main:app --reload` | The FastAPI service, docs at `/docs` |

---

## How it works

```
PDF ──► extract per page ──► strip repeated headers ──► detect document type
          (pypdf)              (running furniture)        (review / policy)
                                                              │
                                                              ▼
                            chunk on the document's own numbered structure
                            (RecursiveCharacterTextSplitter, 1100 / 200)
                                                              │
                                                              ▼
                     embed in batches ──► ONE ChromaDB collection on disk
                                          metadata: file, page, doc_type, section
                                                              │
question ──────────────────────────────────────────────────────┤
                                                              ▼
                        balanced retrieval: a reserved share of the
                        slots for each document, merged by distance
                                                              │
                                                              ▼
                   grounded prompt (figure / clause / action, temperature 0)
                                                              │
                                                              ▼
                        answer + sources grouped by document and page
```

| Module | Responsibility |
|---|---|
| `config.py` | Every tunable value, overridable from `.env` |
| `vector_store.py` | The one Chroma collection and its embedding function |
| `ingest.py` | Extract, classify, chunk, embed, store; per-file counts |
| `rag.py` | Retrieval strategies, the prompt, the answer |
| `app.py` | Streamlit interface |
| `api/main.py` | FastAPI service: `/ingest`, `/ask`, `/stats` |

---

## Chunking decision

| | |
|---|---|
| **Chunk size** | 1100 characters |
| **Overlap** | 200 characters |
| **Total chunks** | 39 |
| **From the review** | 19 chunks (7 pages) |
| **From the handbook** | 20 chunks (8 pages) |

**Why these numbers.** Both sit at the top of the range the guide allows (800–1200 size,
100–200 overlap), deliberately. The handbook's penalty clauses are short and numbered,
and the damage from cutting one is severe: a chunk reading *"a debit note equal to 2% of
the quarterly invoice value"* with no indication of what triggers it is worse than
useless, because the model will attach it to whatever trigger it can imagine. A large
chunk keeps a clause whole, and also keeps most of the review's wide tables in one piece.

**Size alone is not enough, so structure does the real work.** Before splitting,
`mark_structure()` inserts a blank line in front of every numbered section and clause.
`"\n\n"` is the splitter's first separator, so the document's own clause boundaries become
its preferred cut points. It only falls back to a single newline, then a sentence, then a
space, when one clause is longer than the chunk size on its own. Chunks also carry the
section heading they sit under, in metadata and prefixed into the text, so a clause keeps
its context even when the heading itself landed in the previous chunk.

**This is verified, not hoped for.** `scripts/check_chunking.py` extracts every numbered
block from each PDF and checks whether it survives inside a single chunk:

```
Meridian_Procurement_Policy_Handbook_v4.pdf: 40/40 intact in one chunk, 0 split
Meridian_Supply_Chain_Performance_Review_Q2_FY26.pdf: 20/23 intact, 3 split by arithmetic
RESULT: every numbered block that fits in a chunk stayed whole (60 of 63).
```

All 40 handbook clauses stay whole. The 3 split blocks are the review's long tables, which
exceed 1100 characters and *have* to be split — arithmetic, not a bug, and the check
distinguishes the two cases rather than reporting a scary number. The script also
spot-checks the ten clauses the test questions depend on, confirming each appears in a
chunk together with the figure or trigger that makes it usable.

Two extraction bugs were found and fixed by actually reading the output:

- `Likelihood` was extracting as `Likeliho od` — a table column too narrow for its own
  header, so the word broke mid-way. Columns merged.
- The band table header `Band Q1 FY26` wrapped between the two words, and the extracted
  text came back as `Band Q1`, leaving a column of grades nobody could attribute to a
  quarter. Headers now use the unbreakable token `Q1FY26`, and a sentence-form summary of
  the bands sits under the table.

---

## The cross-document problem, and the fix

### The failure

Ask *"Kaveri Metals had 88.1% on-time delivery and 1,150 defects per million — which
clauses does this trigger?"* with a plain similarity search at `top_k=4` and this happens:

```
plain, top_k=4    4 Procurement Policy    EVIDENCE MISSING
                  ! figure not retrieved: ['88.1', '1,150']
```

Four handbook clauses, and not one chunk containing the supplier's actual numbers. The
model gets rules with nothing to apply them to and fills the gap itself.

### Measuring it honestly

Checking *"did chunks arrive from both documents?"* turned out to be too weak a test. A
run can touch both documents and still miss the figure, because a chunk that merely
mentions *"recoveries to be raised"* counts towards coverage while containing none of the
numbers. So `scripts/compare_retrieval.py` checks the thing that matters: **is the
specific figure AND the specific clause the answer needs actually in the retrieved text?**

| Run | Reached both documents | Complete evidence retrieved |
|---|---|---|
| plain, `top_k=4` | 3/5 | **3/5** |
| plain, `top_k=6` | 5/5 | **5/5** |
| balanced, `top_k=6` | 5/5 | **5/5** |

### The fix chosen, and why

Both of the first two fixes from the guide are applied, and the third is not needed:

1. **`top_k` raised from 4 to 6.** On its own this recovers all five cross-document
   questions on this corpus. It is the cheapest fix and it genuinely works here.
2. **A reserved share of the slots for each document** (`RETRIEVAL_MODE=balanced`, the
   default). One filtered similarity search per `doc_type` using the metadata, each asked
   for its quota, merged and de-duplicated by chunk id, then ordered by distance. Any
   slots left over go to whatever is globally closest.
3. Splitting the question into sub-queries was **not** implemented — raising `top_k` was
   already sufficient, and adding it would be complexity with nothing to show for it.

**Being straight about this: `top_k=6` alone matches balanced retrieval on the evidence
test, so balanced mode is not what rescues the score.** What it adds is a guarantee rather
than a coincidence. On the Kaveri question, plain retrieval at `top_k=6` returns 5 handbook
chunks and 1 review chunk; balanced returns 3 and 3. The plain run passes because the one
review chunk it happened to pick up — an actions list — restates the figures. That is luck,
and luck does not survive a third document or a longer handbook. Balanced mode makes the
minimum from each document a property of the retrieval step instead of a property of how
the vocabulary happened to fall. With two documents and `top_k=6` it costs one extra query
and nothing else.

`plain` mode is kept and exposed in the sidebar, so the failure can be reproduced rather
than just described.

Retrieval degrades sensibly at the edges: with only one document indexed, balanced mode
falls back to an ordinary search instead of returning fewer chunks than asked for.

---

## The prompt

Four parts, as the guide sets out: who the model is, the retrieved chunks, the question,
and the rules. Temperature is **0**. Full text is in `rag.py` as `SYSTEM_PROMPT`; this is
it verbatim:

> You are a procurement analyst at Meridian Industrial Systems. You support buyers who act
> on your answers: they raise debit notes against suppliers, set inventory levels, and
> escalate supplier performance. An answer that sounds right but is not in the documents
> causes real financial damage.
>
> You are given numbered context passages taken from two kinds of internal document:
> - a quarterly Supply Chain Performance Review, which contains measured figures: supplier
>   scorecards, spend, on-time delivery, defect rates, freight costs, inventory, line
>   stoppages and risks.
> - a Procurement Policy Handbook, which contains the rules: supplier classification,
>   approval authority, penalty clauses, safety stock calculation and escalation paths.
>
> Follow these rules strictly.
>
> **1. GROUND EVERY WORD IN THE CONTEXT.** Answer only from the context passages below. Do
> not use any outside knowledge of how procurement, penalties, safety stock or approval
> limits usually work. If a figure, a percentage, a rate or a clause number does not appear
> in the context, you may not state it. This matters most for penalties: describing a
> typical industry penalty instead of Meridian's actual clause is the worst error you can
> make.
>
> **2. REFUSE WHEN THE ANSWER IS ABSENT — BUT CHECK FIRST.** Before you refuse, read every
> passage and look for the specific supplier names, figures, bands, dates and clause
> numbers the question mentions. Refuse only when they are genuinely not there. If they are
> there, answer, even when the passages are terse, in a table, or spread across several
> passages. A wrong refusal is as damaging as a wrong answer: it sends the buyer to read the
> PDF by hand.
> When the context truly does not contain the answer, reply exactly: "The information is not
> available in the uploaded documents." You may then name what would be needed to answer.
> Never guess, and never fill a gap with something plausible. A refusal is all-or-nothing:
> if you refuse, that sentence is your entire answer. Never give an answer and then add a
> refusal, and never refuse and then answer anyway — decide one way and say only that.
>
> **3. SEPARATE THE FIGURE, THE CLAUSE AND THE ACTION.** When an answer requires combining a
> measured figure with a rule, set the three out explicitly and label them:
> - Figure: the measured value, and the document it came from.
> - Clause: the clause number and the trigger condition it states.
> - Action: what follows — the consequence, the amount, and who does it.
>
> If several clauses are triggered, do this for each one, and say plainly that more than one
> applies. Show any arithmetic you do, with the inputs you used.
>
> **4. WHEN A RULE HAS A FLOOR, A CAP OR A MINIMUM, APPLY IT.** Some rules give a formula and
> also a minimum or maximum that overrides the calculated result. Check the context for such
> a limit before answering, state both values, and say which one governs.
>
> **5. CITE AS YOU GO.** Refer to the source in line, like (Handbook, p. 5) or (Review, p. 2),
> for each figure and each clause.
>
> **6. BE PRECISE, NOT FLUENT.** Quote figures exactly as written, including units such as
> Rs. lakh or Rs. crore, PPM, days or percent. If the context is incomplete or the passages
> disagree, say what you can support and state what is missing.
>
> **7. ANSWER ONCE, WITHOUT PADDING.** Give each clause and each figure a single time. Do not
> restate the same clause under several headings, and do not compare quantities that are not
> comparable, such as a lead time in days against a buffer requirement in weeks. If a clause
> has lettered sub-requirements, list each one once and say whether the context shows it is
> met.

Rules 4 and 7 were both added in response to observed failures, not written up front.
Rule 4 exists because question 7 has a formula *and* a floor. Rule 7 exists because the
model was restating one clause under five separate "Figure/Clause/Action" headings and
comparing a 46-day lead time against an 8-week buffer requirement as though they were the
same kind of quantity.

---

## Storage and the restart test

| | |
|---|---|
| **Collection name** | `supply_chain_docs` (one collection, both documents) |
| **Persistence folder** | `chroma_db/` (1.1 MB on disk after a clean build) |
| **Chunks after indexing** | 39 |
| **Chunks after restart** | 39 |

Both documents go into **one** collection. Two collections cannot be searched by a single
query, which would make every cross-document question impossible.

The restart test, run as two completely separate interpreter processes with no
re-uploading between them:

```
process 1:  chunks: 39 | by type: {'policy': 20, 'review': 19}
process 2:  chunks after restart: 39
            retrieved 4 chunks without re-indexing:
                policy p.2  3.1 Purchase order approval limits
                policy p.2  3. Approval authority
                review p.1  Front matter
                review p.1  2. Supplier scorecards
```

**Re-indexing does not duplicate.** Chunk ids are an MD5 of file name, page, position and
content, so a repeat run upserts rather than appends. Verified by posting both PDFs to
`/ingest` a second time: the total stayed at 39, not 78. Without this, retrieval returns
the same clause several times and the duplicates crowd out the chunk from the *other*
document that the question actually needed.

---

## The ten test questions

Run with `PROVIDER=ollama` (`llama3.2` 3B + `nomic-embed-text`), `balanced` mode,
`top_k=6`. Full transcript in `test_results.json`.

**Headline: 7 of 10 correct. All 5 cross-document questions retrieved chunks from both
documents. The trap question was refused.**

| # | Question | Correct? | Chunks came from | What the app said |
|---|---|---|---|---|
| 1 | Highest spend supplier and its on-time delivery | **Yes** | 2 review, 1 policy | Sundaram Forge Pvt Ltd, Rs. 38.40 crore, on-time delivery 96.2% |
| 2 | Line stoppages, downtime, causes | **Yes** | 2 review, 2 policy | 7 stoppages totalling 41.5 hours; listed all seven with supplier and cause |
| 3 | Approval authority for Rs. 1.4 crore | **Yes** | 2 policy, 2 review | Procurement Director approves, Chief Financial Officer countersigns (clause 3.1.1) |
| 4 | Supplier classification categories | **Yes** | 1 policy, 2 review | All four: Strategic, Critical, Preferred, Transactional, with each qualifying threshold |
| 5 | Kaveri Metals — clauses triggered | **No** | 1 policy, 2 review | Cited clause 7.3 (5% debit note) and 7.4. Should be **7.1, 7.2 (2%) and 7.5 (3%)** |
| 6 | Single-source microcontrollers — policy requirement | **Partly** | 2 policy, 3 review | Correct on clause 4.2.2 and the eight-week buffer, and correctly concluded non-compliance — but by faulty reasoning, and missed the alternate-source-within-two-quarters requirement |
| 7 | Safety stock for a 46-day imported part | **Yes** | 2 policy, 2 review | Formula gives 0.25 × 46 + 3 = 14.5 days; floor for an imported part over 45 days is 24 days; higher applies, so **24 days** = 5,280 units. Also spotted the 30-day single-sourced floor under clause 5.3.2 |
| 8 | Trident 640 PPM — cost consequence | **Yes** | 2 policy, 3 review | Clause 7.4: full cost of containment, sorting and rework at Rs. 1,850 per hour plus Rs. 40,000 per quality incident |
| 9 | Suppliers below B band, escalation path | **No** | 2 policy, 2 review | Refused, although review p.2 with the band assignments was in its context |
| 10 | **Trap** — penalty for failing an ESG and carbon audit | **Yes — refused** | 2 policy, 1 review | "The information is not available in the uploaded documents." Then named what would be needed |

### Question 7 deserves its own note

This is the one the guide singles out, because the handbook gives a formula *and* a set of
minimum floors and says the higher value applies. A system that retrieves only the formula
answers 15 days and is wrong.

It got this right, and **it got it right for the right reason** — confirmed by reading the
retrieved chunks, not just the answer. Clause 5.2.1 (the formula) and clause 5.3.1 (the
floors) were both retrieved, from handbook pages 3 and 4, alongside the part's 46-day lead
time and 220 units/day demand from review pages 3 and 4. The answer walks through the
calculation, states 14.5 days, states the 24-day floor, applies the higher, converts to
5,280 units, and then notices that clause 5.3.2 raises the floor to 30 days for a part
that is single-sourced *and* imported. That last step was not asked for.

---

## What failed, and why — honestly

**All three failures are in the answering model, not the pipeline.** That distinction is
worth something, and it is provable rather than asserted.

### Question 5 — wrong clause tier

The model cited clause 7.3 (third consecutive quarter, 5% debit note) instead of 7.2
(second consecutive quarter, 2%), and clause 7.4 (500–1,000 PPM) instead of 7.5 (above
1,000 PPM). At 1,150 PPM, 7.5 is the one that applies. Both are off-by-one-tier errors.

Retrieval was not at fault, and here is the check that proves it:

```
Retrieved chunks:
  policy  p.5  7.3 Delivery: third consecutive quarter below the threshold
  policy  p.5  7.5 Quality: defect rate above 1,000 PPM
  policy  p.5  7.1 Delivery: first quarter below the threshold
  review  p.7  9. Actions carried into Q3 FY26
  review  p.7  8. Supply risk register
  review  p.2  2.2 Three-quarter trend for suppliers of concern

  IN CONTEXT  clause 7.2 text (2% debit note)
  IN CONTEXT  clause 7.5 text (3% debit note)
  IN CONTEXT  clause 7.5 trigger (>1,000 PPM)
  IN CONTEXT  clause 7.2 trigger (2 consecutive quarters)
  IN CONTEXT  Kaveri 88.1 figure
  IN CONTEXT  Kaveri 1,150 figure
```

Every clause and every figure needed was in front of the model. Clauses 7.1 and 7.2 share
a chunk, so retrieving either brings both. The model then picked the wrong tiers from a
correct context. Selecting between numeric bands — is 1,150 above or below 1,000, is this
the second or the third consecutive quarter — is exactly what a 3-billion-parameter model
is weakest at.

### Question 9 — refused a question it could answer

Review page 2 was retrieved, and it contains the sentence *"Kaveri Metals Ltd is band C.
Nandi Castings Ltd is band C. Coastal Logistics Services is band C. Rashmi Fasteners Pvt
Ltd is band D."* The handbook's escalation ladder and clause 8.2.1 were also retrieved.
The model refused anyway.

An earlier prompt revision had this question answering correctly — it identified three of
the four suppliers and applied the Watch List clause. Tightening rule 2 to stop the model
combining an answer with a refusal (a real defect: on question 6 it gave a full answer and
then appended *"The information is not available…"*) pushed it the other way on question 9.
A 3B model treats that instruction as a binary switch and sometimes throws it the wrong
way. Two prompt revisions in, the honest conclusion is that this is a capacity limit rather
than a wording problem, so I stopped rewriting the prompt.

### Question 6 — right conclusion, wrong reasoning

It correctly identified clause 4.2.2 and the eight-week buffer requirement, and correctly
concluded non-compliance. But it argued that *"the total replenishment lead time is 46
days, but the policy requires a buffer of eight weeks, which is 56 days"* — comparing a
lead time against a stock quantity, which is not a valid comparison. The real breach is
that 9 days of cover are held where 8 weeks plus 2 more for an imported part are required.
It also missed the alternate-source-within-two-quarters requirement, and the review's note
that the qualification project has been open for four quarters.

### The common cause, and what would fix it

The assignment specifies GPT-4o. These results are from `llama3.2` (3B), chosen so the
whole pipeline could be verified end to end with no API key and no spend. The three
failures are all of one kind: multi-clause synthesis and numeric-tier selection. Questions
1–4, 7, 8 and 10 — single-document lookups, a formula with a floor, a single clause match,
and an honest refusal — all work.

Switching to GPT-4o is two lines in `.env` and a re-index, and would very likely fix all
three. I have not measured that, so I am not claiming it. What is measured is that
retrieval put the correct evidence in front of the model on all five cross-document
questions, which is the part these results can actually speak to.

### Smaller things worth recording

- **The Figure/Clause/Action template is over-applied.** On question 1 the model added
  `Clause: 6.1.1`, which has nothing to do with which supplier spent most. Rule 3 says to
  use the template when combining a figure with a rule; the model applies it to every
  question. Harmless but untidy, and it would mislead a reader who trusted the label.
- **Section numbers collide between the documents.** The review's section 7 is "Quality
  performance"; the handbook's clause 7.5 is a quality penalty. The model occasionally
  cites "Review, p. 5, section 7.5" — a real clause number attached to the wrong document.
  The page number stays correct, so the citation is still checkable.
- **Table extraction is one cell per line.** `pypdf` returns a seven-column scorecard as a
  vertical run of cells in row order. Readable, and the model handles it, but it is why the
  generated review restates its most important rows in sentence form underneath each table
  — which is what a real report does anyway.

---

## Bugs found while building this

Recorded because each one was found by checking output rather than assuming it was fine.

| Symptom | Cause | Fix |
|---|---|---|
| `₹` extracted as a null byte into every chunk | The rupee glyph is missing from the PDF core fonts | Write `Rs.` |
| `Likelihood` became `Likeliho od` | Table column narrower than its own header | Merged two columns |
| Band columns lost their quarter: `Band Q1 FY26` → `Band Q1` | Header wrapped between the two words | Unbreakable token `Q1FY26`, plus a sentence-form summary |
| The whole app hung indefinitely, CPU idle | No timeout on the model call | `REQUEST_TIMEOUT=90`, `MAX_RETRIES=1` |
| A run stalled for 8 minutes and looked like a hang | Generation unbounded; a small model can run to its whole context window repeating itself | `MAX_ANSWER_TOKENS=900` |
| One stalled question abandoned the other nine | No per-question error handling in the batch runner | Catch, record, continue |
| "2 files processed, 39 chunks stored" flashed and vanished | `st.rerun()` refreshed the sidebar count and wiped the message with it | Stash the result in session state, render after the rerun |
| Model gave a full answer then appended a refusal | Nothing said a refusal must be the entire answer | Rule 2 made all-or-nothing |

---

## The interface

`streamlit run app.py`

- **Upload** one or more PDFs, or index whatever is already in `data/`
- **Index** reports files processed, chunks stored, and the document type each file was
  recognised as
- **Ask** stays disabled until something is indexed, so asking too early is impossible
  rather than merely handled
- **Answer** with a spinner during both indexing and answering
- **Sources grouped by document**, each with its page number and section
- A banner above every answer stating whether retrieval **reached both documents** — on a
  cross-document question that is the difference between a real answer and a fluent
  invention
- An expander showing the retrieved chunks with their distances, because a wrong answer
  from the right chunks is a prompt problem and a wrong answer from the wrong chunks is a
  retrieval problem, and they are fixed in completely different places
- Question history kept on screen so answers can be compared
- Sidebar: `top_k`, retrieval mode (switch to `plain` to watch it fail), live chunk counts
  per document, and the models in use
- A dropdown of the ten test questions, so a demo does not depend on typing

### Screenshots

**Not yet captured — these need adding before submission.** To take them:

1. `python ingest.py --reset` then `streamlit run app.py`
2. Capture: the sidebar showing 39 chunks split 20/19; the index confirmation message;
   question 5 answered with both documents in the source list; question 10 being refused.
3. Save into `docs/screenshots/` and link them here.

---

## FastAPI backend

```bash
uvicorn api.main:app --reload
# then open http://localhost:8000/docs
```

| Method | Endpoint | In | Out |
|---|---|---|---|
| `POST` | `/ingest` | multipart PDFs | `files`, `chunks`, `per_file[]`, `skipped[]` |
| `POST` | `/ask` | `question`, `top_k`, `mode` | `answer`, `sources[]`, `coverage`, `retrieval_mode`, `top_k` |
| `GET` | `/stats` | — | collection, chunk counts by document and type, models, chunk size, persist directory |

All three were tested from `/docs` and with `curl` before the interface was pointed at
them. Sample `/stats`:

```json
{
  "provider": "ollama",
  "collection": "supply_chain_docs",
  "total_chunks": 39,
  "chunks_by_document": {
    "Meridian_Procurement_Policy_Handbook_v4.pdf": 20,
    "Meridian_Supply_Chain_Performance_Review_Q2_FY26.pdf": 19
  },
  "chunks_by_type": { "policy": 20, "review": 19 },
  "embedding_model": "nomic-embed-text",
  "llm_model": "llama3.2",
  "chunk_size": 1100,
  "chunk_overlap": 200,
  "retrieval_mode": "balanced",
  "default_top_k": 6
}
```

`/ask` returns a `coverage` block naming which documents the chunks came from, so a caller
can reject an answer whose retrieval never crossed both documents. Empty questions and
unknown modes return `400`; a file that yields no text returns `422` with the reason.

Toggle **Use FastAPI backend** in the Streamlit sidebar to make the interface call the
service over HTTP instead of doing the work itself. The two run independently.

**These endpoints are unauthenticated.** That is fine on localhost for an assignment, but
`/ingest` accepts file uploads and `/ask` spends model credits, so anything reachable from
a network needs an API key or a reverse proxy in front of it first.

---

## Manual verification

Checked by hand against the PDFs, as the guide requires, including one penalty clause.

| Question asked | What the app said | What the PDF says | Document & page | Match |
|---|---|---|---|---|
| Highest spend supplier and its on-time delivery | Sundaram Forge Pvt Ltd, Rs. 38.40 crore, 96.2% | Sundaram Forge Pvt Ltd, 38.40, 96.2 | Review, p. 2, §2.1 | Yes |
| Who approves a Rs. 1.4 crore purchase order | Procurement Director, CFO countersigns | "Above Rs. 1 crore and up to Rs. 2.5 crore — Procurement Director — countersignature of the Chief Financial Officer" | Handbook, p. 2, §3.1 | Yes |
| Trident Polymers at 640 PPM (penalty clause) | Clause 7.4, Rs. 1,850/hour plus Rs. 40,000 per incident | Clause 7.4 triggers from 500 to 1,000 PPM inclusive; Rs. 1,850 per hour; Rs. 40,000 per quality incident | Handbook, p. 5, §7.4 | Yes |
| Safety stock for a 46-day imported part | 24 days, 5,280 units | Formula 0.25 × 46 + 3 = 14.5; floor 24 days for imported over 45 days; higher applies | Handbook, pp. 3–4, §5.2.1 & §5.3.1 | Yes |
| Kaveri Metals — clauses triggered | Clauses 7.3 and 7.4 | 88.1% for a second consecutive quarter → 7.1 and 7.2; 1,150 PPM → 7.5 | Handbook, p. 5, §7.1–7.5 | **No** |

Every figure in the passing rows was opened in the PDF and read, not inferred from the
answer.

---

## Warm-up answers

The three questions the guide asks you to answer by hand before building anything.

1. **Highest spend supplier and its on-time delivery** — Sundaram Forge Pvt Ltd,
   Rs. 38.40 crore, 96.2% on-time. *Review, p. 2.*
2. **Who approves a purchase order worth Rs. 1.4 crore** — the Procurement Director, with
   the Chief Financial Officer countersigning and a documented negotiation record on file.
   *Handbook, p. 2, clause 3.1.* Note that Rs. 1.4 crore sits in the "above Rs. 1 crore and
   up to Rs. 2.5 crore" band, one step above the Head of Procurement's limit.
3. **Kaveri Metals at 1,150 PPM — which clause, and what does it cost** — clause 7.5,
   because the rate is above 1,000 PPM: a debit note of 3% of the quarterly invoice value,
   100% inspection at the supplier's cost until three consecutive lots are accepted,
   probation for two quarters, and the band held at C or lower until the rate falls below
   500 PPM for two consecutive quarters. On Rs. 31.75 crore of quarterly spend the debit
   note alone is Rs. 95.25 lakh. *Handbook, p. 5; figures from Review, pp. 2 and 5.*

---

## Project layout

```
supply-chain/
├── app.py                          Streamlit interface
├── config.py                       every tunable value, .env-overridable
├── ingest.py                       extract, classify, chunk, embed, store
├── rag.py                          retrieval strategies, prompt, answer
├── vector_store.py                 the one Chroma collection
├── api/
│   ├── __init__.py
│   └── main.py                     FastAPI: /ingest, /ask, /stats
├── scripts/
│   ├── make_sample_documents.py    builds the two source PDFs
│   ├── check_chunking.py           proves clauses survive chunking
│   ├── compare_retrieval.py        plain vs balanced, evidence-based
│   └── run_test_questions.py       the ten questions, with a transcript
├── data/                           the two PDFs
├── chroma_db/                      persisted vector store (git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore                      includes .env
└── README.md
```

`.env` and `chroma_db/` are git-ignored. The API key is never in code — it is read from
`.env` through `config.py`, and `config.require_api_key()` raises a clear error naming the
fix if it is missing.

---

## Demo video plan

Three minutes, per the guide:

- **0:00–0:20** — the two documents: the review holds figures, the handbook holds rules,
  and they never share a sentence
- **0:20–1:00** — upload and index, with "2 files processed, 39 chunks stored" and the
  20/19 split visible
- **1:00–2:30** — questions 7 and 8, showing sources from both documents and, on question
  7, the floor overriding the calculated value
- **2:30–3:00** — question 10 refused
