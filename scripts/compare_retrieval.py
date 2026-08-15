"""Diagnose cross-document retrieval: does it fetch the evidence the answer needs?

Stage 6 of the guide describes the characteristic failure of this assignment:
ask a question whose wording appears in only one document, and every retrieved
chunk comes from that document. The model then gets figures without rules and
fills the gap from its own general knowledge.

This script tests two things for each cross-document question, because the first
one on its own is too easy to pass:

  1. COVERAGE - did chunks arrive from both documents?
  2. EVIDENCE - is the specific figure AND the specific clause the answer needs
     actually present in the retrieved text?

Coverage alone is a weak test. A run can touch both documents and still miss the
figure, because a chunk that merely mentions "recoveries to be raised" counts
towards coverage while containing none of the numbers. Evidence is the test that
matters, so that is what the pass/fail verdict uses.

No model is called, so this costs nothing to run.

Run:  python scripts/compare_retrieval.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from rag import retrieve  # noqa: E402
from vector_store import get_collection  # noqa: E402

# Each cross-document question, with the evidence that has to be retrieved for
# the answer to be possible: a figure from the review and a rule from the
# handbook. Every string in a group must be present; the groups are what the
# buyer needs in hand.
CROSS_DOCUMENT_CASES = [
    {
        "question": "Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects "
                    "per million this quarter. Which clauses does this trigger, and "
                    "what does it cost the supplier?",
        "figure": ["88.1", "1,150"],
        # Three clauses are in play: 7.1 and 7.2 on delivery, 7.5 on defects.
        # 7.1 and 7.2 share a chunk, so retrieving either brings both.
        "rule": [
            "below 90%",
            "2% of the quarterly invoice value",
            "1,000 defects per million",
        ],
    },
    {
        "question": "We single-source our microcontrollers from one supplier in Penang. "
                    "What does the policy require us to have in place, and are we "
                    "compliant?",
        "figure": ["MCU-4471", "single source"],
        "rule": ["eight weeks", "two quarters"],
    },
    {
        "question": "How much safety stock should we hold for an imported part with a "
                    "46-day replenishment lead time?",
        "figure": ["46 days"],
        "rule": ["0.25 x total replenishment lead time", "24 days"],
    },
    {
        "question": "Trident Polymers came in at 640 defects per million. What is the "
                    "cost consequence for them?",
        "figure": ["640"],
        "rule": ["Rs. 1,850 per hour"],
    },
    {
        "question": "Which suppliers are below the B band, and what escalation path "
                    "applies to each?",
        # The band actually awarded to a named supplier has to be in the context;
        # the escalation ladder alone is not enough to answer this.
        "figure": ["Kaveri Metals Ltd is band C", "Rashmi Fasteners Pvt Ltd is band D"],
        "rule": ["Watch List", "Head of Supply Chain"],
    },
]

# Answerable from one document. A single-document result is correct here, so
# these exist only to show that balancing has not broken the easy cases.
SINGLE_DOCUMENT_CASES = [
    {
        "question": "Which supplier had the highest spend in the quarter, and what was "
                    "its on-time delivery?",
        "figure": ["38.40", "96.2"],
        "rule": [],
    },
    {
        "question": "Who approves a purchase order worth Rs. 1.4 crore?",
        "figure": [],
        "rule": ["Procurement Director"],
    },
]

# The three runs compared: the naive baseline, the guide's first fix, and the
# fix this project ships with.
RUNS = [
    ("plain, top_k=4", "plain", 4),
    ("plain, top_k=6", "plain", 6),
    ("balanced, top_k=6", "balanced", 6),
]


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def evaluate(case: dict, mode: str, top_k: int) -> dict:
    """Run one question and report coverage and evidence found."""
    chunks = retrieve(case["question"], top_k=top_k, mode=mode)
    context = _flat(" ".join(chunk["text"] for chunk in chunks))

    types = {}
    for chunk in chunks:
        types[chunk["doc_type"]] = types.get(chunk["doc_type"], 0) + 1

    missing_figure = [needle for needle in case["figure"] if needle not in context]
    missing_rule = [needle for needle in case["rule"] if needle not in context]

    return {
        "chunks": chunks,
        "types": types,
        "crossed": len(types) > 1,
        "missing_figure": missing_figure,
        "missing_rule": missing_rule,
        "evidence_ok": not missing_figure and not missing_rule,
    }


def describe_types(types: dict) -> str:
    if not types:
        return "nothing retrieved"
    return ", ".join(
        f"{n} {config.DOC_TYPE_LABELS.get(t, t)}" for t, n in sorted(types.items())
    )


def report(cases: list[dict], heading: str, strict: bool) -> dict:
    print("=" * 78)
    print(heading)
    print("=" * 78)

    tally = {name: {"crossed": 0, "evidence": 0} for name, _, _ in RUNS}

    for case in cases:
        print("-" * 78)
        print(f"Q: {case['question']}")
        print()
        for name, mode, top_k in RUNS:
            result = evaluate(case, mode, top_k)
            tally[name]["crossed"] += int(result["crossed"])
            tally[name]["evidence"] += int(result["evidence_ok"])

            verdict = "evidence complete" if result["evidence_ok"] else "EVIDENCE MISSING"
            print(f"  {name:<18} {describe_types(result['types']):<44} {verdict}")
            if result["missing_figure"]:
                print(f"                     ! figure not retrieved: {result['missing_figure']}")
            if result["missing_rule"]:
                print(f"                     ! rule not retrieved  : {result['missing_rule']}")
            for chunk in result["chunks"]:
                label = config.DOC_TYPE_LABELS.get(chunk["doc_type"], chunk["doc_type"])
                print(
                    f"                       - {label:<20} p.{chunk['page']}  "
                    f"d={chunk['distance']:.4f}  {(chunk.get('section') or '')[:40]}"
                )
            print()
    print()

    total = len(cases)
    print(f"  {'run':<18} {'both documents':<18} {'evidence complete'}")
    for name, _, _ in RUNS:
        print(
            f"  {name:<18} {str(tally[name]['crossed']) + '/' + str(total):<18} "
            f"{tally[name]['evidence']}/{total}"
        )
    print()
    return tally


def main() -> int:
    if get_collection().count() == 0:
        print("Nothing indexed yet. Run: python ingest.py --reset")
        return 1

    cross = report(
        CROSS_DOCUMENT_CASES,
        "CROSS-DOCUMENT QUESTIONS - need a figure from the review AND a rule "
        "from the handbook",
        strict=True,
    )
    report(
        SINGLE_DOCUMENT_CASES,
        "SINGLE-DOCUMENT QUESTIONS - one document is the correct outcome; shown "
        "to confirm balancing did not break them",
        strict=False,
    )

    total = len(CROSS_DOCUMENT_CASES)
    best = max(cross[name]["evidence"] for name, _, _ in RUNS)
    balanced = cross["balanced, top_k=6"]["evidence"]

    print("=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    for name, _, _ in RUNS:
        print(
            f"  {name:<18} retrieved complete evidence on "
            f"{cross[name]['evidence']}/{total} cross-document questions"
        )
    print()
    if balanced == total:
        print("  Balanced retrieval fetched both the figure and the rule on every")
        print("  cross-document question. It is the default in config.RETRIEVAL_MODE.")
    elif balanced == best:
        print("  Balanced retrieval is at least as good as every other run, but did")
        print("  not fetch complete evidence everywhere. Investigate the gaps above")
        print("  before trusting answers to those questions.")
    else:
        print("  Balanced retrieval is NOT the best run here. Read the gaps above.")

    return 0 if balanced == total else 1


if __name__ == "__main__":
    sys.exit(main())
