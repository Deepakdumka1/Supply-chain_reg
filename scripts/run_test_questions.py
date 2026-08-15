"""Run the ten questions from the brief and record what happened (Stage 11).

For each question this records the answer, which documents the retrieved chunks
came from, and an automated indicator of whether the answer contains the facts
it should. Questions 5 to 9 are the cross-document ones and carry the most
marks, so for those the document mix is reported as prominently as the answer.

A WORD ON THE VERDICTS. The "checks" column is a keyword test: it looks for the
figures and clause numbers the answer must contain. It is a smoke test, not
marking. A keyword can appear in a sentence that says the opposite of the truth,
so the guide is right that you have to read the answers - the manual
verification table in the README is where the real check lives. What this script
does reliably is catch a regression, and prove where the chunks came from.

Question 7 gets extra treatment. The handbook gives a formula AND a set of
minimum floors and says the higher value applies. A system that retrieves only
the formula answers 15 days and is wrong; the right answer is 24 days. So the
check requires 24 and separately flags an answer that settles on the calculated
figure.

Run:  python scripts/run_test_questions.py
      python scripts/run_test_questions.py --mode plain     # see it degrade
      python scripts/run_test_questions.py --only 5 7 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from rag import REFUSAL, answer_question  # noqa: E402
from vector_store import get_collection  # noqa: E402

# "expect" strings are matched case-insensitively against the answer. Groups of
# alternatives are given as tuples: any one of them satisfies that requirement.
QUESTIONS: list[dict] = [
    {
        "id": 1,
        "cross": False,
        "topic": "Highest spend supplier and its on-time delivery",
        "question": "Which supplier had the highest spend in the quarter, and what "
                    "was its on-time delivery?",
        "expect": ["sundaram", ("38.40", "38.4"), "96.2"],
    },
    {
        "id": 2,
        "cross": False,
        "topic": "Line stoppages, downtime, causes",
        "question": "What production line stoppages occurred this quarter, how much "
                    "downtime did they cause in total, and what were the causes?",
        "expect": ["41.5", ("seven", "7"), ("late deliver", "defect")],
    },
    {
        "id": 3,
        "cross": False,
        "topic": "Approval authority for Rs. 1.4 crore",
        "question": "Who approves a purchase order worth Rs. 1.4 crore?",
        "expect": ["procurement director", ("chief financial officer", "cfo")],
    },
    {
        "id": 4,
        "cross": False,
        "topic": "Supplier classification categories",
        "question": "What are the supplier classification categories in the "
                    "procurement policy?",
        "expect": ["strategic", "critical", "preferred", "transactional"],
    },
    {
        "id": 5,
        "cross": True,
        "topic": "Kaveri Metals - clauses triggered",
        "question": "Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects "
                    "per million this quarter. Which clauses does this trigger, and "
                    "what does it cost the supplier?",
        "expect": ["88.1", "1,150", ("7.5", "3%"), ("7.2", "2%")],
    },
    {
        "id": 6,
        "cross": True,
        "topic": "Single-source microcontrollers - policy requirement",
        "question": "We single-source our microcontrollers from one supplier in "
                    "Penang. What does the procurement policy require us to have in "
                    "place, and are we compliant?",
        "expect": [("eight weeks", "8 weeks", "ten weeks", "10 weeks"),
                   ("two quarters", "2 quarters"),
                   ("alternate source", "alternate-source")],
    },
    {
        "id": 7,
        "cross": True,
        "topic": "Safety stock for a 46-day imported part",
        "question": "How much safety stock should we hold for an imported part with a "
                    "46-day total replenishment lead time? Show the calculation and "
                    "any minimum that applies.",
        "expect": ["24", ("floor", "minimum")],
        # A system that retrieves only the formula lands here and is wrong.
        "wrong_if_final": ["14.5", "15 days"],
    },
    {
        "id": 8,
        "cross": True,
        "topic": "Trident Polymers 640 PPM - cost consequence",
        "question": "Trident Polymers came in at 640 defects per million this quarter. "
                    "What is the cost consequence for them?",
        "expect": ["640", ("1,850", "1850"), ("40,000", "40000")],
    },
    {
        "id": 9,
        "cross": True,
        "topic": "Suppliers below the B band and the escalation path",
        "question": "Which suppliers are below the B band this quarter, and what "
                    "escalation path applies to each?",
        "expect": ["kaveri", "rashmi", ("watch list", "watchlist")],
    },
    {
        "id": 10,
        "cross": False,
        "topic": "TRAP - must be refused",
        "question": "What penalty applies to a supplier that fails Meridian's annual "
                    "ESG and carbon audit?",
        "must_refuse": True,
        # Nothing in either document mentions an ESG or carbon audit. A confident
        # answer here is the hallucination the whole assignment is testing for.
        "forbid": ["debit note equal to", "2% of the quarterly", "3% of the quarterly"],
    },
]


def _hit(answer_lower: str, requirement) -> bool:
    """True if the requirement is satisfied. Tuples mean 'any one of these'."""
    if isinstance(requirement, tuple):
        return any(option.lower() in answer_lower for option in requirement)
    return requirement.lower() in answer_lower


def _describe(requirement) -> str:
    if isinstance(requirement, tuple):
        return " or ".join(requirement)
    return str(requirement)


def evaluate(case: dict, answer: str) -> dict:
    """Automated indicator of whether the answer holds the facts it should."""
    lowered = (answer or "").lower()
    refused = REFUSAL.lower() in lowered or "not available in the uploaded documents" in lowered

    if case.get("must_refuse"):
        leaked = [term for term in case.get("forbid", []) if term.lower() in lowered]
        return {
            "passed": refused and not leaked,
            "missing": [] if refused else ["an explicit refusal"],
            "notes": [f"invented a penalty: {leaked}"] if leaked else [],
            "refused": refused,
        }

    missing = [_describe(req) for req in case.get("expect", []) if not _hit(lowered, req)]
    notes: list[str] = []
    if refused:
        notes.append("refused a question that the documents can answer")

    # Question 7's specific trap: the calculated value quoted as the final answer.
    for term in case.get("wrong_if_final", []):
        if term.lower() in lowered and "24" not in lowered:
            notes.append(f"settled on the calculated value ({term}) instead of the floor")

    return {
        "passed": not missing and not notes,
        "missing": missing,
        "notes": notes,
        "refused": refused,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ten test questions.")
    parser.add_argument("--mode", default=config.RETRIEVAL_MODE,
                        choices=["balanced", "plain"],
                        help="Retrieval mode (default: whatever config says).")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--only", nargs="*", type=int,
                        help="Run only these question numbers.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "test_results.json"))
    args = parser.parse_args()

    if get_collection().count() == 0:
        print("Nothing indexed yet. Run: python ingest.py --reset")
        return 1

    cases = [c for c in QUESTIONS if not args.only or c["id"] in args.only]

    print("=" * 78)
    print(f"TEN TEST QUESTIONS - mode: {args.mode}, top_k: {args.top_k}, "
          f"model: {config.active_llm_model()}")
    print("=" * 78)

    records = []
    for case in cases:
        started = time.time()
        try:
            result = answer_question(case["question"], top_k=args.top_k, mode=args.mode)
        except Exception as exc:  # noqa: BLE001
            # One model call failing must not abandon the other nine questions.
            # Record it as a failure and carry on, so the run still produces a
            # transcript rather than stopping halfway with nothing to show.
            elapsed = time.time() - started
            print()
            print("-" * 78)
            print(f"Q{case['id']}. {case['topic']}")
            print(f"    MODEL CALL FAILED after {elapsed:.1f}s: "
                  f"{type(exc).__name__}: {exc}")
            records.append(
                {
                    "id": case["id"],
                    "topic": case["topic"],
                    "question": case["question"],
                    "cross_document": case["cross"],
                    "answer": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "sources": [],
                    "coverage": {"doc_types": {}, "documents": {}, "crossed_documents": False},
                    "checks_passed": False,
                    "missing": ["the model call did not return"],
                    "notes": ["model call failed"],
                    "seconds": round(elapsed, 1),
                }
            )
            continue
        elapsed = time.time() - started

        verdict = evaluate(case, result["answer"])
        coverage = result["coverage"]
        mix = ", ".join(
            f"{n} {config.DOC_TYPE_LABELS.get(t, t)}"
            for t, n in sorted(coverage["doc_types"].items())
        ) or "nothing retrieved"

        print()
        print("-" * 78)
        print(f"Q{case['id']}. {case['topic']}")
        print(f"    asked        : {case['question']}")
        print(f"    chunks from  : {mix}")
        if case["cross"]:
            crossed = "YES" if coverage["crossed_documents"] else "NO - retrieval failed"
            print(f"    crossed both : {crossed}")
        print(f"    checks       : {'PASS' if verdict['passed'] else 'REVIEW'}"
              f"   ({elapsed:.1f}s)")
        for item in verdict["missing"]:
            print(f"      ! missing from the answer: {item}")
        for item in verdict["notes"]:
            print(f"      ! {item}")
        print("    sources      :")
        for src in result["sources"]:
            print(f"      - {src['file']} page {src['page']} [{src['doc_type']}]")
        print("    answer       :")
        for line in (result["answer"] or "").splitlines():
            print(f"      {line}")

        records.append(
            {
                "id": case["id"],
                "topic": case["topic"],
                "question": case["question"],
                "cross_document": case["cross"],
                "answer": result["answer"],
                "sources": result["sources"],
                "coverage": coverage,
                "checks_passed": verdict["passed"],
                "missing": verdict["missing"],
                "notes": verdict["notes"],
                "seconds": round(elapsed, 1),
            }
        )

    # Summary table, in the shape the README needs.
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'#':<3} {'checks':<8} {'crossed':<9} {'chunks came from':<34} topic")
    for record in records:
        crossed = "-"
        if record["cross_document"]:
            crossed = "yes" if record["coverage"]["crossed_documents"] else "NO"
        mix = ", ".join(
            f"{n} {t}" for t, n in sorted(record["coverage"]["doc_types"].items())
        )
        print(
            f"  {record['id']:<3} {'PASS' if record['checks_passed'] else 'REVIEW':<8} "
            f"{crossed:<9} {mix:<34} {record['topic'][:34]}"
        )

    passed = sum(1 for r in records if r["checks_passed"])
    cross_cases = [r for r in records if r["cross_document"]]
    crossed_ok = sum(1 for r in cross_cases if r["coverage"]["crossed_documents"])

    print()
    print(f"  automated checks passed      : {passed}/{len(records)}")
    if cross_cases:
        print(f"  cross-document questions that reached both documents: "
              f"{crossed_ok}/{len(cross_cases)}")
    print()
    print("  Read the answers above before recording any of this as correct. A")
    print("  keyword check cannot tell a right answer from a wrong one that")
    print("  happens to contain the right number.")

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(
            {
                "provider": config.PROVIDER,
                "llm_model": config.active_llm_model(),
                "embedding_model": config.active_embed_model(),
                "retrieval_mode": args.mode,
                "top_k": args.top_k,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "results": records,
            },
            indent=2,
        )
    )
    print(f"  Full transcript written to {out_path}")

    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
