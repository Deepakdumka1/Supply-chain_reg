"""Build the two source documents for the Supply Chain RAG assignment.

WHY THIS SCRIPT EXISTS
----------------------
Assignment 2 is meant to ship with two PDFs: a quarterly Supply Chain
Performance Review and a Procurement Policy Handbook. Those files were not
supplied with the copy of the guide in this repository, so this script
generates a faithful stand-in pair for the fictional company
"Meridian Industrial Systems Limited".

Every fact the guide names is reproduced exactly, so the ten test questions
have real, checkable answers:

  * Kaveri Metals - 88.1% on-time delivery and 1,150 defects per million
  * Trident Polymers - 640 defects per million
  * a purchase order worth Rs. 1.4 crore, sitting inside an approval band
  * an imported part with a 46-day replenishment lead time
  * a safety-stock formula PLUS minimum floors, where the higher value applies
  * single-sourced microcontrollers
  * numbered penalty clauses, each stating its own trigger condition
  * a band-below-B escalation path

The two documents deliberately never share a sentence: the review states
figures and holds no rules, the handbook states rules and holds no figures
about actual suppliers. Connecting them is the whole point of the assignment.

DELIBERATE OMISSION: neither document mentions ESG, carbon or sustainability
audits. Test question 10 asks about an ESG-audit penalty, so the honest answer
is a refusal. tests are not the point here - the refusal is graded.

All amounts are written "Rs." rather than with the rupee sign, because the
rupee glyph is missing from the PDF core fonts and extracts as a null byte,
which would put junk characters into every chunk.

Run:  python scripts/make_sample_documents.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

REVIEW_FILE = "Meridian_Supply_Chain_Performance_Review_Q2_FY26.pdf"
POLICY_FILE = "Meridian_Procurement_Policy_Handbook_v4.pdf"


# --- Styles ----------------------------------------------------------------
def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=17, leading=21, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=10.5, leading=14, textColor=colors.HexColor("#444444"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, spaceBefore=14, spaceAfter=6,
            textColor=colors.HexColor("#12355b"),
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, spaceBefore=10, spaceAfter=4,
            textColor=colors.HexColor("#1f4e79"),
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
            leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "clause": ParagraphStyle(
            "Clause", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
            leading=13.5, alignment=TA_JUSTIFY, spaceAfter=8, leftIndent=0.4 * cm,
        ),
        "note": ParagraphStyle(
            "Note", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=9, leading=12.5, spaceAfter=6,
            textColor=colors.HexColor("#333333"),
        ),
        "cell": ParagraphStyle(
            "Cell", parent=base["BodyText"], fontName="Helvetica", fontSize=8.2,
            leading=10.5, spaceAfter=0,
        ),
        "cellhead": ParagraphStyle(
            "CellHead", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=8.2, leading=10.5, spaceAfter=0, textColor=colors.white,
        ),
    }


S = _styles()


def h1(text: str):
    return Paragraph(text, S["h1"])


def h2(text: str):
    return Paragraph(text, S["h2"])


def p(text: str):
    return Paragraph(text, S["body"])


def clause(text: str):
    """A numbered clause. Kept on one page so it is never split from its trigger."""
    return KeepTogether(Paragraph(text, S["clause"]))


def note(text: str):
    return Paragraph(text, S["note"])


def table(rows: list[list[str]], widths: list[float], align_right: tuple = ()):
    """Build a table whose cells wrap, so no text is silently clipped."""
    data = [[Paragraph(str(c), S["cellhead"]) for c in rows[0]]]
    for row in rows[1:]:
        data.append([Paragraph(str(c), S["cell"]) for c in row])

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9bb3c7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ]
    for col in align_right:
        style.append(("ALIGN", (col, 1), (col, -1), "RIGHT"))

    tbl = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle(style))
    return tbl


def _footer_factory(label: str):
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2 * cm, 1.2 * cm, label)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
        canvas.restoreState()

    return _footer


def build(path: Path, label: str, story: list) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=2 * cm,
        title=label, author="Meridian Industrial Systems Limited",
    )
    footer = _footer_factory(label)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ===========================================================================
# Document 1 - Supply Chain Performance Review, Q2 FY26  (doc_type: review)
# ===========================================================================
def review_story() -> list:
    W = A4[0] - 4 * cm  # usable width
    st: list = []

    st.append(Paragraph("Supply Chain Performance Review", S["title"]))
    st.append(
        Paragraph(
            "Meridian Industrial Systems Limited &nbsp;|&nbsp; Quarter 2, FY2025-26 "
            "(01 July 2025 to 30 September 2025) &nbsp;|&nbsp; Issued 18 October 2025 "
            "by the Supply Chain Performance Office. All values in Indian Rupees (Rs.); "
            "1 lakh = 100,000 and 1 crore = 10,000,000.",
            S["subtitle"],
        )
    )

    # --- 1. Executive summary ---
    st.append(h1("1. Executive summary"))
    st.append(p(
        "Total procurement spend for the quarter was <b>Rs. 214.70 crore</b>, 6.2% higher "
        "than Q1 FY26, on a build volume 4.1% higher. Composite on-time delivery across "
        "all direct-material suppliers was <b>92.4%</b> against a target of 95.0%, a "
        "decline of 1.3 percentage points from the previous quarter."
    ))
    st.append(p(
        "Four direct-material suppliers delivered below 90% on-time in the quarter: "
        "Kaveri Metals Ltd, Rashmi Fasteners Pvt Ltd, Nandi Castings Ltd and "
        "Coastal Logistics Services. Seven production line stoppages were recorded, "
        "costing <b>41.5 hours</b> of lost production. The cost of poor quality was "
        "<b>Rs. 3.82 crore</b>, up from Rs. 3.15 crore in Q1 FY26."
    ))
    st.append(p(
        "Two matters are carried forward for procurement action: recoveries against "
        "Kaveri Metals Ltd and Trident Polymers Ltd have not yet been raised, and the "
        "safety stock held on imported microcontroller MCU-4471 has not been revised "
        "since the replenishment lead time lengthened. Both are set out in sections 6 "
        "and 5 respectively. The applicable rules are held in the Procurement Policy "
        "Handbook and are not reproduced in this review."
    ))
    st.append(table(
        [
            ["Indicator", "Q4 FY25", "Q1 FY26", "Q2 FY26", "Target"],
            ["Total procurement spend (Rs. crore)", "195.10", "202.20", "214.70", "-"],
            ["Composite on-time delivery (%)", "94.6", "93.7", "92.4", "95.0"],
            ["Composite defect rate (PPM)", "480", "545", "610", "400"],
            ["Line stoppage hours attributable to suppliers", "22.0", "31.5", "41.5", "0.0"],
            ["Cost of poor quality (Rs. crore)", "2.71", "3.15", "3.82", "-"],
            ["Raw material days of cover", "34", "36", "38", "32"],
        ],
        [W * 0.40, W * 0.15, W * 0.15, W * 0.15, W * 0.15],
        align_right=(1, 2, 3, 4),
    ))

    # --- 2. Supplier scorecards ---
    st.append(h1("2. Supplier scorecards"))
    st.append(h2("2.1 Top ten suppliers by spend"))
    st.append(p(
        "The ten suppliers below account for Rs. 194.30 crore of the Rs. 214.70 crore "
        "total spend. Spend is the value invoiced by the supplier during the quarter. "
        "The composite score and the band are calculated by the Supply Chain "
        "Performance Office using the weightings defined in the Procurement Policy "
        "Handbook."
    ))
    st.append(table(
        [
            ["Supplier", "Commodity", "Spend (Rs. cr)", "On-time delivery (%)",
             "Defects (PPM)", "Composite score", "Band"],
            ["Sundaram Forge Pvt Ltd", "Machined castings", "38.40", "96.2", "210", "94.1", "A"],
            ["Kaveri Metals Ltd", "Sheet metal and extrusions", "31.75", "88.1", "1,150", "74.6", "C"],
            ["Trident Polymers Ltd", "Moulded plastic components", "24.90", "91.4", "640", "83.2", "B"],
            ["Anantha Electronics Ltd", "PCB assemblies", "22.15", "93.8", "480", "86.5", "B"],
            ["Deccan Precision Tools", "Tooling and fixtures", "18.60", "97.1", "95", "95.8", "A"],
            ["Vertex Microsystems Pte Ltd", "Microcontrollers (imported)", "16.85", "90.6", "150", "84.0", "B"],
            ["Rashmi Fasteners Pvt Ltd", "Fasteners", "12.30", "84.7", "1,420", "66.3", "D"],
            ["Coastal Logistics Services", "Inbound freight", "11.20", "89.3", "not applicable", "78.4", "C"],
            ["Girija Rubber Works", "Seals and gaskets", "9.45", "94.9", "320", "87.6", "B"],
            ["Nandi Castings Ltd", "Grey iron castings", "8.70", "86.5", "890", "72.9", "C"],
        ],
        [W * 0.21, W * 0.20, W * 0.11, W * 0.13, W * 0.11, W * 0.13, W * 0.07],
        align_right=(2, 3, 4, 5),
    ))
    st.append(Spacer(1, 6))
    st.append(note(
        "Sundaram Forge Pvt Ltd is the highest-spend supplier of the quarter at "
        "Rs. 38.40 crore, with on-time delivery of 96.2%."
    ))

    st.append(h2("2.2 Three-quarter trend for suppliers of concern"))
    st.append(p(
        "The table records on-time delivery and the awarded band for the three most "
        "recent quarters, so that consecutive-quarter conditions can be established."
    ))
    # Column headings use "Q2FY26" as a single unbreakable token. Written as
    # "Q2 FY26" the narrow column wraps between the two words and the quarter
    # is lost when the PDF is converted back to text, leaving a band column
    # nobody can attribute to a quarter.
    st.append(table(
        [
            ["Supplier", "OTD % Q4FY25", "OTD % Q1FY26", "OTD % Q2FY26",
             "Band Q4FY25", "Band Q1FY26", "Band Q2FY26"],
            ["Kaveri Metals Ltd", "91.2", "89.4", "88.1", "B", "C", "C"],
            ["Rashmi Fasteners Pvt Ltd", "88.0", "86.2", "84.7", "C", "D", "D"],
            ["Nandi Castings Ltd", "90.1", "88.9", "86.5", "B", "C", "C"],
            ["Coastal Logistics Services", "92.4", "90.8", "89.3", "B", "B", "C"],
            ["Trident Polymers Ltd", "93.0", "92.1", "91.4", "B", "B", "B"],
        ],
        [W * 0.22, W * 0.13, W * 0.13, W * 0.13, W * 0.13, W * 0.13, W * 0.13],
        align_right=(1, 2, 3),
    ))
    st.append(Spacer(1, 6))
    st.append(p(
        "<b>Bands awarded for Q2 FY26.</b> Kaveri Metals Ltd is band C. Nandi "
        "Castings Ltd is band C. Coastal Logistics Services is band C. Rashmi "
        "Fasteners Pvt Ltd is band D. Every other supplier listed in section 2.1 "
        "is band A or band B. Four suppliers therefore sit below band B this quarter."
    ))
    st.append(note(
        "Kaveri Metals Ltd and Nandi Castings Ltd have now delivered below 90% "
        "on-time for two consecutive quarters, and have been in band C for two "
        "consecutive quarters. Rashmi Fasteners Pvt Ltd has delivered below 90% for "
        "three consecutive quarters and has been in band D for two consecutive "
        "quarters. Coastal Logistics Services has fallen below 90% for the first "
        "time and is in band C for the first time."
    ))

    st.append(PageBreak())

    # --- 3. Freight ---
    st.append(h1("3. Freight lane cost performance"))
    st.append(p(
        "Freight is reported separately from the material spend in section 2. Total "
        "inbound and inter-plant freight for the quarter was <b>Rs. 5.75 crore</b>."
    ))
    st.append(table(
        [
            ["Lane", "Mode", "Volume", "Cost per unit (Rs.)", "Lane cost (Rs. lakh)",
             "Change vs Q1 FY26"],
            ["Penang to Chennai", "Air freight", "34 shipments", "3.18 lakh per shipment",
             "108.12", "+11.4%"],
            ["Shenzhen to Chennai", "Sea, full container", "96 containers",
             "2.42 lakh per container", "232.32", "-3.1%"],
            ["Chennai to Pune (inter-plant)", "Road, full truck", "412 trips",
             "41,800 per trip", "172.22", "+2.6%"],
            ["Coimbatore to Chennai", "Road, part truck", "288 trips",
             "18,600 per trip", "53.57", "+0.9%"],
            ["Chennai Port to plant (drayage)", "Road", "96 moves",
             "9,400 per move", "9.02", "+4.2%"],
            ["<b>Total</b>", "", "", "", "<b>575.25</b>", ""],
        ],
        [W * 0.23, W * 0.15, W * 0.13, W * 0.20, W * 0.14, W * 0.15],
        align_right=(4,),
    ))
    st.append(Spacer(1, 6))
    st.append(note(
        "The Penang to Chennai air lane is used as the contingency lane for imported "
        "electronics. Its cost rose 11.4% in the quarter because six unplanned air "
        "shipments were raised to cover the microcontroller shortage described in "
        "section 5."
    ))

    # --- 4. Inventory ---
    st.append(h1("4. Inventory position at 30 September 2025"))
    st.append(table(
        [
            ["Category", "Value (Rs. crore)", "Days of cover", "Target days of cover"],
            ["Raw material and bought-out components", "47.30", "38", "32"],
            ["Work in progress", "12.85", "9", "8"],
            ["Finished goods", "21.40", "17", "15"],
            ["<b>Total inventory</b>", "<b>81.55</b>", "-", "-"],
        ],
        [W * 0.40, W * 0.20, W * 0.20, W * 0.20],
        align_right=(1, 2, 3),
    ))
    st.append(Spacer(1, 6))
    st.append(p(
        "Slow-moving and excess stock stood at Rs. 6.12 crore, of which Rs. 2.44 crore "
        "is imported electronic components held as buffer. Raw material days of cover "
        "are 6 days above target, driven mainly by the two sea-freight lanes arriving "
        "in larger, less frequent lots."
    ))

    # --- 5. Critical imported part ---
    st.append(h1("5. Critical imported part - microcontroller MCU-4471"))
    st.append(p(
        "MCU-4471 is a 32-bit motor-control microcontroller used on every variant "
        "built at the Chennai plant. It is supplied only by Vertex Microsystems Pte "
        "Ltd from Penang, Malaysia. No alternate source is qualified. The part is "
        "classified internally as a Class-A part because a shortage stops final "
        "assembly within two working days."
    ))
    st.append(table(
        [
            ["Parameter", "Value"],
            ["Part number and description", "MCU-4471, 32-bit motor-control microcontroller"],
            ["Supplier", "Vertex Microsystems Pte Ltd, Penang, Malaysia"],
            ["Sourcing profile", "Imported, single source, no qualified alternate"],
            ["Supplier production lead time", "24 days"],
            ["Ocean transit time, Penang to Chennai", "16 days"],
            ["Customs clearance and inbound inspection", "6 days"],
            ["<b>Total replenishment lead time</b>", "<b>46 days</b>"],
            ["Average daily demand", "220 units per day"],
            ["Standard deviation of daily demand", "38 units per day"],
            ["Safety stock currently held", "9 days of cover, equal to 1,980 units"],
            ["Last safety stock revision", "March 2024, against a lead time of 28 days"],
        ],
        [W * 0.42, W * 0.58],
    ))
    st.append(Spacer(1, 6))
    st.append(note(
        "The replenishment lead time has lengthened from 28 days to 46 days since the "
        "safety stock was last set. The holding of 9 days of cover has not been "
        "recalculated. The calculation method and the minimum holding required are "
        "defined in the Procurement Policy Handbook, not in this review. A stock-out "
        "of MCU-4471 on 05 August 2025 stopped Sub-assembly Cell C for 9.0 hours "
        "(section 6, event 3)."
    ))

    st.append(PageBreak())

    # --- 6. Line stoppages ---
    st.append(h1("6. Production line stoppages attributable to suppliers"))
    st.append(p(
        "Seven stoppages were recorded in the quarter, totalling <b>41.5 hours</b> of "
        "lost production. Lost contribution margin is valued at Meridian's standard "
        "rate of Rs. 4.5 lakh per hour of line downtime, giving <b>Rs. 1.87 crore</b> "
        "for the quarter. No recovery has yet been raised against any supplier for "
        "these stoppages."
    ))
    st.append(table(
        [
            ["#", "Date", "Line affected", "Downtime (hours)", "Part", "Supplier", "Cause"],
            ["1", "14 Jul 2025", "Assembly Line 2", "6.5", "Sheet metal bracket SM-2210",
             "Kaveri Metals Ltd", "Late delivery; vehicle detained at state check post"],
            ["2", "22 Jul 2025", "Assembly Line 1", "4.0", "Fastener kit FK-118",
             "Rashmi Fasteners Pvt Ltd", "Dimensional rejection at incoming inspection"],
            ["3", "05 Aug 2025", "Sub-assembly Cell C", "9.0", "Microcontroller MCU-4471",
             "Vertex Microsystems Pte Ltd",
             "Stock-out during customs clearance delay at Chennai port"],
            ["4", "18 Aug 2025", "Assembly Line 2", "3.5", "Moulded housing MP-540",
             "Trident Polymers Ltd", "Short shipment; 38% of the ordered quantity received"],
            ["5", "27 Aug 2025", "Paint shop", "5.5", "Grey iron casting GC-77",
             "Nandi Castings Ltd", "Porosity defects found after machining; batch quarantined"],
            ["6", "09 Sep 2025", "Assembly Line 1", "7.0", "Sheet metal panel SM-3140",
             "Kaveri Metals Ltd", "Weld distortion outside tolerance; batch quarantined"],
            ["7", "24 Sep 2025", "Sub-assembly Cell C", "6.0", "Seal kit SK-92",
             "Girija Rubber Works", "Tool breakdown at the supplier's works"],
            ["", "<b>Total</b>", "", "<b>41.5</b>", "", "", ""],
        ],
        [W * 0.04, W * 0.11, W * 0.15, W * 0.10, W * 0.19, W * 0.17, W * 0.24],
        align_right=(3,),
    ))
    st.append(Spacer(1, 8))
    st.append(h2("6.1 Downtime by cause"))
    st.append(table(
        [
            ["Cause category", "Events", "Downtime (hours)", "Share of total"],
            ["Late delivery or logistics delay (events 1, 3)", "2", "15.5", "37.3%"],
            ["Defective material rejected (events 2, 5, 6)", "3", "16.5", "39.8%"],
            ["Short shipment (event 4)", "1", "3.5", "8.4%"],
            ["Supplier plant or tooling breakdown (event 7)", "1", "6.0", "14.5%"],
            ["<b>Total</b>", "<b>7</b>", "<b>41.5</b>", "<b>100.0%</b>"],
        ],
        [W * 0.46, W * 0.14, W * 0.20, W * 0.20],
        align_right=(1, 2, 3),
    ))
    st.append(Spacer(1, 6))
    st.append(note(
        "Kaveri Metals Ltd is the largest single contributor, responsible for two "
        "events and 13.5 hours of downtime, equal to 32.5% of the quarter's total."
    ))

    # --- 7. Quality ---
    st.append(h1("7. Quality performance"))
    st.append(p(
        "Defect rates are stated in defects per million parts received (PPM), measured "
        "at incoming inspection and on the line, for the quarter to which the "
        "scorecard relates."
    ))
    st.append(table(
        [
            ["Supplier", "PPM Q1 FY26", "PPM Q2 FY26", "Trend",
             "Quality incidents recorded in Q2 FY26",
             "Sorting and rework cost recovered (Rs. lakh)"],
            ["Rashmi Fasteners Pvt Ltd", "1,310", "1,420", "Worse", "5", "0.00 (debit note disputed)"],
            ["Kaveri Metals Ltd", "940", "1,150", "Worse", "4", "0.00 (not yet raised)"],
            ["Nandi Castings Ltd", "810", "890", "Worse", "3", "6.40"],
            ["Trident Polymers Ltd", "520", "640", "Worse", "3", "0.00 (not yet raised)"],
            ["Anantha Electronics Ltd", "610", "480", "Better", "2", "4.10"],
            ["Girija Rubber Works", "300", "320", "Broadly flat", "1", "0.00"],
            ["Sundaram Forge Pvt Ltd", "240", "210", "Better", "1", "0.00"],
            ["Vertex Microsystems Pte Ltd", "180", "150", "Better", "1", "0.00"],
            ["Deccan Precision Tools", "120", "95", "Better", "0", "0.00"],
        ],
        [W * 0.21, W * 0.12, W * 0.12, W * 0.12, W * 0.19, W * 0.24],
        align_right=(1, 2),
    ))
    st.append(Spacer(1, 8))
    st.append(h2("7.1 Cost of poor quality"))
    st.append(table(
        [
            ["Element", "Q1 FY26 (Rs. crore)", "Q2 FY26 (Rs. crore)"],
            ["Scrap", "1.21", "1.44"],
            ["Rework", "0.82", "0.98"],
            ["Sorting and containment", "0.57", "0.71"],
            ["Warranty and field returns", "0.55", "0.69"],
            ["<b>Total</b>", "<b>3.15</b>", "<b>3.82</b>"],
        ],
        [W * 0.44, W * 0.28, W * 0.28],
        align_right=(1, 2),
    ))
    st.append(Spacer(1, 6))
    st.append(note(
        "No debit note or cost recovery has been raised against Kaveri Metals Ltd or "
        "Trident Polymers Ltd for Q2 FY26. Procurement is to establish the applicable "
        "clauses in the Procurement Policy Handbook and raise the recoveries in Q3 FY26."
    ))

    st.append(PageBreak())

    # --- 8. Risk register ---
    st.append(h1("8. Supply risk register"))
    st.append(table(
        [
            ["ID", "Risk", "Exposure", "Likelihood / impact", "Owner", "Status note"],
            ["R-01",
             "Single source on 32-bit microcontrollers MCU-4471 and MCU-4480, supplied "
             "only by Vertex Microsystems Pte Ltd, Penang",
             "Rs. 16.85 crore of quarterly spend", "Medium / High",
             "Category Manager, Electronics",
             "No alternate source qualified. The qualification project has been open "
             "since Q3 FY25, which is four quarters."],
            ["R-02",
             "Kaveri Metals Ltd performance decline: 88.1% on-time delivery and "
             "1,150 PPM in Q2 FY26",
             "Rs. 31.75 crore of quarterly spend", "High / High",
             "Head of Supply Chain",
             "Second consecutive quarter below 90% on-time and second consecutive "
             "quarter in band C. Recoveries not yet raised."],
            ["R-03",
             "Replenishment lead time on imported electronics has lengthened from "
             "28 days to 46 days",
             "Assembly stops within 2 working days of a stock-out", "High / High",
             "Head of Materials Planning",
             "Safety stock still set at the FY24 level of 9 days of cover."],
            ["R-04",
             "Air-freight cost escalation on the Penang to Chennai contingency lane",
             "Rs. 1.08 crore in the quarter", "Medium / Medium",
             "Logistics Manager", "Six unplanned air shipments raised in the quarter."],
            ["R-05",
             "Rashmi Fasteners Pvt Ltd in band D for two consecutive quarters at "
             "1,420 PPM",
             "Rs. 12.30 crore of quarterly spend", "High / Medium",
             "Category Manager, Fasteners",
             "Debit note raised in Q1 FY26 is disputed by the supplier."],
            ["R-06",
             "Port congestion at Chennai; average container dwell time up from "
             "3.1 to 4.6 days",
             "All sea-freight lanes", "Medium / Medium",
             "Logistics Manager", "Monitored weekly. No mitigation cost incurred yet."],
        ],
        [W * 0.07, W * 0.25, W * 0.16, W * 0.12, W * 0.15, W * 0.25],
    ))

    # --- 9. Actions ---
    st.append(h1("9. Actions carried into Q3 FY26"))
    st.append(p(
        "1. Establish the clauses triggered by Kaveri Metals Ltd at 88.1% on-time "
        "delivery and 1,150 PPM, and raise the resulting recoveries. Owner: Category "
        "Manager, Sheet Metal. Due 15 November 2025."
    ))
    st.append(p(
        "2. Establish the cost consequence applicable to Trident Polymers Ltd at "
        "640 PPM across three quality incidents, and raise the recovery. Owner: "
        "Category Manager, Plastics. Due 15 November 2025."
    ))
    st.append(p(
        "3. Recalculate the safety stock for MCU-4471 against the current 46-day "
        "replenishment lead time and the applicable minimum floor. Owner: Head of "
        "Materials Planning. Due 31 October 2025."
    ))
    st.append(p(
        "4. Bring the single-source position on microcontrollers into line with the "
        "sourcing policy, including buffer cover and the alternate-source "
        "qualification plan. Owner: Category Manager, Electronics. Due 30 November 2025."
    ))
    st.append(p(
        "5. Enter every supplier whose band is below B on the Watch List and start the "
        "escalation stage applicable to each. Owner: Head of Supply Chain. Due "
        "31 October 2025."
    ))
    st.append(Spacer(1, 8))
    st.append(note(
        "This review reports measured performance only. Approval limits, penalty "
        "clauses, safety stock rules, classification categories and escalation stages "
        "are defined in the Meridian Procurement Policy Handbook, version 4.2."
    ))
    return st


# ===========================================================================
# Document 2 - Procurement Policy Handbook  (doc_type: policy)
# ===========================================================================
def policy_story() -> list:
    W = A4[0] - 4 * cm
    st: list = []

    st.append(Paragraph("Procurement Policy Handbook", S["title"]))
    st.append(
        Paragraph(
            "Meridian Industrial Systems Limited &nbsp;|&nbsp; Version 4.2 &nbsp;|&nbsp; "
            "Effective 01 April 2025 &nbsp;|&nbsp; Owner: Head of Procurement &nbsp;|&nbsp; "
            "Reviewed annually. All values in Indian Rupees (Rs.); 1 lakh = 100,000 and "
            "1 crore = 10,000,000.",
            S["subtitle"],
        )
    )

    # --- 1. Purpose and scope ---
    st.append(h1("1. Purpose and scope"))
    st.append(p(
        "This handbook sets out the rules that govern how Meridian Industrial Systems "
        "Limited buys goods and services, how suppliers are classified and measured, "
        "what happens when performance falls short, how much inventory cover must be "
        "held, and who may approve what. It applies to all direct and indirect "
        "procurement at every Meridian plant and to every employee who commits "
        "Meridian to a supplier."
    ))
    st.append(p(
        "This handbook states rules only. Measured supplier performance for a given "
        "quarter is published separately by the Supply Chain Performance Office in the "
        "quarterly Supply Chain Performance Review. Where a rule refers to a figure "
        "such as on-time delivery or defect rate, the figure is to be taken from the "
        "scorecard published in that review."
    ))

    # --- 2. Supplier classification ---
    st.append(h1("2. Supplier classification"))
    st.append(h2("2.1 Classification categories"))
    st.append(p(
        "Every supplier is placed in exactly one of the following <b>four "
        "classification categories</b>. The category is reviewed annually and "
        "determines governance, contracting and sourcing obligations."
    ))
    st.append(table(
        [
            ["Category", "Qualifying condition", "Governance requirement", "Contracting"],
            ["Strategic",
             "Annual spend above Rs. 25 crore, or sole source for any Class-A part",
             "Executive sponsor appointed; quarterly business review with the "
             "Procurement Director",
             "Three-year agreement with annual price review"],
            ["Critical",
             "Annual spend from Rs. 5 crore up to Rs. 25 crore, or supplies a part "
             "whose shortage would stop a production line within five working days",
             "Monthly scorecard; quarterly performance meeting with the Category Manager",
             "Two-year agreement"],
            ["Preferred",
             "Annual spend from Rs. 50 lakh up to Rs. 5 crore, and band A or band B "
             "sustained for four consecutive quarters",
             "Quarterly scorecard", "One-year agreement or rate contract"],
            ["Transactional",
             "Annual spend below Rs. 50 lakh, catalogue or commodity items",
             "Annual review only", "Purchase order terms only"],
        ],
        [W * 0.13, W * 0.32, W * 0.32, W * 0.23],
    ))
    st.append(Spacer(1, 6))
    st.append(clause(
        "<b>2.1.1</b> Probation is a <b>status</b> and not a classification category. "
        "A supplier placed on probation under clause 7 keeps its existing category but "
        "is not eligible for new part awards while the probation runs."
    ))
    st.append(clause(
        "<b>2.1.2</b> Where a supplier meets the qualifying condition for more than "
        "one category, the higher category applies. Category changes take effect from "
        "the start of the next quarter."
    ))

    st.append(h2("2.2 Performance bands"))
    st.append(p(
        "Separately from its category, every supplier is awarded a performance band "
        "each quarter from its composite score. The band, not the category, drives "
        "escalation under clause 8."
    ))
    st.append(table(
        [
            ["Band", "Composite score", "Standing"],
            ["A", "90.0 and above", "Preferred for new business"],
            ["B", "80.0 to 89.9", "Acceptable"],
            ["C", "70.0 to 79.9", "Below acceptable; Watch List entry required"],
            ["D", "Below 70.0", "Unacceptable; re-sourcing to be planned"],
        ],
        [W * 0.12, W * 0.28, W * 0.60],
    ))
    st.append(Spacer(1, 6))
    st.append(clause(
        "<b>2.2.1</b> The composite score is calculated as 40% on-time delivery, "
        "35% quality, 15% cost competitiveness and 10% responsiveness. Bands B, C and "
        "D are all below band A; the expression \"below band B\" in this handbook "
        "means band C or band D."
    ))

    # --- 3. Approval authority ---
    st.append(h1("3. Approval authority"))
    st.append(h2("3.1 Purchase order approval limits"))
    st.append(p(
        "No purchase order may be released without approval at or above the level "
        "shown below for its total value, inclusive of taxes and freight. The limits "
        "are cumulative over the full committed value of the order, not the value of "
        "an individual release."
    ))
    st.append(table(
        [
            ["Purchase order value", "Approving authority", "Additional requirement"],
            ["Up to Rs. 5 lakh", "Buyer", "None"],
            ["Above Rs. 5 lakh and up to Rs. 25 lakh", "Category Manager",
             "Two written quotations on file"],
            ["Above Rs. 25 lakh and up to Rs. 1 crore", "Head of Procurement",
             "Three written quotations and a price comparison statement"],
            ["Above Rs. 1 crore and up to Rs. 2.5 crore", "Procurement Director",
             "Countersignature of the Chief Financial Officer, plus a documented "
             "negotiation record"],
            ["Above Rs. 2.5 crore and up to Rs. 10 crore", "Executive Committee",
             "Noting by the Board Procurement Sub-committee at its next meeting"],
            ["Above Rs. 10 crore", "Board of Directors",
             "Business case approved by the Chief Executive Officer before the order "
             "is placed"],
        ],
        [W * 0.30, W * 0.24, W * 0.46],
    ))
    st.append(Spacer(1, 6))
    st.append(clause(
        "<b>3.1.1</b> A purchase order worth more than Rs. 1 crore and not more than "
        "Rs. 2.5 crore is approved by the Procurement Director and countersigned by "
        "the Chief Financial Officer. A worked example: an order of Rs. 1.4 crore "
        "falls in this band, so the Procurement Director approves it, the Chief "
        "Financial Officer countersigns it, and a documented negotiation record is "
        "filed with it. Neither the Head of Procurement nor the Category Manager may "
        "release an order of this value."
    ))
    st.append(clause(
        "<b>3.1.2</b> Splitting is prohibited. An order must not be divided to bring "
        "any part of it under an approval limit. Two or more purchase orders placed on "
        "the same supplier for the same part within any 30-day period are treated as a "
        "single order at their combined value for the purpose of clause 3.1."
    ))
    st.append(clause(
        "<b>3.1.3</b> Emergency purchases needed to prevent or end a line stoppage may "
        "be approved by the Plant Head up to Rs. 15 lakh, and must be ratified by the "
        "authority named in clause 3.1 within three working days."
    ))

    st.append(PageBreak())

    # --- 4. Sourcing rules ---
    st.append(h1("4. Sourcing rules"))
    st.append(h2("4.1 Competitive sourcing"))
    st.append(clause(
        "<b>4.1.1</b> Any requirement above Rs. 25 lakh must be competitively bid with "
        "at least three qualified sources, unless a single-source justification is "
        "approved under clause 4.2."
    ))

    st.append(h2("4.2 Single sourcing and sole sourcing"))
    st.append(clause(
        "<b>4.2.1</b> Dual sourcing is mandatory for every part supplied by a "
        "Strategic or Critical supplier. A part is single-sourced when only one "
        "supplier is approved to supply it, and sole-sourced when only one supplier "
        "exists worldwide that is capable of supplying it."
    ))
    st.append(clause(
        "<b>4.2.2</b> Where a part is single-sourced, <b>all</b> of the following are "
        "required, and the position must be reported in the quarterly review until it "
        "is closed: <br/>"
        "(a) a written single-source justification approved by the Procurement "
        "Director and renewed every 12 months; <br/>"
        "(b) a buffer stock of not less than <b>eight weeks</b> of average demand, held "
        "either at Meridian's premises or in a bonded warehouse within 50 km of the "
        "receiving plant; <br/>"
        "(c) an alternate-source qualification plan with a committed completion date "
        "not more than <b>two quarters</b> from the date the single-source status was "
        "recorded; <br/>"
        "(d) a review at every meeting of the Supplier Review Board until an alternate "
        "source is qualified; <br/>"
        "(e) for an <b>imported</b> single-sourced part, a further two weeks of buffer "
        "over and above (b), giving ten weeks in total, and a nominated air-freight "
        "contingency lane kept open at all times."
    ))
    st.append(clause(
        "<b>4.2.3</b> Where the qualification plan required by clause 4.2.2(c) passes "
        "its committed completion date, the Procurement Director must report the "
        "overrun to the Executive Committee at its next meeting, and no new part may "
        "be awarded to that supplier until an alternate source is qualified."
    ))
    st.append(clause(
        "<b>4.2.4</b> A sole-sourced part carries every requirement of clause 4.2.2 "
        "except (c), which may be waived by the Executive Committee, and additionally "
        "requires an annual technology and continuity review with the supplier."
    ))

    # --- 5. Inventory and safety stock ---
    st.append(h1("5. Inventory and safety stock"))
    st.append(h2("5.1 Basis of measurement"))
    st.append(clause(
        "<b>5.1.1</b> Safety stock is expressed in days of cover. One day of cover is "
        "the average daily demand for the part over the trailing quarter. Total "
        "replenishment lead time is the sum of the supplier's production lead time, "
        "the transit time, and the time taken for customs clearance and inbound "
        "inspection."
    ))

    st.append(h2("5.2 Calculated safety stock"))
    st.append(clause(
        "<b>5.2.1</b> The calculated safety stock for every planned part is: <br/>"
        "<b>Safety stock in days of cover = ( 0.25 x total replenishment lead time in "
        "days ) + 3 days</b> <br/>"
        "The result is rounded up to the next whole day. To convert to units, multiply "
        "the days of cover by the average daily demand."
    ))

    st.append(h2("5.3 Minimum floors"))
    st.append(clause(
        "<b>5.3.1</b> Irrespective of the value produced by clause 5.2.1, the minimum "
        "floors in the table below apply. <b>Where the calculated value and the "
        "applicable floor differ, the higher of the two applies.</b> A calculated "
        "value below the floor must never be used."
    ))
    st.append(table(
        [
            ["Part sourcing profile", "Minimum days of cover"],
            ["Domestic, total replenishment lead time up to 15 days", "5 days"],
            ["Domestic, total replenishment lead time above 15 days", "10 days"],
            ["Imported, total replenishment lead time up to 30 days", "15 days"],
            ["Imported, total replenishment lead time above 30 days and up to 45 days", "18 days"],
            ["Imported, total replenishment lead time above 45 days", "24 days"],
            ["Single-sourced and imported, any lead time", "30 days"],
        ],
        [W * 0.68, W * 0.32],
    ))
    st.append(Spacer(1, 6))
    st.append(clause(
        "<b>5.3.2</b> Where more than one floor in clause 5.3.1 applies to the same "
        "part, the <b>highest</b> applicable floor is used. For example, a part that "
        "is imported with a total replenishment lead time of 46 days attracts a floor "
        "of 24 days; if that same part is also single-sourced, the floor of 30 days "
        "applies instead."
    ))
    st.append(clause(
        "<b>5.3.3</b> Buffer stock required by clause 4.2.2(b) or 4.2.2(e) for a "
        "single-sourced part is held in addition to the safety stock required by this "
        "section, and may not be counted towards it."
    ))

    st.append(h2("5.4 Review triggers"))
    st.append(clause(
        "<b>5.4.1</b> Safety stock must be recalculated whenever the total "
        "replenishment lead time for a part changes by more than five days, whenever "
        "average daily demand changes by more than 15%, and in every case at the close "
        "of each financial year. Recalculation is the responsibility of the Head of "
        "Materials Planning."
    ))

    st.append(PageBreak())

    # --- 6. Performance measurement ---
    st.append(h1("6. Supplier performance measurement"))
    st.append(clause(
        "<b>6.1.1</b> Scorecards are published quarterly by the Supply Chain "
        "Performance Office within 20 days of the quarter close, and cover on-time "
        "delivery, quality, cost and responsiveness."
    ))
    st.append(clause(
        "<b>6.2.1</b> A delivery is on time when the full ordered quantity is received "
        "at the nominated dock no earlier than two days before, and no later than, the "
        "confirmed delivery date. A partial delivery is counted as late in full. "
        "On-time delivery for a quarter is the percentage of order lines meeting this "
        "test."
    ))
    st.append(clause(
        "<b>6.3.1</b> The defect rate is stated in defects per million parts received "
        "(PPM), counting defects found at incoming inspection, on the line, and in "
        "warranty returns traced to the supplier, for the quarter to which the "
        "scorecard relates."
    ))

    # --- 7. Consequences ---
    st.append(h1("7. Consequences of underperformance"))
    st.append(p(
        "Each clause in this section states its own trigger condition and the "
        "consequence that follows. More than one clause may be triggered by the same "
        "supplier in the same quarter, in which case every triggered clause applies, "
        "subject only to the cap in clause 7.7. Quarterly invoice value is defined in "
        "clause 10."
    ))

    st.append(h2("7.1 Delivery: first quarter below the threshold"))
    st.append(clause(
        "<b>7.1</b> <b>Trigger:</b> a supplier's on-time delivery falls below 90% in "
        "any quarter. <b>Consequence:</b> the Category Manager issues a formal "
        "Performance Notice within 10 working days of the scorecard being published; "
        "the supplier submits a corrective action plan in 8D format within 15 working "
        "days; and delivery is reviewed weekly until on-time delivery is above 95% for "
        "two consecutive months. No financial recovery arises under this clause alone."
    ))

    st.append(h2("7.2 Delivery: second consecutive quarter below the threshold"))
    st.append(clause(
        "<b>7.2</b> <b>Trigger:</b> a supplier's on-time delivery remains below 90% "
        "for two consecutive quarters. <b>Consequence:</b> in addition to clause 7.1, "
        "(a) a <b>debit note equal to 2% of the quarterly invoice value</b> is raised "
        "against the supplier; (b) the supplier is placed on probation for two "
        "quarters and is not eligible for new part awards during that period; (c) the "
        "supplier's band is reduced by one grade irrespective of its composite score; "
        "and (d) the case is escalated to the Supplier Review Board."
    ))

    st.append(h2("7.3 Delivery: third consecutive quarter below the threshold"))
    st.append(clause(
        "<b>7.3</b> <b>Trigger:</b> a supplier's on-time delivery remains below 90% "
        "for three consecutive quarters. <b>Consequence:</b> the debit note rises to "
        "<b>5% of the quarterly invoice value</b>; not less than 25% of the part "
        "demand is re-allocated to an alternate source; and de-listing is tabled at "
        "the next Supplier Review Board."
    ))

    st.append(h2("7.4 Quality: defect rate from 500 to 1,000 PPM"))
    st.append(clause(
        "<b>7.4</b> <b>Trigger:</b> a supplier's defect rate for the quarter is from "
        "500 to 1,000 defects per million, both figures inclusive. <b>Consequence:</b> "
        "the supplier bears the <b>full cost of containment, sorting and rework</b>, "
        "recovered at Meridian's standard rate of <b>Rs. 1,850 per hour</b>, together "
        "with a fixed <b>quality administration charge of Rs. 40,000 for each quality "
        "incident</b> recorded in the quarter. A corrective action plan in 8D format is "
        "required within 15 working days. No debit note on invoice value arises under "
        "this clause."
    ))

    st.append(h2("7.5 Quality: defect rate above 1,000 PPM"))
    st.append(clause(
        "<b>7.5</b> <b>Trigger:</b> a supplier's defect rate for the quarter is above "
        "1,000 defects per million. <b>Consequence:</b> (a) a <b>debit note equal to "
        "3% of the quarterly invoice value</b> is raised; (b) 100% inspection is "
        "carried out at the supplier's cost, at the supplier's works or by a third "
        "party nominated by Meridian, until three consecutive lots are accepted; "
        "(c) the supplier is placed on probation for two quarters; and (d) the "
        "supplier's band is set to C or lower until the defect rate is below 500 PPM "
        "for two consecutive quarters. The charges in clause 7.4 do not apply in "
        "addition to this clause."
    ))

    st.append(h2("7.6 Line stoppage caused by a supplier"))
    st.append(clause(
        "<b>7.6</b> <b>Trigger:</b> a production line stoppage is directly "
        "attributable to a supplier's late delivery, short shipment or defective "
        "material. <b>Consequence:</b> the supplier is charged the lost contribution "
        "margin at Meridian's standard rate of <b>Rs. 4.5 lakh per hour</b> of "
        "downtime, subject to a cap of 10% of that supplier's quarterly invoice value. "
        "Any single stoppage longer than four hours is additionally reported to the "
        "Supplier Review Board."
    ))

    st.append(h2("7.7 Cap on total recoveries"))
    st.append(clause(
        "<b>7.7</b> Total recoveries raised against a supplier under clauses 7.2, 7.3, "
        "7.4, 7.5 and 7.6 in any single quarter must not exceed <b>12% of that "
        "supplier's quarterly invoice value</b>. Where the sum of the individual "
        "recoveries exceeds the cap, the recoveries are abated proportionately."
    ))

    st.append(h2("7.8 Disputes"))
    st.append(clause(
        "<b>7.8</b> A supplier may dispute a debit note within 15 working days of "
        "issue, in writing and with supporting evidence. The Head of Procurement "
        "decides the dispute within 10 working days. A disputed debit note is held in "
        "abeyance until decided, but the corrective action and any inspection "
        "requirement continue to apply meanwhile."
    ))

    st.append(PageBreak())

    # --- 8. Escalation ---
    st.append(h1("8. Escalation"))
    st.append(h2("8.1 Escalation stages for supplier performance"))
    st.append(p(
        "Escalation is driven by the performance band awarded under clause 2.2. Each "
        "stage carries the stage below it."
    ))
    st.append(table(
        [
            ["Stage", "Trigger", "Owner", "Required action and timeline"],
            ["1", "Band B for two consecutive quarters", "Buyer",
             "Corrective action plan agreed and reviewed fortnightly; closed within 30 days"],
            ["2", "Band C in any quarter", "Category Manager",
             "Performance improvement plan agreed within 15 working days; monthly "
             "review; closed within 60 days"],
            ["3", "Band C for two consecutive quarters, or band D in any quarter",
             "Head of Supply Chain",
             "Formal escalation letter to the supplier's Managing Director; on-site "
             "assessment within 30 days; recovery plan agreed within 45 days"],
            ["4", "No improvement after stage 3, or band D for two consecutive quarters",
             "Supplier Review Board, chaired by the Chief Operating Officer",
             "De-listing decision taken and a re-sourcing plan approved within one quarter"],
        ],
        [W * 0.08, W * 0.28, W * 0.24, W * 0.40],
    ))
    st.append(Spacer(1, 6))
    st.append(h2("8.2 Watch List and the Supplier Review Board"))
    st.append(clause(
        "<b>8.2.1</b> Any supplier whose band falls <b>below band B</b>, that is to "
        "say band C or band D, in any quarter is entered on the Watch List maintained "
        "by the Head of Supply Chain, is reviewed at the monthly Supplier Review Board "
        "meeting, and receives no new part awards until its band has returned to B or "
        "above for two consecutive quarters."
    ))
    st.append(clause(
        "<b>8.2.2</b> The Supplier Review Board meets monthly. It is chaired by the "
        "Chief Operating Officer and attended by the Procurement Director, the Head of "
        "Supply Chain, the Head of Quality and the Category Manager responsible for "
        "the supplier under review."
    ))
    st.append(clause(
        "<b>8.2.3</b> Where a supplier is escalated under both a delivery clause in "
        "section 7 and a band trigger in clause 8.1, the higher stage applies and a "
        "single combined escalation is run."
    ))

    st.append(h2("8.3 Escalation of a supply interruption"))
    st.append(clause(
        "<b>8.3.1</b> A stock-out or stoppage that threatens more than one shift is "
        "escalated the same day to the Head of Supply Chain, and where it threatens "
        "more than three shifts, to the Chief Operating Officer."
    ))

    # --- 9. Records ---
    st.append(h1("9. Records and audit"))
    st.append(clause(
        "<b>9.1.1</b> Purchase orders, approvals, quotations, negotiation records, "
        "scorecards, debit notes and dispute decisions are retained for seven years "
        "and are subject to internal audit twice each financial year."
    ))

    # --- 10. Definitions ---
    st.append(h1("10. Definitions"))
    st.append(table(
        [
            ["Term", "Meaning in this handbook"],
            ["On-time delivery (OTD)",
             "The percentage of order lines received in full at the nominated dock "
             "within the window defined in clause 6.2.1"],
            ["Defect rate (PPM)",
             "Defects per million parts received, counted as set out in clause 6.3.1"],
            ["Quarterly invoice value",
             "The total value invoiced by the supplier during the quarter to which the "
             "scorecard relates, excluding taxes"],
            ["Days of cover",
             "Stock on hand divided by average daily demand for the trailing quarter"],
            ["Composite score", "The weighted score defined in clause 2.2.1"],
            ["Band", "The quarterly grade A, B, C or D awarded under clause 2.2"],
            ["Below band B", "Band C or band D"],
            ["Total replenishment lead time",
             "Supplier production lead time plus transit time plus customs clearance "
             "and inbound inspection time, as defined in clause 5.1.1"],
            ["Class-A part",
             "A part whose shortage stops a production line within two working days"],
            ["Probation",
             "A status under which a supplier receives no new part awards for the "
             "stated period"],
        ],
        [W * 0.26, W * 0.74],
    ))
    st.append(Spacer(1, 8))
    st.append(note(
        "End of version 4.2. This handbook contains no figures for any individual "
        "supplier's measured performance; those are published quarterly in the Supply "
        "Chain Performance Review."
    ))
    return st


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    review_path = config.DATA_DIR / REVIEW_FILE
    policy_path = config.DATA_DIR / POLICY_FILE

    build(review_path, "Meridian Supply Chain Performance Review - Q2 FY26", review_story())
    build(policy_path, "Meridian Procurement Policy Handbook - version 4.2", policy_story())

    for path in (review_path, policy_path):
        size_kb = path.stat().st_size / 1024
        print(f"Wrote {path.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
