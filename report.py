"""
Turn an analysis into something the student can keep.

    to_pdf(results, meta)        the printable report - summary, what to study first, unit by unit
    to_markdown(results, meta)   the same thing as text
    to_csv(results)              one row per syllabus topic, for a spreadsheet

All three take the result list the app builds: each item has topic, unit, level, reasons, kinds,
evidence [(file, location, confidence, text)], closest, and optionally diagram / diagram_pages.

The PDF is built with reportlab rather than Word, so it also works where Word is not installed -
a Linux server, or Hugging Face Spaces.
"""
import csv
import datetime
import io
import os

from depth import advice, study_order

LEVELS = ["PROPERLY TAUGHT", "BARELY TAUGHT", "NOT TAUGHT", "NO MATERIAL PROVIDED"]
BADGE = {"PROPERLY TAUGHT": "✅", "BARELY TAUGHT": "⚠️", "NOT TAUGHT": "❌",
         "NO MATERIAL PROVIDED": "➖"}
SHORT = {"PROPERLY TAUGHT": "Properly taught", "BARELY TAUGHT": "Barely taught",
         "NOT TAUGHT": "Not taught", "NO MATERIAL PROVIDED": "No material provided"}
DIAGRAM = {"diagram": "yes", "none": "no", "not_checked": "not checked"}


def count(rs, level):
    return sum(1 for r in rs if r["level"] == level)


def to_markdown(results, meta=None):
    meta = meta or {}
    out = ["# Syllabus coverage report", ""]
    out.append(f"Generated {datetime.date.today().strftime('%d %B %Y')} by SyllabiSense.")
    if meta.get("syllabus"):
        out.append(f"**Syllabus:** {meta['syllabus']}")
    if meta.get("material"):
        files = meta["material"]
        out.append(f"**Material checked:** {len(files)} file(s) — " + ", ".join(files))
    out.append("")

    out += ["## Summary", "", "| Verdict | Topics |", "| --- | ---: |"]
    for lv in LEVELS:
        n = count(results, lv)
        if n:
            out.append(f"| {BADGE[lv]} {SHORT[lv]} | {n} |")
    out += [f"| **Total** | **{len(results)}** |", ""]

    todo = study_order(results)
    if todo:
        out += ["## What to study first", ""]
        for i, r in enumerate(todo[:10], 1):
            why = ("nothing in the material covers it" if r["level"] == "NOT TAUGHT"
                   else "the material is too thin")
            out.append(f"{i}. **{r['topic']}** — {why}  _({r['unit']})_")
        out.append("")

    out += ["## Unit by unit", ""]
    for unit in dict.fromkeys(r["unit"] for r in results):
        rs = [r for r in results if r["unit"] == unit]
        out += [f"### {unit}", ""]
        if count(rs, "NO MATERIAL PROVIDED") == len(rs):
            out += ["No material was uploaded for this unit, so it could not be judged.", ""]
            continue
        out.append("  ·  ".join(f"{BADGE[lv]} {SHORT[lv].lower()} {count(rs, lv)}"
                                for lv in LEVELS if count(rs, lv)))
        out.append("")
        for lv in LEVELS:
            group = [r for r in rs if r["level"] == lv]
            if not group:
                continue
            out += [f"**{BADGE[lv]} {SHORT[lv]} — {len(group)}**", ""]
            for r in group:
                out.append(f"- **{r['topic']}**")
                if r.get("reasons"):
                    out.append(f"  - {' · '.join(r['reasons'])}")
                out.append(f"  - What to do: {advice(r['level'], r.get('kinds'))}")
                for f, loc, pr, _ in r.get("evidence", [])[:3]:
                    out.append(f"  - Evidence: {f} — {loc} (confidence {pr:.2f})")
                if r.get("diagram") == "diagram" and r.get("diagram_pages"):
                    df, dloc, dp = r["diagram_pages"][0]
                    out.append(f"  - Diagram found: {os.path.basename(df)} — {dloc} "
                               f"(confidence {dp:.2f})")
                elif r.get("diagram") == "none":
                    out.append("  - No diagram on the evidence pages — explained in words only.")
                if not r.get("evidence") and r["level"] == "NOT TAUGHT" and r.get("closest"):
                    f, loc, pr = r["closest"]
                    out.append(f"  - Closest page: {f} — {loc} (scored {pr:.2f}, below the "
                               f"threshold)")
            out.append("")

    out += ["---", "",
            "Verdicts come from an artificial neural network trained on 11,615 (topic, page) "
            "pairs from 12 non-computing subject areas and tested on 109 Computer Science "
            "courses; measured against human-checked labels it is right about 84% of the time "
            "(±5). Diagram detection is a convolutional network, 84% accurate, and reads PDF "
            "pages only. Treat this as a study aid, not a verdict on the teaching."]
    return "\n".join(out)


def to_pdf(results, meta=None):
    """The printable report. Returns PDF bytes."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    meta = meta or {}
    NAVY, BLUE = colors.HexColor("#1F3864"), colors.HexColor("#2E74B5")
    GREY, BAND = colors.HexColor("#595959"), colors.HexColor("#EAF1F8")
    LINE = colors.HexColor("#B7C7D9")

    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def style(name, size, color=colors.black, bold=False, space_before=0, space_after=3,
              left=0, leading=None):
        return ParagraphStyle(name, fontName="Helvetica-Bold" if bold else "Helvetica",
                              fontSize=size, textColor=color, spaceBefore=space_before,
                              spaceAfter=space_after, leftIndent=left,
                              leading=leading or size * 1.35)

    H1 = style("h1", 17, NAVY, True, 0, 4)
    SUB = style("sub", 9, GREY, False, 0, 2)
    H2 = style("h2", 12.5, NAVY, True, 14, 5)
    H3 = style("h3", 10, BLUE, True, 9, 4)
    BODY = style("body", 9.5, colors.black, False, 0, 3)
    BODY.alignment = TA_JUSTIFY
    TOPIC = style("topic", 9.5, colors.black, True, 5, 1)
    DETAIL = style("detail", 8.5, GREY, False, 0, 1, left=10)
    NOTE = style("note", 8, GREY, False, 8, 0)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            title="Syllabus coverage report", author="SyllabiSense")
    width = doc.width

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.4)
        canvas.line(doc_.leftMargin, 13 * mm, doc_.leftMargin + width, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(doc_.leftMargin, 9 * mm, "SyllabiSense — syllabus coverage report")
        canvas.drawRightString(doc_.leftMargin + width, 9 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    def grid(data, widths, header=True):
        t = Table(data, colWidths=widths, hAlign="LEFT")
        s = [("GRID", (0, 0), (-1, -1), 0.4, LINE),
             ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
             ("FONTSIZE", (0, 0), (-1, -1), 8.5),
             ("TOPPADDING", (0, 0), (-1, -1), 4),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
        if header:
            s += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
            for i in range(2, len(data), 2):
                s.append(("BACKGROUND", (0, i), (-1, i), BAND))
        t.setStyle(TableStyle(s))
        return t

    story = [Paragraph("Syllabus coverage report", H1)]
    story.append(Paragraph(
        f"Generated {datetime.date.today().strftime('%d %B %Y')} by SyllabiSense.", SUB))
    if meta.get("syllabus"):
        story.append(Paragraph(f"<b>Syllabus:</b> {esc(meta['syllabus'])}", SUB))
    if meta.get("material"):
        files = meta["material"]
        story.append(Paragraph(
            f"<b>Material checked:</b> {len(files)} file(s) — {esc(', '.join(files))}", SUB))
    story.append(Spacer(1, 10))

    rows = [["Verdict", "Topics"]]
    for lv in LEVELS:
        n = count(results, lv)
        if n:
            rows.append([SHORT[lv], str(n)])
    rows.append(["Total", str(len(results))])
    story.append(grid(rows, [width * 0.72, width * 0.28]))

    todo = study_order(results)
    if todo:
        story.append(Paragraph("What to study first", H2))
        for i, r in enumerate(todo[:10], 1):
            why = ("nothing in the material covers it" if r["level"] == "NOT TAUGHT"
                   else "the material is too thin")
            story.append(Paragraph(
                f"<b>{i}. {esc(r['topic'])}</b> — {why}  <font color='#595959'>"
                f"({esc(r['unit'])})</font>", BODY))

    story.append(Paragraph("Unit by unit", H2))
    for unit in dict.fromkeys(r["unit"] for r in results):
        rs = [r for r in results if r["unit"] == unit]
        block = [Paragraph(esc(unit), H3)]
        if count(rs, "NO MATERIAL PROVIDED") == len(rs):
            block.append(Paragraph("No material was uploaded for this unit, so it could not be "
                                   "judged.", BODY))
            story += block
            continue
        block.append(Paragraph(
            "  ·  ".join(f"{SHORT[lv].lower()} {count(rs, lv)}"
                         for lv in LEVELS if count(rs, lv)), SUB))
        story += block
        for lv in LEVELS:
            group = [r for r in rs if r["level"] == lv]
            if not group:
                continue
            story.append(Paragraph(f"{SHORT[lv]} — {len(group)}", TOPIC))
            for r in group:
                lines = [Paragraph(esc(r["topic"]), TOPIC)]
                if r.get("reasons"):
                    lines.append(Paragraph(esc(" · ".join(r["reasons"])), DETAIL))
                lines.append(Paragraph(
                    f"<b>What to do:</b> {esc(advice(r['level'], r.get('kinds')))}", DETAIL))
                for f, loc, pr, _ in r.get("evidence", [])[:3]:
                    lines.append(Paragraph(
                        f"Evidence: {esc(f)} — {esc(loc)} (confidence {pr:.2f})", DETAIL))
                if r.get("diagram") == "diagram" and r.get("diagram_pages"):
                    df, dloc, dp = r["diagram_pages"][0]
                    lines.append(Paragraph(
                        f"Diagram found: {esc(os.path.basename(df))} — {esc(dloc)} "
                        f"(confidence {dp:.2f})", DETAIL))
                elif r.get("diagram") == "none":
                    lines.append(Paragraph("No diagram on these pages — explained in words only.",
                                           DETAIL))
                if not r.get("evidence") and r["level"] == "NOT TAUGHT" and r.get("closest"):
                    f, loc, pr = r["closest"]
                    lines.append(Paragraph(
                        f"Closest page: {esc(f)} — {esc(loc)} (scored {pr:.2f}, below the "
                        f"threshold)", DETAIL))
                story.append(KeepTogether(lines))

    story.append(Paragraph(
        "Verdicts come from an artificial neural network trained on 11,615 (topic, page) pairs "
        "from 12 non-computing subject areas and tested on 109 Computer Science courses; measured "
        "against human-checked labels it is right about 84% of the time (±5). Diagram detection "
        "is a convolutional network, 84% accurate, and reads PDF pages only. Treat this as a "
        "study aid, not a verdict on the teaching.", NOTE))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def to_csv(results):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["unit", "topic", "verdict", "diagram", "evidence_file", "evidence_page",
                "confidence", "what_to_do", "why"])
    for r in results:
        ev = r.get("evidence") or []
        f, loc, pr = (ev[0][0], ev[0][1], f"{ev[0][2]:.2f}") if ev else ("", "", "")
        w.writerow([r["unit"], r["topic"], SHORT[r["level"]],
                    DIAGRAM.get(r.get("diagram", "not_checked"), ""), f, loc, pr,
                    advice(r["level"], r.get("kinds")),
                    " · ".join(r.get("reasons") or [])])
    return buf.getvalue()
