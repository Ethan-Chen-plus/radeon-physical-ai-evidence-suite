#!/usr/bin/env python3
"""Build the Track 3 English technical report from its Markdown source."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PAGE_W, PAGE_H = A4
NAVY = colors.HexColor("#101820")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#526171")
TEAL = colors.HexColor("#188A88")
RED = colors.HexColor("#E44746")
PALE = colors.HexColor("#EEF3F4")
LINE = colors.HexColor("#D7E0E3")


def clean_text(value: str) -> str:
    return value.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")


def inline_markup(value: str) -> str:
    value = html.escape(clean_text(value), quote=False)
    value = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#188A88">\1</link>', value)
    value = re.sub(r"&lt;(https?://[^&]+)&gt;", r'<link href="\1" color="#188A88">\1</link>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    return value


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=26, leading=30, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=12, leading=16, textColor=TEAL, alignment=TA_LEFT, spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Meta", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=MUTED, spaceAfter=1.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=21, leading=25, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=NAVY, spaceBefore=3 * mm, spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="H3x", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=2.5 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Bodyx", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=13.4, textColor=INK, alignment=TA_LEFT,
        spaceAfter=2.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique",
        fontSize=8, leading=10.5, textColor=MUTED, alignment=TA_CENTER, spaceBefore=1.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=7.7, leading=9.6, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.4, leading=9.4, textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock", parent=styles["Code"], fontName="Courier",
        fontSize=7.1, leading=9.4, textColor=colors.HexColor("#DCE7EA"),
        backColor=NAVY, borderPadding=7, borderRadius=3, spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="PullQuote", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=11, leading=16, textColor=NAVY, leftIndent=6 * mm,
        borderColor=TEAL, borderWidth=0, borderPadding=5, spaceAfter=4 * mm,
    ))
    return styles


STYLES = make_styles()


def table_from_rows(rows: list[list[str]]) -> Table:
    cols = max(len(row) for row in rows)
    normalized = [row + [""] * (cols - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = STYLES["TableHead"] if row_index == 0 else STYLES["TableCell"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    available = PAGE_W - 34 * mm
    col_widths = [available / cols] * cols
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def parse_markdown(source: Path) -> list:
    lines = [clean_text(line.rstrip()) for line in source.read_text(encoding="utf-8").splitlines()]
    story = []
    index = next(i for i, line in enumerate(lines) if line == "## Abstract")
    i = index
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), STYLES["H3x"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), STYLES["H1x"]))
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), STYLES["H1x"]))
            i += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", line)
        if image_match:
            caption, relative = image_match.groups()
            image_path = ROOT / relative
            figure = scaled_image(image_path, PAGE_W - 38 * mm, 76 * mm)
            story.append(KeepTogether([
                figure,
                Paragraph(inline_markup(caption), STYLES["Caption"]),
                Spacer(1, 3 * mm),
            ]))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            story.append(table_from_rows(rows))
            story.append(Spacer(1, 3 * mm))
            continue
        if line.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            story.append(Paragraph("<br/>".join(html.escape(item) or " " for item in code), STYLES["CodeBlock"]))
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(ListItem(Paragraph(inline_markup(lines[i][2:]), STYLES["Bodyx"]), leftIndent=4 * mm))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=6 * mm, bulletColor=TEAL))
            story.append(Spacer(1, 1.5 * mm))
            continue
        if re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                item = re.sub(r"^\d+\. ", "", lines[i])
                items.append(ListItem(Paragraph(inline_markup(item), STYLES["Bodyx"]), leftIndent=4 * mm))
                i += 1
            story.append(ListFlowable(items, bulletType="1", leftIndent=8 * mm, bulletColor=TEAL))
            story.append(Spacer(1, 1.5 * mm))
            continue

        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i] and not (
            lines[i].startswith(("#", "|", "- ", "```", "![", "<!--"))
            or re.match(r"^\d+\. ", lines[i])
        ):
            paragraph.append(lines[i])
            i += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph)), STYLES["Bodyx"]))
    return story


def page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(17 * mm, PAGE_H - 14 * mm, PAGE_W - 17 * mm, PAGE_H - 14 * mm)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(TEAL)
        canvas.drawString(17 * mm, PAGE_H - 10.5 * mm, "DATAWHALE-EAI | RADEON PHYSICAL AI EVIDENCE SUITE")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_W - 17 * mm, 9 * mm, f"TRACK 3 TECHNICAL REPORT | {doc.page}")
    canvas.restoreState()


def cover_story() -> list:
    hero = scaled_image(ROOT / "docs/figures/robocasa-cover-close-fridge.jpg", PAGE_W - 34 * mm, 91 * mm)
    return [
        Spacer(1, 7 * mm),
        Paragraph("AMD AI DEVMASTER HACKATHON 2026 | TRACK 3", STYLES["ReportSubtitle"]),
        Paragraph("Radeon Physical AI<br/>Evidence Suite", STYLES["ReportTitle"]),
        Paragraph(
            "Robot learning, simulation, dexterous control, rendering, and safety evidence on AMD Radeon and ROCm.",
            STYLES["PullQuote"],
        ),
        Spacer(1, 2 * mm),
        hero,
        Spacer(1, 6 * mm),
        Table([
            [Paragraph("TEAM", STYLES["TableHead"]), Paragraph("PLATFORMS", STYLES["TableHead"]), Paragraph("DELIVERY", STYLES["TableHead"])],
            [Paragraph("Datawhale-EAI<br/>Kewei Chen<br/>Yayu Long", STYLES["TableCell"]), Paragraph("Radeon PRO W7900<br/>Ryzen AI MAX+ 395", STYLES["TableCell"]), Paragraph("Source + PDF + 4:59 video<br/>JSON + SHA + website", STYLES["TableCell"])],
        ], colWidths=[(PAGE_W - 34 * mm) / 3] * 3, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("BACKGROUND", (0, 1), (-1, 1), PALE),
            ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])),
        Spacer(1, 4 * mm),
        Paragraph("Submission version v1.0.2-amd-hackathon-final | August 2026", STYLES["Meta"]),
        PageBreak(),
    ]


def build(output: Path, source: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = Frame(17 * mm, 16 * mm, PAGE_W - 34 * mm, PAGE_H - 32 * mm, id="normal")
    template = PageTemplate(id="report", frames=[frame], onPage=page_header_footer)
    doc = BaseDocTemplate(
        str(output), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
        topMargin=17 * mm, bottomMargin=16 * mm,
        title="Radeon Physical AI Evidence Suite - Track 3 Technical Report",
        author="Datawhale-EAI / Kewei Chen / Yayu Long",
        subject="AMD AI DevMaster Hackathon 2026 Track 3",
    )
    doc.addPageTemplates([template])
    doc.build(cover_story() + parse_markdown(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "docs/TECHNICAL_REPORT.md")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/datawhale-eai-radeon-physical-ai-technical-report.pdf")
    args = parser.parse_args()
    build(args.output, args.source)
    print(args.output)


if __name__ == "__main__":
    main()
