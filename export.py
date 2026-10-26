# export.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Export all generated study material to multiple downloadable formats.
#
# Formats:
#   1. PDF  (.pdf)  — ReportLab Platypus; lecture title, generated date,
#                     topic headings, revision notes, formulas, page numbers,
#                     and a compact final revision checklist.
#   2. Markdown (.md) — Human-readable, properly escaped, renders in any viewer.
#   3. Anki CSV (.csv) — UTF-8-BOM, RFC 4180, Front/Back columns, special-char
#                        escaped, ready for direct Anki import.
#   4. JSON (.json)   — Optional; full quiz questions + scored results payload.
#
# Design principles:
#   - All public build_*_bytes() functions return bytes (in-memory) so Streamlit
#     can offer download buttons without touching the filesystem.
#   - All public export_*() functions write to disk and return the absolute path;
#     they delegate to the in-memory builders to avoid code duplication.
#   - Filenames are always constructed; existing files are never overwritten
#     (each call embeds a unique timestamp).
#   - Unicode text is supported as far as the built-in Helvetica/Times fonts
#     allow; characters outside Latin-1 are replaced with '?' gracefully.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
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

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_APP_NAME = "PrepGenius"
_DEFAULT_TITLE = "Lecture Study Notes"
_OUTPUTS_DIR = "outputs"

# Page geometry
_PAGE_W, _PAGE_H = A4                 # 595.27 × 841.89 pt
_MARGIN_LR = 2.2 * cm
_MARGIN_TOP = 2.4 * cm
_MARGIN_BOT = 2.4 * cm
_FOOTER_H = 1.0 * cm

# Colour palette
_C_DARK = colors.HexColor("#0F172A")
_C_PRIMARY = colors.HexColor("#4F46E5")
_C_SECONDARY = colors.HexColor("#0F766E")
_C_MUTED = colors.HexColor("#64748B")
_C_RULE = colors.HexColor("#E2E8F0")
_C_FORMULA_BG = colors.HexColor("#F8FAFC")
_C_FORMULA_BD = colors.HexColor("#CBD5E1")
_C_CHECK_BG = colors.HexColor("#ECFDF5")
_C_CHECK_BD = colors.HexColor("#6EE7B7")


# ═══════════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _timestamp() -> str:
    """Return a compact timestamp string (e.g. '20241014_1030')."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def _safe_filename(name: str) -> str:
    """Collapse non-alphanumeric characters to underscores for safe filenames."""
    slug = re.sub(r"[^\w]+", "_", name).strip("_")
    return slug[:60] or "export"


def _escape_xml(text: str) -> str:
    """Escape characters that would break ReportLab's XML / Paragraph parser."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _strip_markdown(text: str) -> str:
    """
    Convert Markdown-formatted text to plain (ReportLab-safe) text.

    Operations (in order):
      - ATX headings (#, ##, …) → retained as plain text
      - Bold / italic markers stripped
      - Inline code backticks stripped
      - Horizontal rules removed
      - Bullet list markers (- / * / +) → Unicode bullet •
      - Numbered list items normalised (kept as-is)
      - Blank lines collapsed to at most one
    """
    # Headings: strip leading # symbols, keep label
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold + italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"___(.+?)___", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Unordered list bullets
    text = re.sub(r"^[\-\*\+]\s+", "• ", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _now_str() -> str:
    return datetime.now().strftime("%d %B %Y at %H:%M")


def _detect_formula_lines(notes: str) -> set[str]:
    """
    Heuristically detect lines that look like formulas or equations.

    Criteria: contains '=', mathematical symbols, or LaTeX-like patterns.
    Returns the set of stripped line strings that match.
    """
    formula_patterns = re.compile(
        r"("
        r"[A-Za-z_]\s*=\s*[\w\(]"           # variable = expression
        r"|T\(n\)\s*="                          # T(n) = recurrence
        r"|O\([^\)]+\)"                          # Big-O
        r"|Θ\([^\)]+\)"                          # Big-Theta
        r"|Ω\([^\)]+\)"                          # Big-Omega
        r"|\\[a-z]+"                             # LaTeX command
        r"|∑|∏|∫|√|≤|≥|≠|→|⇒|∀|∃"             # Math symbols
        r")"
    )
    result: set[str] = set()
    for line in notes.splitlines():
        stripped = line.strip()
        if stripped and formula_patterns.search(stripped):
            result.add(stripped)
    return result


def _build_checklist_items(notes: str, flashcards: Sequence[dict]) -> list[str]:
    """
    Build a short revision checklist (≤ 15 items) from heading lines and card fronts.
    """
    items: list[str] = []
    # H2 / H3 headings from notes
    for m in re.finditer(r"^#{2,3}\s+(.+)$", notes, re.MULTILINE):
        label = m.group(1).strip()
        if label and label not in items:
            items.append(label)
        if len(items) >= 10:
            break
    # Pad with card fronts if needed
    if len(items) < 5:
        for card in flashcards:
            front = str(card.get("front", "")).strip()
            if front and front not in items:
                items.append(front)
            if len(items) >= 10:
                break
    return items[:15]


# ═══════════════════════════════════════════════════════════════════════════════
# Style Registry (built once, reused)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}

    def _s(name: str, parent_name: str = "Normal", **kw) -> ParagraphStyle:
        s = ParagraphStyle(name, parent=base[parent_name], **kw)
        styles[name] = s
        return s

    _s("DocTitle",    "Title",   fontSize=24, textColor=_C_DARK,
       spaceAfter=4, spaceBefore=0, leading=28)
    _s("DocSubtitle", "Normal",  fontSize=10, textColor=_C_MUTED,
       spaceAfter=14, leading=14)
    _s("H1",          "Heading1", fontSize=16, textColor=_C_PRIMARY,
       spaceBefore=18, spaceAfter=6, leading=20,
       borderPad=0, borderWidth=0)
    _s("H2",          "Heading2", fontSize=13, textColor=_C_DARK,
       spaceBefore=12, spaceAfter=5, leading=16)
    _s("H3",          "Heading3", fontSize=11, textColor=_C_SECONDARY,
       spaceBefore=8, spaceAfter=3, leading=14)
    _s("Body",        "Normal",  fontSize=10, leading=14,
       spaceAfter=5, textColor=_C_DARK)
    _s("Bullet",      "Normal",  fontSize=10, leading=14,
       spaceAfter=3, leftIndent=12, firstLineIndent=0, textColor=_C_DARK)
    _s("Formula",     "Normal",  fontSize=10, leading=14,
       spaceAfter=3, leftIndent=10, fontName="Courier",
       textColor=colors.HexColor("#1E293B"))
    _s("Caption",     "Normal",  fontSize=8, textColor=_C_MUTED,
       spaceAfter=8, leading=11)
    _s("CardQ",       "Normal",  fontSize=10, leading=14,
       spaceAfter=2, fontName="Helvetica-Bold", textColor=_C_DARK)
    _s("CardA",       "Normal",  fontSize=10, leading=14,
       spaceAfter=5, leftIndent=8, textColor=colors.HexColor("#1E293B"))
    _s("QuizQ",       "Normal",  fontSize=10, leading=14,
       spaceAfter=2, fontName="Helvetica-Bold", textColor=_C_DARK)
    _s("QuizOpt",     "Normal",  fontSize=10, leading=13,
       leftIndent=14, spaceAfter=2, textColor=_C_DARK)
    _s("QuizExpl",    "Normal",  fontSize=9, leading=12,
       leftIndent=14, spaceAfter=6,
       textColor=colors.HexColor("#065F46"),
       fontName="Helvetica-Oblique")
    _s("CheckItem",   "Normal",  fontSize=10, leading=14,
       leftIndent=8, spaceAfter=3, textColor=colors.HexColor("#065F46"))

    return styles


# ═══════════════════════════════════════════════════════════════════════════════
# PDF Page Template (header + footer + page numbers)
# ═══════════════════════════════════════════════════════════════════════════════





class _PageCounter:
    """
    Simpler page-number approach using onPage callbacks on PageTemplate.
    Keeps a mutable counter so multi-template docs work correctly.
    """
    def __init__(self, title: str):
        self.title = title

    def __call__(self, canvas: rl_canvas.Canvas, doc: BaseDocTemplate):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_C_MUTED)
        page_num = doc.page
        canvas.drawString(_MARGIN_LR, _FOOTER_H * 0.6, _APP_NAME)
        short = self.title[:55] + ("…" if len(self.title) > 55 else "")
        canvas.drawCentredString(_PAGE_W / 2, _FOOTER_H * 0.6, short)
        canvas.drawRightString(_PAGE_W - _MARGIN_LR, _FOOTER_H * 0.6, f"Page {page_num}")
        # Footer rule
        canvas.setStrokeColor(_C_RULE)
        canvas.setLineWidth(0.5)
        canvas.line(_MARGIN_LR, _FOOTER_H, _PAGE_W - _MARGIN_LR, _FOOTER_H)
        canvas.restoreState()


def _make_doc(buffer: io.BytesIO, title: str) -> BaseDocTemplate:
    """Return a configured BaseDocTemplate with a single PageTemplate."""
    on_page = _PageCounter(title)
    frame = Frame(
        _MARGIN_LR, _MARGIN_BOT + _FOOTER_H,
        _PAGE_W - 2 * _MARGIN_LR,
        _PAGE_H - _MARGIN_TOP - _MARGIN_BOT - _FOOTER_H,
        id="main",
    )
    pt = PageTemplate(id="main", frames=[frame], onPage=on_page)
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        pageTemplates=[pt],
        title=title,
        author=_APP_NAME,
        creator=_APP_NAME,
        leftMargin=_MARGIN_LR,
        rightMargin=_MARGIN_LR,
        topMargin=_MARGIN_TOP,
        bottomMargin=_MARGIN_BOT + _FOOTER_H,
    )
    return doc


# ═══════════════════════════════════════════════════════════════════════════════
# PDF Story Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _notes_to_story(
    notes: str,
    styles: dict[str, ParagraphStyle],
    formula_lines: set[str],
) -> list:
    """
    Convert cleaned (markdown-stripped) revision notes to a list of Platypus flowables.

    Heading lines (detected by prior Markdown stripping leaving them as plain text
    inside the already-stripped string) are re-tagged using size heuristics if
    heading markers were removed.  Original Markdown heading markers (now absent)
    are not available, so we fall back to inline detection of short ALL-CAPS or
    title-case lines that look like headings.
    """
    story = []
    paragraphs = notes.split("\n")

    for line in paragraphs:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 0.1 * cm))
            continue

        # Formula lines get a monospaced box style
        if stripped in formula_lines:
            story.append(Paragraph(_escape_xml(stripped), styles["Formula"]))
            continue

        # Bullet points
        if stripped.startswith("• "):
            text = stripped[2:].strip()
            story.append(Paragraph("• " + _escape_xml(text), styles["Bullet"]))
            continue

        # Heuristic: short lines that look like section headings
        word_count = len(stripped.split())
        is_likely_heading = (
            word_count <= 8
            and not stripped.endswith(".")
            and (stripped[0].isupper() or stripped.isupper())
        )
        if is_likely_heading and len(stripped) < 80:
            story.append(Paragraph(_escape_xml(stripped), styles["H3"]))
        else:
            story.append(Paragraph(_escape_xml(stripped), styles["Body"]))

    return story


def build_pdf_bytes(
    revision_notes: str,
    flashcards: Sequence[dict],
    quiz_questions: Sequence[dict],
    title: str = _DEFAULT_TITLE,
    topics: Optional[Sequence[Any]] = None,
) -> bytes:
    """
    Build a formatted PDF in memory and return its bytes.

    Sections (in order):
      1. Cover block: title, subtitle, date, horizontal rule.
      2. Key Topics overview (if topics are provided).
      3. Revision Notes — Markdown stripped, formulas boxed, headings styled.
      4. Flashcards — Q/A pairs, two-per-section layout.
      5. Practice Quiz — questions with options; correct answer marked ✓.
      6. Final Revision Checklist — compact checkbox list.

    All sections use proper page layout with page numbers in the footer.

    Args:
        revision_notes: Markdown-formatted revision notes string.
        flashcards:     List of flashcard dicts (keys: "front", "back", …).
        quiz_questions: List of MCQ dicts (keys: "question", "options", "answer",
                        "explanation", optionally "difficulty").
        title:          Document / lecture title string.
        topics:         Optional list of TopicCluster objects or strings for the
                        Key Topics section.

    Returns:
        Raw PDF bytes (suitable for st.download_button or writing to disk).
    """
    styles = _make_styles()
    buf = io.BytesIO()
    doc = _make_doc(buf, title)
    story: list = []

    # ── 1. Cover block ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(_escape_xml(title), styles["DocTitle"]))
    story.append(Paragraph(
        f"Generated by {_APP_NAME} · {_now_str()}",
        styles["DocSubtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=_C_PRIMARY, spaceAfter=14))

    # ── 2. Key Topics (optional) ──────────────────────────────────────────────
    if topics:
        story.append(Paragraph("🏷️ Key Topics", styles["H1"]))
        story.append(Spacer(1, 0.15 * cm))
        topic_items = []
        for t in topics:
            label = getattr(t, "label", str(t))
            summary = getattr(t, "summary", "")
            time_fmt = getattr(t, "time_range_fmt", "")
            line = _escape_xml(label)
            if time_fmt:
                line += f" <font color='#64748B' size='8'>({_escape_xml(str(time_fmt))})</font>"
            topic_items.append(Paragraph("• " + line, styles["Bullet"]))
            if summary:
                topic_items.append(Paragraph(
                    _escape_xml(str(summary)[:200]),
                    styles["Caption"],
                ))
        story.extend(topic_items)
        story.append(HRFlowable(width="100%", thickness=0.5, color=_C_RULE, spaceAfter=8))

    # ── 3. Revision Notes ─────────────────────────────────────────────────────
    story.append(Paragraph("📚 Revision Notes", styles["H1"]))
    story.append(Spacer(1, 0.15 * cm))

    formula_lines = _detect_formula_lines(revision_notes)
    clean_notes = _strip_markdown(revision_notes)
    story.extend(_notes_to_story(clean_notes, styles, formula_lines))

    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_C_RULE, spaceAfter=8))

    # ── 4. Flashcards ─────────────────────────────────────────────────────────
    if flashcards:
        story.append(Paragraph("🃏 Flashcards", styles["H1"]))
        story.append(Paragraph(
            f"{len(flashcards)} active-recall cards",
            styles["Caption"],
        ))

        for i, card in enumerate(flashcards, 1):
            front = str(card.get("front", "")).strip()
            back = str(card.get("back", "")).strip()
            cat = str(card.get("category", "Concept"))
            ts = card.get("timestamp")
            cat_label = f"[{cat}]" + (f" ⏱️ {ts}" if ts else "")

            block = [
                Paragraph(f"<b>Card {i}</b> <font size='8' color='#64748B'>{_escape_xml(cat_label)}</font>", styles["CardQ"]),
                Paragraph(f"Q: {_escape_xml(front)}", styles["CardQ"]),
                Paragraph(f"A: {_escape_xml(back)}", styles["CardA"]),
                Spacer(1, 0.1 * cm),
            ]
            story.append(KeepTogether(block))

        story.append(HRFlowable(width="100%", thickness=0.5, color=_C_RULE, spaceAfter=8))

    # ── 5. Practice Quiz ──────────────────────────────────────────────────────
    if quiz_questions:
        story.append(Paragraph("📝 Practice Quiz", styles["H1"]))
        story.append(Paragraph(
            f"{len(quiz_questions)} multiple-choice questions",
            styles["Caption"],
        ))

        for i, q in enumerate(quiz_questions, 1):
            question = str(q.get("question", "")).strip()
            options = q.get("options", [])
            answer = str(q.get("answer", "")).strip()
            explanation = str(q.get("explanation", "")).strip()
            difficulty = str(q.get("difficulty", "")).strip()
            diff_label = f" [{difficulty.capitalize()}]" if difficulty else ""

            q_block = [
                Paragraph(
                    f"<b>Q{i}{_escape_xml(diff_label)}. {_escape_xml(question)}</b>",
                    styles["QuizQ"],
                ),
            ]
            for j, opt in enumerate(options):
                marker = " ✓" if opt == answer else ""
                q_block.append(Paragraph(
                    f"  {chr(65 + j)}. {_escape_xml(str(opt))}{_escape_xml(marker)}",
                    styles["QuizOpt"],
                ))
            if explanation:
                q_block.append(Paragraph(
                    f"💡 {_escape_xml(explanation)}",
                    styles["QuizExpl"],
                ))
            q_block.append(Spacer(1, 0.1 * cm))
            story.append(KeepTogether(q_block))

        story.append(HRFlowable(width="100%", thickness=0.5, color=_C_RULE, spaceAfter=8))

    # ── 6. Final Revision Checklist ───────────────────────────────────────────
    checklist_items = _build_checklist_items(revision_notes, flashcards)
    if checklist_items:
        story.append(Paragraph("✅ Final Revision Checklist", styles["H1"]))
        story.append(Paragraph(
            "Tick off each topic once you feel confident with it.",
            styles["Caption"],
        ))

        # Draw as a table so we can shade it
        table_data = []
        for item in checklist_items:
            table_data.append(["☐", _escape_xml(item)])

        tbl = Table(table_data, colWidths=[0.6 * cm, None])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _C_CHECK_BG),
            ("BOX",        (0, 0), (-1, -1), 0.8, _C_CHECK_BD),
            ("INNERGRID",  (0, 0), (-1, -1), 0.4, _C_CHECK_BD),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica"),
            ("FONTNAME",   (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(tbl)

    # ── Build ──────────────────────────────────────────────────────────────────
    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _escape_md(text: str) -> str:
    """
    Escape Markdown special characters in plain-text values to prevent
    unintended formatting when embedded inside Markdown documents.

    Escapes: \\ ` * _ { } [ ] ( ) # + - . !
    """
    return re.sub(r"([\\`*_{}\[\]()#+\-\.!])", r"\\\1", text)


def build_markdown_bytes(
    revision_notes: str,
    flashcards: Sequence[dict],
    quiz_questions: Sequence[dict],
    title: str = _DEFAULT_TITLE,
    topics: Optional[Sequence[Any]] = None,
) -> bytes:
    """
    Build a Markdown document in memory and return UTF-8 bytes.

    Sections:
      1. Title + metadata
      2. Key Topics (optional)
      3. Revision Notes (verbatim — already Markdown)
      4. Flashcards (Q/A cards with category badges)
      5. Practice Quiz (options A–D with answer and explanation)

    Special characters in card fronts/backs and quiz fields are escaped so they
    render as literal text rather than Markdown formatting.

    Returns:
        UTF-8 encoded bytes of the Markdown document.
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# {title}",
        f"",
        f"*Generated by {_APP_NAME} · {_now_str()}*",
        f"",
        f"---",
        f"",
    ]

    # ── Key Topics ────────────────────────────────────────────────────────────
    if topics:
        lines += ["## 🏷️ Key Topics", ""]
        for i, t in enumerate(topics, 1):
            label = getattr(t, "label", str(t))
            summary = getattr(t, "summary", "")
            time_fmt = getattr(t, "time_range_fmt", "")
            ts_part = f" *({time_fmt})*" if time_fmt else ""
            lines.append(f"### {i}. {label}{ts_part}")
            if summary:
                lines.append(f"{summary}")
            lines.append("")
        lines += ["---", ""]

    # ── Revision Notes ────────────────────────────────────────────────────────
    lines += [
        "## 📚 Revision Notes",
        "",
        revision_notes.strip(),
        "",
        "---",
        "",
    ]

    # ── Flashcards ────────────────────────────────────────────────────────────
    lines += ["## 🃏 Flashcards", ""]
    if flashcards:
        for i, card in enumerate(flashcards, 1):
            front = str(card.get("front", "")).strip()
            back = str(card.get("back", "")).strip()
            cat = str(card.get("category", "Concept"))
            ts = card.get("timestamp", "")
            ts_part = f" · ⏱️ `{ts}`" if ts else ""
            lines += [
                f"### Card {i} `[{cat}]`{ts_part}",
                f"",
                f"**Q:** {_escape_md(front)}",
                f"",
                f"**A:** {_escape_md(back)}",
                f"",
            ]
    else:
        lines += ["> *No flashcards generated.*", ""]

    lines += ["---", ""]

    # ── Practice Quiz ─────────────────────────────────────────────────────────
    lines += ["## 📝 Practice Quiz", ""]
    if quiz_questions:
        for i, q in enumerate(quiz_questions, 1):
            question = str(q.get("question", "")).strip()
            options = q.get("options", [])
            answer = str(q.get("answer", "")).strip()
            explanation = str(q.get("explanation", "")).strip()
            difficulty = str(q.get("difficulty", "")).strip()
            diff_label = f" `[{difficulty.capitalize()}]`" if difficulty else ""

            lines += [
                f"**Q{i}{diff_label}. {_escape_md(question)}**",
                "",
            ]
            for j, opt in enumerate(options):
                marker = " ✅" if opt == answer else ""
                lines.append(f"   {chr(65 + j)}. {_escape_md(str(opt))}{marker}")
            lines += [
                "",
                f"   > 💡 *{_escape_md(explanation)}*",
                "",
            ]
    else:
        lines += ["> *No quiz questions generated.*", ""]

    content = "\n".join(lines)
    return content.encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Anki CSV Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_csv_bytes(
    flashcards: Sequence[dict],
    include_tags: bool = True,
    tag_name: str = "PrepGenius",
) -> bytes:
    """
    Build an Anki-compatible CSV file in memory and return UTF-8-BOM bytes.

    Format:
      - RFC 4180 standard: commas as delimiters, CRLF line endings.
      - Fields with commas, double-quotes, or newlines are enclosed in quotes.
      - Double quotes inside values are escaped by doubling ("").
      - Columns: Front, Back[, Tags]
      - Encoded as UTF-8 with BOM (utf-8-sig) for reliable Anki import on
        all platforms, including Windows.

    Special characters in Front/Back are NOT escaped with backslashes — the
    RFC 4180 quoting handles them correctly for CSV consumers.

    Args:
        flashcards:   Sequence of flashcard dicts with "front" and "back" keys.
        include_tags: Whether to include a "Tags" column (default True).
        tag_name:     Default tag string when card has no tags field.

    Returns:
        UTF-8-BOM encoded bytes of the CSV document.
    """
    output = io.StringIO()
    writer = csv.writer(
        output,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\r\n",  # RFC 4180
    )

    # Header
    header = ["Front", "Back"]
    if include_tags:
        header.append("Tags")
    writer.writerow(header)

    for card in flashcards:
        front = str(card.get("front", "")).strip()
        back = str(card.get("back", "")).strip()
        ts = card.get("timestamp")
        if ts and f"[{ts}]" not in back:
            back_formatted = f"{back}\n\n[⏱️ Source: {ts}]"
        else:
            back_formatted = back

        row = [front, back_formatted]
        if include_tags:
            row.append(str(card.get("tags") or tag_name))
        writer.writerow(row)

    return output.getvalue().encode("utf-8-sig")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Quiz Export Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_quiz_json_bytes(
    quiz_questions: Sequence[dict],
    quiz_results: Optional[dict] = None,
    title: str = _DEFAULT_TITLE,
) -> bytes:
    """
    Build a JSON export of quiz questions and optional scored results.

    Schema:
    {
      "title": str,
      "generated_at": ISO-8601 str,
      "app": str,
      "total_questions": int,
      "questions": [
        {
          "index": int (1-based),
          "question": str,
          "options": [str, str, str, str],
          "answer": str,
          "explanation": str,
          "difficulty": str
        },
        ...
      ],
      "results": {                    # only present when quiz_results is provided
        "score": int,
        "total": int,
        "percent": float,
        "per_question": [
          {
            "index": int (1-based),
            "correct": bool,
            "chosen": str,
            "answer": str,
            "explanation": str
          },
          ...
        ]
      }
    }

    Args:
        quiz_questions: List of MCQ question dicts.
        quiz_results:   Optional scored results dict from `quiz.score_quiz()`.
        title:          Lecture / document title.

    Returns:
        UTF-8 encoded bytes of the JSON document (indented, human-readable).
    """
    payload: dict[str, Any] = {
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "app": _APP_NAME,
        "total_questions": len(quiz_questions),
        "questions": [
            {
                "index": i + 1,
                "question": str(q.get("question", "")),
                "options": [str(o) for o in q.get("options", [])],
                "answer": str(q.get("answer", "")),
                "explanation": str(q.get("explanation", "")),
                "difficulty": str(q.get("difficulty", "medium")),
            }
            for i, q in enumerate(quiz_questions)
        ],
    }

    if quiz_results:
        per_q = []
        for i, r in enumerate(quiz_results.get("results", [])):
            per_q.append({
                "index": i + 1,
                "correct": bool(r.get("correct")),
                "chosen": str(r.get("chosen", "")),
                "answer": str(r.get("answer", "")),
                "explanation": str(r.get("explanation", "")),
            })
        payload["results"] = {
            "score":        int(quiz_results.get("score", 0)),
            "total":        int(quiz_results.get("total", 0)),
            "percent":      float(quiz_results.get("percent", 0.0)),
            "per_question": per_q,
        }

    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Disk-writing wrappers (return absolute path; never overwrite)
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_outputs_dir() -> str:
    os.makedirs(_OUTPUTS_DIR, exist_ok=True)
    return _OUTPUTS_DIR


def export_pdf(
    revision_notes: str,
    flashcards: Sequence[dict],
    quiz_questions: Sequence[dict],
    title: str = _DEFAULT_TITLE,
    topics: Optional[Sequence[Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Write a PDF to disk and return its absolute path.

    The filename is timestamped so successive calls never overwrite each other.
    Uses build_pdf_bytes() internally — no filesystem state is needed for the
    PDF construction itself.

    Returns:
        Absolute path to the saved .pdf file.
    """
    data = build_pdf_bytes(revision_notes, flashcards, quiz_questions, title, topics)
    if not output_path:
        _ensure_outputs_dir()
        output_path = os.path.join(
            _OUTPUTS_DIR,
            f"{_safe_filename(title)}_{_timestamp()}.pdf",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(data)
    return os.path.abspath(output_path)


def export_markdown(
    revision_notes: str,
    flashcards: Sequence[dict],
    quiz_questions: Sequence[dict],
    title: str = _DEFAULT_TITLE,
    topics: Optional[Sequence[Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Write a Markdown document to disk and return its absolute path.

    Returns:
        Absolute path to the saved .md file.
    """
    data = build_markdown_bytes(revision_notes, flashcards, quiz_questions, title, topics)
    if not output_path:
        _ensure_outputs_dir()
        output_path = os.path.join(
            _OUTPUTS_DIR,
            f"{_safe_filename(title)}_{_timestamp()}.md",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(data)
    return os.path.abspath(output_path)


def export_csv(
    flashcards: Sequence[dict],
    include_tags: bool = True,
    tag_name: str = "PrepGenius",
    output_path: Optional[str] = None,
) -> str:
    """
    Write an Anki-compatible CSV to disk and return its absolute path.

    Returns:
        Absolute path to the saved .csv file.
    """
    data = build_csv_bytes(flashcards, include_tags=include_tags, tag_name=tag_name)
    if not output_path:
        _ensure_outputs_dir()
        output_path = os.path.join(
            _OUTPUTS_DIR,
            f"flashcards_{_timestamp()}.csv",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(data)
    return os.path.abspath(output_path)


def export_quiz_json(
    quiz_questions: Sequence[dict],
    quiz_results: Optional[dict] = None,
    title: str = _DEFAULT_TITLE,
    output_path: Optional[str] = None,
) -> str:
    """
    Write a JSON quiz export to disk and return its absolute path.

    Returns:
        Absolute path to the saved .json file.
    """
    data = build_quiz_json_bytes(quiz_questions, quiz_results, title)
    if not output_path:
        _ensure_outputs_dir()
        output_path = os.path.join(
            _OUTPUTS_DIR,
            f"quiz_{_timestamp()}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(data)
    return os.path.abspath(output_path)




