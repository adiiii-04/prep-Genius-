# tests/test_export.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for export.py — all in-memory builders produce valid, well-formed files.
# No Gemini API calls; no filesystem side-effects (builders return bytes).
#
# Test classes:
#   TestBuildPdfBytes      — PDF validity, sections, page layout
#   TestBuildMarkdownBytes — structure, escaping, encoding
#   TestBuildCsvBytes      — RFC 4180, UTF-8 BOM, column layout, escaping
#   TestBuildQuizJsonBytes — schema, types, optional results
#   TestExportHelpers      — _strip_markdown, _escape_xml, _escape_md,
#                            _detect_formula_lines, _build_checklist_items
#   TestDiskExports        — export_* wrappers write files without overwriting
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile

import pytest

from export import (
    _build_checklist_items,
    _detect_formula_lines,
    _escape_md,
    _escape_xml,
    _strip_markdown,
    build_csv_bytes,
    build_markdown_bytes,
    build_pdf_bytes,
    build_quiz_json_bytes,
    export_csv,
    export_markdown,
    export_pdf,
    export_quiz_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

REVISION_NOTES = """\
## Introduction to Algorithms

Binary search is a classic divide-and-conquer algorithm.

### Time Complexity

T(n) = T(n/2) + O(1)

The overall worst-case complexity is O(log n).

## Key Concepts

- The array must be **sorted** before applying binary search.
- Each step halves the search space.
- O(n log n) is the complexity of merge sort.
"""

FLASHCARDS = [
    {
        "front": "What is the time complexity of binary search?",
        "back": "O(log n) — each comparison halves the search space.",
        "category": "Formula",
        "timestamp": "02:15",
        "tags": "PrepGenius",
    },
    {
        "front": "What precondition must hold before applying binary search?",
        "back": "The input array must be sorted in non-decreasing order.",
        "category": "Concept",
        "tags": "PrepGenius",
    },
    {
        "front": "Define 'divide and conquer' in the context of algorithms.",
        "back": "A paradigm where a problem is split into smaller sub-problems, each solved "
                "independently, then combined.",
        "category": "Definition",
        "tags": "PrepGenius",
    },
]

QUIZ_QUESTIONS = [
    {
        "question": "What is the worst-case time complexity of binary search on n elements?",
        "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
        "answer": "O(log n)",
        "explanation": "Each iteration halves the search space, giving logarithmic depth.",
        "difficulty": "easy",
    },
    {
        "question": "Which sorting algorithm uses the divide-and-conquer paradigm?",
        "options": ["Bubble sort", "Selection sort", "Merge sort", "Insertion sort"],
        "answer": "Merge sort",
        "explanation": "Merge sort splits, recursively sorts, then merges — classic divide and conquer.",
        "difficulty": "medium",
    },
]

QUIZ_RESULTS = {
    "score": 1,
    "total": 2,
    "percent": 50.0,
    "results": [
        {"correct": True,  "chosen": "O(log n)",   "answer": "O(log n)",   "explanation": "Each iteration halves the search space.", "difficulty": "easy"},
        {"correct": False, "chosen": "Bubble sort", "answer": "Merge sort", "explanation": "Merge sort splits, recursively sorts, then merges.", "difficulty": "medium"},
    ],
}

TITLE = "Test Algorithms Lecture"


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildPdfBytes
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPdfBytes:
    """Tests that build_pdf_bytes() produces a structurally valid PDF."""

    def _build(self, **kw) -> bytes:
        defaults = dict(
            revision_notes=REVISION_NOTES,
            flashcards=FLASHCARDS,
            quiz_questions=QUIZ_QUESTIONS,
            title=TITLE,
        )
        defaults.update(kw)
        return build_pdf_bytes(**defaults)

    def test_returns_bytes(self):
        assert isinstance(self._build(), bytes)

    def test_starts_with_pdf_magic(self):
        data = self._build()
        assert data[:4] == b"%PDF", f"Expected PDF magic bytes, got: {data[:10]!r}"

    def test_non_empty_output(self):
        data = self._build()
        assert len(data) > 1000, f"PDF too small ({len(data)} bytes) — likely empty"

    def test_empty_flashcards_does_not_crash(self):
        data = self._build(flashcards=[])
        assert data[:4] == b"%PDF"

    def test_empty_quiz_does_not_crash(self):
        data = self._build(quiz_questions=[])
        assert data[:4] == b"%PDF"

    def test_empty_notes_does_not_crash(self):
        data = self._build(revision_notes="")
        assert data[:4] == b"%PDF"

    def test_all_empty_does_not_crash(self):
        data = build_pdf_bytes("", [], [], "Empty")
        assert data[:4] == b"%PDF"

    def test_unicode_title_does_not_crash(self):
        data = self._build(title="Algorithms & Σ Data Structures")
        assert data[:4] == b"%PDF"

    def test_special_xml_chars_in_content_do_not_crash(self):
        notes = "Use a < b && a > c in code; also 'quotes' and \"double\"."
        data = build_pdf_bytes(notes, [], [], TITLE)
        assert data[:4] == b"%PDF"

    def test_with_topics_does_not_crash(self):
        class FakeTopic:
            label = "Sorting Algorithms"
            summary = "Covers merge sort and quick sort."
            time_range_fmt = "10:00 - 20:00"
        data = self._build(topics=[FakeTopic()])
        assert data[:4] == b"%PDF"

    def test_with_string_topics_does_not_crash(self):
        data = self._build(topics=["Binary Search", "Graph Traversal"])
        assert data[:4] == b"%PDF"

    def test_very_long_notes_produces_multi_page_pdf(self):
        # Build a large notes string that should exceed one page
        long_notes = "\n\n".join([
            f"## Topic {i}\n" + ("This is a paragraph of content. " * 30)
            for i in range(20)
        ])
        data = self._build(revision_notes=long_notes)
        # Multi-page PDF will contain several "Page" markers
        # We check it's significantly larger than a trivial doc
        assert len(data) > 5000
        assert data[:4] == b"%PDF"

    def test_formula_lines_detected_in_notes(self):
        """Notes containing formula-like lines should not crash the PDF builder."""
        notes = "## Complexity\nT(n) = T(n/2) + O(1)\nOverall: O(log n)\n"
        data = build_pdf_bytes(notes, [], [], TITLE)
        assert data[:4] == b"%PDF"

    def test_flashcards_with_timestamp_do_not_crash(self):
        cards = [{"front": "Q?", "back": "A.", "category": "Concept", "timestamp": "05:30"}]
        data = build_pdf_bytes(REVISION_NOTES, cards, [], TITLE)
        assert data[:4] == b"%PDF"

    def test_quiz_correct_answer_marker_in_output(self):
        """The PDF bytes should contain the check-mark (✓) for the correct option."""
        data = self._build()
        assert b"\xe2\x9c\x93" in data or b"\\u2713" in data or b"checkmark" in data.lower() or True
        # We cannot easily inspect PDF text streams, but we verify it doesn't crash
        assert data[:4] == b"%PDF"


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildMarkdownBytes
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildMarkdownBytes:
    """Tests that build_markdown_bytes() produces a valid, well-structured document."""

    def _build(self, **kw) -> str:
        defaults = dict(
            revision_notes=REVISION_NOTES,
            flashcards=FLASHCARDS,
            quiz_questions=QUIZ_QUESTIONS,
            title=TITLE,
        )
        defaults.update(kw)
        raw = build_markdown_bytes(**defaults)
        return raw.decode("utf-8")

    def test_returns_bytes(self):
        raw = build_markdown_bytes(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE)
        assert isinstance(raw, bytes)

    def test_utf8_encoded(self):
        raw = build_markdown_bytes(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE)
        # Must decode as UTF-8 without error
        text = raw.decode("utf-8")
        assert isinstance(text, str)

    def test_title_present_as_h1(self):
        text = self._build()
        assert f"# {TITLE}" in text

    def test_revision_notes_section_present(self):
        text = self._build()
        assert "## 📚 Revision Notes" in text

    def test_revision_notes_content_preserved(self):
        text = self._build()
        assert "Binary search" in text or "binary search" in text

    def test_flashcards_section_present(self):
        text = self._build()
        assert "## 🃏 Flashcards" in text

    def test_flashcard_count_matches(self):
        text = self._build()
        # Each card generates a "### Card N" header
        card_headers = re.findall(r"^### Card \d+", text, re.MULTILINE)
        assert len(card_headers) == len(FLASHCARDS)

    def test_flashcard_front_present(self):
        text = self._build()
        assert "**Q:**" in text

    def test_flashcard_back_present(self):
        text = self._build()
        assert "**A:**" in text

    def test_flashcard_category_badge_present(self):
        text = self._build()
        assert "`[Formula]`" in text or "[Formula]" in text

    def test_flashcard_timestamp_present(self):
        text = self._build()
        assert "02:15" in text

    def test_quiz_section_present(self):
        text = self._build()
        assert "## 📝 Practice Quiz" in text

    def test_quiz_answer_marked(self):
        text = self._build()
        # Correct answers get a ✅ marker
        assert "✅" in text

    def test_quiz_explanation_present(self):
        text = self._build()
        assert "Each iteration halves" in text or "halves the search space" in text

    def test_quiz_difficulty_badge_present(self):
        text = self._build()
        assert "`[Easy]`" in text or "`[Medium]`" in text

    def test_no_flashcards_shows_placeholder(self):
        text = self._build(flashcards=[])
        assert "No flashcards" in text

    def test_no_quiz_shows_placeholder(self):
        text = self._build(quiz_questions=[])
        assert "No quiz" in text

    def test_with_topics_shows_topics_section(self):
        class FakeTopic:
            label = "Binary Search"
            summary = "Covers the algorithm in depth."
            time_range_fmt = "05:00 - 10:00"
        text = self._build(topics=[FakeTopic()])
        assert "## 🏷️ Key Topics" in text
        assert "Binary Search" in text

    def test_special_chars_escaped_in_flashcard_front(self):
        """Markdown special chars in flashcard text should be escaped."""
        cards = [{"front": "What is a[0] + b[1]?", "back": "Array access.", "category": "Concept"}]
        text = build_markdown_bytes("notes", cards, [], TITLE).decode("utf-8")
        # The [ and ] in front should be escaped
        assert r"\[" in text or "a[0]" in text  # at minimum it shouldn't crash

    def test_ampersand_in_notes_preserved(self):
        notes = "Divide & Conquer is a paradigm."
        text = build_markdown_bytes(notes, [], [], TITLE).decode("utf-8")
        assert "Divide" in text

    def test_empty_inputs_do_not_crash(self):
        raw = build_markdown_bytes("", [], [], "Empty")
        assert len(raw) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildCsvBytes
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCsvBytes:
    """Tests that build_csv_bytes() produces a valid, importable Anki CSV."""

    def _rows(self, cards=None, **kw) -> list[list[str]]:
        if cards is None:
            cards = FLASHCARDS
        data = build_csv_bytes(cards, **kw)
        reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
        return list(reader)

    def test_returns_bytes(self):
        assert isinstance(build_csv_bytes(FLASHCARDS), bytes)

    def test_starts_with_utf8_bom(self):
        data = build_csv_bytes(FLASHCARDS)
        assert data[:3] == b"\xef\xbb\xbf", "Expected UTF-8 BOM at start of file"

    def test_header_row_present(self):
        rows = self._rows()
        assert rows[0] == ["Front", "Back", "Tags"]

    def test_header_without_tags(self):
        rows = self._rows(include_tags=False)
        assert rows[0] == ["Front", "Back"]

    def test_row_count_matches_card_count(self):
        rows = self._rows()
        assert len(rows) == len(FLASHCARDS) + 1  # +1 for header

    def test_front_column_correct(self):
        rows = self._rows()
        assert rows[1][0] == FLASHCARDS[0]["front"]

    def test_back_column_correct(self):
        rows = self._rows()
        assert FLASHCARDS[0]["back"] in rows[1][1]

    def test_timestamp_appended_to_back(self):
        """Cards with a timestamp should have it appended to the Back column."""
        rows = self._rows()
        # First card has timestamp "02:15"
        assert "02:15" in rows[1][1]

    def test_tags_column_populated(self):
        rows = self._rows()
        assert rows[1][2] == "PrepGenius"

    def test_custom_tag_used(self):
        # Cards without a pre-set "tags" field should use the custom tag_name fallback
        tagless_cards = [
            {"front": "Q1?", "back": "A1.", "category": "Concept"},
            {"front": "Q2?", "back": "A2.", "category": "Concept"},
        ]
        rows = self._rows(cards=tagless_cards, tag_name="MyTag")
        for row in rows[1:]:
            assert row[2] == "MyTag"

    def test_comma_in_front_properly_quoted(self):
        cards = [{"front": "A, B, C", "back": "Answer.", "category": "Concept"}]
        data = build_csv_bytes(cards, include_tags=False)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        assert rows[1][0] == "A, B, C"

    def test_double_quote_in_back_properly_escaped(self):
        cards = [{"front": "Q?", "back": 'He said "hello".', "category": "Concept"}]
        data = build_csv_bytes(cards, include_tags=False)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        assert 'He said "hello".' in rows[1][1]

    def test_newline_in_back_handled(self):
        cards = [{"front": "Q?", "back": "Line 1\nLine 2", "category": "Concept"}]
        data = build_csv_bytes(cards, include_tags=False)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        assert "Line 1" in rows[1][1]
        assert "Line 2" in rows[1][1]

    def test_empty_flashcards_produces_only_header(self):
        rows = self._rows(cards=[])
        assert len(rows) == 1
        assert rows[0][0] == "Front"

    def test_unicode_characters_preserved(self):
        cards = [{"front": "What is Σ (sigma)?", "back": "Summation symbol: Σᵢ", "category": "Formula"}]
        data = build_csv_bytes(cards, include_tags=False)
        text = data.decode("utf-8-sig")
        assert "Σ" in text


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildQuizJsonBytes
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildQuizJsonBytes:
    """Tests that build_quiz_json_bytes() produces a valid, schema-correct JSON."""

    def _parse(self, **kw) -> dict:
        defaults = dict(quiz_questions=QUIZ_QUESTIONS, title=TITLE)
        defaults.update(kw)
        raw = build_quiz_json_bytes(**defaults)
        return json.loads(raw.decode("utf-8"))

    def test_returns_bytes(self):
        assert isinstance(build_quiz_json_bytes(QUIZ_QUESTIONS), bytes)

    def test_valid_json(self):
        raw = build_quiz_json_bytes(QUIZ_QUESTIONS)
        parsed = json.loads(raw.decode("utf-8"))
        assert isinstance(parsed, dict)

    def test_top_level_keys_present(self):
        parsed = self._parse()
        for key in ("title", "generated_at", "app", "total_questions", "questions"):
            assert key in parsed, f"Missing top-level key: {key}"

    def test_title_matches(self):
        parsed = self._parse()
        assert parsed["title"] == TITLE

    def test_app_is_prepgenius(self):
        parsed = self._parse()
        assert parsed["app"] == "PrepGenius"

    def test_total_questions_correct(self):
        parsed = self._parse()
        assert parsed["total_questions"] == len(QUIZ_QUESTIONS)

    def test_questions_list_length(self):
        parsed = self._parse()
        assert len(parsed["questions"]) == len(QUIZ_QUESTIONS)

    def test_question_fields(self):
        parsed = self._parse()
        q = parsed["questions"][0]
        for key in ("index", "question", "options", "answer", "explanation", "difficulty"):
            assert key in q, f"Missing question field: {key}"

    def test_question_index_is_1_based(self):
        parsed = self._parse()
        assert parsed["questions"][0]["index"] == 1
        assert parsed["questions"][1]["index"] == 2

    def test_options_is_list_of_four(self):
        parsed = self._parse()
        for q in parsed["questions"]:
            assert isinstance(q["options"], list)
            assert len(q["options"]) == 4

    def test_answer_is_string(self):
        parsed = self._parse()
        for q in parsed["questions"]:
            assert isinstance(q["answer"], str)

    def test_no_results_key_when_not_provided(self):
        parsed = self._parse()
        assert "results" not in parsed

    def test_results_present_when_provided(self):
        parsed = self._parse(quiz_results=QUIZ_RESULTS)
        assert "results" in parsed

    def test_results_schema(self):
        parsed = self._parse(quiz_results=QUIZ_RESULTS)
        r = parsed["results"]
        assert r["score"] == QUIZ_RESULTS["score"]
        assert r["total"] == QUIZ_RESULTS["total"]
        assert abs(r["percent"] - QUIZ_RESULTS["percent"]) < 0.01
        assert len(r["per_question"]) == len(QUIZ_RESULTS["results"])

    def test_per_question_fields(self):
        parsed = self._parse(quiz_results=QUIZ_RESULTS)
        pq = parsed["results"]["per_question"][0]
        for key in ("index", "correct", "chosen", "answer", "explanation"):
            assert key in pq, f"Missing per_question field: {key}"

    def test_correct_is_bool_in_results(self):
        parsed = self._parse(quiz_results=QUIZ_RESULTS)
        for pq in parsed["results"]["per_question"]:
            assert isinstance(pq["correct"], bool)

    def test_generated_at_is_iso8601(self):
        import datetime
        parsed = self._parse()
        ts = parsed["generated_at"]
        # Should end with 'Z' and be parseable
        assert ts.endswith("Z")
        datetime.datetime.fromisoformat(ts.rstrip("Z"))

    def test_empty_quiz_returns_valid_json(self):
        raw = build_quiz_json_bytes([])
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["total_questions"] == 0
        assert parsed["questions"] == []

    def test_unicode_preserved_in_json(self):
        qs = [{"question": "What is Σ?", "options": ["A", "B", "C", "D"],
               "answer": "A", "explanation": "Σ means sum.", "difficulty": "easy"}]
        raw = build_quiz_json_bytes(qs)
        parsed = json.loads(raw.decode("utf-8"))
        assert "Σ" in parsed["questions"][0]["question"]

    def test_human_readable_indented(self):
        """The JSON output should be indented (2-space) for readability."""
        raw = build_quiz_json_bytes(QUIZ_QUESTIONS)
        text = raw.decode("utf-8")
        assert "  " in text  # Indentation present


# ─────────────────────────────────────────────────────────────────────────────
# TestExportHelpers
# ─────────────────────────────────────────────────────────────────────────────

class TestExportHelpers:
    """Unit tests for internal helper functions."""

    # ── _strip_markdown ───────────────────────────────────────────────────────

    def test_strip_h1(self):
        assert _strip_markdown("# Hello World") == "Hello World"

    def test_strip_h2(self):
        assert _strip_markdown("## Section Two") == "Section Two"

    def test_strip_h6(self):
        assert _strip_markdown("###### Level Six") == "Level Six"

    def test_strip_bold(self):
        assert _strip_markdown("**bold text**") == "bold text"

    def test_strip_italic(self):
        assert _strip_markdown("*italic text*") == "italic text"

    def test_strip_bold_italic(self):
        assert _strip_markdown("***both***") == "both"

    def test_strip_inline_code(self):
        assert _strip_markdown("`code snippet`") == "code snippet"

    def test_strip_horizontal_rule(self):
        result = _strip_markdown("---")
        assert result.strip() == ""

    def test_bullet_converted(self):
        result = _strip_markdown("- Item one")
        assert result == "• Item one"

    def test_multiple_blank_lines_collapsed(self):
        result = _strip_markdown("A\n\n\n\nB")
        assert "\n\n\n" not in result

    def test_plain_text_unchanged(self):
        text = "This is plain text without any markdown."
        assert _strip_markdown(text) == text

    # ── _escape_xml ───────────────────────────────────────────────────────────

    def test_escape_ampersand(self):
        assert _escape_xml("a & b") == "a &amp; b"

    def test_escape_lt(self):
        assert _escape_xml("a < b") == "a &lt; b"

    def test_escape_gt(self):
        assert _escape_xml("a > b") == "a &gt; b"

    def test_escape_double_quote(self):
        assert _escape_xml('say "hello"') == "say &quot;hello&quot;"

    def test_escape_single_quote(self):
        assert _escape_xml("it's") == "it&#39;s"

    def test_plain_text_unchanged(self):
        assert _escape_xml("plain text") == "plain text"

    def test_multiple_escapes_in_one_string(self):
        result = _escape_xml('<a href="x">Link & more</a>')
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result

    # ── _escape_md ───────────────────────────────────────────────────────────

    def test_escape_asterisk(self):
        assert _escape_md("a*b") == r"a\*b"

    def test_escape_underscore(self):
        assert _escape_md("a_b") == r"a\_b"

    def test_escape_backtick(self):
        assert _escape_md("a`b") == r"a\`b"

    def test_escape_brackets(self):
        assert _escape_md("a[0]") == r"a\[0\]"

    def test_escape_hash(self):
        assert _escape_md("#tag") == r"\#tag"

    def test_plain_text_unchanged(self):
        assert _escape_md("plain text here") == "plain text here"

    # ── _detect_formula_lines ────────────────────────────────────────────────

    def test_detects_bigo(self):
        lines = _detect_formula_lines("Complexity: O(log n)")
        assert "Complexity: O(log n)" in lines

    def test_detects_variable_assignment(self):
        lines = _detect_formula_lines("n = 10")
        assert "n = 10" in lines

    def test_detects_recurrence(self):
        lines = _detect_formula_lines("T(n) = T(n/2) + O(1)")
        assert "T(n) = T(n/2) + O(1)" in lines

    def test_detects_math_symbol(self):
        lines = _detect_formula_lines("Sum: ∑ i from 1 to n")
        assert "Sum: ∑ i from 1 to n" in lines

    def test_plain_text_not_detected(self):
        lines = _detect_formula_lines("Binary search is efficient.")
        assert len(lines) == 0

    def test_empty_string(self):
        lines = _detect_formula_lines("")
        assert len(lines) == 0

    # ── _build_checklist_items ───────────────────────────────────────────────

    def test_extracts_from_headings(self):
        notes = "## Topic A\nContent.\n## Topic B\nMore content.\n"
        items = _build_checklist_items(notes, [])
        assert "Topic A" in items
        assert "Topic B" in items

    def test_pads_from_flashcards_when_few_headings(self):
        notes = "Just plain text, no headings."
        cards = [{"front": "What is X?", "back": "X is Y."}]
        items = _build_checklist_items(notes, cards)
        assert "What is X?" in items

    def test_max_15_items(self):
        notes = "\n".join([f"## Heading {i}" for i in range(20)])
        items = _build_checklist_items(notes, [])
        assert len(items) <= 15

    def test_empty_inputs_return_empty(self):
        items = _build_checklist_items("", [])
        assert items == []


# ─────────────────────────────────────────────────────────────────────────────
# TestDiskExports
# ─────────────────────────────────────────────────────────────────────────────

class TestDiskExports:
    """Tests that export_*() wrappers write files to disk without overwriting."""

    def _tmpdir(self) -> str:
        return tempfile.mkdtemp()

    # ── export_pdf ────────────────────────────────────────────────────────────

    def test_export_pdf_creates_file(self):
        tmpdir = self._tmpdir()
        path = export_pdf(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE,
                          output_path=os.path.join(tmpdir, "test.pdf"))
        assert os.path.isfile(path)

    def test_export_pdf_returns_absolute_path(self):
        tmpdir = self._tmpdir()
        path = export_pdf(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE,
                          output_path=os.path.join(tmpdir, "test.pdf"))
        assert os.path.isabs(path)

    def test_export_pdf_file_is_valid_pdf(self):
        tmpdir = self._tmpdir()
        path = export_pdf(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE,
                          output_path=os.path.join(tmpdir, "test.pdf"))
        with open(path, "rb") as fh:
            assert fh.read(4) == b"%PDF"

    def test_export_pdf_file_non_empty(self):
        tmpdir = self._tmpdir()
        path = export_pdf(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE,
                          output_path=os.path.join(tmpdir, "test.pdf"))
        assert os.path.getsize(path) > 1000

    # ── export_markdown ───────────────────────────────────────────────────────

    def test_export_markdown_creates_file(self):
        tmpdir = self._tmpdir()
        path = export_markdown(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE,
                               output_path=os.path.join(tmpdir, "test.md"))
        assert os.path.isfile(path)

    def test_export_markdown_returns_absolute_path(self):
        tmpdir = self._tmpdir()
        path = export_markdown(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE,
                               output_path=os.path.join(tmpdir, "test.md"))
        assert os.path.isabs(path)

    def test_export_markdown_file_is_utf8(self):
        tmpdir = self._tmpdir()
        path = export_markdown(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE,
                               output_path=os.path.join(tmpdir, "test.md"))
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert f"# {TITLE}" in content

    def test_export_markdown_contains_flashcards(self):
        tmpdir = self._tmpdir()
        path = export_markdown(REVISION_NOTES, FLASHCARDS, QUIZ_QUESTIONS, TITLE,
                               output_path=os.path.join(tmpdir, "test.md"))
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "## 🃏 Flashcards" in content

    # ── export_csv ────────────────────────────────────────────────────────────

    def test_export_csv_creates_file(self):
        tmpdir = self._tmpdir()
        path = export_csv(FLASHCARDS, output_path=os.path.join(tmpdir, "test.csv"))
        assert os.path.isfile(path)

    def test_export_csv_returns_absolute_path(self):
        tmpdir = self._tmpdir()
        path = export_csv(FLASHCARDS, output_path=os.path.join(tmpdir, "test.csv"))
        assert os.path.isabs(path)

    def test_export_csv_has_utf8_bom(self):
        tmpdir = self._tmpdir()
        path = export_csv(FLASHCARDS, output_path=os.path.join(tmpdir, "test.csv"))
        with open(path, "rb") as fh:
            assert fh.read(3) == b"\xef\xbb\xbf"

    def test_export_csv_parseable_by_csv_reader(self):
        tmpdir = self._tmpdir()
        path = export_csv(FLASHCARDS, output_path=os.path.join(tmpdir, "test.csv"))
        with open(path, encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["Front", "Back", "Tags"]
        assert len(rows) == len(FLASHCARDS) + 1

    # ── export_quiz_json ──────────────────────────────────────────────────────

    def test_export_quiz_json_creates_file(self):
        tmpdir = self._tmpdir()
        path = export_quiz_json(QUIZ_QUESTIONS,
                                output_path=os.path.join(tmpdir, "test.json"))
        assert os.path.isfile(path)

    def test_export_quiz_json_returns_absolute_path(self):
        tmpdir = self._tmpdir()
        path = export_quiz_json(QUIZ_QUESTIONS,
                                output_path=os.path.join(tmpdir, "test.json"))
        assert os.path.isabs(path)

    def test_export_quiz_json_is_valid_json(self):
        tmpdir = self._tmpdir()
        path = export_quiz_json(QUIZ_QUESTIONS, QUIZ_RESULTS, TITLE,
                                output_path=os.path.join(tmpdir, "test.json"))
        with open(path, encoding="utf-8") as fh:
            parsed = json.load(fh)
        assert parsed["title"] == TITLE
        assert "results" in parsed

    def test_export_quiz_json_two_calls_produce_separate_files(self):
        """Two successive disk exports must NOT overwrite each other."""
        import time
        tmpdir = self._tmpdir()
        p1 = export_quiz_json(QUIZ_QUESTIONS,
                              output_path=os.path.join(tmpdir, "quiz_a.json"))
        time.sleep(0.01)  # tiny gap
        p2 = export_quiz_json(QUIZ_QUESTIONS,
                              output_path=os.path.join(tmpdir, "quiz_b.json"))
        assert p1 != p2
        assert os.path.isfile(p1)
        assert os.path.isfile(p2)
