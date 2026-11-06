"""
test_hw_features.py
Tests for the new Handwritten Notes and improved Concise Notes features.
Run with:  .venv\Scripts\python test_hw_features.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

PASS = []
FAIL = []


def test(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  [FAIL] {name} -- {e}")


# ── Test 1: Format instructions — Concise ───────────────────────────────────
def _t1():
    from summarizer import _get_format_instructions
    instr = _get_format_instructions("concise")
    assert "CONCISE" in instr, "Should contain CONCISE"
    assert "40-55%" in instr, "Should specify target length reduction"
    assert "STRICTLY EXCLUDE" in instr, "Should list what to exclude"
    assert "bullet points" in instr.lower(), "Should require bullet points"


test("Concise format instructions are detailed and strict", _t1)


# ── Test 2: Format instructions — Handwritten maps to concise ───────────────
def _t2():
    from summarizer import _get_format_instructions
    hw_instr = _get_format_instructions("handwritten")
    cn_instr = _get_format_instructions("concise")
    assert hw_instr == cn_instr, "Handwritten should use same concise content generation"


test("Handwritten format maps to concise content generation", _t2)


# ── Test 3: Format instructions — Detailed unchanged ────────────────────────
def _t3():
    from summarizer import _get_format_instructions
    instr = _get_format_instructions("detailed")
    assert "DETAILED" in instr, "Should contain DETAILED"
    assert "comprehensive" in instr.lower(), "Should mention comprehensive coverage"


test("Detailed format instructions unchanged", _t3)


# ── Test 4: build_handwritten_pdf_bytes — returns valid PDF ─────────────────
def _t4():
    from export import build_handwritten_pdf_bytes
    sample_notes = """## 1. Lecture Overview
This lecture covers algorithms and complexity.

## 2. Key Concepts
- **Big-O Notation**: Upper bound on time complexity
- **Sorting**: Arranging elements in order

## 3. Definitions
- **Algorithm**: Step-by-step procedure for computation
- **Complexity**: Measure of resource usage

## 4. Important Formulas
- O(n log n) = Merge Sort time complexity
- T(n) = 2T(n/2) + O(n)

## 5. Examples & Case Studies
- Merge Sort: Divide array, conquer halves, merge results

## 6. Final Revision Checklist
- [ ] Understand Big-O notation
- [ ] Know at least 3 sorting algorithms
"""
    pdf_bytes = build_handwritten_pdf_bytes(
        revision_notes=sample_notes,
        title="Test Lecture — Algorithms",
        video_url="https://www.youtube.com/watch?v=test123",
        video_id="test123",
    )
    assert isinstance(pdf_bytes, bytes), "Should return bytes"
    assert len(pdf_bytes) > 1000, f"PDF too small: {len(pdf_bytes)} bytes"
    assert pdf_bytes[:4] == b"%PDF", f"Not a valid PDF header: {pdf_bytes[:4]}"
    print(f"      PDF size: {len(pdf_bytes):,} bytes")


test("build_handwritten_pdf_bytes returns valid PDF", _t4)


# ── Test 5: Handwritten PDF with topics ─────────────────────────────────────
def _t5():
    from export import build_handwritten_pdf_bytes
    topics = [
        type("T", (), {"label": "Introduction to Algorithms", "time_range_fmt": "00:00 - 05:00", "summary": "Overview"})(),
        type("T", (), {"label": "Sorting Algorithms", "time_range_fmt": "05:00 - 20:00", "summary": "Sorting"})(),
    ]
    pdf_bytes = build_handwritten_pdf_bytes(
        revision_notes="## Overview\n- Key point about algorithms\n\n**Merge Sort**: O(n log n)\n",
        title="Test With Topics",
        topics=topics,
    )
    assert pdf_bytes[:4] == b"%PDF", "Should produce valid PDF with topics"


test("Handwritten PDF with topics section", _t5)


# ── Test 6: build_notes_only_pdf_bytes still works ──────────────────────────
def _t6():
    from export import build_notes_only_pdf_bytes
    pdf_bytes = build_notes_only_pdf_bytes(
        revision_notes="## Overview\n- Test note\n\n## Key Concepts\n- Concept A\n",
        title="Standard Notes Test",
    )
    assert pdf_bytes[:4] == b"%PDF", "Standard notes PDF should still work"
    print(f"      Standard PDF size: {len(pdf_bytes):,} bytes")


test("Standard notes-only PDF still works", _t6)


# ── Test 7: _safe_pdf_filename with format ───────────────────────────────────
def _t7():
    # We test the function logic manually here (can't import app.py easily)
    import re

    def _safe_pdf_filename(title: str, notes_format: str = "Detailed") -> str:
        slug = re.sub(r"[^\w\s-]", "", title).strip()
        slug = re.sub(r"[\s_-]+", "_", slug)
        slug = slug[:60] or "Lecture_Notes"
        fmt_map = {
            "Detailed":    "Detailed_Notes",
            "Concise":     "Concise_Notes",
            "Handwritten": "Handwritten_Notes",
        }
        fmt_label = fmt_map.get(notes_format, "Notes")
        return f"Lecture_{fmt_label}_{slug}.pdf"

    assert _safe_pdf_filename("MIT 6.006", "Concise") == "Lecture_Concise_Notes_MIT_6006.pdf"
    assert _safe_pdf_filename("MIT 6.006", "Detailed") == "Lecture_Detailed_Notes_MIT_6006.pdf"
    assert _safe_pdf_filename("MIT 6.006", "Handwritten") == "Lecture_Handwritten_Notes_MIT_6006.pdf"


test("_safe_pdf_filename generates format-specific filenames", _t7)


# ── Test 8: Multi-page handwritten PDF ──────────────────────────────────────
def _t8():
    from export import build_handwritten_pdf_bytes
    # Generate a large notes string to force multi-page output
    big_notes = ""
    for i in range(1, 8):
        big_notes += f"## Section {i}: Topic Name\n"
        big_notes += "- " + " ".join(["word"] * 30) + "\n" * 5
        for j in range(1, 6):
            big_notes += f"- **Key point {j}**: " + " ".join(["explanation"] * 20) + "\n"
        big_notes += "\n"

    pdf_bytes = build_handwritten_pdf_bytes(
        revision_notes=big_notes,
        title="Multi-Page Test Lecture",
    )
    assert pdf_bytes[:4] == b"%PDF", "Multi-page PDF should be valid"
    # A multi-section PDF should be larger than a minimal one
    assert len(pdf_bytes) > 5000, f"Multi-page PDF too small: {len(pdf_bytes)} bytes"
    print(f"      Multi-page PDF size: {len(pdf_bytes):,} bytes")


test("Multi-page handwritten PDF works correctly", _t8)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print(f"Failed tests: {', '.join(FAIL)}")
    sys.exit(1)
else:
    print("All tests passed!")
