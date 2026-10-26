# test_flashcards.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit and integration tests for flashcards.py
#
# Covers:
#   - Validation: Definitions, concepts, formulas, relationships
#   - Rejection: Missing answers, empty questions, vague cards, placeholders
#   - Deduplication: Exact duplicates, case/punctuation variants, token overlap
#   - Anki CSV Export: Front/Back columns, RFC 4180 escaping, UTF-8 encoding
#   - Round-trip CSV parsing with standard library csv.reader
#   - Generation: User-chosen card counts, mock LLM generation, offline safety
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from flashcards import (
    Flashcard,
    FlashcardConfigError,
    categorize_card,
    deduplicate_flashcards,
    export_anki_csv_bytes,
    export_anki_csv_string,
    export_to_anki,
    export_to_csv,
    extract_timestamp_from_text,
    generate_flashcards,
    is_single_objective,
    is_vague_question,
    validate_and_clean_flashcards,
    validate_flashcard,
    MIN_QUESTION_CHARS,
    MIN_ANSWER_CHARS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Validation Tests (Missing Answers, Empty Questions, Placeholders, Vague Cards)
# ─────────────────────────────────────────────────────────────────────────────

class TestFlashcardValidation:
    """Tests for validate_flashcard and quality filters."""

    def test_valid_concept_card_accepted(self):
        raw = {
            "front": "How does binary search achieve logarithmic time complexity?",
            "back": "It iteratively divides the sorted search space in half with each comparison.",
            "category": "Concept",
            "timestamp": "04:15 - 05:00",
        }
        card = validate_flashcard(raw)
        assert card is not None
        assert card["front"] == raw["front"]
        assert card["back"] == raw["back"]
        assert card["category"] == "Concept"
        assert card["timestamp"] == "04:15 - 05:00"

    def test_valid_definition_card_accepted(self):
        raw = {
            "front": "What is the definition of a recurrence relation?",
            "back": "An equation that expresses a sequence or running time in terms of smaller input subproblems.",
        }
        card = validate_flashcard(raw)
        assert card is not None
        assert card["category"] == "Definition"

    def test_valid_formula_card_accepted(self):
        raw = {
            "front": "What is the Master Theorem recurrence for divide-and-conquer algorithms?",
            "back": "T(n) = a * T(n/b) + f(n), where a >= 1 and b > 1.",
        }
        card = validate_flashcard(raw)
        assert card is not None
        assert card["category"] == "Formula"

    def test_valid_relationship_card_accepted(self):
        raw = {
            "front": "What is the difference between Merge Sort and Quick Sort in terms of worst-case complexity?",
            "back": "Merge Sort is guaranteed O(n log n) whereas Quick Sort can degrade to O(n^2) without good pivots.",
        }
        card = validate_flashcard(raw)
        assert card is not None
        assert card["category"] == "Relationship"

    def test_missing_answer_rejected(self):
        # back is None
        assert validate_flashcard({"front": "What is asymptotic notation?", "back": None}) is None
        # back is missing key
        assert validate_flashcard({"front": "What is asymptotic notation?"}) is None

    def test_empty_answer_rejected(self):
        assert validate_flashcard({"front": "What is asymptotic notation?", "back": ""}) is None
        assert validate_flashcard({"front": "What is asymptotic notation?", "back": "   \n\t "}) is None

    def test_placeholder_answer_rejected(self):
        placeholders = ["N/A", "n/a", "None", "TODO", "TBD", "unknown", "?", "missing", "[missing]", "not mentioned"]
        for ph in placeholders:
            res = validate_flashcard({
                "front": "What is the time complexity of quicksort?",
                "back": ph,
            })
            assert res is None, f"Failed to reject placeholder answer: '{ph}'"

    def test_too_short_answer_rejected(self):
        assert validate_flashcard({"front": "What is an algorithm?", "back": "ok"}) is None

    def test_missing_question_rejected(self):
        assert validate_flashcard({"front": None, "back": "Valid detailed answer."}) is None
        assert validate_flashcard({"back": "Valid detailed answer."}) is None

    def test_empty_question_rejected(self):
        assert validate_flashcard({"front": "", "back": "Valid detailed answer."}) is None
        assert validate_flashcard({"front": "   \n ", "back": "Valid detailed answer."}) is None

    def test_too_short_question_rejected(self):
        # Less than 10 characters or less than 3 words
        assert validate_flashcard({"front": "What?", "back": "Valid detailed answer."}) is None
        assert validate_flashcard({"front": "Big O", "back": "Valid detailed answer."}) is None

    def test_vague_question_rejected(self):
        vague_examples = [
            "What is this?",
            "What is discussed?",
            "What did the speaker say?",
            "What did the professor talk about?",
            "What is the main topic?",
            "Explain everything",
            "Introduction",
            "Overview",
            "Summary?",
        ]
        for vq in vague_examples:
            res = validate_flashcard({
                "front": vq,
                "back": "This is a detailed explanation of the concept.",
            })
            assert res is None, f"Failed to reject vague question: '{vq}'"

    def test_strips_q_and_a_prefixes(self):
        raw = {
            "front": "Q: What is amortized analysis?",
            "back": "A: Averaging running time per operation over a sequence of operations.",
        }
        card = validate_flashcard(raw)
        assert card is not None
        assert card["front"] == "What is amortized analysis?"
        assert card["back"] == "Averaging running time per operation over a sequence of operations."

    def test_extracts_timestamp_from_text_if_missing_in_metadata(self):
        raw = {
            "front": "What is explained at [05:20] regarding recursion trees?",
            "back": "Each level represents subproblem work [05:20 - 06:10].",
        }
        card = validate_flashcard(raw)
        assert card is not None
        assert card["timestamp"] == "05:20"

    def test_flashcard_dataclass_dict_compatibility(self):
        fc = Flashcard(front="What is BFS?", back="Breadth-first search using a FIFO queue.", timestamp="03:00")
        assert fc["front"] == "What is BFS?"
        assert fc["back"] == "Breadth-first search using a FIFO queue."
        assert fc.get("timestamp") == "03:00"
        d = fc.to_dict()
        assert isinstance(d, dict)
        assert d["front"] == fc.front


# ─────────────────────────────────────────────────────────────────────────────
# 2. Deduplication Tests (Exact, Case/Punctuation, Near-Duplicates)
# ─────────────────────────────────────────────────────────────────────────────

class TestFlashcardDeduplication:
    """Tests for deduplicate_flashcards and token-similarity filtering."""

    def test_exact_duplicates_removed(self):
        cards = [
            {"front": "What is binary search?", "back": "Halves the search space."},
            {"front": "What is binary search?", "back": "Halves the search space."},
        ]
        deduped = deduplicate_flashcards(cards)
        assert len(deduped) == 1

    def test_case_and_punctuation_insensitive_duplicates_removed(self):
        cards = [
            {"front": "What is Binary Search?", "back": "Halves the search space."},
            {"front": "what is binary search", "back": "Halves the search interval."},
            {"front": "WHAT IS BINARY SEARCH???", "back": "Different wording."},
        ]
        deduped = deduplicate_flashcards(cards)
        assert len(deduped) == 1

    def test_near_duplicates_with_token_overlap_removed(self):
        cards = [
            {"front": "How does the merge sort algorithm work on an array?", "back": "Divide and conquer."},
            # Heavily overlapping wording:
            {"front": "How does the merge sort algorithm work on an array of numbers?", "back": "Divide and conquer."},
        ]
        deduped = deduplicate_flashcards(cards, similarity_threshold=0.70)
        assert len(deduped) == 1

    def test_distinct_cards_retained(self):
        cards = [
            {"front": "What is the time complexity of Quick Sort?", "back": "Average O(n log n)."},
            {"front": "What is the space complexity of Merge Sort?", "back": "O(n) auxiliary space."},
            {"front": "What is the definition of a heap data structure?", "back": "A complete binary tree satisfying the heap property."},
        ]
        deduped = deduplicate_flashcards(cards)
        assert len(deduped) == 3

    def test_validate_and_clean_caps_max_cards(self):
        raw = [
            {"front": "How does Merge Sort divide the input array?", "back": "Recursively splits array in half."},
            {"front": "What is the average time complexity of Quick Sort?", "back": "O(n log n)."},
            {"front": "How does a binary heap maintain order?", "back": "Every parent node is less than or equal to its children."},
            {"front": "Why does Depth First Search use a LIFO stack?", "back": "To explore as deep as possible before backtracking."},
            {"front": "What is the base case in binary search?", "back": "When low index exceeds high index."},
            {"front": "How does dynamic programming optimize recursion?", "back": "By storing overlapping subproblem results."},
        ]
        cleaned = validate_and_clean_flashcards(raw, max_cards=4)
        assert len(cleaned) == 4


# ─────────────────────────────────────────────────────────────────────────────
# 3. Anki-Compatible CSV Export Tests (Columns, Quoting, UTF-8, Round-Trip)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnkiCSVExport:
    """Tests for Anki-compatible CSV export adhering to RFC 4180 and UTF-8."""

    SAMPLE_CARDS = [
        {
            "front": "What is Big-O notation?",
            "back": "Upper asymptotic bound on algorithm growth.",
            "category": "Definition",
            "timestamp": "02:15",
        },
        {
            "front": 'What is a "divide-and-conquer" recurrence?',
            "back": "T(n) = 2T(n/2) + O(n), used in Merge Sort.",
            "category": "Formula",
            "timestamp": "08:30 - 10:00",
        },
        {
            "front": "What are the trade-offs of Quick Sort vs Merge Sort?",
            "back": "Quick Sort has O(1) extra space,\nbut Merge Sort offers guaranteed O(n log n).",
            "category": "Relationship",
        },
        {
            "front": "What is the omega bound: Ω(n)?",
            "back": "Lower bound: f(n) ≥ c * g(n) for large n.",
            "category": "Formula",
        },
    ]

    def test_csv_header_has_front_and_back_columns(self):
        csv_str = export_anki_csv_string(self.SAMPLE_CARDS, include_header=True)
        lines = csv_str.strip().split("\r\n")
        assert len(lines) >= 2
        header = lines[0].split(",")
        assert header[0] == "Front"
        assert header[1] == "Back"

    def test_csv_escaping_embedded_quotes(self):
        cards = [{"front": 'Explain the term "memoization" in DP.', "back": 'Caching "subproblem" solutions.'}]
        csv_str = export_anki_csv_string(cards, include_header=False)
        # RFC 4180: double quotes inside fields must be escaped as ""
        assert '""memoization""' in csv_str
        assert '""subproblem""' in csv_str

    def test_csv_escaping_commas(self):
        cards = [{"front": "What are the sorting steps: divide, conquer, and combine?", "back": "Step 1, step 2, and step 3."}]
        csv_str = export_anki_csv_string(cards, include_header=False)
        # Row must be enclosed in quotes because it contains commas
        assert '"What are the sorting steps: divide, conquer, and combine?"' in csv_str

    def test_csv_escaping_newlines(self):
        cards = [{"front": "List two properties:\n1. Overlapping subproblems\n2. Optimal substructure", "back": "Line 1\nLine 2"}]
        csv_str = export_anki_csv_string(cards, include_header=False)
        # Must be validly parsed back by csv.reader without breaking row count
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert "Overlapping subproblems" in rows[0][0]
        assert "Line 2" in rows[0][1]

    def test_csv_utf8_encoding_preserves_special_characters(self):
        cards = [
            {"front": "What is the Big-Omega notation: Ω(n)?", "back": "Lower bound: T(n) ≥ c * g(n)."},
            {"front": "Theta notation: Θ(n log n)", "back": "Tight bound: c1 ≤ f(n) ≤ c2."},
        ]
        csv_bytes = export_anki_csv_bytes(cards, include_header=True)
        # Should start with UTF-8 BOM or be valid UTF-8
        decoded = csv_bytes.decode("utf-8-sig")
        assert "Ω(n)" in decoded
        assert "Θ(n log n)" in decoded
        assert "≥" in decoded
        assert "≤" in decoded

    def test_csv_roundtrip_integrity(self):
        """Verify that csv.reader reconstructs the exact original fronts and backs."""
        csv_str = export_anki_csv_string(self.SAMPLE_CARDS, include_header=False)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == len(self.SAMPLE_CARDS)

        for original, parsed in zip(self.SAMPLE_CARDS, rows):
            assert parsed[0] == original["front"]
            # Back contains the answer (and timestamp tag if present)
            assert original["back"] in parsed[1]

    def test_export_to_csv_creates_file_on_disk(self, tmp_path):
        target_file = os.path.join(str(tmp_path), "test_anki_deck.csv")
        saved_path = export_to_csv(self.SAMPLE_CARDS, output_path=target_file)
        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 0

        # Verify content can be read back
        with open(saved_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # 1 header + 4 cards = 5 rows
            assert len(rows) == 5
            assert rows[0] == ["Front", "Back"]

    def test_export_to_anki_apkg_works(self, tmp_path):
        target_file = os.path.join(str(tmp_path), "test_deck.apkg")
        saved_path = export_to_anki(self.SAMPLE_CARDS, deck_name="Unit Test Deck", output_path=target_file)
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".apkg")
        assert os.path.getsize(saved_path) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Flashcard Generation Tests (User-Chosen Cards, Mock LLM, JSON Handling)
# ─────────────────────────────────────────────────────────────────────────────

class TestFlashcardGeneration:
    """Tests for generate_flashcards with mock models and offline safety."""

    SAMPLE_NOTES = """## 1. Lecture Overview
This lecture covers sorting algorithms and divide-and-conquer paradigms.

## 2. Key Concepts
- Merge Sort divides arrays into halves and merges sorted sub-arrays [02:15 - 04:30].
- Quick Sort uses partition around a pivot element [05:00 - 07:15].

## 3. Important Formulas
- Merge Sort recurrence: T(n) = 2T(n/2) + O(n)
- Solution via Master Theorem: O(n log n)
"""

    MOCK_LLM_JSON_RESPONSE = json.dumps([
        {
            "front": "What is the primary recurrence relation for Merge Sort?",
            "back": "T(n) = 2T(n/2) + O(n), describing dividing the array in half and combining.",
            "category": "Formula",
            "timestamp": "02:15 - 04:30",
        },
        {
            "front": "How does the divide-and-conquer paradigm operate?",
            "back": "It breaks a problem into smaller subproblems, solves them recursively, and combines the results.",
            "category": "Concept",
            "timestamp": "02:15",
        },
        {
            "front": "What role does the pivot play in Quick Sort?",
            "back": "The pivot partitions elements so that smaller elements are to the left and larger to the right.",
            "category": "Concept",
            "timestamp": "05:00 - 07:15",
        },
        {
            "front": "What is the asymptotic time complexity of Merge Sort in the worst case?",
            "back": "O(n log n), because each level takes O(n) work and there are log n levels.",
            "category": "Formula",
        },
    ])

    def test_generate_flashcards_with_mock_model(self):
        mock_chat = FakeListChatModel(responses=[self.MOCK_LLM_JSON_RESPONSE])
        cards = generate_flashcards(
            self.SAMPLE_NOTES,
            num_cards=4,
            model=mock_chat,
        )
        assert len(cards) == 4
        for card in cards:
            assert "front" in card
            assert "back" in card
            assert len(card["front"]) >= MIN_QUESTION_CHARS
            assert len(card["back"]) >= MIN_ANSWER_CHARS

    def test_num_cards_selection_respected(self):
        mock_chat = FakeListChatModel(responses=[self.MOCK_LLM_JSON_RESPONSE])
        cards = generate_flashcards(
            self.SAMPLE_NOTES,
            num_cards=2,  # User requests only 2 cards
            model=mock_chat,
        )
        assert len(cards) == 2

    def test_markdown_code_fence_json_parsing(self):
        fenced_json = f"```json\n{self.MOCK_LLM_JSON_RESPONSE}\n```"
        mock_chat = FakeListChatModel(responses=[fenced_json])
        cards = generate_flashcards(
            self.SAMPLE_NOTES,
            num_cards=3,
            model=mock_chat,
        )
        assert len(cards) == 3

    def test_json_with_conversational_preamble(self):
        prose_wrapped = f"Here are your flashcards:\n\n{self.MOCK_LLM_JSON_RESPONSE}\n\nHope this helps you study!"
        mock_chat = FakeListChatModel(responses=[prose_wrapped])
        cards = generate_flashcards(
            self.SAMPLE_NOTES,
            num_cards=4,
            model=mock_chat,
        )
        assert len(cards) == 4

    def test_empty_notes_returns_empty_list(self):
        mock_chat = FakeListChatModel(responses=["[]"])
        cards = generate_flashcards("", model=mock_chat)
        assert cards == []
        cards2 = generate_flashcards("   \n\t ", model=mock_chat)
        assert cards2 == []

    def test_missing_api_key_raises_config_error(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "GEMINI_API_KEY": ""}, clear=True):
            with pytest.raises(FlashcardConfigError):
                generate_flashcards(self.SAMPLE_NOTES, num_cards=5, model=None)

    def test_filters_invalid_cards_from_llm(self):
        # LLM returns 1 valid card, 1 duplicate, 1 with missing answer, 1 vague question
        dirty_response = json.dumps([
            {
                "front": "What is the time complexity of Merge Sort?",
                "back": "O(n log n).",
            },
            {
                "front": "What is the time complexity of Merge Sort?",  # Duplicate
                "back": "O(n log n).",
            },
            {
                "front": "What is a pivot?",
                "back": "N/A",  # Missing answer placeholder
            },
            {
                "front": "What is this?",  # Vague
                "back": "It is an algorithm.",
            },
        ])
        mock_chat = FakeListChatModel(responses=[dirty_response])
        cards = generate_flashcards(self.SAMPLE_NOTES, num_cards=5, model=mock_chat)
        # Only the 1 valid, non-duplicate card should survive
        assert len(cards) == 1
        assert cards[0]["front"] == "What is the time complexity of Merge Sort?"
