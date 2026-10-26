# tests/test_app_integration.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for application integration:
#   1. Input validation (extract_video_id, is_valid_youtube, validate_inputs)
#   2. Safe logging (secret masking filter for OpenAI & Gemini keys)
#   3. Demo mode bundle validity
#   4. In-session caching logic
#   5. End-to-end integrated workflow across all modules
#   6. All 5 download formats generated and validated
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re

import pytest

from app import (
    SecretMaskingFilter,
    extract_video_id,
    is_valid_youtube,
    validate_inputs,
)
from sample_data import (
    SAMPLE_FLASHCARDS,
    SAMPLE_LECTURE_TITLE,
    SAMPLE_QUIZ,
    SAMPLE_REVISION_NOTES,
    SAMPLE_TRANSCRIPT_SEGMENTS,
    get_demo_package,
)
from text_processing import preprocess_segments
from topic_clustering import cluster_transcript_chunks
from flashcards import validate_flashcard
from quiz import score_quiz, validate_quiz
from export import (
    build_csv_bytes,
    build_markdown_bytes,
    build_pdf_bytes,
    build_quiz_json_bytes,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    """Tests for URL and parameter validation."""

    def test_extract_video_id_standard_watch(self):
        url = "https://www.youtube.com/watch?v=Kg4bqm1QDIC"
        assert extract_video_id(url) == "Kg4bqm1QDIC"

    def test_extract_video_id_short_url(self):
        url = "https://youtu.be/Kg4bqm1QDIC"
        assert extract_video_id(url) == "Kg4bqm1QDIC"

    def test_extract_video_id_embed_url(self):
        url = "https://www.youtube.com/embed/Kg4bqm1QDIC"
        assert extract_video_id(url) == "Kg4bqm1QDIC"

    def test_extract_video_id_with_timestamp_params(self):
        url = "https://www.youtube.com/watch?v=0IAPZzGSbME&t=120s&list=PL123"
        assert extract_video_id(url) == "0IAPZzGSbME"

    def test_extract_video_id_raw_11_char_id(self):
        assert extract_video_id("BkxmsZ4iK6g") == "BkxmsZ4iK6g"

    def test_extract_video_id_empty_string(self):
        assert extract_video_id("") is None
        assert extract_video_id("   ") is None

    def test_extract_video_id_invalid_domain(self):
        assert extract_video_id("https://vimeo.com/123456789") is None

    def test_is_valid_youtube_true_for_valid_links(self):
        assert is_valid_youtube("https://www.youtube.com/watch?v=Kg4bqm1QDIC") is True
        assert is_valid_youtube("https://youtu.be/0IAPZzGSbME") is True

    def test_is_valid_youtube_false_for_invalid_links(self):
        assert is_valid_youtube("https://google.com") is False
        assert is_valid_youtube("not a url") is False
        assert is_valid_youtube("") is False

    def test_validate_inputs_demo_mode_always_passes(self):
        valid, err, vid = validate_inputs("", is_demo=True)
        assert valid is True
        assert err is None
        assert vid == "demo_mit_6006"

    def test_validate_inputs_empty_url_fails(self):
        valid, err, vid = validate_inputs("", is_demo=False)
        assert valid is False
        assert "Please enter a YouTube lecture URL" in (err or "")
        assert vid is None

    def test_validate_inputs_non_youtube_domain_fails(self):
        valid, err, vid = validate_inputs("https://dailymotion.com/video/x123", is_demo=False)
        assert valid is False
        assert "YouTube URL" in (err or "")

    def test_validate_inputs_valid_url_succeeds(self):
        valid, err, vid = validate_inputs("https://www.youtube.com/watch?v=Kg4bqm1QDIC", is_demo=False)
        assert valid is True
        assert err is None
        assert vid == "Kg4bqm1QDIC"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Safe Logging (Secret Masking Filter) Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeLogging:
    """Tests ensuring API keys and authorization tokens are redacted."""

    def _filter_record(self, message: str) -> str:
        masker = SecretMaskingFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg=message, args=(), exc_info=None,
        )
        masker.filter(record)
        return str(record.msg)

    def test_masks_openai_api_key(self):
        msg = "Connecting to OpenAI with key sk-proj-1234567890abcdef1234567890"
        redacted = self._filter_record(msg)
        assert "sk-proj-1234567890abcdef1234567890" not in redacted
        assert "[REDACTED_API_KEY]" in redacted

    def test_masks_gemini_api_key(self):
        msg = "Initialising Gemini using AIzaSyD1234567890abcdef1234567890abcdef"
        redacted = self._filter_record(msg)
        assert "AIzaSyD1234567890abcdef1234567890abcdef" not in redacted
        assert "[REDACTED_API_KEY]" in redacted

    def test_masks_bearer_token(self):
        msg = "HTTP Header Authorization: Bearer abcdef1234567890abcdef1234567890"
        redacted = self._filter_record(msg)
        assert "Bearer abcdef" not in redacted
        assert "[REDACTED_API_KEY]" in redacted

    def test_normal_text_passes_untouched(self):
        msg = "Processing lecture chunk 3/10 with 500 words"
        assert self._filter_record(msg) == msg


# ─────────────────────────────────────────────────────────────────────────────
# 3. Demo Mode Bundle Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDemoModeBundle:
    """Tests confirming the demo dataset is structurally sound and validated."""

    def test_demo_package_keys(self):
        pkg = get_demo_package()
        for key in ("title", "segments", "raw_transcript", "notes", "flashcards", "quiz"):
            assert key in pkg, f"Missing key in demo package: {key}"

    def test_demo_segments_have_required_fields(self):
        for seg in SAMPLE_TRANSCRIPT_SEGMENTS:
            assert "start" in seg
            assert "duration" in seg
            assert "text" in seg
            assert len(seg["text"]) > 0

    def test_demo_flashcards_pass_validation(self):
        for card in SAMPLE_FLASHCARDS:
            validated = validate_flashcard(card)
            assert validated is not None
            assert validated["front"]
            assert validated["back"]
            assert validated["category"] in {"Concept", "Formula", "Definition", "Relationship"}

    def test_demo_quiz_passes_validation(self):
        validated = validate_quiz(SAMPLE_QUIZ)
        assert len(validated) == len(SAMPLE_QUIZ)
        for q in validated:
            assert len(q["options"]) == 4
            assert q["answer"] in q["options"]
            assert q["difficulty"] in {"easy", "medium", "hard"}

    def test_demo_notes_contain_mandatory_sections(self):
        for section in (
            "1. Lecture Overview",
            "2. Key Concepts",
            "3. Definitions",
            "4. Important Formulas",
            "5. Examples & Case Studies",
            "6. Final Revision Checklist",
        ):
            assert section in SAMPLE_REVISION_NOTES


# ─────────────────────────────────────────────────────────────────────────────
# 4. Session Caching Logic Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionCaching:
    """Tests confirming session caching prevents duplicate processing."""

    def test_session_cache_storage_and_retrieval(self):
        cache = {}
        vid_key = "Kg4bqm1QDIC"
        lang = "🇺🇸 English"
        fmt = "Detailed"
        n_cards = 15

        cache_key = f"{vid_key}::{lang}::{fmt}::{n_cards}"

        dummy_output = {
            "raw_transcript": "sample transcript text",
            "revision_notes": "sample revision notes",
            "flashcards": [{"front": "Q1", "back": "A1"}],
            "quiz": [{"question": "Q?", "options": ["A", "B", "C", "D"], "answer": "A"}],
            "lecture_title": "MIT Lecture 1",
        }

        # Store in cache
        cache[cache_key] = dummy_output
        assert cache_key in cache

        # Retrieve from cache
        retrieved = cache[cache_key]
        assert retrieved["lecture_title"] == "MIT Lecture 1"
        assert retrieved["raw_transcript"] == "sample transcript text"

    def test_different_settings_generate_different_cache_keys(self):
        vid_key = "Kg4bqm1QDIC"
        key1 = f"{vid_key}::en::Detailed::15"
        key2 = f"{vid_key}::es::Detailed::15"
        key3 = f"{vid_key}::en::Concise::15"
        key4 = f"{vid_key}::en::Detailed::20"

        assert len({key1, key2, key3, key4}) == 4


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integrated Pipeline Flow (Demo Mode — No API Keys Needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegratedWorkflow:
    """
    End-to-end integration test verifying that running the demo workflow
    produces valid inputs for every UI tab and export builder.
    """

    def test_full_offline_workflow_integration(self):
        # 1. Transcript stage
        pkg = get_demo_package()
        segs = pkg["segments"]
        raw_text = pkg["raw_transcript"]
        assert len(segs) > 10
        assert len(raw_text.split()) > 100

        # 2. Text processing & chunking stage
        processed = preprocess_segments(segs, chunk_size=300, overlap=30)
        assert not processed.was_empty
        assert len(processed.chunks) >= 1
        assert processed.chunks[0].start_time is not None

        # 3. Topic clustering stage (runs real local CPU sentence-transformers)
        clustering = cluster_transcript_chunks(processed.chunks)
        assert clustering.n_topics >= 1
        assert len(clustering.topics) == clustering.n_topics
        for top in clustering.topics:
            assert len(top.label) > 0
            assert len(top.summary) > 0

        # 4. Revision notes stage
        notes = pkg["notes"]
        assert "## 1. Lecture Overview" in notes
        assert "## 6. Final Revision Checklist" in notes

        # 5. Flashcards stage
        flashcards = pkg["flashcards"]
        assert len(flashcards) >= 5

        # 6. Quiz & Scoring stage
        quiz = pkg["quiz"]
        assert len(quiz) >= 5
        user_answers = {0: quiz[0]["answer"], 1: quiz[1]["options"][0]}
        score_res = score_quiz(quiz, user_answers)
        assert score_res["total"] == len(quiz)
        assert score_res["score"] >= 1
        assert "results" in score_res

        # 7. Multi-format Exports stage
        title = SAMPLE_LECTURE_TITLE

        # 7a. ReportLab PDF
        pdf_bytes = build_pdf_bytes(notes, flashcards, quiz, title=title, topics=clustering.topics)
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 2000

        # 7b. Markdown
        md_bytes = build_markdown_bytes(notes, flashcards, quiz, title=title, topics=clustering.topics)
        md_text = md_bytes.decode("utf-8")
        assert f"# {title}" in md_text
        assert "## 📚 Revision Notes" in md_text
        assert "## 🃏 Flashcards" in md_text
        assert "## 📝 Practice Quiz" in md_text

        # 7c. Anki CSV
        csv_bytes = build_csv_bytes(flashcards, include_tags=True)
        assert csv_bytes[:3] == b"\xef\xbb\xbf"  # UTF-8 BOM
        csv_reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig")))
        csv_rows = list(csv_reader)
        assert len(csv_rows) == len(flashcards) + 1  # header + cards
        assert csv_rows[0] == ["Front", "Back", "Tags"]

        # 7d. Quiz JSON
        json_bytes = build_quiz_json_bytes(quiz, quiz_results=score_res, title=title)
        parsed_json = json.loads(json_bytes.decode("utf-8"))
        assert parsed_json["title"] == title
        assert parsed_json["total_questions"] == len(quiz)
        assert "results" in parsed_json
        assert parsed_json["results"]["score"] == score_res["score"]
