# tests/test_text_processing.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for the text cleaning and chunking module.
# ─────────────────────────────────────────────────────────────────────────────

import pytest
from text_processing import clean_transcript, chunk_text, preprocess


class TestCleanTranscript:
    """Tests for filler-word removal and whitespace normalisation."""

    def test_removes_brackets(self):
        result = clean_transcript("Hello [Music] world")
        assert "[Music]" not in result

    def test_removes_parens(self):
        result = clean_transcript("This is (laughter) a lecture")
        assert "(laughter)" not in result

    def test_removes_uh_um(self):
        result = clean_transcript("So uh this is um a test")
        assert "uh" not in result
        assert "um" not in result

    def test_collapses_whitespace(self):
        result = clean_transcript("hello   world   foo")
        assert "  " not in result

    def test_strips_edges(self):
        result = clean_transcript("  hello world  ")
        assert result == result.strip()

    def test_empty_string(self):
        assert clean_transcript("") == ""


class TestChunkText:
    """Tests for the overlapping windowed chunker."""

    def test_single_chunk_short_text(self):
        text = " ".join(["word"] * 100)  # 100 words
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        text = " ".join(["word"] * 1800)
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        # Expect ceil(1800 / 700) ≈ 3 chunks
        assert len(chunks) >= 2

    def test_overlap_present(self):
        words = [str(i) for i in range(1000)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # The last 50 words of chunk[0] should appear at start of chunk[1]
        end_of_first = chunks[0].split()[-50:]
        start_of_second = chunks[1].split()[:50]
        assert end_of_first == start_of_second

    def test_no_empty_chunks(self):
        text = " ".join(["a"] * 50)
        for chunk in chunk_text(text):
            assert chunk.strip() != ""


class TestPreprocess:
    """Integration test for the preprocess convenience function."""

    def test_returns_tuple(self):
        raw = "Hello [Music] world um this is a test sentence."
        cleaned, chunks = preprocess(raw)
        assert isinstance(cleaned, str)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_cleaned_has_no_filler(self):
        raw = "So uh we are um discussing [Music] things"
        cleaned, _ = preprocess(raw)
        assert "uh" not in cleaned
        assert "um" not in cleaned
