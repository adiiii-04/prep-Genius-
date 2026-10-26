# tests/test_transcript.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for the transcript extraction module.
# Run with:  pytest tests/
# ─────────────────────────────────────────────────────────────────────────────

import pytest
from transcript import extract_video_id


class TestExtractVideoId:
    """Unit tests for extract_video_id — no network calls needed."""

    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s&list=PL123"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Could not extract"):
            extract_video_id("https://www.google.com")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            extract_video_id("")
