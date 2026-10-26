# test_transcript.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for transcript.py
#
# Run with:
#   py -m pytest test_transcript.py -v
#
# These tests are offline-safe — all YouTube API calls are mocked.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from youtube_transcript_api import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
)

from transcript import (
    TranscriptResult,
    TranscriptSegment,
    extract_video_id,
    fetch_transcript,
    friendly_error_message,
    get_transcript,
    get_transcript_with_timestamps,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ═══════════════════════════════════════════════════════════════════════════════

VIDEO_ID = "dQw4w9WgXcQ"

def _make_snippet(text: str, start: float = 0.0, duration: float = 2.0):
    """Build a mock FetchedTranscriptSnippet."""
    snip = MagicMock()
    snip.text = text
    snip.start = start
    snip.duration = duration
    return snip


def _make_fetched(snippets, language="English", language_code="en", is_generated=True):
    """Build a mock FetchedTranscript."""
    fetched = MagicMock()
    fetched.language = language
    fetched.language_code = language_code
    fetched.is_generated = is_generated
    fetched.__iter__ = MagicMock(return_value=iter(snippets))
    fetched.__len__ = MagicMock(return_value=len(snippets))
    return fetched


# ═══════════════════════════════════════════════════════════════════════════════
# 1. extract_video_id — URL parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractVideoId:
    """Validate ID extraction across all supported URL formats."""

    def test_standard_watch_url(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        assert extract_video_id(url) == VIDEO_ID

    def test_watch_url_with_extra_params(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}&t=42s&list=PLxxx"
        assert extract_video_id(url) == VIDEO_ID

    def test_short_youtu_be_url(self):
        url = f"https://youtu.be/{VIDEO_ID}"
        assert extract_video_id(url) == VIDEO_ID

    def test_youtu_be_with_query(self):
        url = f"https://youtu.be/{VIDEO_ID}?si=abc123"
        assert extract_video_id(url) == VIDEO_ID

    def test_embed_url(self):
        url = f"https://www.youtube.com/embed/{VIDEO_ID}"
        assert extract_video_id(url) == VIDEO_ID

    def test_shorts_url(self):
        url = f"https://www.youtube.com/shorts/{VIDEO_ID}"
        assert extract_video_id(url) == VIDEO_ID

    def test_http_scheme(self):
        url = f"http://www.youtube.com/watch?v={VIDEO_ID}"
        assert extract_video_id(url) == VIDEO_ID

    def test_no_www(self):
        url = f"https://youtube.com/watch?v={VIDEO_ID}"
        assert extract_video_id(url) == VIDEO_ID

    # --- Invalid inputs ---

    def test_plain_video_id_raises(self):
        with pytest.raises(ValueError, match="Could not extract"):
            extract_video_id(VIDEO_ID)

    def test_non_youtube_url_raises(self):
        with pytest.raises(ValueError, match="Could not extract"):
            extract_video_id("https://vimeo.com/123456")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            extract_video_id("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            extract_video_id("   ")

    def test_none_type_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            extract_video_id(None)  # type: ignore[arg-type]

    def test_wrong_id_length_not_matched(self):
        # 10-char ID — too short
        with pytest.raises(ValueError):
            extract_video_id("https://www.youtube.com/watch?v=shortid123")

    def test_returns_exactly_11_chars(self):
        url = f"https://youtu.be/{VIDEO_ID}"
        vid_id = extract_video_id(url)
        assert len(vid_id) == 11


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TranscriptSegment & TranscriptResult — data model
# ═══════════════════════════════════════════════════════════════════════════════

class TestTranscriptSegment:
    def test_start_formatted_zero(self):
        seg = TranscriptSegment(text="Hello", start=0.0, duration=2.5)
        assert seg.start_formatted == "00:00"

    def test_start_formatted_minutes(self):
        seg = TranscriptSegment(text="Hi", start=75.0, duration=3.0)
        assert seg.start_formatted == "01:15"

    def test_start_formatted_hour_range(self):
        seg = TranscriptSegment(text="X", start=3661.0, duration=1.0)
        assert seg.start_formatted == "61:01"

    def test_attributes_stored(self):
        seg = TranscriptSegment(text="Test", start=10.5, duration=2.0)
        assert seg.text == "Test"
        assert seg.start == 10.5
        assert seg.duration == 2.0


class TestTranscriptResult:
    @pytest.fixture
    def result(self):
        segments = [
            TranscriptSegment("Hello", 0.0, 2.0),
            TranscriptSegment("World", 2.0, 2.0),
        ]
        return TranscriptResult(
            video_id=VIDEO_ID,
            language="English",
            language_code="en",
            is_generated=True,
            segments=segments,
        )

    def test_full_text(self, result):
        assert result.full_text == "Hello World"

    def test_segment_count(self, result):
        assert result.segment_count == 2

    def test_to_raw_data_keys(self, result):
        raw = result.to_raw_data()
        assert len(raw) == 2
        for item in raw:
            assert set(item.keys()) == {"text", "start", "duration"}

    def test_to_raw_data_values(self, result):
        raw = result.to_raw_data()
        assert raw[0] == {"text": "Hello", "start": 0.0, "duration": 2.0}
        assert raw[1] == {"text": "World", "start": 2.0, "duration": 2.0}

    def test_empty_segments(self):
        r = TranscriptResult(VIDEO_ID, "English", "en", True, [])
        assert r.full_text == ""
        assert r.segment_count == 0
        assert r.to_raw_data() == []


# ═══════════════════════════════════════════════════════════════════════════════
# 3. get_transcript — success paths
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTranscriptSuccess:
    """Happy-path tests with mocked YouTubeTranscriptApi."""

    @pytest.fixture
    def snippets(self):
        return [
            _make_snippet("Never gonna give you up", 0.0, 3.0),
            _make_snippet("Never gonna let you down", 3.5, 3.0),
        ]

    @pytest.fixture
    def fetched(self, snippets):
        return _make_fetched(snippets)

    def test_returns_transcript_result(self, fetched):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.return_value = fetched
            result = get_transcript(url)

        assert isinstance(result, TranscriptResult)
        assert result.video_id == VIDEO_ID

    def test_segment_texts(self, fetched):
        url = f"https://youtu.be/{VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.return_value = fetched
            result = get_transcript(url)

        assert result.segments[0].text == "Never gonna give you up"
        assert result.segments[1].text == "Never gonna let you down"

    def test_segment_timestamps(self, fetched):
        url = f"https://www.youtube.com/shorts/{VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.return_value = fetched
            result = get_transcript(url)

        assert result.segments[0].start == 0.0
        assert result.segments[1].start == 3.5

    def test_language_metadata(self, fetched):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        fetched.language = "Spanish"
        fetched.language_code = "es"
        fetched.is_generated = False
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.return_value = fetched
            result = get_transcript(url, languages=["es"])

        assert result.language == "Spanish"
        assert result.language_code == "es"
        assert result.is_generated is False

    def test_default_languages_passed_to_api(self, fetched):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.return_value = fetched
            get_transcript(url)
            instance.fetch.assert_called_once_with(
                VIDEO_ID, languages=["en", "en-US", "en-GB"]
            )

    def test_custom_languages_forwarded(self, fetched):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.return_value = fetched
            get_transcript(url, languages=["fr", "de"])
            instance.fetch.assert_called_once_with(
                VIDEO_ID, languages=["fr", "de"]
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. get_transcript — fallback path (NoTranscriptFound → fallback language)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTranscriptFallback:
    def _make_transcript_entry(self, lang_code, is_translatable=True):
        t = MagicMock()
        t.language_code = lang_code
        t.is_translatable = is_translatable
        translated = MagicMock()
        translated.fetch.return_value = _make_fetched(
            [_make_snippet("Translated text")], language_code=lang_code
        )
        t.translate.return_value = translated
        t.fetch.return_value = _make_fetched(
            [_make_snippet("Raw text")], language_code=lang_code
        )
        return t

    def test_falls_back_to_available_transcript(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        de_transcript = self._make_transcript_entry("de")

        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.side_effect = NoTranscriptFound(VIDEO_ID, ["en"], MagicMock())
            transcript_list = MagicMock()
            transcript_list.__iter__ = MagicMock(return_value=iter([de_transcript]))
            transcript_list.find_transcript.return_value = de_transcript
            instance.list.return_value = transcript_list

            result = get_transcript(url)

        assert result is not None
        assert len(result.segments) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. get_transcript — error paths
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTranscriptErrors:
    def _patch_api_error(self, error):
        """Return a context manager that makes api.fetch() raise *error*."""
        return patch("transcript.YouTubeTranscriptApi", return_value=MagicMock(
            fetch=MagicMock(side_effect=error)
        ))

    def test_invalid_url_raises_value_error(self):
        with pytest.raises(ValueError):
            get_transcript("https://vimeo.com/12345")

    def test_transcripts_disabled_propagates(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.side_effect = TranscriptsDisabled(VIDEO_ID)
            with pytest.raises(TranscriptsDisabled):
                get_transcript(url)

    def test_video_unavailable_propagates(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.side_effect = VideoUnavailable(VIDEO_ID)
            with pytest.raises(VideoUnavailable):
                get_transcript(url)

    def test_request_blocked_propagates(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.side_effect = RequestBlocked(VIDEO_ID)
            with pytest.raises(RequestBlocked):
                get_transcript(url)

    def test_ip_blocked_propagates(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.side_effect = IpBlocked(VIDEO_ID)
            with pytest.raises(IpBlocked):
                get_transcript(url)

    def test_youtube_request_failed_becomes_youtube_request_failed(self):
        """YouTubeRequestFailed is re-raised unchanged (re-constructing it would
        require the exact (video_id, http_error) arguments)."""
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.side_effect = YouTubeRequestFailed(VIDEO_ID, MagicMock())
            with pytest.raises(YouTubeRequestFailed):
                get_transcript(url)

    def test_generic_exception_becomes_runtime_error(self):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.YouTubeTranscriptApi") as MockApi:
            instance = MockApi.return_value
            instance.fetch.side_effect = Exception("mystery error")
            with pytest.raises(RuntimeError, match="Unexpected error"):
                get_transcript(url)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Convenience wrappers
# ═══════════════════════════════════════════════════════════════════════════════

class TestConvenienceWrappers:
    @pytest.fixture
    def mock_result(self):
        segs = [
            TranscriptSegment("Hello", 0.0, 2.0),
            TranscriptSegment("World", 2.0, 2.0),
        ]
        return TranscriptResult(VIDEO_ID, "English", "en", True, segs)

    def test_fetch_transcript_returns_string(self, mock_result):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.get_transcript", return_value=mock_result):
            text = fetch_transcript(url)
        assert isinstance(text, str)
        assert text == "Hello World"

    def test_get_transcript_with_timestamps_returns_dicts(self, mock_result):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.get_transcript", return_value=mock_result):
            data = get_transcript_with_timestamps(url)
        assert isinstance(data, list)
        assert all(isinstance(d, dict) for d in data)
        assert data[0] == {"text": "Hello", "start": 0.0, "duration": 2.0}

    def test_get_transcript_with_timestamps_has_required_keys(self, mock_result):
        url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
        with patch("transcript.get_transcript", return_value=mock_result):
            data = get_transcript_with_timestamps(url)
        for item in data:
            assert "text" in item
            assert "start" in item
            assert "duration" in item


# ═══════════════════════════════════════════════════════════════════════════════
# 7. friendly_error_message
# ═══════════════════════════════════════════════════════════════════════════════

class TestFriendlyErrorMessage:
    def test_value_error(self):
        msg = friendly_error_message(ValueError("bad url"))
        assert "Invalid YouTube URL" in msg

    def test_transcripts_disabled(self):
        msg = friendly_error_message(TranscriptsDisabled(VIDEO_ID))
        assert "disabled" in msg.lower()
        assert "Alternatives" in msg

    def test_no_transcript_found(self):
        exc = NoTranscriptFound(VIDEO_ID, ["en"], MagicMock())
        msg = friendly_error_message(exc)
        assert "No captions" in msg
        assert "Alternatives" in msg

    def test_video_unavailable(self):
        msg = friendly_error_message(VideoUnavailable(VIDEO_ID))
        assert "unavailable" in msg.lower()

    def test_request_blocked(self):
        msg = friendly_error_message(RequestBlocked(VIDEO_ID))
        assert "blocked" in msg.lower() or "rate" in msg.lower()

    def test_ip_blocked(self):
        msg = friendly_error_message(IpBlocked(VIDEO_ID))
        assert "blocked" in msg.lower()

    def test_youtube_request_failed(self):
        exc = YouTubeRequestFailed(MagicMock(), "Connection timeout")
        msg = friendly_error_message(exc)
        assert "Network error" in msg

    def test_generic_exception(self):
        msg = friendly_error_message(RuntimeError("boom"))
        assert "unexpected" in msg.lower()

    def test_all_messages_are_strings(self):
        errors = [
            ValueError("bad"),
            TranscriptsDisabled(VIDEO_ID),
            VideoUnavailable(VIDEO_ID),
            RequestBlocked(VIDEO_ID),
            IpBlocked(VIDEO_ID),
            RuntimeError("x"),
        ]
        for exc in errors:
            assert isinstance(friendly_error_message(exc), str)
