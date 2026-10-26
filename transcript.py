# transcript.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Extract transcripts (captions) from YouTube videos.
#
# Uses youtube-transcript-api ≥ 1.0 (instance-based API).
# Downloads auto-generated or manually created captions directly from YouTube —
# no headless browser or scraping required. Does NOT bypass access controls or
# download restricted content.
#
# Supported URL formats:
#   - https://www.youtube.com/watch?v=VIDEO_ID
#   - https://youtu.be/VIDEO_ID
#   - https://www.youtube.com/embed/VIDEO_ID
#   - https://www.youtube.com/shorts/VIDEO_ID
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    FetchedTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeDataUnparsable,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """A single captioned segment of a YouTube video."""

    text: str
    """The caption text for this segment."""

    start: float
    """Start time in seconds from the beginning of the video."""

    duration: float
    """How long (in seconds) this caption stays on screen."""

    @property
    def start_formatted(self) -> str:
        """Return start time as MM:SS string."""
        mins, secs = divmod(int(self.start), 60)
        return f"{mins:02d}:{secs:02d}"


@dataclass
class TranscriptResult:
    """The complete result of a successful transcript fetch."""

    video_id: str
    language: str
    language_code: str
    is_generated: bool
    segments: list[TranscriptSegment]

    @property
    def full_text(self) -> str:
        """All segment texts joined by a single space."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def to_raw_data(self) -> list[dict]:
        """Return segments as plain dicts (compatible with old dict-based code)."""
        return [
            {"text": seg.text, "start": seg.start, "duration": seg.duration}
            for seg in self.segments
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Video ID extraction
# ─────────────────────────────────────────────────────────────────────────────

# All patterns capture exactly 11 alphanumeric/dash/underscore characters.
_VIDEO_ID_PATTERNS: list[str] = [
    r"(?:v=)([0-9A-Za-z_-]{11})",          # /watch?v=VIDEO_ID
    r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",  # youtu.be/VIDEO_ID
    r"(?:embed/)([0-9A-Za-z_-]{11})",      # /embed/VIDEO_ID
    r"(?:shorts/)([0-9A-Za-z_-]{11})",     # /shorts/VIDEO_ID  ← YouTube Shorts
]


def extract_video_id(url: str) -> str:
    """
    Extract the 11-character video ID from any recognised YouTube URL.

    Handles standard watch links, youtu.be short links, embed links,
    and YouTube Shorts links.

    Args:
        url: A YouTube video URL string.

    Returns:
        The 11-character video ID.

    Raises:
        ValueError: If no valid video ID can be extracted from the URL.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    for pattern in _VIDEO_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(
        f"Could not extract a YouTube video ID from: '{url}'\n"
        "Supported formats:\n"
        "  • https://www.youtube.com/watch?v=VIDEO_ID\n"
        "  • https://youtu.be/VIDEO_ID\n"
        "  • https://www.youtube.com/embed/VIDEO_ID\n"
        "  • https://www.youtube.com/shorts/VIDEO_ID"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core transcript fetching
# ─────────────────────────────────────────────────────────────────────────────

def get_transcript(
    url: str,
    languages: Optional[list[str]] = None,
) -> TranscriptResult:
    """
    Fetch the transcript for a YouTube video and return structured data.

    The function first tries to fetch captions in the requested languages.
    If none of those are available it lists all available transcripts and
    falls back to the first one it can access (translated to English where
    the video supports translation).

    This function does NOT download video content, bypass access controls,
    or circumvent age-restriction or region-lock mechanisms.  If captions
    are disabled by the video owner, a clear error is raised so the caller
    can offer the user an alternative workflow (e.g. providing their own
    transcript text).

    Args:
        url:       Full YouTube video URL (watch, short, embed, or Shorts).
        languages: Ordered list of BCP-47 language codes to try, e.g.
                   ["en", "en-US", "en-GB"].  Defaults to English variants.

    Returns:
        A :class:`TranscriptResult` containing language metadata and a list
        of :class:`TranscriptSegment` objects (text, start, duration).

    Raises:
        ValueError:              If the video ID cannot be parsed from the URL.
        TranscriptsDisabled:     If the video owner has disabled captions —
                                 the caller should explain this to the user and
                                 offer a manual transcript input option.
        NoTranscriptFound:       If no transcript exists in any language.
        VideoUnavailable:        If the video is private, deleted, or geo-blocked.
        RequestBlocked / IpBlocked:
                                 If YouTube has rate-limited this IP.
        YouTubeRequestFailed:    On network-level failures (DNS, timeout, etc.).
        RuntimeError:            Wraps any other unexpected API errors.
    """
    if languages is None:
        languages = ["en", "en-US", "en-GB"]

    video_id = extract_video_id(url)  # raises ValueError on bad URL
    api = YouTubeTranscriptApi()

    fetched: FetchedTranscript = _fetch_with_fallback(api, video_id, languages)
    segments = [
        TranscriptSegment(text=snip.text, start=snip.start, duration=snip.duration)
        for snip in fetched
    ]

    return TranscriptResult(
        video_id=video_id,
        language=fetched.language,
        language_code=fetched.language_code,
        is_generated=fetched.is_generated,
        segments=segments,
    )


def _fetch_with_fallback(
    api: YouTubeTranscriptApi,
    video_id: str,
    languages: list[str],
) -> FetchedTranscript:
    """
    Try preferred languages first; fall back to any available transcript.

    Internal helper — not part of the public API.
    """
    try:
        return api.fetch(video_id, languages=languages)

    except NoTranscriptFound:
        # Preferred languages not available — find whatever IS available.
        try:
            transcript_list = api.list(video_id)
            transcript = transcript_list.find_transcript(
                [t.language_code for t in transcript_list]
            )
            # Translate to English if supported, otherwise return as-is.
            if transcript.is_translatable:
                try:
                    return transcript.translate("en").fetch()
                except Exception:
                    pass  # Translation failed — fall through to raw fetch
            return transcript.fetch()
        except (TranscriptsDisabled, VideoUnavailable):
            raise  # re-raise with original message
        except NoTranscriptFound:
            raise NoTranscriptFound(
                video_id,
                languages,
            )

    except TranscriptsDisabled:
        raise TranscriptsDisabled(video_id)

    except VideoUnavailable:
        raise

    except (RequestBlocked, IpBlocked):
        raise

    except YouTubeRequestFailed:
        raise  # constructor requires (video_id, http_error) — re-raise as-is

    except YouTubeDataUnparsable as exc:
        raise RuntimeError(
            f"YouTube returned unexpected data for video '{video_id}'. "
            "The page structure may have changed. "
            f"Details: {exc}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error fetching transcript for video '{video_id}': {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers  (keep backward-compat with app.py call sites)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_transcript(url: str, languages: Optional[list[str]] = None) -> str:
    """
    Return the full transcript as a single plain-text string.

    Convenience wrapper around :func:`get_transcript` for callers that only
    need the raw text (e.g. for AI summarisation).

    Args:
        url:       Full YouTube video URL.
        languages: BCP-47 language codes to prefer (default: English variants).

    Returns:
        All caption segments joined by spaces.

    Raises:
        See :func:`get_transcript` for the full list of possible exceptions.
    """
    result = get_transcript(url, languages)
    return result.full_text


def get_transcript_with_timestamps(
    url: str,
    languages: Optional[list[str]] = None,
) -> list[dict]:
    """
    Return transcript segments as a list of plain dicts.

    Each dict has the keys ``text``, ``start`` (seconds), and ``duration``
    (seconds).  This format is compatible with the old dict-based code in
    ``app.py`` so no changes to the Streamlit rendering layer are required.

    Args:
        url:       Full YouTube video URL.
        languages: BCP-47 language codes to prefer (default: English variants).

    Returns:
        List of ``{"text": str, "start": float, "duration": float}`` dicts.

    Raises:
        See :func:`get_transcript` for the full list of possible exceptions.
    """
    if languages is None:
        languages = ["en", "en-US", "en-GB"]
    result = get_transcript(url, languages)
    return result.to_raw_data()


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit display helper
# ─────────────────────────────────────────────────────────────────────────────

def display_transcript_in_streamlit(result: TranscriptResult) -> None:
    """
    Render a :class:`TranscriptResult` inside a Streamlit app.

    Displays:
    - Language metadata and segment count
    - A scrollable, timestamp-annotated transcript block
    - A plain-text download button

    This function imports Streamlit lazily so that ``transcript.py`` can be
    imported in non-Streamlit contexts (tests, CLI scripts) without error.

    Args:
        result: A :class:`TranscriptResult` returned by :func:`get_transcript`.
    """
    import streamlit as st  # lazy import — only needed when rendering

    generated_label = "auto-generated" if result.is_generated else "manual"
    st.caption(
        f"🌐 Language: **{result.language}** (`{result.language_code}`, {generated_label}) · "
        f"**{result.segment_count}** caption segments"
    )

    # Scrollable transcript block (reuses the .transcript-block CSS from app.py)
    lines_html = ""
    for seg in result.segments:
        lines_html += (
            f'<span class="ts-time">[{seg.start_formatted}]</span>'
            f'{seg.text}<br>'
        )

    st.markdown(
        f'<div class="transcript-block">{lines_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Plain-text download
    plain_text = "\n".join(
        f"[{seg.start_formatted}] {seg.text}" for seg in result.segments
    )
    st.download_button(
        "📥 Download Transcript (.txt)",
        data=plain_text,
        file_name="transcript.txt",
        mime="text/plain",
    )


# ─────────────────────────────────────────────────────────────────────────────
# User-facing error messages
# ─────────────────────────────────────────────────────────────────────────────

def friendly_error_message(exc: Exception) -> str:
    """
    Convert a transcript API exception into a user-friendly message.

    Args:
        exc: Any exception raised during transcript fetching.

    Returns:
        A human-readable string explaining the problem and suggesting next steps.
    """
    if isinstance(exc, ValueError):
        return (
            f"⚠️ Invalid YouTube URL\n\n{exc}\n\n"
            "Please check the link and try again."
        )
    if isinstance(exc, TranscriptsDisabled):
        return (
            "🚫 Captions are disabled for this video.\n\n"
            "The video owner has turned off captions, so automatic extraction "
            "is not possible.\n\n"
            "**Alternatives:**\n"
            "- Paste the transcript text manually into the text area below.\n"
            "- Use an authorised audio-transcription service (e.g. Whisper) "
            "if you have permission to process the audio."
        )
    if isinstance(exc, NoTranscriptFound):
        return (
            "🔍 No captions found for this video.\n\n"
            "YouTube does not have a transcript in any language for this video.\n\n"
            "**Alternatives:**\n"
            "- Paste the transcript text manually into the text area below.\n"
            "- Use an authorised audio-transcription service if you have "
            "permission to process the audio."
        )
    if isinstance(exc, VideoUnavailable):
        return (
            "📵 This video is unavailable.\n\n"
            "The video may be private, deleted, or restricted in your region."
        )
    if isinstance(exc, (RequestBlocked, IpBlocked)):
        return (
            "🚧 YouTube has temporarily blocked requests from this IP address.\n\n"
            "This usually happens due to rate-limiting. Please wait a few minutes "
            "and try again, or use a different network connection."
        )
    if isinstance(exc, YouTubeRequestFailed):
        return (
            f"🌐 Network error while contacting YouTube.\n\n{exc}\n\n"
            "Check your internet connection and try again."
        )
    # Generic fallback
    return f"❌ An unexpected error occurred: {exc}"
