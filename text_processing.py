# text_processing.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Clean raw transcript text and split it into manageable chunks.
#
# YouTube auto-captions contain noise: filler words, repeated phrases,
# bracketed sound cues, HTML entities, etc.  This module normalises the text
# conservatively — preserving technical terms, equations, definitions,
# examples, and negations — then divides it into overlapping chunks that
# stay within the selected model's context limit.
#
# A list of dicts is used internally to organise timestamped segments
# so that each cleaned segment retains its start/duration metadata.
#
# This module does NOT summarise, generate flashcards, or produce quizzes.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import nltk

# ── NLTK resource bootstrap ───────────────────────────────────────────────────
# Downloads are silent and cached after the first run.
for _res in ("punkt", "punkt_tab"):
    nltk.download(_res, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model context limits (conservative word-count budgets, not token counts).
# Gemini tokenises at ~1.3 tokens/word on average; we keep a 20 % safety
# margin so chunks never approach the actual limit.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_WORD_LIMITS: dict[str, int] = {
    "gemini-1.5-flash":     24_000,   # 1 M tokens  ÷ 1.3 × 0.8 ≈ 615 k words → cap at 24 k for practical chunks
    "gemini-1.5-pro":       24_000,
    "gemini-2.0-flash-exp": 24_000,
    # Fallback for unknown models
    "_default":              8_000,
}

# Hard per-chunk budget used when the caller doesn't override chunk_size.
DEFAULT_CHUNK_WORDS = 1_200
DEFAULT_OVERLAP_WORDS = 80


# ─────────────────────────────────────────────────────────────────────────────
# Filler / noise patterns
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that are ALWAYS safe to strip (sound cues, HTML entities).
_SAFE_STRIP_PATTERNS: list[str] = [
    r"\[(?:Music|Applause|Laughter|Inaudible|Background noise|Silence|Sound|Noise)[^\]]*\]",
    r"\((?:laughter|applause|music|cheering|clapping|sighing)[^)]*\)",
    r"&amp;",
    r"&#\d+;",
    r"&[a-z]+;",           # any remaining HTML entity
]

# Filler words / phrases that are safe to remove only when they appear as
# isolated discourse markers, not inside longer technical phrases.
# Each pattern is anchored with \b (word boundary) and checked against
# the preservation guard before substitution.
_FILLER_WORD_PATTERNS: list[str] = [
    r"\buh+h?\b",          # uh, uhh, uhhh
    r"\bum+\b",            # um, umm
    r"\bhm+\b",            # hm, hmm
    r"\bhmm+\b",
    r"\byou know\b",
    r"\bkind of\b(?!\s+like)",       # "kind of" as filler, not "kind of like X"
    r"\bsort of\b(?!\s+like)",       # same pattern
    r"\bI mean\b",
    r"\bbasically\b",
    r"\bliterally\b",
    r"\bactually\b(?!\s+\w+\s+not)", # keep "actually X not Y" (negation context)
    r"\bokay so\b",
    r"\bso yeah\b",
    r"\byeah so\b",
    r"\balright\b(?!\s*,)",          # keep "alright," as sentence starter signal
    r"\bright\s*\?",                 # "right?" as filler question
]

# Patterns that identify content we must NEVER remove.
# If a candidate filler match overlaps with any of these, it is kept.
_PRESERVE_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    # Equations / maths
    r"(?:[a-zA-Z]\s*[=+\-*/^]\s*[a-zA-Z0-9])",
    r"\b(?:equals?|plus|minus|times|divided by|modulo|factorial)\b",
    r"\b(?:theorem|lemma|corollary|proof|equation|formula|integral|derivative|limit)\b",
    # Definitions
    r"\b(?:is defined as|is called|refers to|known as|denoted by|denoted as)\b",
    r"\b(?:definition|example|instance|case|scenario|suppose|assume|given that)\b",
    # Negations
    r"\b(?:not|never|no|nor|neither|without|except|unless|but not|rather than)\b",
    # Technical / scientific vocabulary (common stems)
    r"\b(?:algorithm|function|class|method|object|variable|parameter|argument)\b",
    r"\b(?:hypothesis|experiment|data|result|analysis|conclusion|observation)\b",
    r"\b(?:i\.e\.|e\.g\.|etc\.|vs\.)\b",
]]

_COMPILED_SAFE = [re.compile(p, re.IGNORECASE) for p in _SAFE_STRIP_PATTERNS]
_COMPILED_FILLER = [re.compile(p, re.IGNORECASE) for p in _FILLER_WORD_PATTERNS]

# Consecutive-duplicate sentence guard: removes exact adjacent repetitions
# (common in auto-captions where the same phrase appears twice in a row).
_DUP_SEP = re.compile(r"(\b.{10,}\b)\s+\1", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    A contiguous slice of cleaned transcript text.

    Attributes:
        text:         The chunk text (cleaned, overlapping with neighbours).
        chunk_index:  Zero-based position of this chunk in the sequence.
        start_time:   Approximate start time (seconds) of the first segment
                      whose text begins this chunk.  ``None`` if timestamps
                      were not provided.
        end_time:     Approximate end time of the last segment in this chunk.
        word_count:   Number of whitespace-separated words in the chunk.
    """
    text: str
    chunk_index: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())


@dataclass
class ProcessedTranscript:
    """
    The result of the full pre-processing pipeline.

    Attributes:
        cleaned_text:  The full cleaned transcript as a single string.
        chunks:        Ordered list of :class:`TextChunk` objects.
        segment_df:    A list of dicts of the cleaned per-segment data
                       (keys: ``start``, ``duration``, ``end``, ``text_raw``,
                       ``text``, ``word_count``).
                       Empty list when no segments were provided.
        word_count:    Word count of the cleaned full text.
        was_empty:     ``True`` if the input transcript was blank.
        was_short:     ``True`` if the transcript was too short to chunk
                       (returned as a single chunk).
    """
    cleaned_text: str
    chunks: list[TextChunk]
    segment_df: list[dict]
    word_count: int
    was_empty: bool = False
    was_short: bool = False

    @property
    def chunk_texts(self) -> list[str]:
        """Return chunk texts as a plain list of strings (for AI callers)."""
        return [c.text for c in self.chunks]


# ─────────────────────────────────────────────────────────────────────────────
# Core cleaning functions
# ─────────────────────────────────────────────────────────────────────────────

def _is_protected(text: str, match_start: int, match_end: int) -> bool:
    """
    Return True if the region [match_start, match_end) in *text* overlaps
    with any preserve-pattern match.  Used to guard filler removal.
    """
    window = text[max(0, match_start - 40): match_end + 40]
    return any(p.search(window) for p in _PRESERVE_PATTERNS)


def clean_segment_text(text: str) -> str:
    """
    Clean a single transcript segment's text string.

    Operations (in order):
    1. Normalise Unicode to NFC (handles curly quotes, em-dashes, etc.)
    2. Strip always-safe bracketed/HTML noise.
    3. Remove filler words — ONLY when no preserve-pattern overlaps.
    4. Remove consecutive duplicate short phrases.
    5. Collapse whitespace.

    Args:
        text: Raw segment text from the transcript API.

    Returns:
        Cleaned segment text.  May be an empty string if the segment
        contained only noise.
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # 2. Always-safe strip (sound cues, HTML entities)
    for pat in _COMPILED_SAFE:
        text = pat.sub(" ", text)

    # 3. Filler removal with preservation guard
    for pat in _COMPILED_FILLER:
        def _replace(m: re.Match) -> str:
            if _is_protected(text, m.start(), m.end()):
                return m.group(0)   # keep it
            return " "
        text = pat.sub(_replace, text)

    # 4. Consecutive duplicate phrases
    text = _DUP_SEP.sub(r"\1", text)

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_transcript(raw_text: str) -> str:
    """
    Clean a full transcript string (not segment-by-segment).

    Suitable when only the concatenated text is available (no timestamps).
    For segment-level cleaning with timestamp preservation use
    :func:`clean_segments_dataframe`.

    Args:
        raw_text: Raw full transcript text.

    Returns:
        Cleaned transcript string.
    """
    return clean_segment_text(raw_text)


# ─────────────────────────────────────────────────────────────────────────────
# Segment processing (pure Python — no pandas dependency)
# ─────────────────────────────────────────────────────────────────────────────

def build_segment_dataframe(segments: list[dict]) -> list[dict]:
    """
    Convert a list of transcript segment dicts into a list of cleaned
    segment records (replaces the previous pandas DataFrame).

    Expected input format (from ``transcript.get_transcript_with_timestamps``):
    ``[{"text": str, "start": float, "duration": float}, ...]``

    Each returned dict has the keys:
        ``start``, ``duration``, ``end``, ``text_raw``, ``text``, ``word_count``

    Returns an empty list when *segments* is empty or contains only noise.
    """
    if not segments:
        return []

    rows: list[dict] = []
    for seg in segments:
        try:
            start = float(seg.get("start", 0) or 0)
        except (TypeError, ValueError):
            start = 0.0
        try:
            duration = float(seg.get("duration", 0) or 0)
        except (TypeError, ValueError):
            duration = 0.0
        text_raw = str(seg.get("text", "") or "")
        text_clean = clean_segment_text(text_raw)
        rows.append({
            "start":      start,
            "duration":   duration,
            "end":        start + duration,
            "text_raw":   text_raw,
            "text":       text_clean,
            "word_count": len(text_clean.split()) if text_clean else 0,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_chunk_size(chunk_size: int, model: str) -> int:
    """Ensure chunk_size never exceeds the model's word-count budget."""
    limit = MODEL_WORD_LIMITS.get(model, MODEL_WORD_LIMITS["_default"])
    return min(chunk_size, limit)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
    model: str = "_default",
) -> list[str]:
    """
    Split *text* into overlapping word-based chunks.

    Each chunk is at most *chunk_size* words and shares *overlap* words with
    the previous chunk so that sentence context is not lost at boundaries.
    The chunk size is silently clamped to the model's word budget.

    Args:
        text:       Cleaned transcript text.
        chunk_size: Target number of words per chunk (default 1 200).
        overlap:    Number of words repeated at the start of each new chunk
                    (default 80).  Must be less than *chunk_size*.
        model:      Gemini model name — used to clamp *chunk_size* against the
                    known context limit.

    Returns:
        List of chunk strings.  Returns ``[""]`` for blank input and a
        single-element list for short transcripts.
    """
    if not text or not text.strip():
        return [""]

    chunk_size = _clamp_chunk_size(max(1, chunk_size), model)
    overlap    = max(0, min(overlap, chunk_size - 1))

    words = text.split()
    if not words:
        return [""]

    # Short transcript — return as a single chunk, no splitting needed
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    step  = chunk_size - overlap

    while start < len(words):
        chunk_words = words[start: start + chunk_size]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def chunk_with_timestamps(
    df: list[dict],
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
    model: str = "_default",
) -> list[TextChunk]:
    """
    Split a cleaned segment list into :class:`TextChunk` objects that
    carry the approximate timestamp of each chunk's first segment.

    Args:
        df:         List of segment dicts produced by :func:`build_segment_dataframe`.
        chunk_size: Target words per chunk.
        overlap:    Overlap words between consecutive chunks.
        model:      Model name for context-limit clamping.

    Returns:
        List of :class:`TextChunk` objects in order.
    """
    # Filter to non-empty segments
    valid_rows = [r for r in df if r.get("text", "").strip()]
    if not valid_rows:
        return [TextChunk(text="", chunk_index=0, start_time=None, end_time=None)]

    chunk_size = _clamp_chunk_size(max(1, chunk_size), model)
    overlap    = max(0, min(overlap, chunk_size - 1))

    # Build word→segment index map
    word_positions: list[int] = []   # word_positions[i] = row index for word i
    all_words: list[str] = []

    for seg_idx, row in enumerate(df):
        seg_words = row["text"].split()
        all_words.extend(seg_words)
        word_positions.extend([seg_idx] * len(seg_words))

    if not all_words:
        return [TextChunk(text="", chunk_index=0, start_time=None, end_time=None)]

    if len(all_words) <= chunk_size:
        start_row = df[word_positions[0]]
        end_row   = df[word_positions[-1]]
        return [TextChunk(
            text        = " ".join(all_words),
            chunk_index = 0,
            start_time  = float(start_row["start"]),
            end_time    = float(end_row["end"]),
        )]

    chunks: list[TextChunk] = []
    step   = chunk_size - overlap
    start  = 0

    while start < len(all_words):
        end = min(start + chunk_size, len(all_words))
        chunk_words = all_words[start:end]

        first_seg_idx = word_positions[start]
        last_seg_idx  = word_positions[end - 1]

        start_time = float(df[first_seg_idx]["start"])
        end_time   = float(df[last_seg_idx]["end"])

        chunks.append(TextChunk(
            text        = " ".join(chunk_words),
            chunk_index = len(chunks),
            start_time  = start_time,
            end_time    = end_time,
        ))
        start += step

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline entry points
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
    model: str = "_default",
) -> tuple[str, list[str]]:
    """
    Clean a raw transcript string and return a (cleaned_text, chunks) tuple.

    This is the **backward-compatible entry point** used by ``app.py``'s
    pipeline step 2.  It accepts a plain string and returns plain strings —
    no DataFrames or timestamps are involved.

    Args:
        text:       Raw transcript string (from ``transcript.fetch_transcript``).
        chunk_size: Words per chunk.
        overlap:    Overlap words between consecutive chunks.
        model:      Gemini model name for context-limit clamping.

    Returns:
        ``(cleaned_text, list_of_chunk_strings)``
    """
    cleaned = clean_transcript(text)
    if not cleaned:
        return ("", [""])
    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap, model=model)
    return cleaned, chunks


def preprocess_segments(
    segments: list[dict],
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
    model: str = "_default",
) -> ProcessedTranscript:
    """
    Full pipeline that accepts timestamped segment dicts and returns a
    :class:`ProcessedTranscript` with typed chunks and metadata flags.

    Args:
        segments:   ``list[{"text": str, "start": float, "duration": float}]``
                    as returned by ``transcript.get_transcript_with_timestamps``.
        chunk_size: Words per chunk.
        overlap:    Overlap words between consecutive chunks.
        model:      Gemini model name for context-limit clamping.

    Returns:
        A :class:`ProcessedTranscript` instance.
    """
    # Empty transcript guard
    if not segments:
        return ProcessedTranscript(
            cleaned_text = "",
            chunks       = [TextChunk(text="", chunk_index=0)],
            segment_df   = [],
            word_count   = 0,
            was_empty    = True,
        )

    rows = build_segment_dataframe(segments)

    # Cleaned full text (join non-empty segment texts)
    cleaned_text = " ".join(r["text"] for r in rows if r["text"])
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    if not cleaned_text:
        return ProcessedTranscript(
            cleaned_text = "",
            chunks       = [TextChunk(text="", chunk_index=0)],
            segment_df   = rows,
            word_count   = 0,
            was_empty    = True,
        )

    word_count = len(cleaned_text.split())
    chunks = chunk_with_timestamps(rows, chunk_size=chunk_size, overlap=overlap, model=model)

    # Short-transcript flag: single chunk, no splitting occurred
    was_short = len(chunks) == 1 and word_count > 0

    return ProcessedTranscript(
        cleaned_text = cleaned_text,
        chunks       = chunks,
        segment_df   = rows,
        word_count   = word_count,
        was_short    = was_short,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit progress helper
# ─────────────────────────────────────────────────────────────────────────────

def report_processing_stats(result: ProcessedTranscript) -> None:
    """
    Display a concise processing summary inside a Streamlit progress step.

    Imports Streamlit lazily so this module can be used outside Streamlit
    (tests, CLI) without error.

    Args:
        result: A :class:`ProcessedTranscript` returned by
                :func:`preprocess_segments`.
    """
    import streamlit as st  # lazy import

    if result.was_empty:
        st.warning("⚠️ The transcript appears to be empty — nothing to process.")
        return

    if result.was_short:
        st.info(
            f"ℹ️ Short transcript ({result.word_count} words) — "
            "processed as a single chunk."
        )
    else:
        st.caption(
            f"⚙️ Cleaned to **{result.word_count:,}** words · "
            f"Split into **{len(result.chunks)}** chunks · "
            f"Segment table: **{len(result.segment_df)}** rows"
        )
