# test_text_processing.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for text_processing.py
#
# Run with:   py -m pytest test_text_processing.py -v
#
# All tests are offline-safe (no network calls, no Streamlit).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from text_processing import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    MODEL_WORD_LIMITS,
    ProcessedTranscript,
    TextChunk,
    build_segment_dataframe,
    chunk_text,
    chunk_with_timestamps,
    clean_segment_text,
    clean_transcript,
    preprocess,
    preprocess_segments,
    _clamp_chunk_size,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _seg(text: str, start: float = 0.0, duration: float = 2.0) -> dict:
    return {"text": text, "start": start, "duration": duration}


def _segs(*texts: str) -> list[dict]:
    """Build a list of consecutive 2-second segments."""
    return [_seg(t, i * 2.0) for i, t in enumerate(texts)]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  clean_segment_text — basic cleaning
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanSegmentText:

    # ── Noise removal ──────────────────────────────────────────────────────────

    def test_removes_music_annotation(self):
        assert "[Music]" not in clean_segment_text("[Music] Hello world")

    def test_removes_applause_annotation(self):
        assert "[Applause]" not in clean_segment_text("[Applause] Great answer")

    def test_removes_laughter_parens(self):
        result = clean_segment_text("That's funny (laughter) right?")
        assert "(laughter)" not in result

    def test_removes_html_amp_entity(self):
        result = clean_segment_text("cats &amp; dogs")
        assert "&amp;" not in result

    def test_removes_numeric_html_entity(self):
        result = clean_segment_text("value &#39; is here")
        assert "&#39;" not in result

    def test_removes_named_html_entity(self):
        result = clean_segment_text("a &lt; b")
        assert "&lt;" not in result

    # ── Filler word removal ────────────────────────────────────────────────────

    def test_removes_um(self):
        result = clean_segment_text("um we need to start")
        assert "um" not in result.lower()

    def test_removes_uh(self):
        result = clean_segment_text("uh this is important")
        assert result.lower().startswith("this") or "uh" not in result.lower()

    def test_removes_you_know(self):
        result = clean_segment_text("you know the answer")
        # "you know" filler removed, content preserved
        assert "answer" in result

    def test_removes_i_mean(self):
        result = clean_segment_text("I mean the function returns")
        assert "function" in result

    def test_removes_basically(self):
        result = clean_segment_text("basically the loop runs twice")
        assert "loop" in result

    # ── Whitespace normalisation ───────────────────────────────────────────────

    def test_collapses_multiple_spaces(self):
        result = clean_segment_text("hello    world   here")
        assert "  " not in result

    def test_strips_leading_trailing(self):
        result = clean_segment_text("   hello world   ")
        assert result == result.strip()

    def test_empty_string_returns_empty(self):
        assert clean_segment_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_segment_text("   ") == ""

    def test_noise_only_returns_empty_or_short(self):
        result = clean_segment_text("[Music] (applause) uh um")
        # After stripping all noise, result should be empty or just whitespace
        assert result.strip() == ""

    # ── Consecutive duplicate removal ─────────────────────────────────────────

    def test_removes_exact_duplicate_phrase(self):
        result = clean_segment_text(
            "the algorithm runs in O of n the algorithm runs in O of n"
        )
        # Duplicate should be collapsed
        count = result.lower().count("algorithm")
        assert count <= 1

    # ── Unicode normalisation ─────────────────────────────────────────────────

    def test_normalises_unicode(self):
        # Curly apostrophe → should survive
        result = clean_segment_text("it\u2019s a function")
        assert "function" in result

    def test_handles_em_dash(self):
        result = clean_segment_text("the result\u2014it matters")
        assert "result" in result
        assert "matters" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Content preservation — technical terms, negations, definitions
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreservation:

    def test_preserves_negation_not(self):
        text = "the function does not return a value"
        result = clean_segment_text(text)
        assert "not" in result

    def test_preserves_negation_never(self):
        result = clean_segment_text("you should never call this method")
        assert "never" in result

    def test_preserves_definition_phrase(self):
        result = clean_segment_text("a recursion is defined as calling itself")
        assert "defined as" in result or "defined" in result

    def test_preserves_theorem_keyword(self):
        result = clean_segment_text("the theorem states that P equals NP")
        assert "theorem" in result

    def test_preserves_equation_letters(self):
        result = clean_segment_text("so we have a equals b plus c")
        assert "equals" in result

    def test_preserves_algorithm_term(self):
        result = clean_segment_text("the algorithm runs in linear time")
        assert "algorithm" in result

    def test_preserves_example_keyword(self):
        result = clean_segment_text("for example suppose x is greater than zero")
        assert "example" in result

    def test_preserves_hypothesis_keyword(self):
        result = clean_segment_text("our hypothesis is that the data fits")
        assert "hypothesis" in result

    def test_preserves_ie_abbreviation(self):
        result = clean_segment_text("use a hash table i.e. a dictionary")
        assert "i.e." in result

    def test_preserves_eg_abbreviation(self):
        result = clean_segment_text("e.g. Python Java or C++")
        assert "e.g." in result

    def test_preserves_unless_negation(self):
        result = clean_segment_text("don't do this unless the flag is set")
        assert "unless" in result

    def test_preserves_except_negation(self):
        result = clean_segment_text("all values except zero are valid")
        assert "except" in result

    def test_filler_inside_technical_context_kept(self):
        # "kind of" is a filler, but surrounded by a definition phrase —
        # even if it is removed, the surrounding technical words must survive.
        text = "this is kind of like a binary search"
        result = clean_segment_text(text)
        assert "binary" in result
        assert "search" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  clean_transcript — full-text cleaning
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanTranscript:

    def test_returns_string(self):
        result = clean_transcript("hello world")
        assert isinstance(result, str)

    def test_removes_noise(self):
        result = clean_transcript("[Music] welcome to the lecture")
        assert "[Music]" not in result
        assert "welcome" in result

    def test_empty_input(self):
        assert clean_transcript("") == ""

    def test_full_noise_input(self):
        result = clean_transcript("[Music] [Applause] (laughter)")
        assert result.strip() == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  build_segment_dataframe — Segment list integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildSegmentDataframe:

    @pytest.fixture
    def sample_df(self):
        segs = [
            _seg("Hello world", 0.0, 2.5),
            _seg("[Music] um", 2.5, 1.0),          # should clean to empty
            _seg("This is a theorem", 3.5, 3.0),
        ]
        return build_segment_dataframe(segs)

    def test_returns_list_of_dicts(self, sample_df):
        assert isinstance(sample_df, list)
        assert all(isinstance(r, dict) for r in sample_df)

    def test_required_columns_present(self, sample_df):
        for col in ("start", "duration", "end", "text_raw", "text", "word_count"):
            for row in sample_df:
                assert col in row

    def test_row_count_matches_input(self, sample_df):
        assert len(sample_df) == 3

    def test_end_equals_start_plus_duration(self, sample_df):
        for row in sample_df:
            assert abs(row["end"] - (row["start"] + row["duration"])) < 1e-9

    def test_noise_segment_text_is_empty(self, sample_df):
        # Second segment ([Music] um) should clean to ""
        assert sample_df[1]["text"].strip() == ""

    def test_word_count_for_noise_segment_is_zero(self, sample_df):
        assert sample_df[1]["word_count"] == 0

    def test_text_raw_preserved(self, sample_df):
        assert sample_df[0]["text_raw"] == "Hello world"

    def test_technical_content_preserved_in_text(self, sample_df):
        assert "theorem" in sample_df[2]["text"]

    def test_empty_segments_returns_empty_list(self):
        df = build_segment_dataframe([])
        assert isinstance(df, list)
        assert len(df) == 0

    def test_missing_duration_defaults_to_zero(self):
        segs = [{"text": "hello", "start": 1.0}]
        df = build_segment_dataframe(segs)
        assert df[0]["duration"] == 0.0

    def test_missing_start_defaults_to_zero(self):
        segs = [{"text": "hello", "duration": 2.0}]
        df = build_segment_dataframe(segs)
        assert df[0]["start"] == 0.0

    def test_numeric_coercion_for_string_values(self):
        segs = [{"text": "hi", "start": "1.5", "duration": "2.0"}]
        df = build_segment_dataframe(segs)
        assert df[0]["start"] == 1.5
        assert df[0]["duration"] == 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  chunk_text — splitting and overlap
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkText:

    def _make_text(self, n_words: int) -> str:
        return " ".join(f"word{i}" for i in range(n_words))

    def test_returns_list_of_strings(self):
        result = chunk_text("hello world how are you", chunk_size=3, overlap=1)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_empty_text_returns_single_empty_chunk(self):
        assert chunk_text("") == [""]

    def test_whitespace_only_returns_single_empty_chunk(self):
        assert chunk_text("   ") == [""]

    def test_short_text_is_single_chunk(self):
        result = chunk_text("hello world", chunk_size=100, overlap=10)
        assert len(result) == 1
        assert "hello" in result[0]

    def test_chunk_count_correct(self):
        # 100 words, chunk=30, overlap=10, step=20 → ceil((100-30)/20)+1 = 5
        text = self._make_text(100)
        result = chunk_text(text, chunk_size=30, overlap=10)
        # Each chunk except possibly the last has chunk_size words
        assert len(result) >= 4

    def test_first_chunk_has_correct_size(self):
        text = self._make_text(200)
        result = chunk_text(text, chunk_size=50, overlap=10)
        assert len(result[0].split()) == 50

    def test_overlap_shared_words(self):
        text = self._make_text(60)
        result = chunk_text(text, chunk_size=30, overlap=10)
        if len(result) >= 2:
            tail = result[0].split()[-10:]
            head = result[1].split()[:10]
            assert tail == head, "Overlap words should be shared between chunks"

    def test_all_words_covered(self):
        # Verify every word from the original appears in at least one chunk
        text = self._make_text(85)
        result = chunk_text(text, chunk_size=30, overlap=5)
        all_chunk_words = set(w for c in result for w in c.split())
        assert all(w in all_chunk_words for w in text.split())

    def test_no_empty_chunks(self):
        text = self._make_text(50)
        result = chunk_text(text, chunk_size=20, overlap=5)
        assert all(c.strip() for c in result)

    def test_overlap_zero_no_repetition(self):
        text = self._make_text(40)
        result = chunk_text(text, chunk_size=20, overlap=0)
        assert len(result) == 2
        assert result[0].split()[-1] != result[1].split()[0]

    def test_chunk_size_clamped_to_model_limit(self):
        # Requesting more than the model allows should be silently clamped
        limit = MODEL_WORD_LIMITS["gemini-1.5-flash"]
        text = self._make_text(limit + 100)
        result = chunk_text(text, chunk_size=limit + 10_000, overlap=0,
                            model="gemini-1.5-flash")
        assert all(len(c.split()) <= limit for c in result)

    def test_overlap_clamps_to_chunk_size_minus_one(self):
        # overlap >= chunk_size should not crash
        text = self._make_text(20)
        result = chunk_text(text, chunk_size=5, overlap=100)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_single_word_text(self):
        result = chunk_text("hello", chunk_size=10, overlap=2)
        assert result == ["hello"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  chunk_with_timestamps — TextChunk objects
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkWithTimestamps:

    @pytest.fixture
    def df_10segs(self):
        segs = [_seg(f"word{i} content here", i * 3.0, 3.0) for i in range(10)]
        return build_segment_dataframe(segs)

    def test_returns_list_of_text_chunks(self, df_10segs):
        result = chunk_with_timestamps(df_10segs, chunk_size=10, overlap=2)
        assert isinstance(result, list)
        assert all(isinstance(c, TextChunk) for c in result)

    def test_chunk_indices_are_sequential(self, df_10segs):
        result = chunk_with_timestamps(df_10segs, chunk_size=10, overlap=2)
        for i, c in enumerate(result):
            assert c.chunk_index == i

    def test_start_time_is_float(self, df_10segs):
        result = chunk_with_timestamps(df_10segs, chunk_size=10, overlap=2)
        for c in result:
            assert isinstance(c.start_time, float)

    def test_end_time_is_float(self, df_10segs):
        result = chunk_with_timestamps(df_10segs, chunk_size=10, overlap=2)
        for c in result:
            assert isinstance(c.end_time, float)

    def test_start_time_non_negative(self, df_10segs):
        result = chunk_with_timestamps(df_10segs, chunk_size=10, overlap=2)
        assert result[0].start_time >= 0.0

    def test_word_count_populated(self, df_10segs):
        result = chunk_with_timestamps(df_10segs, chunk_size=10, overlap=2)
        for c in result:
            assert c.word_count > 0

    def test_empty_df_returns_empty_chunk(self):
        df = build_segment_dataframe([])
        result = chunk_with_timestamps(df)
        assert result[0].text == ""

    def test_all_noise_df_returns_empty_chunk(self):
        segs = [_seg("[Music]", 0.0), _seg("um uh", 2.0)]
        df = build_segment_dataframe(segs)
        result = chunk_with_timestamps(df)
        assert result[0].text.strip() == ""

    def test_short_transcript_single_chunk(self):
        segs = [_seg("hello world", 0.0, 2.0)]
        df = build_segment_dataframe(segs)
        result = chunk_with_timestamps(df, chunk_size=100)
        assert len(result) == 1

    def test_text_chunk_chunk_texts_property(self, df_10segs):
        result = chunk_with_timestamps(df_10segs, chunk_size=10, overlap=2)
        # Wrap in ProcessedTranscript to test chunk_texts property
        pt = ProcessedTranscript(
            cleaned_text="x",
            chunks=result,
            segment_df=df_10segs,
            word_count=30,
        )
        assert isinstance(pt.chunk_texts, list)
        assert all(isinstance(t, str) for t in pt.chunk_texts)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  preprocess — backward-compatible entry point
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocess:

    def test_returns_tuple(self):
        result = preprocess("hello world how are you today sir")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_string(self):
        cleaned, _ = preprocess("hello world")
        assert isinstance(cleaned, str)

    def test_second_element_is_list_of_strings(self):
        _, chunks = preprocess("hello world")
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_empty_returns_empty_chunk(self):
        cleaned, chunks = preprocess("")
        assert cleaned == ""
        assert chunks == [""]

    def test_filler_removed_from_cleaned(self):
        cleaned, _ = preprocess("um so basically hello world")
        assert "um" not in cleaned
        assert "hello" in cleaned

    def test_chunk_size_respected(self):
        text = " ".join(["word"] * 300)
        _, chunks = preprocess(text, chunk_size=100, overlap=10)
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_model_kwarg_accepted(self):
        text = " ".join(["word"] * 50)
        cleaned, chunks = preprocess(text, model="gemini-1.5-flash")
        assert isinstance(chunks, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  preprocess_segments — rich pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessSegments:

    @pytest.fixture
    def sample_segs(self):
        return [
            _seg("Hello this is a lecture about algorithms", 0.0, 3.0),
            _seg("[Music] um uh", 3.0, 1.5),
            _seg("The algorithm is defined as a step-by-step procedure", 4.5, 4.0),
            _seg("We never skip the base case", 8.5, 3.0),
            _seg("For example suppose n equals ten", 11.5, 3.5),
        ]

    def test_returns_processed_transcript(self, sample_segs):
        result = preprocess_segments(sample_segs)
        assert isinstance(result, ProcessedTranscript)

    def test_cleaned_text_is_string(self, sample_segs):
        result = preprocess_segments(sample_segs)
        assert isinstance(result.cleaned_text, str)

    def test_word_count_positive(self, sample_segs):
        result = preprocess_segments(sample_segs)
        assert result.word_count > 0

    def test_segment_df_is_list(self, sample_segs):
        result = preprocess_segments(sample_segs)
        assert isinstance(result.segment_df, list)

    def test_segment_df_has_correct_row_count(self, sample_segs):
        result = preprocess_segments(sample_segs)
        assert len(result.segment_df) == len(sample_segs)

    def test_chunks_list_of_text_chunks(self, sample_segs):
        result = preprocess_segments(sample_segs)
        assert isinstance(result.chunks, list)
        assert all(isinstance(c, TextChunk) for c in result.chunks)

    def test_chunk_texts_property(self, sample_segs):
        result = preprocess_segments(sample_segs)
        texts = result.chunk_texts
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)

    def test_was_empty_false_for_real_content(self, sample_segs):
        result = preprocess_segments(sample_segs)
        assert result.was_empty is False

    def test_technical_content_in_cleaned_text(self, sample_segs):
        result = preprocess_segments(sample_segs)
        assert "algorithm" in result.cleaned_text
        assert "never" in result.cleaned_text
        assert "example" in result.cleaned_text

    def test_empty_segments_returns_was_empty_true(self):
        result = preprocess_segments([])
        assert result.was_empty is True
        assert result.cleaned_text == ""
        assert result.word_count == 0

    def test_noise_only_segments_returns_was_empty_true(self):
        segs = [_seg("[Music]"), _seg("um uh"), _seg("[Applause]")]
        result = preprocess_segments(segs)
        assert result.was_empty is True

    def test_single_segment_is_short(self):
        segs = [_seg("hello world this is short", 0.0, 2.0)]
        result = preprocess_segments(segs, chunk_size=1000)
        assert result.was_short is True
        assert len(result.chunks) == 1

    def test_long_transcript_produces_multiple_chunks(self):
        # 600 words total, chunk_size=200, overlap=20 → multiple chunks
        words = " ".join([f"word{i}" for i in range(600)])
        segs = [_seg(words[:200], 0.0, 60.0),
                _seg(words[200:400], 60.0, 60.0),
                _seg(words[400:], 120.0, 60.0)]
        result = preprocess_segments(segs, chunk_size=200, overlap=20)
        assert len(result.chunks) >= 2

    def test_timestamps_attach_to_chunks(self):
        segs = [
            _seg("first part here now", 0.0, 5.0),
            _seg("second part here now", 5.0, 5.0),
            _seg("third part here now", 10.0, 5.0),
        ]
        result = preprocess_segments(segs, chunk_size=4, overlap=0)
        # First chunk must have start_time near 0
        assert result.chunks[0].start_time is not None
        assert result.chunks[0].start_time >= 0.0

    def test_model_parameter_accepted(self, sample_segs):
        result = preprocess_segments(
            sample_segs, chunk_size=500, model="gemini-1.5-pro"
        )
        assert isinstance(result, ProcessedTranscript)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  _clamp_chunk_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestClampChunkSize:

    def test_known_model_clamps(self):
        limit = MODEL_WORD_LIMITS["gemini-1.5-flash"]
        assert _clamp_chunk_size(limit + 5000, "gemini-1.5-flash") == limit

    def test_known_model_below_limit_unchanged(self):
        assert _clamp_chunk_size(500, "gemini-1.5-flash") == 500

    def test_unknown_model_uses_default(self):
        default = MODEL_WORD_LIMITS["_default"]
        assert _clamp_chunk_size(default + 1000, "some-future-model") == default

    def test_zero_chunk_size_clamps_to_one(self):
        # chunk_size is clamped to max(1, chunk_size) before _clamp_chunk_size
        result = _clamp_chunk_size(1, "_default")
        assert result == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TextChunk dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextChunk:

    def test_word_count_auto_computed(self):
        c = TextChunk(text="hello world how are you", chunk_index=0)
        assert c.word_count == 5

    def test_empty_text_word_count_zero(self):
        c = TextChunk(text="", chunk_index=0)
        assert c.word_count == 0

    def test_optional_timestamps_default_none(self):
        c = TextChunk(text="hello", chunk_index=0)
        assert c.start_time is None
        assert c.end_time is None

    def test_timestamps_stored(self):
        c = TextChunk(text="hello", chunk_index=0, start_time=1.5, end_time=3.0)
        assert c.start_time == 1.5
        assert c.end_time == 3.0
