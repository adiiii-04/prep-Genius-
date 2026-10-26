# test_summarizer.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for summarizer.py
#
# Run with:
#   & "C:\Users\Aditya\AppData\Local\Python\bin\python.exe" -m pytest test_summarizer.py -v
#
# Offline-safe: All LLM calls and API responses are mocked.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import openai
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from text_processing import TextChunk
from summarizer import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    SummarizerAPIError,
    SummarizerConfigError,
    SummarizerError,
    SummarizerRateLimitError,
    _get_format_instructions,
    _invoke_with_retry,
    extract_chunk_info,
    format_seconds,
    generate_revision_notes,
    get_chat_model,
    merge_summaries,
    partition_oversized_text,
    summarise_chunk,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Configuration & Model Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfiguration:
    """Test model configuration through environment variables and arguments."""

    def test_missing_api_key_raises_config_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SummarizerConfigError) as exc_info:
                get_chat_model(api_key=None)
            assert "OPENAI_API_KEY is not set" in str(exc_info.value)
            # Also should be an EnvironmentError subclass
            assert isinstance(exc_info.value, EnvironmentError)

    def test_explicit_api_key_used(self):
        with patch.dict(os.environ, {}, clear=True):
            model = get_chat_model(api_key="sk-test-explicit-key")
            assert model.openai_api_key.get_secret_value() == "sk-test-explicit-key"

    def test_default_model_and_temperature(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}, clear=True):
            model = get_chat_model()
            assert model.model_name == DEFAULT_MODEL
            assert model.temperature == DEFAULT_TEMPERATURE

    def test_env_model_and_temperature_overrides(self):
        env_vars = {
            "OPENAI_API_KEY": "sk-env-key",
            "OPENAI_MODEL": "gpt-4o",
            "OPENAI_TEMPERATURE": "0.7",
            "OPENAI_BASE_URL": "https://api.custom-ai.com/v1",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            model = get_chat_model()
            assert model.model_name == "gpt-4o"
            assert abs(model.temperature - 0.7) < 1e-6
            assert str(model.openai_api_base) == "https://api.custom-ai.com/v1"

    def test_explicit_arguments_override_env(self):
        env_vars = {
            "OPENAI_API_KEY": "sk-env-key",
            "OPENAI_MODEL": "gpt-3.5-turbo",
            "OPENAI_TEMPERATURE": "0.8",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            model = get_chat_model(
                api_key="sk-explicit",
                model_name="gpt-4o-mini",
                temperature=0.1,
                base_url="http://localhost:11434/v1",
            )
            assert model.openai_api_key.get_secret_value() == "sk-explicit"
            assert model.model_name == "gpt-4o-mini"
            assert abs(model.temperature - 0.1) < 1e-6
            assert str(model.openai_api_base) == "http://localhost:11434/v1"

    def test_temperature_clamped_between_0_and_2(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key"}, clear=True):
            m_low = get_chat_model(temperature=-0.5)
            assert m_low.temperature == 0.0

            m_high = get_chat_model(temperature=3.5)
            assert m_high.temperature == 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    """Test chunk text/metadata extraction and text partitioning."""

    def test_format_seconds_standard(self):
        assert format_seconds(0) == "00:00"
        assert format_seconds(75) == "01:15"
        assert format_seconds(3665) == "01:01:05"

    def test_format_seconds_none_returns_empty(self):
        assert format_seconds(None) == ""

    def test_extract_chunk_info_text_chunk(self):
        chunk = TextChunk(
            text="Clean transcript text here",
            chunk_index=2,
            start_time=135.0,
            end_time=210.0,
        )
        text, ref = extract_chunk_info(chunk)
        assert text == "Clean transcript text here"
        assert ref == "[02:15 - 03:30]"

    def test_extract_chunk_info_text_chunk_without_end_time(self):
        chunk = TextChunk(
            text="Clean transcript text",
            chunk_index=0,
            start_time=45.0,
            end_time=None,
        )
        text, ref = extract_chunk_info(chunk)
        assert text == "Clean transcript text"
        assert ref == "[00:45]"

    def test_extract_chunk_info_dict(self):
        d = {"text": "Segment text", "start": 60.0, "end": 120.0}
        text, ref = extract_chunk_info(d)
        assert text == "Segment text"
        assert ref == "[01:00 - 02:00]"

    def test_extract_chunk_info_string_with_existing_timestamp(self):
        s = "[04:20 - 06:15] Discussing binary search trees"
        text, ref = extract_chunk_info(s)
        assert "Discussing binary search trees" in text
        assert ref == "[04:20 - 06:15]"

    def test_extract_chunk_info_string_default_index(self):
        s = "Plain text without timestamp"
        text, ref = extract_chunk_info(s, default_index=4)
        assert text == "Plain text without timestamp"
        assert ref == "[Segment 5]"

    def test_partition_oversized_text_not_exceeding_limit(self):
        text = "word " * 500
        parts = partition_oversized_text(text, max_words=1000)
        assert len(parts) == 1
        assert parts[0] == text.strip()

    def test_partition_oversized_text_exceeding_limit(self):
        text = " ".join([f"word{i}" for i in range(1200)])
        parts = partition_oversized_text(text, max_words=500, subchunk_words=300, overlap=50)
        assert len(parts) >= 3
        # Check words are covered
        assert "word0" in parts[0]
        assert "word1199" in parts[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Format Selection (Concise vs Detailed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatInstructions:
    def test_concise_instructions(self):
        instr = _get_format_instructions("concise")
        assert "CONCISE" in instr
        assert "bullet points" in instr.lower()

    def test_detailed_instructions(self):
        instr = _get_format_instructions("detailed")
        assert "DETAILED" in instr
        assert "comprehensive" in instr.lower()

    def test_default_is_detailed(self):
        instr = _get_format_instructions("anything_else")
        assert "DETAILED" in instr


# ═══════════════════════════════════════════════════════════════════════════════
# 4. summarise_chunk
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummariseChunk:
    """Test single chunk summarisation with mocked chat models."""

    def test_empty_chunk_returns_empty_string(self):
        mock_model = FakeListChatModel(responses=["Not called"])
        result = summarise_chunk("", model=mock_model)
        assert result == ""

    def test_summarise_chunk_invokes_model(self):
        expected_output = "## Key Topics\n- **Algorithm**: O(n log n) sort [01:20]"
        mock_model = FakeListChatModel(responses=[expected_output])

        chunk = TextChunk(
            text="We discuss merge sort which runs in O(n log n).",
            chunk_index=0,
            start_time=80.0,
            end_time=120.0,
        )

        result = summarise_chunk(chunk, model=mock_model, format_type="detailed")
        assert result == expected_output

    def test_summarise_chunk_concise_mode(self):
        expected_output = "- Quick sort: average O(n log n)"
        mock_model = FakeListChatModel(responses=[expected_output])

        result = summarise_chunk(
            "Quick sort analysis",
            model=mock_model,
            format_type="concise",
            timestamp_ref="[00:15]",
        )
        assert result == expected_output

    def test_summarise_chunk_oversized_subchunking(self):
        # 800 words with max_words=300 triggers subchunking
        oversized_text = " ".join([f"lecture{i}" for i in range(800)])
        mock_model = FakeListChatModel(responses=[
            "Subchunk 1 summary",
            "Subchunk 2 summary",
            "Subchunk 3 summary",
            "Subchunk 4 summary",
        ])

        with patch("summarizer.MAX_CHUNK_WORDS", 300), \
             patch("summarizer.SUBCHUNK_WORDS", 250), \
             patch("summarizer.SUBCHUNK_OVERLAP", 20):
            result = summarise_chunk(oversized_text, model=mock_model)

        assert "Subchunk" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. merge_summaries (Combining & 6 Standard Sections)
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_MASTER_NOTES = """## 1. Lecture Overview
This lecture explores algorithmic complexity and recursion principles.

## 2. Key Concepts
- Recursion requires a base case and inductive step [02:15].
- Divide and conquer partitions problems into subproblems [05:40].

## 3. Definitions
- **Recursion**: A technique where a function calls itself.
- **Base Case**: The condition terminating recursive calls.

## 4. Important Formulas
- Master Theorem: T(n) = aT(n/b) + f(n)
- Merge Sort complexity: O(n log n)

## 5. Examples & Case Studies
- Binary search on sorted arrays demonstrating logarithmic time [08:10].
- [Uncertain / Missing from audio: Inaudible student question at 11:20].

## 6. Final Revision Checklist
- [ ] Define the base case requirement for recursive functions
- [ ] State and apply the Master Theorem
- [ ] Calculate the recurrence relation for merge sort
"""

class TestMergeSummaries:
    """Test combining chunk summaries into the cohesive 6-section guide."""

    def test_empty_summaries_list_returns_default_sections(self):
        result = merge_summaries([])
        assert "## 1. Lecture Overview" in result
        assert "## 2. Key Concepts" in result
        assert "## 3. Definitions" in result
        assert "## 4. Important Formulas" in result
        assert "## 5. Examples & Case Studies" in result
        assert "## 6. Final Revision Checklist" in result

    def test_merge_summaries_combines_and_preserves_sections(self):
        mock_model = FakeListChatModel(responses=[SAMPLE_MASTER_NOTES])
        chunk_notes = [
            "Part 1: Recursion basics [02:15]",
            "Part 2: Master Theorem and complexities [05:40]",
        ]

        result = merge_summaries(chunk_notes, model=mock_model, format_type="detailed")
        assert "## 1. Lecture Overview" in result
        assert "## 2. Key Concepts" in result
        assert "## 3. Definitions" in result
        assert "## 4. Important Formulas" in result
        assert "## 5. Examples & Case Studies" in result
        assert "## 6. Final Revision Checklist" in result
        assert "[Uncertain / Missing from audio" in result
        assert "[02:15]" in result

    def test_merge_summaries_preserves_no_formula_statement(self):
        notes_without_formulas = SAMPLE_MASTER_NOTES.replace(
            "Master Theorem: T(n) = aT(n/b) + f(n)\n- Merge Sort complexity: O(n log n)",
            "*No mathematical formulas were introduced in this lecture.*"
        )
        mock_model = FakeListChatModel(responses=[notes_without_formulas])
        result = merge_summaries(["chunk 1"], model=mock_model)
        assert "*No mathematical formulas were introduced in this lecture.*" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Error Handling, Rate Limits & Retries
# ═══════════════════════════════════════════════════════════════════════════════

class TestResilienceAndRetries:
    """Test exponential backoff and error handling for API failures and rate limits."""

    def test_rate_limit_retry_succeeds_on_second_attempt(self):
        mock_chain = MagicMock()
        rate_limit_err = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_chain.invoke.side_effect = [
            rate_limit_err,
            AIMessage(content="Recovered after rate limit"),
        ]

        with patch("time.sleep") as mock_sleep:
            result = _invoke_with_retry(mock_chain, {"input": "test"}, max_attempts=3)
            assert result == "Recovered after rate limit"
            assert mock_sleep.called

    def test_rate_limit_exhausted_raises_summarizer_rate_limit_error(self):
        mock_chain = MagicMock()
        rate_limit_err = openai.RateLimitError(
            message="Rate limit reached",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_chain.invoke.side_effect = rate_limit_err

        with patch("time.sleep"):
            with pytest.raises(SummarizerRateLimitError) as exc_info:
                _invoke_with_retry(mock_chain, {"input": "test"}, max_attempts=2)
            assert "rate limit exceeded" in str(exc_info.value).lower()

    def test_timeout_and_connection_error_raises_summarizer_api_error(self):
        mock_chain = MagicMock()
        timeout_err = openai.APITimeoutError(request=MagicMock())
        mock_chain.invoke.side_effect = timeout_err

        with patch("time.sleep"):
            with pytest.raises(SummarizerAPIError) as exc_info:
                _invoke_with_retry(mock_chain, {"input": "test"}, max_attempts=2)
            assert "network or server error" in str(exc_info.value).lower()

    def test_unexpected_exception_wrapped_in_summarizer_error(self):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = TypeError("Unexpected data type")

        with pytest.raises(SummarizerError):
            _invoke_with_retry(mock_chain, {"input": "test"}, max_attempts=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Full Pipeline (generate_revision_notes)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateRevisionNotesPipeline:
    """Test end-to-end notes generation pipeline with mock models."""

    def test_empty_chunks_returns_default_structure(self):
        result = generate_revision_notes([])
        assert "## 1. Lecture Overview" in result
        assert "## 6. Final Revision Checklist" in result

    def test_single_chunk_flow(self):
        # 1 chunk runs chunk summary then merge
        mock_model = FakeListChatModel(responses=[
            "Single chunk draft notes [00:10]",
            SAMPLE_MASTER_NOTES,
        ])

        progress_calls = []
        def _prog(cur, tot):
            progress_calls.append((cur, tot))

        chunk = TextChunk("Single short lecture chunk", chunk_index=0, start_time=10.0, end_time=60.0)
        result = generate_revision_notes(
            [chunk],
            model=mock_model,
            format_type="detailed",
            show_progress=_prog,
        )

        assert "## 1. Lecture Overview" in result
        assert "## 4. Important Formulas" in result
        assert progress_calls == [(1, 1)]

    def test_multi_chunk_parallel_flow(self):
        # 3 chunks: 3 chunk summaries + 1 merge summary = 4 responses
        mock_model = FakeListChatModel(responses=[
            "Summary chunk 1 [00:00 - 02:00]",
            "Summary chunk 2 [02:00 - 04:00]",
            "Summary chunk 3 [04:00 - 06:00]",
            SAMPLE_MASTER_NOTES,
        ])

        chunks = [
            TextChunk(f"Chunk text {i}", chunk_index=i, start_time=i * 120.0, end_time=(i + 1) * 120.0)
            for i in range(3)
        ]

        progress_calls = []
        def _prog(cur, tot):
            progress_calls.append((cur, tot))

        result = generate_revision_notes(
            chunks,
            model=mock_model,
            format_type="concise",
            show_progress=_prog,
            max_workers=2,
        )

        assert "## 1. Lecture Overview" in result
        assert "## 6. Final Revision Checklist" in result
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    def test_string_chunks_supported(self):
        mock_model = FakeListChatModel(responses=[
            "Summary string 1",
            "Summary string 2",
            SAMPLE_MASTER_NOTES,
        ])
        result = generate_revision_notes(
            ["[01:00] First segment", "[03:00] Second segment"],
            model=mock_model,
        )
        assert "## 1. Lecture Overview" in result
        assert "## 3. Definitions" in result
