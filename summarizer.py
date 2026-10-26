# summarizer.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Generate structured revision notes from transcript chunks using
#          LangChain and an OpenAI-compatible chat model (default: NVIDIA
#          Nemotron 3.5 via https://integrate.api.nvidia.com/v1), configured
#          via environment variables or direct parameters.
#
# Requirements satisfied:
#   1. Generate notes from transcript chunks.
#   2. Combine chunk summaries into a coherent set of final revision notes.
#   3. Standard sections: Lecture Overview, Key Concepts, Definitions,
#      Important Formulas, Examples & Case Studies, Final Revision Checklist.
#   4. Preserve technical accuracy; avoid inventing or extrapolating information.
#   5. Preserve timestamps and segment references where available.
#   6. Clearly mark uncertain or missing information: [Uncertain / Missing: ...].
#   7. Handle API failures, rate limits, and oversized inputs gracefully.
#   8. Selectable format: concise vs. detailed.
#
# Environment variables (priority order):
#   NVIDIA_API_KEY   — NVIDIA / Nemotron API key (preferred)
#   OPENAI_API_KEY   — Fallback (works with custom base URL)
#   NEMOTRON_MODEL   — Model identifier (preferred)
#   OPENAI_MODEL     — Fallback model
#   NVIDIA_BASE_URL  — API endpoint (preferred, default: NVIDIA endpoint)
#   OPENAI_BASE_URL  — Fallback endpoint
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import time
import random
from typing import Any, Callable, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_FORMAT = "detailed"
MAX_CHUNK_WORDS = 3500
SUBCHUNK_WORDS = 2000
SUBCHUNK_OVERLAP = 100


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class SummarizerError(Exception):
    """Base exception for all summarizer operations."""
    pass


class SummarizerConfigError(SummarizerError, EnvironmentError):
    """Raised when mandatory configuration (such as NVIDIA_API_KEY) is missing."""
    pass


class SummarizerAPIError(SummarizerError, RuntimeError):
    """Raised when an API call fails after retrying."""
    pass


class SummarizerRateLimitError(SummarizerAPIError):
    """Raised when rate limits are exceeded after retrying."""
    pass


class SummarizerInputError(SummarizerError, ValueError):
    """Raised when input chunks or parameters are invalid."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper Utilities
# ─────────────────────────────────────────────────────────────────────────────

def format_seconds(seconds: Optional[Union[float, int]]) -> str:
    """Format seconds into MM:SS or HH:MM:SS string."""
    if seconds is None:
        return ""
    total = max(0, int(round(float(seconds))))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def extract_chunk_info(chunk: Any, default_index: int = 0) -> tuple[str, str]:
    """
    Extract chunk text and timestamp reference label.

    Handles:
      - Plain string chunks (and extracts existing '[MM:SS]' if present)
      - TextChunk objects (with start_time, end_time, chunk_index, text)
      - Dicts with 'text', 'start'/'start_time', 'end'/'end_time'
    """
    # TextChunk dataclass (from text_processing.py) or object with attributes
    if hasattr(chunk, "text"):
        text = str(chunk.text).strip()
        start = getattr(chunk, "start_time", None)
        end = getattr(chunk, "end_time", None)
        idx = getattr(chunk, "chunk_index", default_index)

        if start is not None:
            start_fmt = format_seconds(start)
            end_fmt = format_seconds(end) if end is not None and end > start else ""
            ref = f"[{start_fmt} - {end_fmt}]" if end_fmt else f"[{start_fmt}]"
        else:
            ref = f"[Segment {idx + 1}]"
        return text, ref

    # Dictionary chunk
    if isinstance(chunk, dict):
        text = str(chunk.get("text", "")).strip()
        start = chunk.get("start_time", chunk.get("start"))
        end = chunk.get("end_time", chunk.get("end"))
        if start is not None:
            start_fmt = format_seconds(start)
            end_fmt = format_seconds(end) if end is not None and float(end) > float(start) else ""
            ref = f"[{start_fmt} - {end_fmt}]" if end_fmt else f"[{start_fmt}]"
        else:
            ref = f"[Segment {default_index + 1}]"
        return text, ref

    # Plain string chunk
    text = str(chunk).strip()
    match = re.match(r"^\[(\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?)\]", text)
    if match:
        ref = match.group(0)
    else:
        ref = f"[Segment {default_index + 1}]"
    return text, ref


def partition_oversized_text(
    text: str,
    max_words: int = MAX_CHUNK_WORDS,
    subchunk_words: int = SUBCHUNK_WORDS,
    overlap: int = SUBCHUNK_OVERLAP,
) -> list[str]:
    """Split an oversized text block into smaller sub-chunks to avoid context limit overflow."""
    text_clean = text.strip()
    words = text_clean.split()
    if len(words) <= max_words:
        return [text_clean]

    subchunks: list[str] = []
    step = max(1, subchunk_words - overlap)
    start = 0
    while start < len(words):
        end = min(start + subchunk_words, len(words))
        subchunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return subchunks


# ─────────────────────────────────────────────────────────────────────────────
# Model Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def get_chat_model(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    request_timeout: float = 120.0,
) -> BaseChatModel:
    """
    Initialise and return an OpenAI-compatible LangChain Chat model.

    Default backend: NVIDIA Nemotron 3.5 (https://integrate.api.nvidia.com/v1).
    Falls back gracefully to OpenAI-compatible env vars for backwards compatibility.

    Configuration hierarchy:
      1. Explicit arguments passed to this function.
      2. Environment variables (NVIDIA preferred, OpenAI as fallback):
         - NVIDIA_API_KEY  → OPENAI_API_KEY
         - NEMOTRON_MODEL  → OPENAI_MODEL / OPENAI_MODEL_NAME
         - NVIDIA_BASE_URL → OPENAI_BASE_URL / OPENAI_API_BASE → NVIDIA default endpoint
         - OPENAI_TEMPERATURE / SUMMARIZER_TEMPERATURE
      3. Built-in defaults (model=nvidia/nemotron-3.5-super-vl-120b-a12b, temperature=0.2).

    Raises:
        SummarizerConfigError: If neither NVIDIA_API_KEY nor OPENAI_API_KEY is set.
    """
    # API Key: prefer NVIDIA, fall back to OpenAI
    resolved_api_key = (
        api_key
        or os.getenv("NVIDIA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not resolved_api_key:
        raise SummarizerConfigError(
            "NVIDIA_API_KEY is not set. "
            "Please provide your NVIDIA API key in the sidebar or your .env file. "
            "Get a free key at https://build.nvidia.com"
        )

    # Model: prefer NEMOTRON_MODEL, fall back to OPENAI_MODEL
    resolved_model = (
        model_name
        or os.getenv("NEMOTRON_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL_NAME")
        or DEFAULT_MODEL
    )

    if temperature is None:
        env_temp = os.getenv("OPENAI_TEMPERATURE") or os.getenv("SUMMARIZER_TEMPERATURE")
        if env_temp is not None:
            try:
                temperature = float(env_temp)
            except ValueError:
                temperature = DEFAULT_TEMPERATURE
        else:
            temperature = DEFAULT_TEMPERATURE

    clamped_temp = max(0.0, min(2.0, float(temperature)))

    # Base URL: prefer NVIDIA, fall back to OPENAI, then default to NVIDIA endpoint
    resolved_base_url = (
        base_url
        or os.getenv("NVIDIA_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or NVIDIA_BASE_URL
    )

    return ChatOpenAI(
        api_key=resolved_api_key,
        model=resolved_model,
        temperature=clamped_temp,
        base_url=resolved_base_url,
        max_retries=max_retries,
        timeout=request_timeout,
    )


def _get_model() -> BaseChatModel:
    """Convenience alias for default model initialisation."""
    return get_chat_model()


# ─────────────────────────────────────────────────────────────────────────────
# Resilient Invocation with Retries & Rate-Limit Backoff
# ─────────────────────────────────────────────────────────────────────────────

def _invoke_with_retry(
    chain_or_model: Any,
    input_data: Any,
    max_attempts: int = 4,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> str:
    """
    Invoke a LangChain runnable with exponential backoff on rate limits
    and transient network errors.
    """
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            result = chain_or_model.invoke(input_data)
            if hasattr(result, "content"):
                return str(result.content).strip()
            return str(result).strip()

        except openai.RateLimitError as exc:
            if attempt == max_attempts:
                raise SummarizerRateLimitError(
                    f"OpenAI API rate limit exceeded after {max_attempts} attempts: {exc}"
                ) from exc
            jitter = random.uniform(0.1, 0.6)
            time.sleep(delay + jitter)
            delay *= backoff_factor

        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
            if attempt == max_attempts:
                raise SummarizerAPIError(
                    f"OpenAI API network or server error after {max_attempts} attempts: {exc}"
                ) from exc
            time.sleep(delay)
            delay *= backoff_factor

        except openai.APIError as exc:
            msg = str(exc).lower()
            if "rate limit" in msg or "429" in msg:
                if attempt == max_attempts:
                    raise SummarizerRateLimitError(f"Rate limit exceeded: {exc}") from exc
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise SummarizerAPIError(f"OpenAI API error: {exc}") from exc

        except Exception as exc:
            msg = str(exc).lower()
            if "rate limit" in msg or "429" in msg:
                if attempt == max_attempts:
                    raise SummarizerRateLimitError(f"Rate limit exceeded: {exc}") from exc
                time.sleep(delay)
                delay *= backoff_factor
            elif "timeout" in msg or "connection" in msg:
                if attempt == max_attempts:
                    raise SummarizerAPIError(f"Connection failure: {exc}") from exc
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise SummarizerError(f"Summarizer failed during execution: {exc}") from exc

    raise SummarizerAPIError("Unexpected termination of retry loop.")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SYSTEM_PROMPT = (
    "You are an expert academic tutor and university-level technical note-taker.\n"
    "Your objective is to read a transcript excerpt from a lecture and produce structured, "
    "technically rigorous revision notes.\n\n"
    "CRITICAL INTEGRITY GUIDELINES:\n"
    "1. PRESERVE TECHNICAL ACCURACY: Preserve exact technical terminology, mathematical expressions, "
    "equations, parameters, proofs, and definitions as stated in the excerpt.\n"
    "2. NO FABRICATION: Do NOT invent, extrapolate, or hallucinate facts, data, citations, or formulas "
    "not present in the transcript.\n"
    "3. MARK UNCERTAIN OR MISSING INFORMATION: If audio cuts off, captions are garbled, reasoning is "
    "truncated, or a concept is ambiguous, explicitly mark it with: "
    "[Uncertain / Missing from audio: <brief description>].\n"
    "4. SOURCE REFERENCES & TIMESTAMPS: Always tie points to the provided segment timestamp reference ({timestamp_ref}).\n"
    "5. FORMAT REQUIREMENT:\n{format_instructions}"
)

CHUNK_HUMAN_PROMPT = (
    "Segment Reference: {timestamp_ref}\n\n"
    "Transcript Excerpt:\n"
    "\"\"\"\n{text}\n\"\"\"\n\n"
    "Generate structured revision notes for this segment."
)

MERGE_SYSTEM_PROMPT = (
    "You are a master academic editor and subject matter expert.\n"
    "Your task is to combine and synthesize per-chunk lecture notes into a single, cohesive, "
    "unified master revision guide.\n\n"
    "REQUIRED DOCUMENT STRUCTURE:\n"
    "You MUST organize the document strictly into the following six major sections using standard Markdown headings:\n\n"
    "## 1. Lecture Overview\n"
    "- A coherent high-level synthesis of the lecture's core themes, context, learning outcomes, and scope.\n\n"
    "## 2. Key Concepts\n"
    "- Core theoretical principles, mechanisms, and models discussed. Retain segment timestamp references "
    "(e.g., [MM:SS] or [MM:SS - MM:SS]) next to key concepts.\n\n"
    "## 3. Definitions\n"
    "- Formal, precise definitions of all technical keywords and domain terminology introduced.\n\n"
    "## 4. Important Formulas\n"
    "- Explicit mathematical equations, algorithms, computational complexities, or scientific laws.\n"
    "- NOTE: If NO mathematical formulas were introduced in this lecture, write exactly:\n"
    "  *No mathematical formulas were introduced in this lecture.*\n\n"
    "## 5. Examples & Case Studies\n"
    "- Real-world examples, code illustrations, diagrams described, or practical walkthroughs provided by the lecturer.\n\n"
    "## 6. Final Revision Checklist\n"
    "- A concise, high-yield checklist with checkable items ('- [ ] ...') summarizing the critical questions "
    "and competencies students must master for revision.\n\n"
    "INTEGRITY & QUALITY RULES:\n"
    "- Preserve technical accuracy; never invent formulas, citations, or concepts.\n"
    "- Preserve all '[Uncertain / Missing from audio: ...]' flags from chunk notes.\n"
    "- Remove duplicate points and synthesize overlapping explanations smoothly.\n"
    "- Adhere to the requested format style:\n{format_instructions}"
)

MERGE_HUMAN_PROMPT = (
    "Below are revision notes extracted from different parts of the lecture:\n\n"
    "{combined_notes}\n\n"
    "Combine and synthesize these into the final master revision guide with all six required sections."
)


def _get_format_instructions(format_type: str) -> str:
    """Return specific prompt instructions based on concise vs detailed format."""
    normalized = str(format_type).strip().lower()
    if normalized == "concise":
        return (
            "- STYLE: CONCISE.\n"
            "- Use tight bullet points, high-yield takeaways, condensed definitions, and direct formulas.\n"
            "- Eliminate conversational filler and verbose prose while keeping all technical facts intact.\n"
            "- Keep checklist focused on the most critical high-yield exam points."
        )
    return (
        "- STYLE: DETAILED.\n"
        "- Provide comprehensive coverage with in-depth explanations, background context, step-by-step logic, "
        "and extensive walkthroughs of examples.\n"
        "- Ensure all nuances, edge cases, and explanations are thoroughly documented for deep study."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core Summarisation Functions
# ─────────────────────────────────────────────────────────────────────────────

def summarise_chunk(
    chunk: Any,
    model: Optional[BaseChatModel] = None,
    format_type: str = DEFAULT_FORMAT,
    chunk_index: int = 0,
    timestamp_ref: Optional[str] = None,
    **model_kwargs: Any,
) -> str:
    """
    Produce revision notes for a single transcript chunk using LangChain.

    Args:
        chunk:         A string, TextChunk object, or segment dict.
        model:         Initialised LangChain chat model. If None, `get_chat_model()` is used.
        format_type:   'concise' or 'detailed' (default: 'detailed').
        chunk_index:   Zero-based index of the chunk.
        timestamp_ref: Optional timestamp string (e.g. '[02:15 - 04:30]').
        **model_kwargs: Passed to `get_chat_model` if model is not provided.

    Returns:
        A Markdown-formatted string of notes for the chunk.
    """
    raw_text, auto_ref = extract_chunk_info(chunk, default_index=chunk_index)
    ref = timestamp_ref or auto_ref

    if not raw_text or not raw_text.strip():
        return ""

    # Check for oversized inputs and partition if necessary
    subchunks = partition_oversized_text(raw_text, max_words=MAX_CHUNK_WORDS)
    if len(subchunks) > 1:
        # Process sub-chunks and combine
        sub_summaries = []
        for i, sub in enumerate(subchunks):
            sub_ref = f"{ref} (Part {i + 1}/{len(subchunks)})"
            sub_summary = summarise_chunk(
                sub,
                model=model,
                format_type=format_type,
                chunk_index=chunk_index,
                timestamp_ref=sub_ref,
                **model_kwargs,
            )
            if sub_summary:
                sub_summaries.append(sub_summary)
        return "\n\n".join(sub_summaries)

    if model is None:
        model = get_chat_model(**model_kwargs)

    format_instructions = _get_format_instructions(format_type)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHUNK_SYSTEM_PROMPT),
        ("human", CHUNK_HUMAN_PROMPT),
    ])

    chain = prompt | model | StrOutputParser()

    input_data = {
        "timestamp_ref": ref,
        "text": raw_text,
        "format_instructions": format_instructions,
    }

    return _invoke_with_retry(chain, input_data)


def merge_summaries(
    summaries: list[str],
    model: Optional[BaseChatModel] = None,
    format_type: str = DEFAULT_FORMAT,
    **model_kwargs: Any,
) -> str:
    """
    Merge a list of per-chunk summaries into a single cohesive master revision guide.

    The final output contains all 6 required sections:
      1. Lecture Overview
      2. Key Concepts (with timestamps)
      3. Definitions
      4. Important Formulas (or explicit notice if none)
      5. Examples & Case Studies
      6. Final Revision Checklist

    Args:
        summaries:    List of Markdown chunk notes.
        model:        Initialised LangChain chat model.
        format_type:  'concise' or 'detailed'.
        **model_kwargs: Passed to `get_chat_model` if model is None.

    Returns:
        Unified Markdown revision notes string.
    """
    valid_summaries = [s.strip() for s in summaries if s and s.strip()]
    if not valid_summaries:
        return (
            "## 1. Lecture Overview\n*No transcript content was available to summarize.*\n\n"
            "## 2. Key Concepts\n- None\n\n"
            "## 3. Definitions\n- None\n\n"
            "## 4. Important Formulas\n*No mathematical formulas were introduced in this lecture.*\n\n"
            "## 5. Examples & Case Studies\n- None\n\n"
            "## 6. Final Revision Checklist\n- [ ] Review lecture audio once available\n"
        )

    if model is None:
        model = get_chat_model(**model_kwargs)

    # If only one valid summary exists and already has standard sections, reuse it directly
    if len(valid_summaries) == 1:
        first = valid_summaries[0]
        if "## 1." in first or "## Lecture Overview" in first:
            return first

    combined_notes = "\n\n---\n\n".join(valid_summaries)
    words = combined_notes.split()

    # Modern context windows (128k tokens) support up to ~35,000 words easily in 1 prompt.
    # Only batch if combined notes exceed 35,000 words.
    MAX_COMBINED_WORDS = 35_000
    if len(words) > MAX_COMBINED_WORDS and len(valid_summaries) > 1:
        mid = len(valid_summaries) // 2
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut1 = pool.submit(merge_summaries, valid_summaries[:mid], model, format_type)
            fut2 = pool.submit(merge_summaries, valid_summaries[mid:], model, format_type)
            first_half = fut1.result()
            second_half = fut2.result()
        combined_notes = f"{first_half}\n\n---\n\n{second_half}"

    format_instructions = _get_format_instructions(format_type)

    prompt = ChatPromptTemplate.from_messages([
        ("system", MERGE_SYSTEM_PROMPT),
        ("human", MERGE_HUMAN_PROMPT),
    ])

    chain = prompt | model | StrOutputParser()

    input_data = {
        "combined_notes": combined_notes,
        "format_instructions": format_instructions,
    }

    return _invoke_with_retry(chain, input_data)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def generate_revision_notes(
    chunks: list[Any],
    format_type: str = DEFAULT_FORMAT,
    model: Optional[BaseChatModel] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    show_progress: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 5,
) -> str:
    """
    Full pipeline: summarize all transcript chunks concurrently, then merge
    into a coherent, university-standard set of final revision notes.

    Args:
        chunks:        List of transcript chunk strings, TextChunk objects, or dicts.
        format_type:   'concise' or 'detailed' (default: 'detailed').
        model:         Optional pre-configured LangChain chat model.
        model_name:    Optional model name (e.g., 'gpt-4o-mini', 'gpt-4o').
        temperature:   Optional temperature (0.0 to 1.0).
        api_key:       Optional API key (defaults to OPENAI_API_KEY env var).
        base_url:      Optional base URL for OpenAI-compatible endpoints.
        show_progress: Optional callback callable(done_count, total_count).
        max_workers:   Maximum concurrent workers for chunk summarisation.

    Returns:
        Complete Markdown revision notes string with the 6 standard sections.
    """
    if not chunks:
        return (
            "## 1. Lecture Overview\n*No transcript content provided for summarisation.*\n\n"
            "## 2. Key Concepts\n- [Uncertain / Missing from audio: Transcript is empty]\n\n"
            "## 3. Definitions\n- None\n\n"
            "## 4. Important Formulas\n*No mathematical formulas were introduced in this lecture.*\n\n"
            "## 5. Examples & Case Studies\n- None\n\n"
            "## 6. Final Revision Checklist\n- [ ] Ensure lecture transcript is correctly loaded\n"
        )

    # Initialise model once for the pipeline if not provided
    if model is None:
        model = get_chat_model(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    total = len(chunks)
    worker_count = max(1, min(total, max_workers))

    # Fast path: single chunk
    if total == 1:
        chunk_text, ref = extract_chunk_info(chunks[0], default_index=0)
        if not chunk_text or not chunk_text.strip():
            return generate_revision_notes([])

        # Summarise single chunk directly into full merged notes structure
        single_chunk_summary = summarise_chunk(
            chunks[0],
            model=model,
            format_type=format_type,
            chunk_index=0,
            timestamp_ref=ref,
        )
        if show_progress:
            show_progress(1, 1)

        # Pass through merge_summaries to guarantee the 6-section structure
        return merge_summaries([single_chunk_summary], model=model, format_type=format_type)

    # Multi-chunk concurrent processing
    summaries: list[Optional[str]] = [None] * total
    done_count = 0

    def _worker(idx_and_item: tuple[int, Any]) -> tuple[int, str]:
        idx, item = idx_and_item
        res = summarise_chunk(
            item,
            model=model,
            format_type=format_type,
            chunk_index=idx,
        )
        return idx, res

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_worker, (i, c)): i for i, c in enumerate(chunks)
        }
        for fut in as_completed(futures):
            idx, result = fut.result()
            summaries[idx] = result
            done_count += 1
            if show_progress:
                show_progress(done_count, total)

    valid_summaries = [s for s in summaries if s and s.strip()]

    if not valid_summaries:
        return generate_revision_notes([])

    # Notify caller that chunks are done and final synthesis is underway
    if show_progress:
        try:
            show_progress(total, total, "✨ Synthesising master revision notes guide…")
        except TypeError:
            show_progress(total, total)

    # Merge per-chunk summaries into the final structured revision guide
    return merge_summaries(valid_summaries, model=model, format_type=format_type)
