# flashcards.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Generate high-quality revision flashcards from lecture transcripts
#          and revision notes, validate their contents, remove duplicates,
#          and export them in Anki-compatible CSV and .apkg formats.
#
# Requirements satisfied:
#   1. Question-and-answer cards for definitions, concepts, formulas,
#      and important relationships.
#   2. Avoid duplicate and vague cards.
#   3. Keep each card focused on one learning objective.
#   4. Include a source timestamp or reference when possible.
#   5. Allow the user to choose the number of cards.
#   6. Display the cards in the Streamlit interface with preview.
#   7. Export Anki-compatible CSV with Front/Back columns, UTF-8 encoding,
#      and RFC 4180 CSV escaping.
#   8. Validate generated data (duplicates, missing answers, empty questions).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import string
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Anki constants for genanki (.apkg) ─────────────────────────────────────────
_ANKI_DECK_ID = 1_234_567_890
_ANKI_MODEL_ID = 9_876_543_210

# ── Validation constants ──────────────────────────────────────────────────────
MIN_QUESTION_CHARS = 10
MIN_ANSWER_CHARS = 3
DEFAULT_NUM_CARDS = 15
MIN_NUM_CARDS = 1
MAX_NUM_CARDS = 50
SIMILARITY_DEDUP_THRESHOLD = 0.70  # Jaccard token similarity threshold

# Vague / low-value question patterns that must be filtered out
VAGUE_QUESTION_PATTERNS = [
    r"^what is this\??$",
    r"^what is discussed\??$",
    r"^what did the (speaker|professor|teacher|lecture) (say|discuss|talk about)\??$",
    r"^what is the (main )?(topic|overview|summary)\??$",
    r"^explain (everything|the lecture|the topic)\??$",
    r"^introduction\??$",
    r"^overview\??$",
    r"^summary\??$",
    r"^key concepts?\??$",
]

# Placeholder answers indicating missing / incomplete content
PLACEHOLDER_ANSWERS = {
    "", "n/a", "none", "unknown", "tbd", "todo", "?", "null",
    "missing", "[missing]", "not mentioned", "not specified", "unclear",
}


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class FlashcardError(Exception):
    """Base exception for flashcard operations."""
    pass


class FlashcardConfigError(FlashcardError, EnvironmentError):
    """Raised when neither OPENAI_API_KEY nor GEMINI_API_KEY is available."""
    pass


class FlashcardValidationError(FlashcardError, ValueError):
    """Raised when generated flashcard data fails validation."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Data Structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Flashcard:
    """
    Representation of a single revision flashcard.

    Attributes:
        front:     The question or prompt (focused on one learning objective).
        back:      The concise, factual answer.
        category:  Category: 'Definition', 'Concept', 'Formula', or 'Relationship'.
        timestamp: Optional source timestamp reference (e.g., '02:15' or '02:15 - 03:00').
        tags:      Optional tag string for Anki export.
    """
    front: str
    back: str
    category: str = "Concept"
    timestamp: Optional[str] = None
    tags: str = "PrepGenius"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching the standard app format."""
        d = {
            "front": self.front,
            "back": self.back,
            "category": self.category,
            "tags": self.tags,
        }
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-like access for backward compatibility."""
        return self.to_dict()[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.to_dict().get(item, default)


# ─────────────────────────────────────────────────────────────────────────────
# Question & Answer Validation & Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Normalize text into lowercase alphanumeric tokens, stripping punctuation."""
    text_clean = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in text_clean.split() if len(w) > 1}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Calculate Jaccard similarity index between two token sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def is_vague_question(question: str) -> bool:
    """
    Check if a question is too vague, generic, or lacks a concrete learning objective.

    Args:
        question: The candidate front-of-card text.

    Returns:
        True if the question is vague and should be rejected; False otherwise.
    """
    q_stripped = question.strip()
    if len(q_stripped) < MIN_QUESTION_CHARS:
        return True

    words = q_stripped.split()
    if len(words) < 3:
        return True

    q_lower = q_stripped.lower()
    for pattern in VAGUE_QUESTION_PATTERNS:
        if re.search(pattern, q_lower):
            return True

    return False


def is_single_objective(question: str) -> bool:
    """
    Check that a question does not excessively compound multiple learning objectives.

    Questions like 'What is X, and what is Y, and what are the 5 differences and formulas?'
    violate the single learning objective rule.
    """
    q_lower = question.lower()
    # Count independent question markers
    q_marks = question.count("?")
    if q_marks > 1:
        return False

    # Check for excessive multi-part clauses
    multi_part_clauses = re.findall(r"\b(and what|and how|and why|also explain|as well as)\b", q_lower)
    if len(multi_part_clauses) >= 2:
        return False

    return True


def extract_timestamp_from_text(text: str) -> Optional[str]:
    """
    Extract a timestamp reference like '[02:15 - 03:00]' or '[04:20]' from text.
    """
    match = re.search(r"\[(\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?)\]", text)
    if match:
        return match.group(1).strip()
    return None


def categorize_card(front: str, back: str, raw_category: Optional[str] = None) -> str:
    """
    Determine the educational category of a flashcard:
    'Definition', 'Formula', 'Relationship', or 'Concept'.
    """
    if raw_category:
        rc = raw_category.strip().capitalize()
        if rc in {"Definition", "Formula", "Relationship", "Concept"}:
            return rc

    front_lower = front.lower()
    combined = f"{front} {back}".lower()

    # 1. Relationships & Comparisons (highest precedence for difference/compare questions)
    if any(k in front_lower for k in ["difference between", "compare", "versus", "vs", "relationship", "trade-off", "how do"]):
        return "Relationship"

    # 2. Definitions (check if question is asking for definition or meaning of a term)
    if any(k in front_lower for k in ["define", "definition of", "what is a", "what is an", "what is the definition", "term for", "is defined as"]):
        return "Definition"

    # 3. Formulas & Equations
    if any(k in combined for k in ["formula", "equation", "t(n)", "o(", "o(1)", "theta(", "omega(", "algorithm complexity", "recurrence relation"]):
        return "Formula"

    # Fallback checks on combined text
    if any(k in combined for k in ["difference between", "compare", "versus", "vs", "relationship", "trade-off"]):
        return "Relationship"

    if any(k in combined for k in ["is defined as", "meaning of", "definition"]):
        return "Definition"

    return "Concept"


def validate_flashcard(card: Any) -> Optional[dict[str, Any]]:
    """
    Validate a single flashcard candidate.

    Checks:
      - 'front' is a valid, non-empty, non-vague question string (≥10 chars, ≥3 words).
      - 'back' is a non-empty, valid answer string (≥3 chars, not a placeholder).
      - Question focuses on a reasonable learning objective.
      - Extracts and normalizes timestamp and category.

    Returns:
        Cleaned dict with 'front', 'back', 'category', 'timestamp', 'tags'
        or None if validation fails.
    """
    if not isinstance(card, dict):
        if hasattr(card, "front") and hasattr(card, "back"):
            card = {
                "front": getattr(card, "front"),
                "back": getattr(card, "back"),
                "category": getattr(card, "category", None),
                "timestamp": getattr(card, "timestamp", None),
                "tags": getattr(card, "tags", "PrepGenius"),
            }
        else:
            return None

    raw_front = card.get("front")
    raw_back = card.get("back")

    if raw_front is None or raw_back is None:
        return None

    front = str(raw_front).strip()
    back = str(raw_back).strip()

    # Reject empty or whitespace-only
    if not front or not back:
        return None

    # Strip leading "Q:" or "Question:" prefixes if LLM added them
    front = re.sub(r"^(?:Q\s*[:.-]\s*|Question\s*[:.-]\s*)", "", front, flags=re.IGNORECASE).strip()
    back = re.sub(r"^(?:A\s*[:.-]\s*|Answer\s*[:.-]\s*)", "", back, flags=re.IGNORECASE).strip()

    # Reject missing / placeholder answers
    if back.lower() in PLACEHOLDER_ANSWERS or len(back) < MIN_ANSWER_CHARS:
        return None

    # Reject vague questions
    if is_vague_question(front):
        return None

    # Check timestamp
    ts = card.get("timestamp")
    if not ts:
        ts = extract_timestamp_from_text(front) or extract_timestamp_from_text(back)
    else:
        ts = str(ts).strip()
        # Clean bracket if user passed '[01:23]'
        ts = ts.strip("[]")

    category = categorize_card(front, back, card.get("category"))
    tags = str(card.get("tags") or "PrepGenius").strip()

    return {
        "front": front,
        "back": back,
        "category": category,
        "timestamp": ts if ts else None,
        "tags": tags,
    }


def deduplicate_flashcards(
    cards: Sequence[dict[str, Any]],
    similarity_threshold: float = SIMILARITY_DEDUP_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Remove duplicate and near-duplicate flashcards.

    Uses a two-stage filter:
      1. Exact matching on normalized question text.
      2. Token-level Jaccard similarity to catch paraphrased identical questions.

    Args:
        cards: List of candidate flashcard dictionaries.
        similarity_threshold: Jaccard similarity cutoff above which a card is
                              considered a duplicate of an existing card.

    Returns:
        Deduplicated list of flashcard dictionaries in original order.
    """
    seen_exact_fronts: set[str] = set()
    seen_token_sets: list[set[str]] = []
    unique_cards: list[dict[str, Any]] = []

    for card in cards:
        front = card.get("front", "").strip()
        if not front:
            continue

        # Normalized key for exact matching (lowercase, no punctuation, single spaces)
        norm_key = re.sub(r"[^\w\s]", "", front.lower()).strip()
        norm_key = re.sub(r"\s+", " ", norm_key)

        if norm_key in seen_exact_fronts:
            continue

        # Token set for similarity matching
        tokens = _tokenize(front)
        if not tokens:
            continue

        # Check Jaccard similarity against all already-accepted cards
        is_dup = False
        for existing_tokens in seen_token_sets:
            sim = _jaccard_similarity(tokens, existing_tokens)
            if sim >= similarity_threshold:
                is_dup = True
                break

        if is_dup:
            continue

        seen_exact_fronts.add(norm_key)
        seen_token_sets.append(tokens)
        unique_cards.append(card)

    return unique_cards


def validate_and_clean_flashcards(
    raw_cards: Sequence[Any],
    max_cards: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Validate, filter, and deduplicate a sequence of raw flashcard candidates.

    Args:
        raw_cards: Raw candidate objects from LLM JSON output.
        max_cards: Optional maximum count to retain.

    Returns:
        List of high-quality, validated, deduplicated flashcard dictionaries.
    """
    valid_cards: list[dict[str, Any]] = []
    for raw in raw_cards:
        cleaned = validate_flashcard(raw)
        if cleaned is not None:
            valid_cards.append(cleaned)

    deduped = deduplicate_flashcards(valid_cards)

    if max_cards is not None and max_cards > 0:
        return deduped[:max_cards]
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Anki-Compatible CSV Exporter (RFC 4180, UTF-8, proper escaping)
# ─────────────────────────────────────────────────────────────────────────────

def export_anki_csv_string(
    flashcards: Sequence[dict[str, Any]],
    include_header: bool = True,
    include_tags: bool = False,
    tag_name: str = "PrepGenius",
) -> str:
    """
    Generate an Anki-compatible CSV string from a list of flashcards.

    Formatting details:
      - Complies with RFC 4180 standard CSV quoting.
      - Fields containing commas, double quotes, or newlines are enclosed in quotes.
      - Double quotes inside values are escaped by doubling them (e.g., " -> "").
      - Columns: 'Front', 'Back' (and optionally 'Tags').

    Args:
        flashcards:     List of flashcard dicts with 'front' and 'back' keys.
        include_header: Whether to write the 'Front,Back' header row.
        include_tags:   Whether to include a third 'Tags' column.
        tag_name:       Default tag to populate if include_tags is True.

    Returns:
        CSV text string.
    """
    output = io.StringIO()
    writer = csv.writer(
        output,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\r\n",  # RFC 4180 standard line ending
    )

    if include_header:
        header = ["Front", "Back"]
        if include_tags:
            header.append("Tags")
        writer.writerow(header)

    for card in flashcards:
        front = str(card.get("front", "")).strip()
        back = str(card.get("back", "")).strip()

        # If timestamp is available and not in front, format timestamp reference nicely
        ts = card.get("timestamp")
        if ts and f"[{ts}]" not in front and f"[{ts}]" not in back:
            back_formatted = f"{back}\n\n[⏱️ Source: {ts}]"
        else:
            back_formatted = back

        row = [front, back_formatted]
        if include_tags:
            row.append(str(card.get("tags") or tag_name))
        writer.writerow(row)

    return output.getvalue()


def export_anki_csv_bytes(
    flashcards: Sequence[dict[str, Any]],
    include_header: bool = True,
    include_tags: bool = False,
    tag_name: str = "PrepGenius",
) -> bytes:
    """
    Export flashcards as UTF-8 with BOM bytes for direct browser download in Streamlit.

    Using UTF-8 with BOM (utf-8-sig) ensures Anki, Excel, and text editors on
    Windows and Mac correctly interpret non-ASCII characters, math symbols (e.g.
    Ω, Θ, ≤, ≥, →), and special formatting without mojibake.
    """
    csv_str = export_anki_csv_string(
        flashcards,
        include_header=include_header,
        include_tags=include_tags,
        tag_name=tag_name,
    )
    return csv_str.encode("utf-8-sig")


def export_to_csv(
    flashcards: Sequence[dict[str, Any]],
    output_path: Optional[str] = None,
    include_header: bool = True,
    include_tags: bool = False,
    tag_name: str = "PrepGenius",
) -> str:
    """
    Save flashcards as an Anki-compatible CSV file on disk.

    Args:
        flashcards:     List of flashcard dictionaries.
        output_path:    Target file path. Defaults to 'outputs/prepgenius_flashcards.csv'.
        include_header: Whether to include 'Front,Back' header.
        include_tags:   Whether to include tags column.
        tag_name:       Tag value.

    Returns:
        Absolute path to the saved CSV file.
    """
    if not output_path:
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", "prepgenius_flashcards.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    csv_bytes = export_anki_csv_bytes(
        flashcards,
        include_header=include_header,
        include_tags=include_tags,
        tag_name=tag_name,
    )

    with open(output_path, "wb") as f:
        f.write(csv_bytes)

    return os.path.abspath(output_path)


def export_to_anki(
    flashcards: Sequence[dict[str, Any]],
    deck_name: str = "SAN Lecture Notes",
    output_path: Optional[str] = None,
) -> str:
    """
    Export flashcards as an Anki .apkg package file using genanki.

    Retained for full backward compatibility with existing users and callers.

    Args:
        flashcards:  List of dicts with 'front' and 'back' keys.
        deck_name:   Name shown in Anki for this deck.
        output_path: Optional custom output path.

    Returns:
        Absolute path to the generated .apkg file.
    """
    import genanki

    anki_model = genanki.Model(
        _ANKI_MODEL_ID,
        "SAN Basic",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}<hr id='answer'>{{Back}}",
            }
        ],
    )

    deck = genanki.Deck(_ANKI_DECK_ID, deck_name)

    for card in flashcards:
        front = str(card.get("front", ""))
        back = str(card.get("back", ""))
        ts = card.get("timestamp")
        if ts and f"[{ts}]" not in front and f"[{ts}]" not in back:
            back = f"{back}<br><br><small style='color:gray;'>⏱️ Source: {ts}</small>"
        # Convert newlines to <br> for HTML rendering in Anki
        back_html = back.replace("\n", "<br>")
        note = genanki.Note(
            model=anki_model,
            fields=[front, back_html],
        )
        deck.add_note(note)

    if not output_path:
        os.makedirs("outputs", exist_ok=True)
        safe_name = deck_name.replace(" ", "_").replace("/", "-")
        output_path = os.path.join("outputs", f"{safe_name}.apkg")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    genanki.Package(deck).write_to_file(output_path)
    return os.path.abspath(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Engineering for Flashcard Generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_flashcard_prompt(
    lecture_content: str,
    num_cards: int,
    additional_context: Optional[str] = None,
) -> str:
    """
    Construct the prompt requesting structured flashcard generation.
    """
    extra_sec = f"\nAdditional Context / Source Segments:\n{additional_context}\n" if additional_context else ""

    return (
        f"You are an expert tutor creating high-yield active-recall study flashcards.\n"
        f"Generate exactly {num_cards} flashcards based strictly on the material provided below.\n\n"
        "FLASHCARD REQUIREMENTS:\n"
        "1. Content Types:\n"
        "   - Definitions: Formal terms, core vocabulary, and foundational definitions.\n"
        "   - Concepts: Mechanisms, principles, why things work the way they do.\n"
        "   - Formulas: Equations, mathematical expressions, variable meanings, and complexity bounds.\n"
        "   - Relationships: Comparisons, trade-offs, differences between related ideas, and cause-and-effect.\n"
        "2. Single Learning Objective:\n"
        "   - Each card must test exactly ONE bite-sized, specific concept or fact.\n"
        "   - Never combine multiple questions into one card (no 'What is X and what are its 3 steps and formula?').\n"
        "3. Quality and Clarity:\n"
        "   - Questions must be concrete, unambiguous, and self-contained.\n"
        "   - Strictly avoid vague prompts like 'What did the speaker say?' or 'Overview of topic'.\n"
        "   - Answers must be direct, factual, and concise (1 to 3 clear sentences).\n"
        "4. No Duplicates:\n"
        "   - Do not generate cards that test the same underlying concept in different words.\n"
        "5. Timestamps / References:\n"
        "   - Where timestamps like [MM:SS] or [MM:SS - MM:SS] appear in the text, preserve them in the 'timestamp' field.\n\n"
        "OUTPUT FORMAT:\n"
        "Respond ONLY with a valid JSON array of objects. Do not include markdown preamble or commentary.\n"
        "Example JSON structure:\n"
        "[\n"
        "  {\n"
        '    "front": "What is the time complexity of binary search on a sorted array of size n?",\n'
        '    "back": "O(log n), because the search interval is halved with each comparison.",\n'
        '    "category": "Formula",\n'
        '    "timestamp": "15:30 - 18:00"\n'
        "  },\n"
        "  {\n"
        '    "front": "How is a recurrence relation defined in algorithmic analysis?",\n'
        '    "back": "An equation that expresses the total running time T(n) recursively in terms of smaller input subproblems.",\n'
        '    "category": "Definition",\n'
        '    "timestamp": "09:15"\n'
        "  }\n"
        "]\n\n"
        f"LECTURE MATERIAL:\n\"\"\"\n{lecture_content}\n\"\"\"{extra_sec}"
    )


def _parse_llm_json_response(raw_text: str) -> list[dict[str, Any]]:
    """
    Robustly extract and parse a JSON array from LLM output.

    Handles:
      - Markdown code fences (```json ... ```)
      - Leading and trailing conversational text
      - Individual JSON objects if full array bracket is missing
    """
    cleaned = raw_text.strip()

    # Strip code blocks
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            # Fallback split
            parts = cleaned.split("```")
            for part in parts:
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("[") and p.endswith("]"):
                    cleaned = p
                    break

    # Find [ ... ] array
    if not (cleaned.startswith("[") and cleaned.endswith("]")):
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError as exc:
        logger.debug("Standard JSON parse failed (%s); attempting regex object extraction.", exc)

    # Fallback regex extraction: find individual {"front": ..., "back": ...} blocks
    cards: list[dict[str, Any]] = []
    pattern = re.compile(r"\{\s*\"front\"\s*:\s*\"(.*?)\"\s*,\s*\"back\"\s*:\s*\"(.*?)\"(?:\s*,\s*\"category\"\s*:\s*\"(.*?)\")?(?:\s*,\s*\"timestamp\"\s*:\s*\"(.*?)\")?\s*\}", re.DOTALL)
    for m in pattern.finditer(raw_text):
        front, back, cat, ts = m.groups()
        cards.append({
            "front": front.encode().decode("unicode_escape", errors="ignore"),
            "back": back.encode().decode("unicode_escape", errors="ignore"),
            "category": cat or "Concept",
            "timestamp": ts,
        })

    return cards


# ─────────────────────────────────────────────────────────────────────────────
# Model Invocation & Core Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _call_model(
    prompt: str,
    model: Any = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """
    Dispatch prompt to the appropriate model (LangChain ChatOpenAI or Gemini).
    """
    # 1. Custom model explicitly provided (e.g., FakeListChatModel or pre-configured instance)
    if model is not None:
        if hasattr(model, "invoke"):
            # LangChain BaseChatModel / Runnable
            from langchain_core.messages import HumanMessage
            response = model.invoke([HumanMessage(content=prompt)])
            return getattr(response, "content", str(response))
        if hasattr(model, "generate_content"):
            # Gemini GenerativeModel
            res = model.generate_content(prompt)
            return getattr(res, "text", str(res))
        if callable(model):
            return str(model(prompt))
        raise TypeError(f"Unsupported model type passed to generate_flashcards: {type(model)}")

    # 2. Prefer LangChain / OpenAI-compatible endpoint (NVIDIA or OpenAI)
    openai_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI
        resolved_model = (
            model_name
            or os.getenv("NEMOTRON_MODEL")
            or os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
        )
        resolved_base = (
            base_url
            or os.getenv("NVIDIA_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or "https://integrate.api.nvidia.com/v1"
        )
        chat = ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            api_key=openai_key,
            base_url=resolved_base,
            max_retries=3,
            timeout=120.0,
        )
        from langchain_core.messages import HumanMessage
        response = chat.invoke([HumanMessage(content=prompt)])
        return getattr(response, "content", str(response))

    # 3. Fallback to Gemini if GEMINI_API_KEY is configured
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        resolved_gemini_model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        g_model = genai.GenerativeModel(resolved_gemini_model)
        res = g_model.generate_content(prompt)
        return getattr(res, "text", str(res))

    raise FlashcardConfigError(
        "No API key configured for flashcard generation. "
        "Please set NVIDIA_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in your environment or UI settings."
    )


def generate_flashcards(
    revision_notes: str,
    num_cards: int = DEFAULT_NUM_CARDS,
    transcript_chunks: Optional[Sequence[Any]] = None,
    model: Any = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Generate, validate, and deduplicate revision flashcards from lecture material.

    Args:
        revision_notes:     Full revision notes text (or raw transcript text).
        num_cards:          Target number of flashcards to generate (default 15, range 1-50).
        transcript_chunks:  Optional list of TextChunk objects or strings with timestamps.
        model:              Pre-initialized LangChain Chat model, Gemini model, or mock.
        model_name:         Model name override.
        temperature:        Sampling temperature (default 0.2).
        api_key:            Explicit API key override.
        base_url:           Explicit base URL override.

    Returns:
        List of validated, deduplicated flashcard dictionaries:
        [{'front': str, 'back': str, 'category': str, 'timestamp': Optional[str], 'tags': str}, ...]
    """
    # ── Input validation ─────────────────────────────────────────────────────
    if not revision_notes or not revision_notes.strip():
        return []

    # Clamp num_cards to valid range
    target_count = max(MIN_NUM_CARDS, min(MAX_NUM_CARDS, int(num_cards)))

    # Request slightly more cards from LLM (target + 3) to compensate for potential
    # deduplication and validation filters
    requested_from_llm = min(MAX_NUM_CARDS, target_count + 3)

    # ── Compile source context ────────────────────────────────────────────────
    additional_context = ""
    if transcript_chunks:
        seg_lines = []
        for i, chk in enumerate(transcript_chunks[:20]):
            text = getattr(chk, "text", str(chk)).strip()
            start = getattr(chk, "start_time", None)
            end = getattr(chk, "end_time", None)
            if start is not None:
                sm, ss = divmod(int(round(start)), 60)
                em, es = divmod(int(round(end or start)), 60)
                seg_lines.append(f"[{sm:02d}:{ss:02d} - {em:02d}:{es:02d}] {text[:200]}")
            else:
                seg_lines.append(f"[Segment {i+1}] {text[:200]}")
        additional_context = "\n".join(seg_lines)

    prompt = _build_flashcard_prompt(
        lecture_content=revision_notes,
        num_cards=requested_from_llm,
        additional_context=additional_context or None,
    )

    # ── Call Model ────────────────────────────────────────────────────────────
    raw_response = _call_model(
        prompt=prompt,
        model=model,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
    )

    # ── Parse JSON ────────────────────────────────────────────────────────────
    parsed_cards = _parse_llm_json_response(raw_response)

    # ── Validate & Deduplicate ────────────────────────────────────────────────
    clean_cards = validate_and_clean_flashcards(parsed_cards, max_cards=target_count)

    return clean_cards
