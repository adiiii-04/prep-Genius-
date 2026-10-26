# quiz.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Generate multiple-choice questions (MCQs) from the revision notes
#          and lecture transcript using Gemini, validate question structure,
#          and score user answers in the Streamlit UI.
#
# Each question dict has:
#   - "question"    : The question text.
#   - "options"     : A list of exactly 4 answer strings (no A/B/C/D prefix).
#   - "answer"      : The correct option text (must exactly match one item in "options").
#   - "explanation" : A brief explanation of why the answer is correct.
#   - "difficulty"  : One of "easy", "medium", or "hard".
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
VALID_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium", "hard"})
MIN_QUESTIONS = 1
MAX_QUESTIONS = 30
DEFAULT_QUESTIONS = 10

DifficultyType = Literal["easy", "medium", "hard", "mixed"]


# ─────────────────────────────────────────────────────────────────────────────
# Model initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_model() -> genai.GenerativeModel:
    """Initialise the Gemini model using the configured API key and model name."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set. See .env.example.")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)


# ─────────────────────────────────────────────────────────────────────────────
# Question Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_question(q: object) -> dict | None:
    """
    Validate a single MCQ question dict.

    Checks enforced:
      - Must be a dict with keys "question", "options", "answer", "explanation".
      - "options" must be a list of exactly 4 non-empty strings.
      - "answer" must exactly match one of the option strings.
      - "question" must be a non-empty string (≥ 10 characters).
      - "explanation" must be a non-empty string.
      - All option strings must be distinct (no duplicates).
      - "difficulty" (optional) is normalised to lowercase; defaults to "medium".

    Args:
        q: Candidate question object from LLM output.

    Returns:
        Cleaned and normalised question dict, or None if validation fails.
    """
    if not isinstance(q, dict):
        return None

    required_keys = {"question", "options", "answer", "explanation"}
    if not required_keys.issubset(q.keys()):
        logger.debug("Question missing required keys: %s", required_keys - q.keys())
        return None

    question_text = str(q.get("question", "")).strip()
    if len(question_text) < 10:
        logger.debug("Question text too short: %r", question_text)
        return None

    raw_options = q.get("options")
    if not isinstance(raw_options, list) or len(raw_options) != 4:
        logger.debug("Options must be a list of exactly 4 items; got: %r", raw_options)
        return None

    options = [str(o).strip() for o in raw_options]

    # All options must be non-empty
    if any(len(o) == 0 for o in options):
        logger.debug("One or more options are empty strings.")
        return None

    # No duplicate options
    if len(set(options)) != 4:
        logger.debug("Duplicate options detected: %r", options)
        return None

    answer = str(q.get("answer", "")).strip()
    if answer not in options:
        # Attempt case-insensitive rescue
        lower_options = [o.lower() for o in options]
        if answer.lower() in lower_options:
            answer = options[lower_options.index(answer.lower())]
        else:
            logger.debug("Answer %r does not match any option.", answer)
            return None

    explanation = str(q.get("explanation", "")).strip()
    if not explanation:
        logger.debug("Explanation is empty.")
        return None

    raw_difficulty = str(q.get("difficulty", "medium")).strip().lower()
    difficulty = raw_difficulty if raw_difficulty in VALID_DIFFICULTIES else "medium"

    return {
        "question":    question_text,
        "options":     options,
        "answer":      answer,
        "explanation": explanation,
        "difficulty":  difficulty,
    }


def validate_quiz(questions: list) -> list[dict]:
    """
    Validate and filter a list of raw MCQ question objects.

    Args:
        questions: List of candidate question objects.

    Returns:
        List of validated question dicts (invalid entries are silently dropped).
    """
    validated: list[dict] = []
    for q in questions:
        cleaned = validate_question(q)
        if cleaned is not None:
            validated.append(cleaned)
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_quiz_prompt(
    source_material: str,
    num_questions: int,
    difficulty: DifficultyType,
) -> str:
    """Build the Gemini prompt for MCQ generation."""
    if difficulty == "mixed":
        difficulty_instruction = (
            "Mix the difficulties evenly across questions: "
            f"approximately {num_questions // 3 or 1} easy, "
            f"{num_questions // 3 or 1} medium, and "
            f"{num_questions - 2 * (num_questions // 3 or 1)} hard questions."
        )
    else:
        difficulty_instruction = f"All questions must be {difficulty} difficulty."

    difficulty_guide = (
        "Difficulty guidelines:\n"
        "  - easy:   Recall of a single fact or definition directly stated in the material.\n"
        "  - medium: Application or understanding of a concept; requires some inference.\n"
        "  - hard:   Analysis, synthesis, or comparison of multiple concepts.\n"
    )

    return (
        "You are an expert university exam setter.\n"
        f"Create exactly {num_questions} multiple-choice questions STRICTLY based on the "
        "source material provided below. Do NOT invent facts or make unsupported claims.\n\n"
        "Requirements for EACH question:\n"
        '  - "question"   : A clear, unambiguous, self-contained question string (≥ 10 characters).\n'
        '  - "options"    : A JSON array of exactly 4 distinct answer strings '
        '(no A/B/C/D prefix; just plain text).\n'
        '  - "answer"     : The FULL TEXT of the correct option. '
        'It MUST exactly match one item in "options" character-for-character.\n'
        '  - "explanation": 1-2 sentences explaining why the answer is correct. '
        'Must be grounded in the source material.\n'
        f'  - "difficulty" : One of "easy", "medium", or "hard".\n\n'
        f"{difficulty_guide}\n"
        f"{difficulty_instruction}\n\n"
        "Additional rules:\n"
        "  - Generate questions ONLY from information present in the source material.\n"
        "  - Avoid ambiguous, trick, or double-barrelled questions.\n"
        "  - Distractors (wrong options) must be plausible but clearly incorrect.\n"
        "  - Each question must test a distinct concept (no near-duplicates).\n\n"
        "Respond ONLY with a valid JSON array. Do NOT include any text outside the JSON.\n\n"
        f'Source Material:\n"""\n{source_material}\n"""'
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON Response Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> list:
    """
    Robustly extract and parse a JSON array from raw LLM output.

    Handles:
      - Markdown code fences (```json ... ```)
      - Leading/trailing prose outside the JSON array
    """
    cleaned = raw.strip()

    # Strip markdown code fences
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            # Fallback: split on ``` and look for JSON array
            for part in cleaned.split("```"):
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("[") and p.endswith("]"):
                    cleaned = p
                    break

    # Find the first [ ... ] array if prose still surrounds it
    if not (cleaned.startswith("[") and cleaned.endswith("]")):
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0).strip()

    return json.loads(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_quiz(
    revision_notes: str,
    num_questions: int = DEFAULT_QUESTIONS,
    difficulty: DifficultyType = "mixed",
) -> list[dict]:
    """
    Generate and validate multiple-choice questions from revision notes.

    Args:
        revision_notes: The full Markdown revision notes string (or raw transcript).
        num_questions:  Number of MCQ questions to generate (1–30, default 10).
        difficulty:     Target difficulty — "easy", "medium", "hard", or "mixed"
                        (default "mixed" blends all three levels evenly).

    Returns:
        List of validated question dicts with keys:
          "question", "options" (list of 4), "answer", "explanation", "difficulty".

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
        ValueError:       If Gemini's response cannot be parsed as valid JSON.
    """
    if not revision_notes or not revision_notes.strip():
        return []

    # Clamp to allowed range
    num_questions = max(MIN_QUESTIONS, min(MAX_QUESTIONS, int(num_questions)))

    # Normalise difficulty
    difficulty_normalised: DifficultyType = (
        difficulty if difficulty in {*VALID_DIFFICULTIES, "mixed"} else "mixed"
    )

    model = _get_model()
    prompt = _build_quiz_prompt(revision_notes, num_questions, difficulty_normalised)

    response = model.generate_content(prompt)
    raw = response.text.strip()

    try:
        raw_questions = _parse_json_response(raw)
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(
            f"Gemini returned invalid JSON for the quiz.\n"
            f"Raw response:\n{raw}\n\nError: {exc}"
        ) from exc

    if not isinstance(raw_questions, list):
        raise ValueError(
            f"Expected a JSON array from Gemini; got {type(raw_questions).__name__}.\n"
            f"Raw response:\n{raw}"
        )

    validated = validate_quiz(raw_questions)
    logger.info(
        "Quiz generation: requested %d, received %d, validated %d (difficulty=%s)",
        num_questions, len(raw_questions), len(validated), difficulty_normalised,
    )
    return validated


def score_quiz(questions: list[dict], user_answers: dict[int, str]) -> dict:
    """
    Calculate the user's quiz score.

    Args:
        questions:    List of question dicts from `generate_quiz`.
        user_answers: Dict mapping question index (0-based) → chosen option string.

    Returns:
        A dict with keys:
          - "score"   : Number of correct answers (int).
          - "total"   : Total number of questions (int).
          - "percent" : Score as a percentage (float, 0–100, rounded to 1 dp).
          - "results" : List of per-question result dicts, each with keys:
                        "correct" (bool), "chosen" (str), "answer" (str),
                        "explanation" (str), "difficulty" (str).
    """
    results: list[dict] = []
    correct_count = 0

    for i, q in enumerate(questions):
        chosen = user_answers.get(i, "")
        is_correct = chosen == q["answer"]
        if is_correct:
            correct_count += 1
        results.append({
            "correct":     is_correct,
            "chosen":      chosen,
            "answer":      q["answer"],
            "explanation": q["explanation"],
            "difficulty":  q.get("difficulty", "medium"),
        })

    total = len(questions)
    percent = (correct_count / total * 100) if total > 0 else 0.0

    return {
        "score":   correct_count,
        "total":   total,
        "percent": round(percent, 1),
        "results": results,
    }
