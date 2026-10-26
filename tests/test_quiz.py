# tests/test_quiz.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for quiz.py — pure logic, no Gemini API calls.
#
# Coverage:
#   - validate_question: structure, answer correctness, edge cases
#   - validate_quiz: batch filtering
#   - score_quiz: all-correct, all-wrong, partial, empty answers, results schema
#   - generate_quiz prompt plumbing: mocked Gemini response parsing
# ─────────────────────────────────────────────────────────────────────────────

import json
import pytest
from quiz import validate_question, validate_quiz, score_quiz, _parse_json_response


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_q(**overrides) -> dict:
    """Return a minimal valid question dict, with optional field overrides."""
    base = {
        "question":    "What is the primary purpose of photosynthesis?",
        "options":     [
            "Producing glucose from CO₂ and sunlight",
            "Breaking down glucose to release energy",
            "Absorbing water from the soil",
            "Transporting nutrients in the xylem",
        ],
        "answer":      "Producing glucose from CO₂ and sunlight",
        "explanation": "Photosynthesis converts CO₂ and water into glucose using light energy.",
        "difficulty":  "easy",
    }
    base.update(overrides)
    return base


SAMPLE_QUESTIONS = [
    _make_q(),
    _make_q(
        question="What does Newton's first law of motion state?",
        options=[
            "F = ma",
            "An object at rest stays at rest unless acted upon by a net force",
            "Energy is always conserved",
            "Every action has an equal and opposite reaction",
        ],
        answer="An object at rest stays at rest unless acted upon by a net force",
        explanation="Newton's first law (inertia): objects resist changes in motion.",
        difficulty="medium",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# TestValidateQuestion — question structure validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateQuestion:
    """Tests for quiz.validate_question."""

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_valid_question_returns_dict(self):
        result = validate_question(_make_q())
        assert result is not None
        assert isinstance(result, dict)

    def test_valid_question_has_all_required_keys(self):
        result = validate_question(_make_q())
        assert result is not None
        for key in ("question", "options", "answer", "explanation", "difficulty"):
            assert key in result, f"Missing key: {key}"

    def test_answer_matches_one_option(self):
        result = validate_question(_make_q())
        assert result is not None
        assert result["answer"] in result["options"]

    def test_exactly_four_options_returned(self):
        result = validate_question(_make_q())
        assert result is not None
        assert len(result["options"]) == 4

    def test_difficulty_normalised_to_lowercase(self):
        result = validate_question(_make_q(difficulty="HARD"))
        assert result is not None
        assert result["difficulty"] == "hard"

    def test_difficulty_defaults_to_medium_when_missing(self):
        q = _make_q()
        del q["difficulty"]
        result = validate_question(q)
        assert result is not None
        assert result["difficulty"] == "medium"

    def test_difficulty_defaults_to_medium_for_invalid_value(self):
        result = validate_question(_make_q(difficulty="extreme"))
        assert result is not None
        assert result["difficulty"] == "medium"

    def test_answer_case_insensitive_rescue(self):
        """If answer differs only in case from an option, it should be rescued."""
        opts = [
            "Producing glucose from CO₂ and sunlight",
            "Breaking down glucose",
            "Absorbing water",
            "Transporting nutrients",
        ]
        result = validate_question(_make_q(
            options=opts,
            answer="producing glucose from CO₂ and sunlight",  # lowercase first char
        ))
        # rescue should work
        assert result is not None
        assert result["answer"] == "Producing glucose from CO₂ and sunlight"

    def test_whitespace_stripped_from_question(self):
        result = validate_question(_make_q(question="   What is ATP?   "))
        assert result is not None
        assert result["question"] == "What is ATP?"

    def test_whitespace_stripped_from_options(self):
        opts = ["  Option A  ", "Option B", "Option C", "Option D"]
        result = validate_question(_make_q(options=opts, answer="  Option A  "))
        assert result is not None
        assert result["options"][0] == "Option A"
        assert result["answer"] == "Option A"

    # ── Rejection cases ───────────────────────────────────────────────────────

    def test_rejects_non_dict(self):
        assert validate_question("not a dict") is None
        assert validate_question(42) is None
        assert validate_question(None) is None
        assert validate_question(["q", "opts", "ans", "expl"]) is None

    def test_rejects_missing_question_key(self):
        q = _make_q()
        del q["question"]
        assert validate_question(q) is None

    def test_rejects_missing_options_key(self):
        q = _make_q()
        del q["options"]
        assert validate_question(q) is None

    def test_rejects_missing_answer_key(self):
        q = _make_q()
        del q["answer"]
        assert validate_question(q) is None

    def test_rejects_missing_explanation_key(self):
        q = _make_q()
        del q["explanation"]
        assert validate_question(q) is None

    def test_rejects_question_too_short(self):
        assert validate_question(_make_q(question="Short?")) is None

    def test_rejects_empty_question(self):
        assert validate_question(_make_q(question="")) is None

    def test_rejects_options_not_a_list(self):
        assert validate_question(_make_q(options="A, B, C, D")) is None

    def test_rejects_fewer_than_four_options(self):
        assert validate_question(_make_q(options=["A", "B", "C"])) is None

    def test_rejects_more_than_four_options(self):
        assert validate_question(_make_q(options=["A", "B", "C", "D", "E"])) is None

    def test_rejects_duplicate_options(self):
        opts = ["Same option", "Same option", "Option C", "Option D"]
        assert validate_question(_make_q(options=opts, answer="Same option")) is None

    def test_rejects_empty_option_string(self):
        opts = ["Option A", "", "Option C", "Option D"]
        assert validate_question(_make_q(options=opts, answer="Option A")) is None

    def test_rejects_answer_not_in_options(self):
        q = _make_q(answer="This option does not exist in the list at all")
        assert validate_question(q) is None

    def test_rejects_empty_explanation(self):
        assert validate_question(_make_q(explanation="")) is None
        assert validate_question(_make_q(explanation="   ")) is None


# ─────────────────────────────────────────────────────────────────────────────
# TestValidateQuiz — batch validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateQuiz:
    """Tests for quiz.validate_quiz (batch)."""

    def test_all_valid_questions_pass(self):
        result = validate_quiz(SAMPLE_QUESTIONS)
        assert len(result) == len(SAMPLE_QUESTIONS)

    def test_invalid_questions_are_dropped(self):
        mixed = [
            _make_q(),
            {"question": "bad", "options": [], "answer": "", "explanation": ""},
            None,
            "not a dict",
            _make_q(
                question="What is the capital of France?",
                options=["Berlin", "London", "Paris", "Rome"],
                answer="Paris",
                explanation="Paris is the capital.",
            ),
        ]
        result = validate_quiz(mixed)
        assert len(result) == 2  # only two valid questions

    def test_empty_list_returns_empty(self):
        assert validate_quiz([]) == []

    def test_all_invalid_returns_empty(self):
        invalid = [None, "string", 42, {"x": 1}]
        assert validate_quiz(invalid) == []


# ─────────────────────────────────────────────────────────────────────────────
# TestScoreQuiz — scoring logic
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreQuiz:
    """Tests for quiz.score_quiz — pure logic, no Gemini calls."""

    def test_all_correct(self):
        answers = {
            0: SAMPLE_QUESTIONS[0]["answer"],
            1: SAMPLE_QUESTIONS[1]["answer"],
        }
        result = score_quiz(SAMPLE_QUESTIONS, answers)
        assert result["score"] == 2
        assert result["total"] == 2
        assert result["percent"] == 100.0

    def test_all_wrong(self):
        answers = {
            0: SAMPLE_QUESTIONS[0]["options"][1],  # a wrong option
            1: SAMPLE_QUESTIONS[1]["options"][0],  # a wrong option
        }
        result = score_quiz(SAMPLE_QUESTIONS, answers)
        assert result["score"] == 0
        assert result["percent"] == 0.0

    def test_partial_score(self):
        answers = {
            0: SAMPLE_QUESTIONS[0]["answer"],   # correct
            1: SAMPLE_QUESTIONS[1]["options"][0],  # wrong
        }
        result = score_quiz(SAMPLE_QUESTIONS, answers)
        assert result["score"] == 1
        assert result["percent"] == 50.0

    def test_empty_answers(self):
        result = score_quiz(SAMPLE_QUESTIONS, {})
        assert result["score"] == 0
        assert result["percent"] == 0.0

    def test_results_length_matches_questions(self):
        result = score_quiz(SAMPLE_QUESTIONS, {})
        assert len(result["results"]) == len(SAMPLE_QUESTIONS)

    def test_result_dict_has_required_fields(self):
        result = score_quiz(SAMPLE_QUESTIONS, {0: SAMPLE_QUESTIONS[0]["answer"]})
        for r in result["results"]:
            for key in ("correct", "chosen", "answer", "explanation", "difficulty"):
                assert key in r, f"Missing key '{key}' in result dict"

    def test_percent_rounded_to_one_decimal(self):
        # 1 correct out of 3 → 33.33…% → should round to 33.3
        three_qs = [_make_q() for _ in range(3)]
        answers = {0: three_qs[0]["answer"]}
        result = score_quiz(three_qs, answers)
        assert result["percent"] == 33.3

    def test_zero_questions_returns_zero_percent(self):
        result = score_quiz([], {})
        assert result["score"] == 0
        assert result["total"] == 0
        assert result["percent"] == 0.0
        assert result["results"] == []

    def test_unanswered_question_counts_as_wrong(self):
        answers = {}  # no answers provided
        result = score_quiz(SAMPLE_QUESTIONS, answers)
        for r in result["results"]:
            assert r["correct"] is False
            assert r["chosen"] == ""

    def test_correct_flag_is_bool(self):
        answers = {0: SAMPLE_QUESTIONS[0]["answer"]}
        result = score_quiz(SAMPLE_QUESTIONS, answers)
        for r in result["results"]:
            assert isinstance(r["correct"], bool)

    def test_difficulty_propagated_in_results(self):
        answers = {}
        result = score_quiz(SAMPLE_QUESTIONS, answers)
        assert result["results"][0]["difficulty"] == "easy"
        assert result["results"][1]["difficulty"] == "medium"


# ─────────────────────────────────────────────────────────────────────────────
# TestParseJsonResponse — internal JSON parser
# ─────────────────────────────────────────────────────────────────────────────

class TestParseJsonResponse:
    """Tests for quiz._parse_json_response (internal helper)."""

    def test_plain_json_array(self):
        raw = json.dumps([{"a": 1}])
        result = _parse_json_response(raw)
        assert result == [{"a": 1}]

    def test_json_inside_markdown_fence(self):
        payload = [{"question": "Q?", "options": ["A", "B", "C", "D"],
                    "answer": "A", "explanation": "E."}]
        raw = f"```json\n{json.dumps(payload)}\n```"
        result = _parse_json_response(raw)
        assert result == payload

    def test_json_inside_plain_code_fence(self):
        payload = [{"x": 42}]
        raw = f"```\n{json.dumps(payload)}\n```"
        result = _parse_json_response(raw)
        assert result == payload

    def test_json_with_surrounding_prose(self):
        payload = [{"q": "test"}]
        raw = f"Here are your questions:\n{json.dumps(payload)}\nHope that helps!"
        result = _parse_json_response(raw)
        assert result == payload

    def test_raises_on_completely_invalid_json(self):
        with pytest.raises((ValueError, Exception)):
            _parse_json_response("this is not json at all")
