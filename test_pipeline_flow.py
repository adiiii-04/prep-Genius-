# test_pipeline_flow.py
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end integration test of the complete PrepGenius flow:
# Transcript -> Text Processing -> LangChain Summarizer -> Revision Notes & Exports
#
# Offline-safe: Mocks YouTube and OpenAI network calls.
# ─────────────────────────────────────────────────────────────────────────────

import os
from unittest.mock import MagicMock, patch
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from text_processing import preprocess_segments
from summarizer import generate_revision_notes
from topic_clustering import cluster_transcript_chunks
from export import export_markdown, export_pdf


MOCK_TRANSCRIPT_SEGMENTS = [
    {"text": "Welcome to our lecture on Algorithm Design and Big O notation.", "start": 0.0, "duration": 4.5},
    {"text": "[Music] Today we analyze time and space complexity.", "start": 4.5, "duration": 5.0},
    {"text": "A recurrence relation is defined as an equation expressing T of n in terms of smaller inputs.", "start": 9.5, "duration": 6.0},
    {"text": "For example, binary search splits the search space in half at each step.", "start": 15.5, "duration": 5.5},
    {"text": "The formula for binary search recurrence is T(n) = T(n/2) + O(1).", "start": 21.0, "duration": 5.0},
    {"text": "[Uncertain / Missing from audio: student microphone cut off].", "start": 26.0, "duration": 4.0},
    {"text": "In summary, master the recurrence tree method and Master Theorem for the exam.", "start": 30.0, "duration": 6.0},
]

MOCK_MASTER_REVISION_NOTES = """## 1. Lecture Overview
This lecture introduces fundamental concepts of algorithm design and asymptotic analysis, focusing on recurrence relations and binary search efficiency.

## 2. Key Concepts
- Recurrence relations express computational complexity recursively [00:09 - 00:15].
- Binary search operates by iteratively halving the search space [00:15 - 00:21].

## 3. Definitions
- **Recurrence Relation**: An equation that defines a sequence or function based on a rule to find subsequent terms from earlier ones.
- **Asymptotic Analysis**: Evaluating algorithm behavior as the input size grows towards infinity.

## 4. Important Formulas
- Binary search recurrence: T(n) = T(n/2) + O(1)
- Solution via Master Theorem: T(n) = O(log n)

## 5. Examples & Case Studies
- Binary search application on sorted arrays [00:15].
- [Uncertain / Missing from audio: Student microphone cut off at 00:26].

## 6. Final Revision Checklist
- [ ] Write the recurrence relation for binary search
- [ ] State and apply the Master Theorem cases
- [ ] Distinguish between best-case, average-case, and worst-case bounds
"""


def test_complete_flow_integration(tmp_path):
    """Test full pipeline from raw segments to structured revision notes and file exports."""

    # 1. Text processing step
    processed = preprocess_segments(MOCK_TRANSCRIPT_SEGMENTS, chunk_size=50, overlap=10)
    assert not processed.was_empty
    assert processed.word_count > 0
    assert len(processed.chunks) >= 1
    assert processed.chunks[0].start_time is not None

    # 2. AI Summarisation step with LangChain model
    mock_model = FakeListChatModel(responses=[
        "Chunk 1 draft notes [00:00 - 00:15]",
        MOCK_MASTER_REVISION_NOTES,
    ])

    progress_log = []
    def _prog(cur, tot):
        progress_log.append((cur, tot))

    notes = generate_revision_notes(
        processed.chunks,
        format_type="detailed",
        model=mock_model,
        show_progress=_prog,
    )

    # 3. Verify all 6 mandatory sections are present in Revision Notes
    assert "## 1. Lecture Overview" in notes
    assert "## 2. Key Concepts" in notes
    assert "## 3. Definitions" in notes
    assert "## 4. Important Formulas" in notes
    assert "## 5. Examples & Case Studies" in notes
    assert "## 6. Final Revision Checklist" in notes

    # 4. Verify technical accuracy and uncertainty preservation
    assert "T(n) = T(n/2) + O(1)" in notes
    assert "[Uncertain / Missing from audio" in notes
    assert "[00:09 - 00:15]" in notes

    # 4b. Topic clustering step with Agglomerative Clustering & Sentence-Transformers
    clustering = cluster_transcript_chunks(processed.chunks)
    assert clustering.n_topics >= 1
    assert len(clustering.topics) == clustering.n_topics
    for topic in clustering.topics:
        assert len(topic.label) > 0
        assert len(topic.summary) > 0
        assert topic.word_count > 0
        assert topic.timestamps is not None
        assert topic.first_timestamp is not None
        assert topic.time_range_fmt != ""

    # 5. Verify exports work seamlessly with the generated revision notes
    mock_fc = [{"front": "What is binary search complexity?", "back": "O(log n)"}]
    mock_quiz = [{
        "question": "What is the recurrence relation for binary search?",
        "options": ["T(n) = T(n/2) + O(1)", "T(n) = 2T(n/2) + O(n)", "T(n) = T(n-1) + O(1)", "T(n) = O(n)"],
        "answer": "T(n) = T(n/2) + O(1)",
        "explanation": "Binary search halves the input and performs constant work per step.",
    }]

    md_path = export_markdown(notes, mock_fc, mock_quiz, title="Test Integration Notes")
    assert md_path.endswith(".md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    assert "## 1. Lecture Overview" in md_content
    assert "## 6. Final Revision Checklist" in md_content

    pdf_path = export_pdf(notes, mock_fc, mock_quiz, title="Test Integration Notes")
    assert pdf_path.endswith(".pdf")
    assert os.path.exists(pdf_path)
