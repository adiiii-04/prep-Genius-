# test_topic_clustering.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit & integration tests for topic_clustering.py.
#
# Run with:
#   & "C:\Users\Aditya\AppData\Local\Python\bin\python.exe" -m pytest test_topic_clustering.py -v
#
# All tests are offline-safe:
#   - SentenceTransformer.encode() is mocked with deterministic random vectors.
#   - No GPU, no network calls, no API keys needed.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from text_processing import TextChunk


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_model(n_dim: int = 32, seed: int = 42) -> MagicMock:
    """
    Return a mock SentenceTransformer whose .encode() returns deterministic
    random unit vectors — same behaviour as the real model but instant & free.
    """
    rng = np.random.default_rng(seed)

    def _encode(texts, show_progress_bar=False, convert_to_numpy=True,
                 batch_size=32, normalize_embeddings=False):
        vecs = rng.standard_normal((len(texts), n_dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    mock = MagicMock()
    mock.encode.side_effect = _encode
    return mock


def _make_text_chunks(n: int, start_offset: float = 0.0) -> list[TextChunk]:
    """Create n synthetic TextChunk objects with 30-second intervals."""
    chunks = []
    for i in range(n):
        start = start_offset + i * 30.0
        chunks.append(TextChunk(
            text=f"Lecture content segment {i}: discussing important topic area {i % 3}.",
            chunk_index=i,
            start_time=start,
            end_time=start + 30.0,
        ))
    return chunks


def _make_plain_strings(n: int) -> list[str]:
    return [f"[{i//2:02d}:{(i%2)*30:02d}] Content about subject area number {i}." for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data structures
# ─────────────────────────────────────────────────────────────────────────────

class TestTopicCluster:
    from topic_clustering import TopicCluster

    def _make_tc(self, topic_id=0, timestamps=None):
        from topic_clustering import TopicCluster
        return TopicCluster(
            topic_id=topic_id,
            label="Introduction to Algorithms",
            summary="Overview of algorithm complexity.",
            chunk_texts=["text A", "text B"],
            timestamps=timestamps or [(10.0, 40.0), (50.0, 80.0)],
            chunk_indices=[0, 1],
            representative="text A",
        )

    def test_first_timestamp(self):
        tc = self._make_tc(timestamps=[(30.0, 60.0), (10.0, 40.0)])
        assert tc.first_timestamp == 10.0

    def test_last_timestamp(self):
        tc = self._make_tc(timestamps=[(10.0, 40.0), (50.0, 80.0)])
        assert tc.last_timestamp == 80.0

    def test_time_range_fmt_standard(self):
        tc = self._make_tc(timestamps=[(135.0, 210.0)])
        fmt = tc.time_range_fmt
        assert "02:15" in fmt
        assert "03:30" in fmt

    def test_time_range_fmt_no_timestamps(self):
        tc = self._make_tc(timestamps=[(None, None)])
        assert tc.time_range_fmt == ""  # both unknown -> empty string

    def test_word_count(self):
        tc = self._make_tc()
        assert tc.word_count == 4  # "text A" + "text B"

    def test_clustering_result_as_topic_list(self):
        from topic_clustering import ClusteringResult, TopicCluster
        tc1 = TopicCluster(0, "Topic A", "Sum A", ["a"], [], [], "a")
        tc2 = TopicCluster(1, "Topic B", "Sum B", ["b"], [], [], "b")
        result = ClusteringResult(
            topics=[tc1, tc2], n_topics=2, embedding_model="model",
            auto_selected_k=True,
        )
        assert result.as_topic_list() == ["Topic A", "Topic B"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. embed_chunks
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedChunks:

    def test_returns_float32_array(self):
        from topic_clustering import embed_chunks
        mock_model = _make_fake_model()
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            result = embed_chunks(["hello world", "foo bar"])
        assert result.dtype == np.float32

    def test_shape_matches_inputs(self):
        from topic_clustering import embed_chunks
        mock_model = _make_fake_model()
        texts = ["a", "b", "c", "d"]
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            result = embed_chunks(texts, model=mock_model)
        assert result.shape[0] == 4

    def test_rows_are_unit_vectors(self):
        from topic_clustering import embed_chunks
        mock_model = _make_fake_model()
        result = embed_chunks(["hello", "world"], model=mock_model)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. find_optimal_k
# ─────────────────────────────────────────────────────────────────────────────

class TestFindOptimalK:

    def _random_embeddings(self, n: int, dim: int = 32, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_returns_int(self):
        from topic_clustering import find_optimal_k
        embs = self._random_embeddings(10)
        k, score = find_optimal_k(embs, max_k=4)
        assert isinstance(k, int)
        assert isinstance(score, float)

    def test_k_within_range(self):
        from topic_clustering import find_optimal_k
        embs = self._random_embeddings(10)
        k, _ = find_optimal_k(embs, max_k=5, min_k=2)
        assert 1 <= k <= 5

    def test_returns_one_for_too_few_samples(self):
        from topic_clustering import find_optimal_k
        embs = self._random_embeddings(2)
        k, score = find_optimal_k(embs, max_k=8)
        assert k == 1
        assert score == 0.0

    def test_score_is_non_negative(self):
        from topic_clustering import find_optimal_k
        embs = self._random_embeddings(15)
        _, score = find_optimal_k(embs, max_k=5)
        assert score >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. generate_topic_label
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateTopicLabel:

    def _random_unit(self, n: int, dim: int = 32) -> np.ndarray:
        rng = np.random.default_rng(7)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_label_is_string(self):
        from topic_clustering import generate_topic_label
        texts = ["Binary search divides the array in half.", "Log n complexity."]
        embs = self._random_unit(2)
        centroid = embs.mean(axis=0)
        label, summary = generate_topic_label(texts, embs, centroid)
        assert isinstance(label, str)
        assert isinstance(summary, str)

    def test_label_not_empty(self):
        from topic_clustering import generate_topic_label
        texts = ["First point.", "Second point is about algorithms."]
        embs = self._random_unit(2)
        centroid = embs.mean(axis=0)
        label, _ = generate_topic_label(texts, embs, centroid)
        assert len(label.strip()) > 0

    def test_label_respects_max_length(self):
        from topic_clustering import generate_topic_label, MAX_LABEL_CHARS
        long_text = "A " * 200  # very long text
        texts = [long_text, "Short."]
        embs = self._random_unit(2)
        centroid = embs[0:1]  # bias toward first chunk
        label, _ = generate_topic_label(texts, embs, centroid)
        assert len(label) <= MAX_LABEL_CHARS + 1  # +1 for ellipsis

    def test_uncertainty_markers_stripped_from_label(self):
        from topic_clustering import generate_topic_label
        texts = ["[Uncertain / Missing from audio: inaudible]. Core concept explained here."]
        embs = self._random_unit(1)
        label, _ = generate_topic_label(texts, embs, embs[0])
        assert "[Uncertain" not in label

    def test_summary_respects_200_chars(self):
        from topic_clustering import generate_topic_label
        texts = ["Word " * 100]
        embs = self._random_unit(1)
        _, summary = generate_topic_label(texts, embs, embs[0])
        assert len(summary) <= 204  # 200 + "…" + small buffer


# ─────────────────────────────────────────────────────────────────────────────
# 5. cluster_transcript_chunks — the main entry point
# ─────────────────────────────────────────────────────────────────────────────

class TestClusterTranscriptChunks:

    def _run(self, chunks, n_topics=None, max_topics=5):
        from topic_clustering import cluster_transcript_chunks
        mock_model = _make_fake_model()
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            return cluster_transcript_chunks(
                chunks,
                n_topics=n_topics,
                max_topics=max_topics,
                model=mock_model,
            )

    # ── Empty & short inputs ─────────────────────────────────────────────────

    def test_empty_input_returns_single_topic(self):
        result = self._run([])
        assert result.n_topics == 1
        assert result.was_short is True
        assert result.topics[0].label == "No content available"

    def test_single_chunk_returns_single_topic(self):
        result = self._run(["One lone chunk of text from the lecture."])
        assert result.n_topics == 1
        assert result.was_short is True

    def test_two_chunks_returns_single_topic(self):
        result = self._run(["Chunk A text here.", "Chunk B text here."])
        assert result.n_topics == 1
        assert result.was_short is True

    # ── Standard inputs ──────────────────────────────────────────────────────

    def test_returns_clustering_result(self):
        from topic_clustering import ClusteringResult
        chunks = _make_plain_strings(8)
        result = self._run(chunks)
        assert isinstance(result, ClusteringResult)

    def test_topics_count_within_max(self):
        chunks = _make_plain_strings(12)
        result = self._run(chunks, max_topics=4)
        assert result.n_topics <= 4

    def test_topics_count_positive(self):
        chunks = _make_plain_strings(10)
        result = self._run(chunks)
        assert result.n_topics >= 1

    def test_explicit_n_topics_respected(self):
        chunks = _make_plain_strings(10)
        result = self._run(chunks, n_topics=3, max_topics=6)
        assert result.n_topics == 3

    def test_auto_selected_flag_set_when_n_topics_none(self):
        chunks = _make_plain_strings(8)
        result = self._run(chunks, n_topics=None)
        assert result.auto_selected_k is True

    def test_auto_selected_flag_false_when_n_topics_given(self):
        chunks = _make_plain_strings(8)
        result = self._run(chunks, n_topics=2)
        assert result.auto_selected_k is False

    # ── TextChunk inputs (timestamps preserved) ───────────────────────────────

    def test_text_chunk_timestamps_preserved(self):
        chunks = _make_text_chunks(6)
        result = self._run(chunks, n_topics=2)
        # Every topic must have at least one chunk with real timestamps
        for topic in result.topics:
            for start, end in topic.timestamps:
                # Timestamps might be None for chunks without them, but
                # our TextChunks always have timestamps
                assert start is not None or end is not None

    def test_text_chunk_first_timestamp_non_negative(self):
        chunks = _make_text_chunks(6, start_offset=120.0)
        result = self._run(chunks)
        for topic in result.topics:
            if topic.first_timestamp is not None:
                assert topic.first_timestamp >= 0.0

    def test_time_range_fmt_populated_for_text_chunks(self):
        chunks = _make_text_chunks(6)
        result = self._run(chunks, n_topics=2)
        for topic in result.topics:
            # All our TextChunks have timestamps, so fmt should not be ""
            assert topic.time_range_fmt != ""

    # ── Topic content ────────────────────────────────────────────────────────

    def test_all_input_chunks_covered(self):
        chunks = _make_plain_strings(10)
        result = self._run(chunks, n_topics=3)
        all_texts = [t for topic in result.topics for t in topic.chunk_texts]
        # Every chunk assigned to at least one topic
        assert len(all_texts) == 10

    def test_no_topic_is_empty(self):
        chunks = _make_plain_strings(9)
        result = self._run(chunks, n_topics=3)
        for topic in result.topics:
            assert len(topic.chunk_texts) > 0

    def test_every_topic_has_label(self):
        chunks = _make_plain_strings(9)
        result = self._run(chunks)
        for topic in result.topics:
            assert isinstance(topic.label, str)
            assert len(topic.label.strip()) > 0

    def test_every_topic_has_summary(self):
        chunks = _make_plain_strings(9)
        result = self._run(chunks)
        for topic in result.topics:
            assert isinstance(topic.summary, str)
            assert len(topic.summary.strip()) > 0

    def test_every_topic_has_representative(self):
        chunks = _make_plain_strings(9)
        result = self._run(chunks)
        for topic in result.topics:
            assert isinstance(topic.representative, str)

    def test_topic_indices_are_sequential(self):
        chunks = _make_plain_strings(8)
        result = self._run(chunks, n_topics=3)
        for i, topic in enumerate(result.topics):
            assert topic.topic_id == i

    def test_as_topic_list_length_matches_n_topics(self):
        chunks = _make_plain_strings(9)
        result = self._run(chunks)
        assert len(result.as_topic_list()) == result.n_topics

    # ── dict chunk inputs ────────────────────────────────────────────────────

    def test_dict_chunks_supported(self):
        chunks = [
            {"text": f"Segment {i} about topic {i%2}.", "start": i*30.0, "end": i*30.0+30.0}
            for i in range(6)
        ]
        result = self._run(chunks)
        assert result.n_topics >= 1
        for topic in result.topics:
            assert len(topic.chunk_texts) > 0

    # ── Embedding model name ─────────────────────────────────────────────────

    def test_embedding_model_recorded(self):
        chunks = _make_plain_strings(5)
        from topic_clustering import cluster_transcript_chunks
        mock_model = _make_fake_model()
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            result = cluster_transcript_chunks(
                chunks, model=mock_model, model_name="custom-bert"
            )
        assert result.embedding_model == "custom-bert"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Backward-compatible API
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatibleAPI:

    def test_cluster_sentences_returns_dict(self):
        from topic_clustering import cluster_sentences
        mock_model = _make_fake_model()
        sentences = [f"Sentence number {i} about subject." for i in range(8)]
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            result = cluster_sentences(sentences, n_clusters=2, max_k=4)
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_cluster_sentences_covers_all_inputs(self):
        from topic_clustering import cluster_sentences
        mock_model = _make_fake_model()
        sentences = [f"Sentence {i}." for i in range(6)]
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            result = cluster_sentences(sentences, n_clusters=2)
        all_sents = [s for lst in result.values() for s in lst]
        assert len(all_sents) == 6

    def test_get_cluster_labels_returns_dict_of_strings(self):
        from topic_clustering import get_cluster_labels
        mock_model = _make_fake_model()
        clusters = {
            0: ["Binary search is logarithmic.", "Divide and conquer approach."],
            1: ["Dynamic programming uses memoization.", "Overlapping subproblems."],
        }
        result = get_cluster_labels(clusters, model_instance=mock_model)
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1}
        for v in result.values():
            assert isinstance(v, str)
            assert len(v.strip()) > 0

    def test_get_cluster_labels_empty_cluster_fallback(self):
        from topic_clustering import get_cluster_labels
        mock_model = _make_fake_model()
        clusters = {0: [], 1: ["Real content here."]}
        result = get_cluster_labels(clusters, model_instance=mock_model)
        assert result[0] == "Topic 1"
        assert isinstance(result[1], str)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Edge cases and error resilience
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCasesAndResilience:

    def test_all_empty_strings_returns_single_topic(self):
        from topic_clustering import cluster_transcript_chunks
        mock_model = _make_fake_model()
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            result = cluster_transcript_chunks(
                ["", "   ", "\n"], model=mock_model
            )
        assert result.n_topics == 1
        assert result.was_short is True

    def test_n_topics_larger_than_inputs_clamped(self):
        from topic_clustering import cluster_transcript_chunks
        mock_model = _make_fake_model()
        chunks = _make_plain_strings(4)
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            result = cluster_transcript_chunks(
                chunks, n_topics=50, model=mock_model
            )
        assert result.n_topics <= len(chunks)

    def test_embedding_failure_returns_single_topic(self):
        from topic_clustering import cluster_transcript_chunks
        bad_model = MagicMock()
        bad_model.encode.side_effect = RuntimeError("CUDA OOM")
        with patch("topic_clustering.load_embedding_model", return_value=bad_model):
            result = cluster_transcript_chunks(
                _make_plain_strings(6), model=bad_model
            )
        assert result.n_topics == 1
        assert result.was_short is True

    def test_env_max_topics_respected(self):
        from topic_clustering import cluster_transcript_chunks
        mock_model = _make_fake_model()
        with patch.dict(os.environ, {"MAX_TOPICS": "2"}):
            with patch("topic_clustering.load_embedding_model", return_value=mock_model):
                result = cluster_transcript_chunks(
                    _make_plain_strings(10), model=mock_model
                )
        assert result.n_topics <= 2

    def test_env_embedding_model_used_when_no_model_kwarg(self):
        from topic_clustering import cluster_transcript_chunks
        mock_model = _make_fake_model()
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "paraphrase-MiniLM-L3-v2"}):
            with patch("topic_clustering.load_embedding_model", return_value=mock_model) as loader:
                cluster_transcript_chunks(_make_plain_strings(5))
                loader.assert_called_once_with("paraphrase-MiniLM-L3-v2")

    def test_no_gpu_required(self):
        """Verify load_embedding_model forces CPU device."""
        from topic_clustering import load_embedding_model
        with patch("topic_clustering.SentenceTransformer") as MockST:
            MockST.return_value = _make_fake_model()
            load_embedding_model("all-MiniLM-L6-v2")
            MockST.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Integration test with realistic segment data (small transcript)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationSmallTranscript:
    """
    End-to-end: realistic data flow from text_processing.preprocess_segments
    → cluster_transcript_chunks → ClusteringResult.
    """

    SAMPLE_SEGMENTS = [
        {"text": "Welcome to our lecture on data structures and algorithms.", "start": 0.0,  "duration": 5.0},
        {"text": "We start with sorting algorithms such as merge sort and quick sort.", "start": 5.0,  "duration": 6.0},
        {"text": "Merge sort is a divide-and-conquer algorithm with O(n log n) complexity.", "start": 11.0, "duration": 5.5},
        {"text": "Quick sort has average O(n log n) but worst-case O(n squared).", "start": 16.5, "duration": 5.5},
        {"text": "Now we move to graph traversal: breadth-first and depth-first search.", "start": 22.0, "duration": 6.0},
        {"text": "BFS uses a queue and explores neighbours level by level.", "start": 28.0, "duration": 5.0},
        {"text": "DFS uses a stack or recursion and explores as far as possible before backtracking.", "start": 33.0, "duration": 6.0},
        {"text": "Finally, dynamic programming solves problems by storing subproblem solutions.", "start": 39.0, "duration": 5.0},
        {"text": "The Fibonacci series is a classic DP example using memoization.", "start": 44.0, "duration": 5.0},
        {"text": "The knapsack problem is another key DP example covered in exams.", "start": 49.0, "duration": 5.0},
    ]

    def _cluster_with_mock(self, chunks, n_topics=None):
        from topic_clustering import cluster_transcript_chunks
        mock_model = _make_fake_model(seed=99)
        with patch("topic_clustering.load_embedding_model", return_value=mock_model):
            return cluster_transcript_chunks(chunks, n_topics=n_topics, max_topics=5, model=mock_model)

    def test_preprocess_then_cluster(self):
        from text_processing import preprocess_segments
        processed = preprocess_segments(self.SAMPLE_SEGMENTS, chunk_size=30, overlap=5)
        result = self._cluster_with_mock(processed.chunks)
        assert result.n_topics >= 1
        assert not result.was_short or len(processed.chunks) < 3

    def test_all_chunks_covered_after_integration(self):
        from text_processing import preprocess_segments
        processed = preprocess_segments(self.SAMPLE_SEGMENTS, chunk_size=30, overlap=5)
        result = self._cluster_with_mock(processed.chunks)
        total_chunks_in_topics = sum(len(t.chunk_texts) for t in result.topics)
        # Every non-empty chunk must appear in exactly one topic
        non_empty_inputs = [c for c in processed.chunks if c.text.strip()]
        assert total_chunks_in_topics == len(non_empty_inputs)

    def test_timestamps_preserved_from_text_chunks(self):
        from text_processing import preprocess_segments
        processed = preprocess_segments(self.SAMPLE_SEGMENTS, chunk_size=30, overlap=5)
        result = self._cluster_with_mock(processed.chunks)
        for topic in result.topics:
            # Topics built from TextChunks with timestamps should have them
            for start, end in topic.timestamps:
                if start is not None:
                    assert start >= 0.0
                if end is not None:
                    assert end >= 0.0

    def test_topics_in_transcript_order(self):
        """Topics should be ordered by their first appearance in the transcript."""
        from text_processing import preprocess_segments
        processed = preprocess_segments(self.SAMPLE_SEGMENTS, chunk_size=30, overlap=5)
        if len(processed.chunks) < 3:
            pytest.skip("Not enough chunks for ordering test")
        result = self._cluster_with_mock(processed.chunks, n_topics=3)
        # Verify first_timestamp is non-decreasing across topics
        first_times = [t.first_timestamp for t in result.topics if t.first_timestamp is not None]
        assert first_times == sorted(first_times)
