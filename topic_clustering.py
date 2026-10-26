# topic_clustering.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Identify the main topics discussed in a lecture by clustering
#          transcript chunks using local sentence embeddings and agglomerative
#          hierarchical clustering.
#
# Key design decisions:
#   - Sentence-Transformers runs fully on CPU — no GPU required.
#   - Agglomerative clustering is preferred over KMeans because it:
#       * does not require the number of clusters as a fixed input,
#       * is deterministic (no random seed dependency),
#       * uses Ward linkage which minimises within-cluster variance and
#         naturally avoids tiny/meaningless clusters.
#   - Silhouette analysis is used to select the best number of topics
#     when the caller does not specify one.
#   - Each topic carries its chunk texts, a readable label, a one-sentence
#     summary, and all preserved timestamps from the original TextChunk objects.
#   - Short transcripts (< MIN_CHUNKS_FOR_CLUSTERING chunks) are returned as
#     a single topic without attempting clustering.
#
# Configurable via environment variables:
#   EMBEDDING_MODEL   — Sentence-Transformers model name
#                       (default: "all-MiniLM-L6-v2"; cpu-only, 22 MB)
#   MAX_TOPICS        — Hard upper bound on the number of topics (default: 8)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MAX_TOPICS = 8
MIN_CHUNKS_FOR_CLUSTERING = 3   # below this → single "overview" topic
MAX_LABEL_CHARS = 90            # truncate representative sentence in label
MIN_SILHOUETTE_IMPROVEMENT = 0.01  # require this gain before accepting extra cluster


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicCluster:
    """
    A single identified topic, with all associated metadata.

    Attributes:
        topic_id:          Zero-based topic index.
        label:             Short human-readable topic name (≤ 90 chars).
        summary:           One-sentence summary of the topic's content.
        chunk_texts:       The raw text of every chunk assigned to this topic.
        timestamps:        List of (start_time, end_time) tuples in seconds.
                           May be empty if input had no timestamps.
        chunk_indices:     Original chunk indices (position in the input list).
        representative:    The single chunk text closest to the topic centroid.
    """
    topic_id: int
    label: str
    summary: str
    chunk_texts: list[str]
    timestamps: list[tuple[Optional[float], Optional[float]]] = field(default_factory=list)
    chunk_indices: list[int] = field(default_factory=list)
    representative: str = ""

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def first_timestamp(self) -> Optional[float]:
        """Start time of the earliest chunk in this topic."""
        starts = [s for s, _ in self.timestamps if s is not None]
        return min(starts) if starts else None

    @property
    def last_timestamp(self) -> Optional[float]:
        """End time of the latest chunk in this topic."""
        ends = [e for _, e in self.timestamps if e is not None]
        return max(ends) if ends else None

    @property
    def time_range_fmt(self) -> str:
        """Return a formatted time range string like '02:15 – 06:40'."""
        def _fmt(secs: Optional[float]) -> str:
            if secs is None:
                return "?"
            t = max(0, int(round(float(secs))))
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        s = _fmt(self.first_timestamp)
        e = _fmt(self.last_timestamp)
        if s == "?" and e == "?":
            return ""
        return f"{s} – {e}"

    @property
    def word_count(self) -> int:
        return sum(len(t.split()) for t in self.chunk_texts)


@dataclass
class ClusteringResult:
    """
    The complete output of `cluster_transcript_chunks`.

    Attributes:
        topics:            Ordered list of :class:`TopicCluster` objects.
        n_topics:          Number of topics found.
        embedding_model:   Name of the embedding model used.
        auto_selected_k:   True if the number of topics was chosen automatically.
        was_short:         True if the transcript was too short to cluster.
        silhouette:        Best silhouette score achieved (None for single topic).
    """
    topics: list[TopicCluster]
    n_topics: int
    embedding_model: str
    auto_selected_k: bool
    was_short: bool = False
    silhouette: Optional[float] = None

    def as_topic_list(self) -> list[str]:
        """Return just the topic labels — compatible with `st.session_state.topics`."""
        return [t.label for t in self.topics]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Load (or hot-reload) a Sentence-Transformers model on CPU.

    The model is cached by the library after the first download.

    Args:
        model_name: Model name / path.  Defaults to ``EMBEDDING_MODEL`` env var
                    or ``"all-MiniLM-L6-v2"`` (22 MB, CPU-only, very fast).

    Returns:
        A loaded :class:`SentenceTransformer` instance on CPU.
    """
    resolved = (
        model_name
        or os.getenv("EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
    # Force CPU device so no GPU is required
    return SentenceTransformer(resolved, device="cpu")


def embed_chunks(
    texts: list[str],
    model: Optional[SentenceTransformer] = None,
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Convert a list of text strings into L2-normalised dense embeddings.

    Args:
        texts:      List of strings to embed.
        model:      Pre-loaded SentenceTransformer.  If ``None``, one is loaded.
        model_name: Model name passed to :func:`load_embedding_model` when
                    ``model`` is not provided.
        batch_size: Encoding batch size.

    Returns:
        Float32 NumPy array of shape ``(len(texts), embedding_dim)``,
        L2-normalised so cosine similarity equals dot product.
    """
    if model is None:
        model = load_embedding_model(model_name)
    raw = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=batch_size,
        normalize_embeddings=False,
    )
    return normalize(raw.astype(np.float32), norm="l2")


# ─────────────────────────────────────────────────────────────────────────────
# Optimal-k selection via silhouette analysis
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(
    embeddings: np.ndarray,
    max_k: int = DEFAULT_MAX_TOPICS,
    min_k: int = 2,
) -> tuple[int, float]:
    """
    Select the number of clusters that maximises the silhouette score using
    agglomerative clustering with Ward linkage.

    The search range is ``[min_k, min(max_k, n-1)]`` where ``n`` is the number
    of samples.  If the range is empty, returns ``(1, 0.0)`` (single cluster).

    Args:
        embeddings: L2-normalised embedding matrix.
        max_k:      Upper bound on the number of clusters to evaluate.
        min_k:      Lower bound (must be ≥ 2).

    Returns:
        ``(best_k, best_silhouette_score)``
    """
    n = len(embeddings)
    effective_max = min(max_k, n - 1)

    if effective_max < min_k:
        return 1, 0.0

    best_k = min_k
    best_score = -1.0

    # Agglomerative Ward linkage requires euclidean metric
    for k in range(min_k, effective_max + 1):
        try:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
                metric="euclidean",
            )
            labels = model.fit_predict(embeddings)
            # Silhouette requires at least 2 distinct labels
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            # Only accept a higher k if it gives a meaningful improvement
            if score > best_score + MIN_SILHOUETTE_IMPROVEMENT or k == min_k:
                best_score = score
                best_k = k
        except Exception as exc:  # noqa: BLE001
            logger.debug("Silhouette evaluation failed for k=%d: %s", k, exc)
            continue

    return best_k, max(best_score, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Label & summary generation (fully local, no API calls)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_for_label(text: str) -> str:
    """Strip noise markers and collapse whitespace for use in a topic label."""
    text = re.sub(r"\[Uncertain\s*/\s*Missing[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_topic_label(
    chunk_texts: list[str],
    embeddings: np.ndarray,
    centroid: np.ndarray,
) -> tuple[str, str]:
    """
    Derive a readable label and one-sentence summary from the cluster's content.

    Strategy:
    1.  Find the chunk whose embedding is closest to the cluster centroid
        (the "representative chunk").
    2.  Extract the **first complete sentence** from that chunk as the label.
    3.  If that sentence is very short, fall back to the first 90 characters.
    4.  The summary is the first 200 characters of the representative chunk.

    Args:
        chunk_texts: List of text strings in the cluster.
        embeddings:  Embeddings corresponding to ``chunk_texts``.
        centroid:    Mean embedding of the cluster.

    Returns:
        ``(label, summary)`` — both are plain strings.
    """
    # Find representative chunk (closest to centroid)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    rep_idx = int(np.argmin(distances))
    rep_text = _clean_for_label(chunk_texts[rep_idx])

    # Extract first sentence for the label
    sentences = re.split(r"(?<=[.!?])\s+", rep_text)
    first_sent = sentences[0].strip() if sentences else rep_text

    # Truncate or pad
    if len(first_sent) > MAX_LABEL_CHARS:
        label = first_sent[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + "…"
    elif len(first_sent) < 15 and len(rep_text) > 15:
        # Too short — use more text
        label = rep_text[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + "…"
    else:
        label = first_sent

    summary = rep_text[:200] + ("…" if len(rep_text) > 200 else "")
    return label, summary


# ─────────────────────────────────────────────────────────────────────────────
# Core clustering pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _extract_chunk_fields(chunk: Any, index: int) -> tuple[str, Optional[float], Optional[float]]:
    """
    Extract (text, start_time, end_time) from a chunk of any supported type:
    - TextChunk object (has .text, .start_time, .end_time attributes)
    - dict with "text", "start"/"start_time", "end"/"end_time" keys
    - plain string
    """
    if hasattr(chunk, "text"):
        return (
            str(chunk.text).strip(),
            getattr(chunk, "start_time", None),
            getattr(chunk, "end_time", None),
        )
    if isinstance(chunk, dict):
        return (
            str(chunk.get("text", "")).strip(),
            chunk.get("start_time", chunk.get("start")),
            chunk.get("end_time", chunk.get("end")),
        )
    return str(chunk).strip(), None, None


def cluster_transcript_chunks(
    chunks: list[Any],
    n_topics: Optional[int] = None,
    max_topics: Optional[int] = None,
    model: Optional[SentenceTransformer] = None,
    model_name: Optional[str] = None,
) -> ClusteringResult:
    """
    Cluster transcript chunks into semantic topic groups.

    This is the **main entry point** for this module.

    Args:
        chunks:     List of transcript chunks.  Each element may be a:
                    - :class:`text_processing.TextChunk` object
                    - plain string
                    - dict with at least a ``"text"`` key
        n_topics:   If provided, use exactly this many topics.
                    If ``None``, the best number is chosen automatically via
                    silhouette analysis.
        max_topics: Upper bound for auto-selection.  Defaults to the
                    ``MAX_TOPICS`` env var or ``DEFAULT_MAX_TOPICS`` (8).
        model:      Pre-loaded :class:`SentenceTransformer`.  Loaded on first
                    call if omitted.
        model_name: Model name forwarded to :func:`load_embedding_model` when
                    ``model`` is ``None``.

    Returns:
        A :class:`ClusteringResult` with all topics, timestamps, labels, and
        metadata.
    """
    effective_max = (
        max_topics
        or int(os.getenv("MAX_TOPICS", DEFAULT_MAX_TOPICS))
    )
    embedding_model_name = (
        (model_name if model_name else None)
        or os.getenv("EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )

    # ── Extract texts and timestamps ─────────────────────────────────────────
    texts: list[str] = []
    starts: list[Optional[float]] = []
    ends: list[Optional[float]] = []

    for i, chunk in enumerate(chunks):
        text, start, end = _extract_chunk_fields(chunk, i)
        if text:
            texts.append(text)
            starts.append(start)
            ends.append(end)

    # ── Guard: empty input ───────────────────────────────────────────────────
    if not texts:
        return ClusteringResult(
            topics=[TopicCluster(
                topic_id=0,
                label="No content available",
                summary="The transcript appears to be empty.",
                chunk_texts=[],
                timestamps=[],
                chunk_indices=[],
                representative="",
            )],
            n_topics=1,
            embedding_model=embedding_model_name,
            auto_selected_k=False,
            was_short=True,
            silhouette=None,
        )

    # ── Guard: too short to cluster meaningfully ─────────────────────────────
    if len(texts) < MIN_CHUNKS_FOR_CLUSTERING:
        combined = " ".join(texts)
        topic = TopicCluster(
            topic_id=0,
            label=_clean_for_label(texts[0])[:MAX_LABEL_CHARS],
            summary=combined[:200] + ("…" if len(combined) > 200 else ""),
            chunk_texts=texts,
            timestamps=list(zip(starts, ends)),
            chunk_indices=list(range(len(texts))),
            representative=texts[0],
        )
        return ClusteringResult(
            topics=[topic],
            n_topics=1,
            embedding_model=embedding_model_name,
            auto_selected_k=False,
            was_short=True,
            silhouette=None,
        )

    # ── Load model if not provided ───────────────────────────────────────────
    if model is None:
        model = load_embedding_model(embedding_model_name)

    # ── Embed all chunks ─────────────────────────────────────────────────────
    try:
        embeddings = embed_chunks(texts, model=model)
    except Exception as exc:
        logger.warning("Embedding failed: %s — returning single topic.", exc)
        return _fallback_single_topic(texts, starts, ends, embedding_model_name)

    # ── Determine number of clusters ─────────────────────────────────────────
    auto_selected = n_topics is None
    best_silhouette: Optional[float] = None

    if n_topics is None:
        n_topics, best_silhouette = find_optimal_k(embeddings, max_k=effective_max)
    else:
        n_topics = max(1, min(n_topics, len(texts) - 1) if len(texts) > 1 else 1)

    # ── Cluster ──────────────────────────────────────────────────────────────
    if n_topics == 1 or len(texts) == 1:
        labels_arr = np.zeros(len(texts), dtype=int)
    else:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_topics,
                linkage="ward",
                metric="euclidean",
            )
            labels_arr = clustering.fit_predict(embeddings)
        except Exception as exc:
            logger.warning("Clustering failed: %s — returning single topic.", exc)
            return _fallback_single_topic(texts, starts, ends, embedding_model_name)

    # ── Build TopicCluster objects (sorted by first appearance in transcript) ─
    cluster_map: dict[int, dict] = {}
    for i, (label_id, text, start, end) in enumerate(
        zip(labels_arr.tolist(), texts, starts, ends)
    ):
        label_id = int(label_id)
        if label_id not in cluster_map:
            cluster_map[label_id] = {
                "texts": [],
                "emb_indices": [],
                "timestamps": [],
                "chunk_indices": [],
                "first_index": i,
            }
        cluster_map[label_id]["texts"].append(text)
        cluster_map[label_id]["emb_indices"].append(i)
        cluster_map[label_id]["timestamps"].append((start, end))
        cluster_map[label_id]["chunk_indices"].append(i)

    # Sort clusters by first appearance index so topics appear in transcript order
    sorted_ids = sorted(cluster_map.keys(), key=lambda cid: cluster_map[cid]["first_index"])

    topics: list[TopicCluster] = []
    for new_id, cluster_id in enumerate(sorted_ids):
        info = cluster_map[cluster_id]
        cluster_texts = info["texts"]
        cluster_emb_indices = info["emb_indices"]
        cluster_embeddings = embeddings[cluster_emb_indices]
        centroid = cluster_embeddings.mean(axis=0)

        label, summary = generate_topic_label(cluster_texts, cluster_embeddings, centroid)

        # Representative = chunk closest to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        rep_idx = int(np.argmin(distances))
        representative = cluster_texts[rep_idx]

        topics.append(TopicCluster(
            topic_id=new_id,
            label=label,
            summary=summary,
            chunk_texts=cluster_texts,
            timestamps=info["timestamps"],
            chunk_indices=info["chunk_indices"],
            representative=representative,
        ))

    return ClusteringResult(
        topics=topics,
        n_topics=len(topics),
        embedding_model=embedding_model_name,
        auto_selected_k=auto_selected,
        was_short=False,
        silhouette=best_silhouette,
    )


def _fallback_single_topic(
    texts: list[str],
    starts: list[Optional[float]],
    ends: list[Optional[float]],
    model_name: str,
) -> ClusteringResult:
    """Return a single-topic ClusteringResult when clustering fails."""
    label = _clean_for_label(texts[0])[:MAX_LABEL_CHARS] if texts else "Lecture Content"
    combined = " ".join(texts)
    topic = TopicCluster(
        topic_id=0,
        label=label,
        summary=combined[:200] + ("…" if len(combined) > 200 else ""),
        chunk_texts=texts,
        timestamps=list(zip(starts, ends)),
        chunk_indices=list(range(len(texts))),
        representative=texts[0] if texts else "",
    )
    return ClusteringResult(
        topics=[topic],
        n_topics=1,
        embedding_model=model_name,
        auto_selected_k=False,
        was_short=True,
        silhouette=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy / backward-compatible API  (keeps old callers working)
# ─────────────────────────────────────────────────────────────────────────────

def embed_sentences(sentences: list[str], model: SentenceTransformer) -> np.ndarray:
    """Backward-compatible wrapper around :func:`embed_chunks`."""
    return embed_chunks(sentences, model=model)


def cluster_sentences(
    sentences: list[str],
    n_clusters: Optional[int] = None,
    max_k: int = DEFAULT_MAX_TOPICS,
) -> dict[int, list[str]]:
    """
    Backward-compatible API: group sentences into clusters.

    Returns a dict mapping cluster_id → list of sentences.
    """
    result = cluster_transcript_chunks(sentences, n_topics=n_clusters, max_topics=max_k)
    return {t.topic_id: t.chunk_texts for t in result.topics}


def get_cluster_labels(
    clusters: dict[int, list[str]],
    model_instance: Optional[SentenceTransformer] = None,
) -> dict[int, str]:
    """
    Backward-compatible API: return a representative label per cluster.

    Returns a dict mapping cluster_id → label string.
    """
    if model_instance is None:
        model_instance = load_embedding_model()

    labels: dict[int, str] = {}
    for cluster_id, texts in clusters.items():
        if not texts:
            labels[cluster_id] = f"Topic {cluster_id + 1}"
            continue
        embs = embed_chunks(texts, model=model_instance)
        centroid = embs.mean(axis=0)
        label, _ = generate_topic_label(texts, embs, centroid)
        labels[cluster_id] = label
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load model once and reuse (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

_CACHED_MODEL: Optional[SentenceTransformer] = None
_CACHED_MODEL_NAME: Optional[str] = None


def get_cached_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Return a module-level cached SentenceTransformer, loading it only once.

    Subsequent calls with the same model name reuse the cached instance.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    resolved = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    if _CACHED_MODEL is None or _CACHED_MODEL_NAME != resolved:
        _CACHED_MODEL = load_embedding_model(resolved)
        _CACHED_MODEL_NAME = resolved
    return _CACHED_MODEL
