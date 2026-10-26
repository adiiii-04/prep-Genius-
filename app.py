"""
PrepGenius — Lecture-to-Study Notes Converter
Real pipeline: YouTube transcript → Gemini AI → Notes / Flashcards / Quiz / Export
"""

import streamlit as st
import time
import io
import csv
import os
import re
import logging

from dotenv import load_dotenv

# Load .env FIRST — before any os.getenv() calls
load_dotenv()

# ── Safe Logging Configuration (masks secrets) ────────────────────────────────
class SecretMaskingFilter(logging.Filter):
    """Filter that automatically masks API keys and tokens from all log messages."""
    _PATTERNS = [
        re.compile(r"sk-[A-Za-z0-9_\-]{15,}"),
        re.compile(r"AIza[A-Za-z0-9_\-]{30,}"),
        re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE),
        re.compile(r"nvapi-[A-Za-z0-9_\-]{10,}"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pat in self._PATTERNS:
                record.msg = pat.sub("[REDACTED_API_KEY]", record.msg)
        return True

def setup_app_logging() -> logging.Logger:
    app_logger = logging.getLogger("prepgenius")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        handler.addFilter(SecretMaskingFilter())
        app_logger.addHandler(handler)
        app_logger.setLevel(logging.INFO)
    return app_logger

logger = setup_app_logging()

# ── Page config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="PrepGenius",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  -webkit-font-smoothing: antialiased;
}
:root {
  --primary: #6366F1;
  --primary-hover: #4F46E5;
  --primary-light: rgba(99, 102, 241, 0.15);
  --secondary: #10B981;
  --accent: #F59E0B;
  --danger: #EF4444;
  --surface: #131B2E;
  --surface-2: #0B0F19;
  --border: #1E293B;
  --text-primary: #F8FAFC;
  --text-secondary: #CBD5E1;
  --text-muted: #94A3B8;
}
.stApp {
  background: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.14) 0px, transparent 50%),
              radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%),
              #0B0F19;
  color: #F8FAFC;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3.5rem; max-width: 1140px; }

/* Hero */
.hero-wrap {
  background: linear-gradient(135deg, #0B0F19 0%, #131B2E 35%, #1E1B4B 75%, #312E81 100%);
  border-radius: 24px;
  padding: 2.8rem 2.8rem 2.2rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.6);
  border: 1px solid rgba(99, 102, 241, 0.25);
}
.hero-wrap::before {
  content: "";
  position: absolute;
  top: -80px;
  right: -80px;
  width: 280px;
  height: 280px;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.25) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-wrap::after {
  content: "";
  position: absolute;
  bottom: -60px;
  left: 200px;
  width: 200px;
  height: 200px;
  background: radial-gradient(circle, rgba(16, 185, 129, 0.18) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(99, 102, 241, 0.25);
  color: #C7D2FE;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 9999px;
  margin-bottom: 0.9rem;
  backdrop-filter: blur(8px);
  border: 1px solid rgba(199, 210, 254, 0.3);
}
.hero-title {
  font-size: 2.75rem;
  font-weight: 800;
  color: #FFF;
  margin: 0 0 0.5rem;
  letter-spacing: -0.8px;
  line-height: 1.15;
}
.hero-sub {
  font-size: 1.05rem;
  color: #CBD5E1;
  margin: 0 0 1.6rem;
  max-width: 650px;
  line-height: 1.55;
}
.hero-stats {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  padding-top: 1.2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}
.stat-item {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 0.6rem 1.1rem;
  backdrop-filter: blur(4px);
}
.stat-num {
  font-size: 1.35rem;
  font-weight: 800;
  color: #FFF;
  display: block;
}
.stat-lbl {
  font-size: 0.68rem;
  color: #94A3B8;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-weight: 600;
}

/* Input card */
.input-card {
  background: #131B2E;
  border-radius: 20px;
  padding: 2rem;
  border: 1px solid #1E293B;
  box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35);
  margin-bottom: 1.8rem;
  transition: all 0.2s ease;
}
.input-card:hover {
  border-color: #312E81;
  box-shadow: 0 15px 35px -5px rgba(99, 102, 241, 0.15);
}
.section-label {
  font-size: 0.75rem;
  font-weight: 700;
  color: #94A3B8;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}

/* Inputs */
.stTextInput > div > div > input {
  background: #0B0F19 !important;
  color: #F8FAFC !important;
  border: 1px solid #334155 !important;
  border-radius: 12px !important;
  padding: 0.75rem 1rem !important;
  font-size: 0.95rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
  border-color: #6366F1 !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}
.stSelectbox > div > div {
  background: #0B0F19 !important;
  color: #F8FAFC !important;
  border: 1px solid #334155 !important;
  border-radius: 12px !important;
}

/* Primary button */
div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #7C3AED 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.75rem 1.8rem !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  box-shadow: 0 6px 20px -2px rgba(79, 70, 229, 0.5) !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 25px -2px rgba(79, 70, 229, 0.7) !important;
}

/* Progress steps */
.progress-wrap {
  background: #131B2E;
  border-radius: 16px;
  padding: 1.5rem 1.8rem;
  border: 1px solid #1E293B;
  margin-bottom: 1.8rem;
  box-shadow: 0 4px 15px -2px rgba(0, 0, 0, 0.25);
}
.progress-title {
  font-size: 0.82rem;
  font-weight: 700;
  color: #94A3B8;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.steps-row { display: flex; align-items: center; }
.step-item { display: flex; flex-direction: column; align-items: center; flex: 1; position: relative; }
.step-item:not(:last-child)::after {
  content: "";
  position: absolute;
  top: 16px;
  left: calc(50% + 16px);
  width: calc(100% - 32px);
  height: 3px;
  background: #1E293B;
  z-index: 0;
  border-radius: 3px;
}
.step-item.done:not(:last-child)::after { background: var(--secondary); }
.step-item.active:not(:last-child)::after {
  background: linear-gradient(90deg, var(--secondary), #1E293B);
}
.step-circle {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  font-weight: 700;
  z-index: 1;
  border: 2px solid #334155;
  background: #0B0F19;
  color: #94A3B8;
  transition: all 0.3s ease;
}
.step-item.done .step-circle {
  background: var(--secondary);
  border-color: var(--secondary);
  color: white;
}
.step-item.active .step-circle {
  background: var(--primary);
  border-color: var(--primary);
  color: white;
  box-shadow: 0 0 0 5px rgba(99, 102, 241, 0.25);
}
.step-lbl { font-size: 0.72rem; color: #94A3B8; margin-top: 0.5rem; text-align: center; font-weight: 600; }
.step-item.done .step-lbl { color: var(--secondary); }
.step-item.active .step-lbl { color: #A5B4FC; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: #131B2E !important;
  border-radius: 14px !important;
  padding: 6px !important;
  gap: 4px !important;
  border: 1px solid #1E293B !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  color: #94A3B8 !important;
  padding: 0.55rem 1.2rem !important;
  transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
  background: #4F46E5 !important;
  color: white !important;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.45) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem !important; }

/* Metric row */
.metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1rem; margin-bottom: 1.8rem; }
.metric-card {
  background: #131B2E;
  border: 1px solid #1E293B;
  border-radius: 16px;
  padding: 1.2rem 1.4rem;
  box-shadow: 0 4px 15px -2px rgba(0, 0, 0, 0.25);
  position: relative;
  overflow: hidden;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
  transform: translateY(-2px);
  border-color: #312E81;
  box-shadow: 0 8px 25px -4px rgba(99, 102, 241, 0.2);
}
.metric-card::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #6366F1, #10B981);
}
.metric-val { font-size: 1.85rem; font-weight: 800; color: #FFFFFF; line-height: 1.1; margin-bottom: 0.2rem; }
.metric-name { font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700; }

/* Topic pill & cards */
.topic-pill {
  display: inline-flex;
  align-items: center;
  background: #1E293B;
  border: 1px solid #334155;
  border-radius: 9999px;
  padding: 0.35rem 0.9rem;
  font-size: 0.8rem;
  font-weight: 600;
  color: #E2E8F0;
  margin: 0.25rem;
}
.topic-card {
  background: #131B2E;
  border: 1px solid #1E293B;
  border-left: 5px solid #6366F1;
  border-radius: 14px;
  padding: 1.25rem 1.4rem;
  margin-bottom: 1rem;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
  transition: all 0.2s ease;
}
.topic-card:hover {
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
  border-color: #6366F1;
}
.topic-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem; gap: 0.8rem; }
.topic-title { font-size: 0.98rem; font-weight: 800; color: #F8FAFC; margin: 0; }
.topic-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: rgba(99, 102, 241, 0.15);
  color: #A5B4FC;
  border: 1px solid rgba(99, 102, 241, 0.3);
  font-size: 0.72rem;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 9999px;
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
}
.topic-summary { font-size: 0.88rem; color: #CBD5E1; line-height: 1.55; margin-bottom: 0.6rem; }
.topic-meta { display: flex; gap: 1.2rem; font-size: 0.75rem; color: #94A3B8; font-weight: 600; }

/* Flashcards */
.flashcard {
  background: #131B2E;
  border: 1px solid #1E293B;
  border-radius: 16px;
  padding: 1.4rem;
  margin-bottom: 1rem;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
  transition: all 0.2s ease;
}
.flashcard:hover {
  box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
  border-color: #6366F1;
  transform: translateY(-1px);
}
.fc-q { font-weight: 700; color: #F8FAFC; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0.8rem; }
.fc-a {
  background: #0B0F19;
  border: 1px solid #1E293B;
  border-left: 3px solid #10B981;
  border-radius: 10px;
  padding: 0.85rem 1rem;
  color: #E2E8F0;
  font-size: 0.88rem;
  line-height: 1.55;
}
.fc-badge {
  background: #4F46E5;
  color: white;
  font-size: 0.65rem;
  font-weight: 800;
  padding: 3px 9px;
  border-radius: 9999px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

/* Transcript */
.transcript-block {
  background: #0B0F19;
  color: #E2E8F0;
  border-radius: 14px;
  padding: 1.5rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  line-height: 1.7;
  max-height: 420px;
  overflow-y: auto;
  border: 1px solid #1E293B;
}
.ts-time { color: #34D399; font-weight: 700; margin-right: 0.6rem; }

/* Download buttons */
.stDownloadButton > button {
  border-radius: 10px !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  padding: 0.6rem 1.2rem !important;
  transition: all 0.2s ease !important;
  width: 100% !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
}
.stDownloadButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 18px rgba(79, 70, 229, 0.3) !important;
}

/* Alerts */
.err-box {
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.35);
  border-left: 5px solid #EF4444;
  border-radius: 12px;
  padding: 1rem 1.25rem;
  color: #FCA5A5;
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 1.2rem;
}
.err-box strong { display: block; margin-bottom: 0.25rem; color: #F87171; }
.ok-box {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.35);
  border-left: 5px solid #10B981;
  border-radius: 12px;
  padding: 1rem 1.25rem;
  color: #6EE7B7;
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 1.2rem;
}
.warn-box {
  background: rgba(245, 158, 11, 0.12);
  border: 1px solid rgba(245, 158, 11, 0.35);
  border-left: 5px solid #F59E0B;
  border-radius: 12px;
  padding: 1rem 1.25rem;
  color: #FCD34D;
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 1.2rem;
}

/* Empty state / Feature Cards */
.empty-hero {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.2rem;
  margin-top: 1.8rem;
}
.feature-box {
  background: #131B2E;
  border: 1px solid #1E293B;
  border-radius: 18px;
  padding: 1.6rem;
  box-shadow: 0 4px 15px rgba(0,0,0,0.25);
  text-align: left;
  transition: all 0.2s ease;
}
.feature-box:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px rgba(99, 102, 241, 0.2);
  border-color: #6366F1;
}
.feature-icon { font-size: 2.2rem; margin-bottom: 0.8rem; display: block; }
.feature-title { font-size: 1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.4rem; }
.feature-desc { font-size: 0.83rem; color: #94A3B8; line-height: 1.55; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE MAP  →  BCP-47 codes for youtube-transcript-api
# ═══════════════════════════════════════════════════════════════════════════════
LANGUAGES = {
    "🇺🇸 English":             ["en", "en-US", "en-GB"],
    "🇪🇸 Spanish":             ["es", "es-ES", "es-MX"],
    "🇫🇷 French":              ["fr", "fr-FR"],
    "🇩🇪 German":              ["de", "de-DE"],
    "🇮🇳 Hindi":               ["hi"],
    "🇨🇳 Chinese (Simplified)":["zh-Hans", "zh", "zh-CN"],
    "🇯🇵 Japanese":            ["ja"],
    "🇵🇹 Portuguese":          ["pt", "pt-BR", "pt-PT"],
    "🇸🇦 Arabic":              ["ar"],
    "🇷🇺 Russian":             ["ru"],
    "🇮🇹 Italian":             ["it"],
    "🇰🇷 Korean":              ["ko"],
}

PROGRESS_STEPS = [
    "Transcript Extraction",
    "Text Processing",
    "AI Summarisation",
    "Export Preparation",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
def _init():
    defaults = dict(
        generated=False, generating=False, error=None,
        progress_step=-1,
        url="",
        language="🇺🇸 English", notes_format="Detailed",
        raw_transcript="", transcript_segments=[],
        revision_notes="", topics=[], flashcards=[], quiz=[],
        quiz_submitted=False,
        topic_clustering_result=None, target_n_topics=None,
        num_flashcards=15,
        # Quiz interactive-mode state
        quiz_index=0,           # 0-based index of the current question
        quiz_answers={},        # {question_index: chosen_option_str}
        quiz_difficulty="mixed",
        quiz_num_questions=10,
        quiz_show_results=False,
        quiz_generating=False,
        # Caching & Integration
        lecture_cache={},       # {cache_key: result_dict}
        cache_hit_message=None,
        lecture_title="PrepGenius — Study Notes",
        force_reprocess=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS & INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_video_id(url: str) -> str | None:
    """Extract an 11-character YouTube video ID from various standard URL formats."""
    if not url:
        return None
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|embed\/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None

def is_valid_youtube(url: str) -> bool:
    """Validate whether the provided string represents a legitimate YouTube video URL."""
    return extract_video_id(url) is not None

def validate_inputs(url: str) -> tuple[bool, str | None, str | None]:
    """
    Validate input URL and configuration before running the pipeline.
    Returns (is_valid, error_message, video_id).
    """
    url_clean = (url or "").strip()
    if not url_clean:
        return False, "Please enter a YouTube lecture URL before generating.", None

    if not re.search(r"(youtube\.com|youtu\.be)", url_clean, re.IGNORECASE):
        return False, "That doesn't look like a valid YouTube URL. Please use https://www.youtube.com/watch?v=… or https://youtu.be/…", None

    vid = extract_video_id(url_clean)
    if not vid:
        return False, "Could not extract a valid 11-character YouTube video ID from the provided link.", None

    return True, None, vid

def render_steps(current: int):
    icons = ["📥", "⚙️", "🤖", "📦"]
    parts = []
    for i, (step, icon) in enumerate(zip(PROGRESS_STEPS, icons)):
        cls = "step-item done" if i < current else ("step-item active" if i == current else "step-item")
        circle = "✓" if i < current else (icon if i == current else str(i + 1))
        parts.append(
            f'<div class="{cls}"><div class="step-circle">{circle}</div>'
            f'<span class="step-lbl">{step}</span></div>'
        )
    st.markdown(
        f'<div class="progress-wrap"><div class="progress-title">🔄 Processing Pipeline</div>'
        f'<div class="steps-row">{"".join(parts)}</div></div>',
        unsafe_allow_html=True,
    )

def build_markdown_export() -> bytes:
    from export import build_markdown_bytes
    doc_title = st.session_state.get("lecture_title") or "PrepGenius — Study Notes"
    return build_markdown_bytes(
        revision_notes=st.session_state.revision_notes,
        flashcards=st.session_state.flashcards,
        quiz_questions=st.session_state.quiz,
        title=doc_title,
        topics=st.session_state.topics or None,
    )

def build_csv_export() -> bytes:
    from export import build_csv_bytes
    return build_csv_bytes(st.session_state.flashcards, include_tags=True)

def build_anki_package_bytes() -> bytes:
    try:
        from flashcards import export_to_anki
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_anki(st.session_state.flashcards, deck_name="PrepGenius Lecture Deck", output_path=tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return data
    except Exception as e:
        return f"Anki package generation failed: {e}".encode()

def build_pdf_bytes() -> bytes:
    from export import build_pdf_bytes as _build_pdf
    doc_title = st.session_state.get("lecture_title") or "PrepGenius — Study Notes"
    try:
        return _build_pdf(
            revision_notes=st.session_state.revision_notes,
            flashcards=st.session_state.flashcards,
            quiz_questions=st.session_state.quiz,
            title=doc_title,
            topics=st.session_state.topics or None,
        )
    except Exception as e:
        return f"PDF generation failed: {e}".encode()

def build_quiz_json_export() -> bytes:
    """Build a JSON export of the current quiz questions and results (if available)."""
    from export import build_quiz_json_bytes
    quiz = st.session_state.quiz
    answers = st.session_state.get("quiz_answers", {})
    results = None
    if quiz and answers:
        from quiz import score_quiz
        results = score_quiz(quiz, answers)
    doc_title = st.session_state.get("lecture_title") or "PrepGenius — Practice Quiz"
    return build_quiz_json_bytes(
        quiz_questions=quiz,
        quiz_results=results,
        title=doc_title,
    )



# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE  (runs inside st.session_state.generating == True)
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline(url: str, lang_key: str, bar, step_text):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lang_codes = LANGUAGES.get(lang_key, ["en"])

    # ── Step 1: Transcript ────────────────────────────────────────────────────
    st.session_state.progress_step = 0
    bar.progress(5, "📥 Extracting transcript…")
    step_text.markdown("**Step 1/4 — Transcript Extraction**")

    logger.info("Pipeline Step 1: Fetching YouTube transcript for URL: %s", url)
    from transcript import fetch_transcript, get_transcript_with_timestamps
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_raw  = pool.submit(fetch_transcript, url, lang_codes)
            fut_segs = pool.submit(get_transcript_with_timestamps, url)
            raw  = fut_raw.result()
            segs = fut_segs.result()
    except Exception as e:
        logger.error("Transcript extraction failed: %s", e)
        raise RuntimeError(f"Transcript extraction failed: {e}")

    st.session_state.raw_transcript      = raw
    st.session_state.transcript_segments = segs
    vid = extract_video_id(url)
    st.session_state.lecture_title = f"Lecture Study Notes ({vid})" if vid else "PrepGenius — Study Notes"

    # ── Step 2: Text processing ───────────────────────────────────────────────
    st.session_state.progress_step = 1
    bar.progress(22, "⚙️ Cleaning and chunking text…")
    step_text.markdown("**Step 2/4 — Text Processing**")
    logger.info("Pipeline Step 2: Preprocessing %d transcript segments", len(segs))

    from text_processing import preprocess_segments, report_processing_stats
    processed = preprocess_segments(
        segs,
        chunk_size=1200,
        overlap=80,
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    )
    report_processing_stats(processed)
    cleaned = processed.cleaned_text
    chunks  = processed.chunk_texts   # list[str] for the summariser

    # ── Step 3: AI Summarisation & Content Generation ─────────────────────────
    st.session_state.progress_step = 2
    bar.progress(35, "🤖 Generating revision notes…")
    step_text.markdown("**Step 3/4 — AI Summarisation & Flashcards**")

    n_cards = int(st.session_state.get("num_flashcards", 15))
    fc_result, quiz_result = None, None

    logger.info("Pipeline Step 3: Generating revision notes via LangChain model")
    from summarizer import generate_revision_notes
    from flashcards import generate_flashcards
    from quiz      import generate_quiz

    def _prog(done, total, status: str = ""):
        pct = 35 + int((done / total) * 30)
        if status:
            bar.progress(min(66, pct), status)
        elif done >= total:
            bar.progress(65, "✨ Synthesising master revision notes guide…")
        else:
            bar.progress(pct, f"🤖 Summarising chunk {done}/{total}…")

    notes_format = st.session_state.get("notes_format", "Detailed").lower()
    openai_model = (
        os.getenv("NEMOTRON_MODEL")
        or os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    )
    try:
        openai_temp = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    except ValueError:
        openai_temp = 0.2
    openai_base_url = (
        os.getenv("NVIDIA_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or None
    )
    openai_key = (
        os.getenv("NVIDIA_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or None
    )

    input_chunks = processed.chunks if (processed.chunks and any(c.text for c in processed.chunks)) else chunks

    notes = generate_revision_notes(
        input_chunks,
        format_type=notes_format,
        model_name=openai_model,
        temperature=openai_temp,
        api_key=openai_key,
        base_url=openai_base_url,
        show_progress=_prog,
    )
    st.session_state.revision_notes = notes

    # Flashcards + quiz fire simultaneously ── saves ~50% of AI wait time
    bar.progress(67, "⚡ Generating flashcards & quiz in parallel…")
    step_text.markdown("**Step 3/4 — Flashcards + Quiz (parallel)**")

    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_openai = bool(os.getenv("NVIDIA_API_KEY")) or bool(os.getenv("OPENAI_API_KEY"))

    if has_gemini or has_openai:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_fc   = pool.submit(generate_flashcards, notes, num_cards=n_cards, transcript_chunks=processed.chunks)
            fut_quiz = pool.submit(generate_quiz,       notes,  8) if has_gemini else None

            futs = [f for f in [fut_fc, fut_quiz] if f is not None]
            for fut in as_completed(futs):
                if fut is fut_fc:
                    try:
                        fc_result   = fut.result()
                    except Exception as e:
                        logger.warning("Flashcard generation failed: %s", e)
                        fc_result   = []
                    bar.progress(82, "🃏 Flashcards ready…")
                elif fut is fut_quiz:
                    try:
                        quiz_result = fut.result()
                    except Exception as e:
                        logger.warning("Quiz generation failed: %s", e)
                        quiz_result = []
                    bar.progress(90, "❓ Quiz ready…")
            if not has_gemini:
                quiz_result = []
                bar.progress(90, "ℹ️ Practice Quiz skipped (GEMINI_API_KEY not configured)")
    else:
        fc_result, quiz_result = [], []
        bar.progress(90, "ℹ️ Flashcards & Quiz skipped (no API key configured)")

    st.session_state.flashcards = fc_result   or []
    st.session_state.quiz       = quiz_result or []

    # ── Step 3b: Semantic topic clustering (Runs locally on CPU) ─────────────
    bar.progress(91, "🏷️ Clustering topics with Sentence-Transformers…")
    logger.info("Pipeline Step 3b: Running local CPU topic clustering")
    try:
        from topic_clustering import cluster_transcript_chunks, get_cached_model
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        cached_model = get_cached_model(embedding_model_name)
        target_n = st.session_state.get("target_n_topics", None)
        clustering_result = cluster_transcript_chunks(
            processed.chunks,
            n_topics=target_n,
            model=cached_model,
            model_name=embedding_model_name,
        )
        st.session_state.topics = clustering_result.topics            # list[TopicCluster]
        st.session_state.topic_clustering_result = clustering_result
    except Exception as _tc_err:
        logger.warning("Topic clustering failed: %s", _tc_err)
        fallback_topics = re.findall(r"^#{1,3}\s+(.+)$", notes, re.MULTILINE)
        if not fallback_topics and fc_result:
            fallback_topics = [c["front"][:60] for c in fc_result[:10]]
        st.session_state.topics = fallback_topics
        st.session_state.topic_clustering_result = None

    # ── Step 4: Export prep & Session Caching ─────────────────────────────────
    st.session_state.progress_step = 3
    bar.progress(97, "📦 Preparing exports & caching lecture…")
    step_text.markdown("**Step 4/4 — Export Preparation**")

    # Cache completed material to avoid reprocessing same lecture
    vid_key = extract_video_id(url) or url.strip()
    notes_fmt = st.session_state.get("notes_format", "Detailed")
    cache_key = f"{vid_key}::{lang_key}::{notes_fmt}::{n_cards}"
    st.session_state.lecture_cache[cache_key] = {
        "raw_transcript": st.session_state.raw_transcript,
        "transcript_segments": st.session_state.transcript_segments,
        "revision_notes": st.session_state.revision_notes,
        "topics": st.session_state.topics,
        "topic_clustering_result": st.session_state.topic_clustering_result,
        "flashcards": st.session_state.flashcards,
        "quiz": st.session_state.quiz,
        "lecture_title": st.session_state.lecture_title,
    }
    logger.info("Pipeline complete. Saved outputs to session cache under key: %s", cache_key)

    bar.progress(100, "✅ Done!")
    time.sleep(0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Configuration
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ API Configuration")

    st.markdown("#### 🤖 Revision Notes (Nemotron 3.5)")
    nvidia_key_input = st.text_input(
        "NVIDIA API Key",
        type="password",
        value=os.getenv("NVIDIA_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
        placeholder="nvapi-…",
        help="Required for live revision notes generation via NVIDIA Nemotron 3.5. Get a free key at https://build.nvidia.com",
    )
    if nvidia_key_input:
        os.environ["NVIDIA_API_KEY"] = nvidia_key_input.strip()

    nemotron_models = [
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "Custom",
    ]
    cur_model = os.getenv("NEMOTRON_MODEL") or os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    model_idx = nemotron_models.index(cur_model) if cur_model in nemotron_models else 2
    model_choice_box = st.selectbox(
        "Nemotron Model",
        nemotron_models,
        index=model_idx,
    )
    if model_choice_box == "Custom":
        custom_model = st.text_input(
            "Custom Model Name",
            value=cur_model if cur_model not in nemotron_models[:2] else "",
            placeholder="e.g. nvidia/llama-3.1-nemotron-70b-instruct",
        )
        if custom_model:
            os.environ["NEMOTRON_MODEL"] = custom_model.strip()
    else:
        os.environ["NEMOTRON_MODEL"] = model_choice_box

    # Temperature configuration
    try:
        cur_temp = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    except ValueError:
        cur_temp = 0.2
    temp_slider = st.slider(
        "Model Temperature",
        min_value=0.0,
        max_value=1.0,
        value=cur_temp,
        step=0.05,
        help="Lower values (0.0 - 0.3) preserve technical precision and reduce hallucination.",
    )
    os.environ["OPENAI_TEMPERATURE"] = str(temp_slider)

    # Optional Base URL for OpenAI-compatible services
    base_url_input = st.text_input(
        "Base URL (Optional)",
        value=os.getenv("NVIDIA_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", ""),
        placeholder="https://integrate.api.nvidia.com/v1",
        help="NVIDIA endpoint is used by default. Override for Ollama, Groq, OpenRouter, vLLM, or local models.",
    )
    if base_url_input:
        os.environ["NVIDIA_BASE_URL"] = base_url_input.strip()

    st.markdown("---")
    st.markdown("#### 🃏 Flashcards & Quiz (Gemini)")
    gemini_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        placeholder="AIza…",
        help="Optional: used for flashcards and quiz generation. Get free at https://aistudio.google.com/app/apikey",
    )
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input.strip()

    gemini_choice = st.selectbox(
        "Gemini model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        index=0,
    )
    os.environ["GEMINI_MODEL"] = gemini_choice

    num_cards_slider = st.slider(
        "Number of Flashcards",
        min_value=5,
        max_value=30,
        value=int(st.session_state.get("num_flashcards", 15)),
        step=1,
        help="Number of active-recall flashcards to generate.",
    )
    st.session_state.num_flashcards = num_cards_slider

    st.markdown("---")
    st.markdown("#### 🏷️ Topic Clustering (Local CPU)")
    emb_models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2", "all-mpnet-base-v2", "Custom"]
    cur_emb = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    emb_idx = emb_models.index(cur_emb) if cur_emb in emb_models else 3
    emb_choice = st.selectbox(
        "Embedding Model",
        emb_models,
        index=emb_idx,
        help="Runs locally on CPU with Sentence-Transformers. No GPU or API keys needed.",
    )
    if emb_choice == "Custom":
        custom_emb = st.text_input(
            "Custom Embedding Model",
            value=cur_emb if cur_emb not in emb_models[:3] else "",
            placeholder="e.g. BAAI/bge-small-en-v1.5",
        )
        if custom_emb:
            os.environ["EMBEDDING_MODEL"] = custom_emb.strip()
    else:
        os.environ["EMBEDDING_MODEL"] = emb_choice

    clustering_mode = st.radio(
        "Topic Count",
        ["Auto-detect", "Configured count"],
        horizontal=True,
        help="Auto-detect uses silhouette score to find optimal clusters. Configured lets you set a target count.",
    )
    if clustering_mode == "Configured count":
        fixed_topics = st.slider("Target Topics", min_value=2, max_value=10, value=4)
        st.session_state.target_n_topics = fixed_topics
    else:
        st.session_state.target_n_topics = None

    # Cache Management
    st.markdown("---")
    st.markdown("#### ⚡ Session Cache")
    cached_count = len(st.session_state.get("lecture_cache", {}))
    st.caption(f"Currently caching {cached_count} lecture{'s' if cached_count != 1 else ''} in memory.")
    if st.button("🧹 Clear Lecture Cache", use_container_width=True):
        st.session_state.lecture_cache = {}
        st.session_state.cache_hit_message = None
        st.success("Session cache cleared!")
        st.rerun()

    st.markdown("---")
    st.caption("🔒 Keys are stored in-memory for this session.\n\nYou can also add them to a `.env` file.")

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">✨ AI-Powered Study Tool</div>
  <div class="hero-title">🎓 PrepGenius</div>
  <p class="hero-sub">Turn long lectures into smart revision material — in seconds.</p>
  <div class="hero-stats">
    <div class="stat-item"><span class="stat-num">10×</span><span class="stat-lbl">Faster notes</span></div>
    <div class="stat-item"><span class="stat-num">5</span><span class="stat-lbl">Output formats</span></div>
    <div class="stat-item"><span class="stat-num">12+</span><span class="stat-lbl">Languages</span></div>
    <div class="stat-item"><span class="stat-num">100%</span><span class="stat-lbl">AI driven</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT CARD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="input-card">', unsafe_allow_html=True)

col_url, col_lang, col_fmt, col_btn = st.columns([3.5, 1.8, 1.7, 1.5])
with col_url:
    st.markdown('<p class="section-label">📎 YouTube Lecture URL</p>', unsafe_allow_html=True)
    url = st.text_input(
        "url",
        value=st.session_state.url,
        placeholder="https://www.youtube.com/watch?v=… or https://youtu.be/…",
        label_visibility="collapsed",
    )
    st.session_state.url = url
with col_lang:
    st.markdown('<p class="section-label">🌐 Output Language</p>', unsafe_allow_html=True)
    lang = st.selectbox(
        "lang",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.language),
        label_visibility="collapsed",
    )
    st.session_state.language = lang
with col_fmt:
    st.markdown('<p class="section-label">📋 Notes Format</p>', unsafe_allow_html=True)
    fmt = st.selectbox(
        "fmt",
        ["Detailed", "Concise"],
        index=0 if st.session_state.get("notes_format", "Detailed") == "Detailed" else 1,
        label_visibility="collapsed",
    )
    st.session_state.notes_format = fmt
with col_btn:
    st.markdown('<p class="section-label">&nbsp;</p>', unsafe_allow_html=True)
    generate_clicked = st.button("🚀 Generate", type="primary", use_container_width=True)

# 1-Click quick test sample buttons
st.markdown(
    '<div style="margin-top:10px;margin-bottom:4px;"><span style="font-size:0.75rem;font-weight:700;'
    'color:#94A3B8;letter-spacing:0.5px;text-transform:uppercase;">Try Sample Lectures:</span></div>',
    unsafe_allow_html=True,
)
c_ex1, c_ex2, c_ex3, c_ex_clear = st.columns([1.5, 1.5, 1.5, 1.0])
with c_ex1:
    if st.button("⚡ MIT Algo", use_container_width=True, help="Load MIT 6.006 Lecture 1 URL"):
        st.session_state.url = "https://www.youtube.com/watch?v=Kg4bqm1QDIC"
        st.session_state.error = None
        st.rerun()
with c_ex2:
    if st.button("🤖 Stanford ML", use_container_width=True, help="Load Stanford CS229 ML Lecture URL"):
        st.session_state.url = "https://www.youtube.com/watch?v=0IAPZzGSbME"
        st.session_state.error = None
        st.rerun()
with c_ex3:
    if st.button("🧬 CS50", use_container_width=True, help="Load Harvard CS50 Lecture URL"):
        st.session_state.url = "https://www.youtube.com/watch?v=BkxmsZ4iK6g"
        st.session_state.error = None
        st.rerun()
with c_ex_clear:
    if st.button("🧹 Clear", use_container_width=True, help="Clear input URL"):
        st.session_state.url = ""
        st.session_state.error = None
        st.session_state.cache_hit_message = None
        st.rerun()

# Optional force-reprocess checkbox to bypass session cache
c_cache_toggle, _ = st.columns([2.5, 3.5])
with c_cache_toggle:
    force_reprocess_cb = st.checkbox(
        "🔄 Force re-process (bypass session cache)",
        value=st.session_state.get("force_reprocess", False),
        help="Check this to force running the full pipeline again even if this lecture was already processed in this session.",
    )
    st.session_state.force_reprocess = force_reprocess_cb

st.markdown("</div>", unsafe_allow_html=True)

# Cache hit notification
if st.session_state.get("cache_hit_message"):
    st.markdown(
        f'<div class="ok-box">⚡ <strong>Session Cache:</strong> {st.session_state.cache_hit_message}</div>',
        unsafe_allow_html=True,
    )

# Error display
if st.session_state.error:
    st.markdown(
        f'<div class="err-box"><strong>⚠️ Error</strong><br>{st.session_state.error}</div>',
        unsafe_allow_html=True,
    )

# ── API key warning ──
_nvidia_key = os.getenv("NVIDIA_API_KEY", "")
_openai_key = os.getenv("OPENAI_API_KEY", "")
if not _nvidia_key and not _openai_key:
    st.markdown(
        '<div class="warn-box">⚠️ <strong>No NVIDIA API key detected.</strong> '
        'Please enter your NVIDIA API key in the sidebar (←) or add it to your <code>.env</code> file. '
        'Get a free key at <a href="https://build.nvidia.com" target="_blank">build.nvidia.com</a></div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════
if generate_clicked:
    st.session_state.error = None
    st.session_state.cache_hit_message = None
    raw_url = st.session_state.url.strip()

    # Input validation
    valid, err_msg, video_id = validate_inputs(raw_url)
    if not valid:
        st.session_state.error = err_msg
        logger.warning("Input validation failed: %s", err_msg)
        st.rerun()

    has_api_key = (
        bool(os.getenv("NVIDIA_API_KEY"))
        or (bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here")
    )
    if not has_api_key:
        st.session_state.error = (
            "NVIDIA API key is missing. Please enter your NVIDIA_API_KEY in the sidebar (←) or your .env file. "
            "Get a free key at https://build.nvidia.com"
        )
        logger.warning("Generation aborted: No NVIDIA API key")
        st.rerun()

    # Session caching check: avoid processing same lecture repeatedly
    cache_vid = video_id or raw_url
    cache_key = f"{cache_vid}::{st.session_state.language}::{st.session_state.notes_format}::{st.session_state.num_flashcards}"

    if cache_key in st.session_state.lecture_cache and not st.session_state.get("force_reprocess", False):
        cached = st.session_state.lecture_cache[cache_key]
        for k, v in cached.items():
            st.session_state[k] = v
        st.session_state.generated = True
        st.session_state.generating = False
        st.session_state.error = None
        st.session_state.quiz_submitted = False
        st.session_state.quiz_index = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_show_results = False
        st.session_state.cache_hit_message = f"Retrieved from session cache for '{cache_vid}'. Redundant processing avoided!"
        logger.info("Session cache hit for key: %s", cache_key)
        st.rerun()
    else:
        st.session_state.generating = True
        st.session_state.generated = False
        st.session_state.quiz_submitted = False
        st.session_state.quiz_index = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_show_results = False
        st.session_state.force_reprocess = False
        logger.info("Starting fresh pipeline run for: %s", cache_vid)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.generating:
    render_steps(st.session_state.progress_step)
    bar_ph = st.empty()
    step_ph = st.empty()
    try:
        run_pipeline(
            st.session_state.url.strip(),
            st.session_state.language,
            bar_ph,
            step_ph,
        )
        st.session_state.generating = False
        st.session_state.generated = True
        st.session_state.progress_step = len(PROGRESS_STEPS)
        bar_ph.empty()
        step_ph.empty()
        st.rerun()
    except Exception as exc:
        bar_ph.empty()
        step_ph.empty()
        st.session_state.generating = False
        st.session_state.generated = False
        st.session_state.error = str(exc)
        logger.error("Pipeline failed: %s", exc)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.generated:
    render_steps(len(PROGRESS_STEPS))
    st.markdown(
        '<div class="ok-box">✅ Study material generated! Explore the tabs below.</div>',
        unsafe_allow_html=True,
    )

    fc   = st.session_state.flashcards
    quiz = st.session_state.quiz
    tops = st.session_state.topics
    segs = st.session_state.transcript_segments

    # Metrics
    st.markdown(
        f"""<div class="metric-row">
          <div class="metric-card"><div class="metric-val">{len(tops)}</div><div class="metric-name">Topics</div></div>
          <div class="metric-card"><div class="metric-val">{len(fc)}</div><div class="metric-name">Flashcards</div></div>
          <div class="metric-card"><div class="metric-val">{len(quiz)}</div><div class="metric-name">Quiz Qs</div></div>
          <div class="metric-card"><div class="metric-val">{len(segs)}</div><div class="metric-name">Transcript Segs</div></div>
          <div class="metric-card"><div class="metric-val">{len(st.session_state.raw_transcript.split())}</div><div class="metric-name">Words</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Downloads Hub
    st.markdown("#### 📥 Download Center")
    dl1, dl2, dl3, dl4, dl5 = st.columns(5)
    with dl1:
        st.download_button("📄 PDF Report", data=build_pdf_bytes(),
                           file_name="prepgenius_notes.pdf", mime="application/pdf",
                           use_container_width=True)
    with dl2:
        st.download_button("📝 Markdown Notes", data=build_markdown_export(),
                           file_name="prepgenius_notes.md", mime="text/markdown",
                           use_container_width=True)
    with dl3:
        st.download_button("🃏 Anki CSV", data=build_csv_export(),
                           file_name="prepgenius_flashcards.csv", mime="text/csv",
                           use_container_width=True)
    with dl4:
        st.download_button("📦 Anki Deck (.apkg)", data=build_anki_package_bytes(),
                           file_name="prepgenius_deck.apkg", mime="application/octet-stream",
                           use_container_width=True)
    with dl5:
        st.download_button("❓ Quiz JSON", data=build_quiz_json_export(),
                           file_name="prepgenius_quiz.json", mime="application/json",
                           use_container_width=True)
    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_notes, tab_topics, tab_flash, tab_quiz, tab_transcript = st.tabs(
        ["📝 Revision Notes", "🏷️ Topics", "🃏 Flashcards", "❓ Practice Quiz", "📜 Transcript"]
    )

    # ── Revision Notes ─────────────────────────────────────────────────────
    with tab_notes:
        col_hdr, col_badge = st.columns([3.5, 1.5])
        with col_hdr:
            st.markdown("##### 📝 Structured Revision Notes")
            model_used = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            st.caption(f"Language: {st.session_state.language} · Model: {model_used} · LangChain engine")
        with col_badge:
            fmt_lbl = st.session_state.get("notes_format", "Detailed")
            st.markdown(
                f'<div style="text-align:right;"><span class="topic-pill" style="background:#EEF2FF;border-color:#6C63FF;color:#4F46E5;font-weight:600;">📋 {fmt_lbl} Notes</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(st.session_state.revision_notes)

    # ── Topics ────────────────────────────────────────────────────────────
    with tab_topics:
        col_thdr, col_tmeta = st.columns([3.5, 1.5])
        with col_thdr:
            st.markdown("##### 🏷️ Key Topics & Lecture Structure")
            emb_used = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            st.caption(f"Clustering: Agglomerative (Ward linkage) · Model: {emb_used} · Local CPU")
        with col_tmeta:
            st.markdown(
                f'<div style="text-align:right;"><span class="topic-pill" style="background:#ECFDF5;border-color:#10B981;color:#065F46;font-weight:600;">🏷️ {len(tops)} Topics</span></div>',
                unsafe_allow_html=True,
            )

        if tops:
            # Topic overview badges
            pills = []
            for t in tops:
                lbl = getattr(t, "label", str(t))
                pills.append(f'<span class="topic-pill">🔖 {lbl}</span>')
            st.markdown("".join(pills), unsafe_allow_html=True)
            st.markdown("---")

            # Detailed Topic Cards
            for i, t in enumerate(tops, 1):
                if hasattr(t, "label"):
                    ts_badge = f'<span class="topic-badge">⏱️ {t.time_range_fmt}</span>' if t.time_range_fmt else '<span class="topic-badge">Full Lecture</span>'
                    n_chunks = len(t.chunk_texts)
                    w_count = t.word_count
                    st.markdown(
                        f'<div class="topic-card">'
                        f'  <div class="topic-header">'
                        f'    <div class="topic-title">Topic {i}: {t.label}</div>'
                        f'    <div>{ts_badge}</div>'
                        f'  </div>'
                        f'  <div class="topic-summary">{t.summary}</div>'
                        f'  <div class="topic-meta">'
                        f'    <span>📄 {n_chunks} segment{"s" if n_chunks != 1 else ""}</span>'
                        f'    <span>📝 ~{w_count} words</span>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if t.chunk_texts:
                        with st.expander(f"🔍 View {n_chunks} source segment{'s' if n_chunks != 1 else ''} for Topic {i}"):
                            for idx, (ct, (ts_start, ts_end)) in enumerate(zip(t.chunk_texts, t.timestamps), 1):
                                if ts_start is not None and ts_end is not None:
                                    sm, ss = divmod(int(round(ts_start)), 60)
                                    em, es = divmod(int(round(ts_end)), 60)
                                    time_prefix = f"**[{sm:02d}:{ss:02d} - {em:02d}:{es:02d}]** "
                                else:
                                    time_prefix = ""
                                st.markdown(f"{time_prefix}{ct}")
                                if idx < len(t.chunk_texts):
                                    st.divider()
                else:
                    st.markdown(
                        f'<div class="topic-card">'
                        f'  <div class="topic-title">Topic {i}: {t}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No topics found — try a longer lecture.")

    # ── Flashcards ────────────────────────────────────────────────────────
    with tab_flash:
        col_fhdr, col_factions = st.columns([2.8, 2.2])
        with col_fhdr:
            st.markdown("##### 🃏 Active-Recall Flashcards")
            st.caption(f"{len(fc)} cards · Definitions, Concepts, Formulas & Relationships")
        with col_factions:
            from flashcards import export_anki_csv_bytes
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                st.download_button(
                    "📥 Anki CSV",
                    data=export_anki_csv_bytes(fc),
                    file_name="prepgenius_flashcards.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_download_anki_csv_tab",
                )
            with c_f2:
                st.download_button(
                    "📦 Anki Deck",
                    data=build_anki_package_bytes(),
                    file_name="prepgenius_deck.apkg",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="btn_download_anki_apkg_tab",
                )

        if not fc:
            st.info("No flashcards generated. Make sure your API key is configured and try running the pipeline.")
        else:
            # Active Recall Study Mode toggle & Filter bar
            col_study, col_search, col_cat = st.columns([1.6, 2.1, 1.3])
            with col_study:
                study_mode = st.toggle(
                    "🧠 Active Recall Mode",
                    value=False,
                    key="fc_study_mode",
                    help="Hides answers behind an interactive reveal toggle so you can test yourself before checking the answer.",
                )
            with col_search:
                search = st.text_input("🔍 Search flashcards", placeholder="Type a keyword or formula…", key="fc_search", label_visibility="collapsed")
            with col_cat:
                all_cats = ["All Categories"] + sorted(list({c.get("category", "Concept") for c in fc}))
                selected_cat = st.selectbox("Category filter", all_cats, key="fc_cat_filter", label_visibility="collapsed")

            # Category badges with counts
            cat_counts = {}
            for c in fc:
                ct = c.get("category", "Concept")
                cat_counts[ct] = cat_counts.get(ct, 0) + 1
            cat_color_map = {
                "Formula": ("#D97706", "#FFFBEB", "📐"),
                "Definition": ("#059669", "#ECFDF5", "🔖"),
                "Relationship": ("#7C3AED", "#F5F3FF", "🔗"),
                "Concept": ("#2563EB", "#EFF6FF", "💡"),
            }
            cat_badges = []
            for ct, cnt in sorted(cat_counts.items()):
                col, bg, ico = cat_color_map.get(ct, ("#64748B", "#F1F5F9", "📌"))
                cat_badges.append(
                    f'<span style="background:{bg};color:{col};border:1px solid {col}33;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;margin-right:6px;">{ico} {ct}: {cnt}</span>'
                )
            st.markdown(f'<div style="margin: 4px 0 14px 0;">{"".join(cat_badges)}</div>', unsafe_allow_html=True)

            filtered = [
                c for c in fc
                if (not search or search.lower() in c["front"].lower() or search.lower() in c["back"].lower())
                and (selected_cat == "All Categories" or c.get("category", "Concept") == selected_cat)
            ]

            # Anki CSV Preview Expander
            with st.expander("📋 Preview Anki CSV Export (Front & Back Columns)", expanded=False):
                st.caption("Standard 2-column format ready for direct import into Anki desktop (File → Import):")
                preview_data = []
                for i, c in enumerate(fc, 1):
                    preview_data.append({
                        "Card #": i,
                        "Front (Question)": c["front"],
                        "Back (Answer)": c["back"] + (f"\n[⏱️ Source: {c['timestamp']}]" if c.get("timestamp") else ""),
                        "Category": c.get("category", "Concept"),
                    })
                st.dataframe(preview_data, use_container_width=True, hide_index=True)

            if not filtered:
                st.info("No cards match your filter criteria.")
            else:
                col_a, col_b = st.columns(2)
                for i, c in enumerate(filtered):
                    cat = c.get("category", "Concept")
                    ts = c.get("timestamp")
                    ts_badge = f'<span class="topic-badge" style="font-size:0.65rem;margin-left:4px;">⏱️ {ts}</span>' if ts else ""
                    cat_col, cat_bg, cat_ico = cat_color_map.get(cat, ("#64748B", "#F1F5F9", "💡"))
                    cat_badge = f'<span class="topic-pill" style="font-size:0.68rem;padding:2px 8px;margin:0;background:{cat_bg};color:{cat_col};border:1px solid {cat_col}33;font-weight:600;">{cat_ico} {cat}</span>'

                    with (col_a if i % 2 == 0 else col_b):
                        if study_mode:
                            st.markdown(
                                f'<div class="flashcard" style="border-left: 4px solid {cat_col};">'
                                f'  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">'
                                f'    <div>{cat_badge}{ts_badge}</div>'
                                f'    <div class="fc-badge">Card {i+1}</div>'
                                f'  </div>'
                                f'  <div class="fc-q" style="font-size:0.95rem;margin-bottom:0.4rem;">❓ {c["front"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            with st.expander(f"👁️ Reveal Answer #{i+1}", expanded=False):
                                st.markdown(
                                    f'<div style="background:#F8FAFC;border-radius:8px;padding:10px 14px;border:1px solid #E2E8F0;font-size:0.88rem;color:#1E293B;line-height:1.55;">'
                                    f'  <strong>💡 Answer:</strong> {c["back"]}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                f'<div class="flashcard" style="border-left: 4px solid {cat_col};">'
                                f'  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">'
                                f'    <div>{cat_badge}{ts_badge}</div>'
                                f'    <div class="fc-badge">Card {i+1}</div>'
                                f'  </div>'
                                f'  <div class="fc-q">Q: {c["front"]}</div>'
                                f'  <div class="fc-a">💡 {c["back"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    # ── Practice Quiz ──────────────────────────────────────────────────────
    with tab_quiz:
        st.markdown("##### ❓ Practice Quiz")

        # ── Quiz Controls ─────────────────────────────────────────────────
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.8, 1.8, 2.4])
        with ctrl_col1:
            diff_choice = st.selectbox(
                "Difficulty",
                ["mixed", "easy", "medium", "hard"],
                index=["mixed", "easy", "medium", "hard"].index(
                    st.session_state.quiz_difficulty
                ),
                format_func=lambda d: {
                    "mixed": "🎲 Mixed",
                    "easy":  "🟢 Easy",
                    "medium":"🟡 Medium",
                    "hard":  "🔴 Hard",
                }[d],
                key="quiz_diff_sel",
            )
            st.session_state.quiz_difficulty = diff_choice

        with ctrl_col2:
            n_q = st.slider(
                "Number of Questions",
                min_value=3,
                max_value=20,
                value=st.session_state.quiz_num_questions,
                step=1,
                key="quiz_n_slider",
            )
            st.session_state.quiz_num_questions = n_q

        with ctrl_col3:
            st.markdown("<p style='margin-bottom:4px;'>&nbsp;</p>", unsafe_allow_html=True)
            gen_btn = st.button(
                "🔄 Generate Quiz",
                type="primary",
                use_container_width=True,
                key="quiz_generate_btn",
                help="Generate a fresh quiz with the selected difficulty and question count.",
            )

        # ── Generate Quiz on Demand ───────────────────────────────────────
        if gen_btn:
            if not st.session_state.revision_notes:
                st.warning("⚠️ No revision notes found. Run the pipeline first.")
            elif not os.getenv("GEMINI_API_KEY"):
                st.warning("⚠️ GEMINI_API_KEY is not set. Open the sidebar and add it.")
            else:
                st.session_state.quiz_generating = True
                st.session_state.quiz_index = 0
                st.session_state.quiz_answers = {}
                st.session_state.quiz_show_results = False
                with st.spinner("🤖 Generating quiz with Gemini…"):
                    try:
                        from quiz import generate_quiz as _gen_quiz
                        fresh_quiz = _gen_quiz(
                            st.session_state.revision_notes,
                            num_questions=st.session_state.quiz_num_questions,
                            difficulty=st.session_state.quiz_difficulty,
                        )
                        st.session_state.quiz = fresh_quiz
                        st.session_state.quiz_generating = False
                        st.rerun()
                    except Exception as _qe:
                        st.session_state.quiz_generating = False
                        st.error(f"Quiz generation failed: {_qe}")

        quiz = st.session_state.quiz

        # ── Empty / Not-generated state ───────────────────────────────────
        if not quiz:
            st.info(
                "No quiz questions yet. Configure difficulty & count above, "
                "then click **🔄 Generate Quiz**."
            )
        else:
            # ── Difficulty badge helper ───────────────────────────────────
            def _diff_badge(d: str) -> str:
                cfg = {
                    "easy":   ("#10B981", "rgba(16, 185, 129, 0.15)", "🟢"),
                    "medium": ("#F59E0B", "rgba(245, 158, 11, 0.15)", "🟡"),
                    "hard":   ("#EF4444", "rgba(239, 68, 68, 0.15)", "🔴"),
                }
                col, bg, ico = cfg.get(d, ("#94A3B8", "rgba(148, 163, 184, 0.15)", "⚪"))
                return (
                    f'<span style="background:{bg};color:{col};border:1px solid {col}55;'
                    f'padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:700;">'
                    f'{ico} {d.capitalize()}</span>'
                )

            total_q = len(quiz)

            # ── Results View ──────────────────────────────────────────────
            if st.session_state.quiz_show_results:
                from quiz import score_quiz as _score_quiz

                user_answers = st.session_state.quiz_answers
                result = _score_quiz(quiz, user_answers)
                score = result["score"]
                pct = result["percent"]

                # Score badge
                tier_col = "#10B981" if pct >= 80 else ("#3B82F6" if pct >= 50 else "#F59E0B")
                tier_msg = (
                    "🏆 Outstanding Mastery!" if pct >= 80
                    else "📈 Solid Understanding! Review missed concepts."
                    if pct >= 50 else "📚 Keep Practising! Revisit notes and flashcards."
                )
                if pct >= 80:
                    st.balloons()
                elif pct >= 50:
                    st.snow()

                st.markdown(
                    f'<div style="background:#131B2E;border:2px solid {tier_col};border-radius:16px;'
                    f'padding:1.4rem;margin-bottom:1.4rem;text-align:center;'
                    f'box-shadow:0 4px 16px rgba(0,0,0,0.3);">'
                    f'  <div style="font-size:2.4rem;font-weight:800;color:{tier_col};">'
                    f'{score} / {total_q}</div>'
                    f'  <div style="font-size:1.1rem;font-weight:700;color:#F8FAFC;margin-top:0.2rem;">'
                    f'{pct}% Score</div>'
                    f'  <div style="font-size:0.9rem;color:#94A3B8;margin-top:0.4rem;">{tier_msg}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Per-question results
                st.markdown("**📋 Question-by-Question Breakdown:**")
                for qi, (q, res) in enumerate(zip(quiz, result["results"]), 1):
                    ok = res["correct"]
                    box_style = (
                        "background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.35);border-left:5px solid #10B981;"
                        if ok else
                        "background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.35);border-left:5px solid #EF4444;"
                    )
                    icon = "✅" if ok else "❌"
                    chosen_display = res["chosen"] or "*(no answer)*"
                    diff_b = _diff_badge(res.get("difficulty", "medium"))
                    correct_html = (
                        f'<div style="font-size:0.88rem;margin-bottom:0.3rem;color:#E2E8F0;">Correct answer: <code>{res["answer"]}</code></div>'
                        if not ok else ""
                    )
                    st.markdown(
                        f'<div style="{box_style}border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;">'
                        f'  <div style="display:flex;justify-content:space-between;align-items:center;'
                        f'margin-bottom:0.5rem;">'
                        f'    <strong style="color:#F8FAFC;">{icon} Q{qi}. {q["question"]}</strong>'
                        f'    <div>{diff_b}</div>'
                        f'  </div>'
                        f'  <div style="font-size:0.88rem;margin-bottom:0.3rem;color:#E2E8F0;">'
                        f'    Your answer: <code>{chosen_display}</code>'
                        f'  </div>'
                        f'  {correct_html}'
                        f'  <div style="font-size:0.85rem;color:#94A3B8;font-style:italic;">'
                        f'💡 {res["explanation"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                col_res1, col_res2 = st.columns([1, 1])
                with col_res1:
                    if st.button("🔁 Restart Quiz", key="quiz_restart_btn", use_container_width=True):
                        st.session_state.quiz_index = 0
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_show_results = False
                        st.rerun()
                with col_res2:
                    st.download_button(
                        "📥 Download Results (.json)",
                        data=build_quiz_json_export(),
                        file_name="prepgenius_quiz_results.json",
                        mime="application/json",
                        use_container_width=True,
                        key="quiz_download_results_json",
                    )

            # ── One-Question-at-a-Time View ───────────────────────────────
            else:
                idx = st.session_state.quiz_index
                # Guard bounds
                if idx >= total_q:
                    idx = total_q - 1
                    st.session_state.quiz_index = idx

                q = quiz[idx]
                answered = idx in st.session_state.quiz_answers

                # Progress bar & counter
                progress_pct = (idx + (1 if answered else 0)) / total_q
                st.progress(progress_pct)
                st.markdown(
                    f'<div style="text-align:right;font-size:0.8rem;color:#94A3B8;'
                    f'font-weight:600;margin-bottom:0.6rem;">'
                    f'Question {idx + 1} of {total_q} &nbsp;·&nbsp; '
                    f'{len(st.session_state.quiz_answers)} answered'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Question card
                diff_badge_html = _diff_badge(q.get("difficulty", "medium"))
                st.markdown(
                    f'<div style="background:#131B2E;border:1px solid #1E293B;'
                    f'border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:1rem;'
                    f'box-shadow:0 4px 14px rgba(0,0,0,0.25);">'
                    f'  <div style="display:flex;justify-content:space-between;'
                    f'align-items:flex-start;margin-bottom:0.8rem;">'
                    f'    <span style="font-size:0.72rem;font-weight:700;color:#94A3B8;'
                    f'text-transform:uppercase;letter-spacing:0.8px;">Q{idx + 1} / {total_q}</span>'
                    f'    {diff_badge_html}'
                    f'  </div>'
                    f'  <div style="font-size:1.05rem;font-weight:700;color:#F8FAFC;'
                    f'line-height:1.5;">{q["question"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Options — use a form so answer is hidden until Submit is pressed
                with st.form(key=f"quiz_q_form_{idx}"):
                    chosen = st.radio(
                        "Select your answer:",
                        q["options"],
                        index=None,
                        key=f"radio_q{idx}",
                    )
                    submit_ans = st.form_submit_button(
                        "✅ Submit Answer",
                        type="primary",
                        disabled=answered,
                    )

                if submit_ans and not answered:
                    if chosen is None:
                        st.warning("⚠️ Please select an answer before submitting.")
                    else:
                        st.session_state.quiz_answers[idx] = chosen
                        st.rerun()

                # Feedback block — shown only after submission
                if answered:
                    chosen_val = st.session_state.quiz_answers[idx]
                    is_correct = chosen_val == q["answer"]
                    if is_correct:
                        st.markdown(
                            f'<div class="ok-box">'
                            f'  <strong>✅ Correct!</strong><br>'
                            f'  <em>💡 {q["explanation"]}</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="err-box">'
                            f'  <strong>❌ Incorrect.</strong> '
                            f'You chose: <code>{chosen_val}</code><br>'
                            f'  Correct answer: <code>{q["answer"]}</code><br>'
                            f'  <em>💡 {q["explanation"]}</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Navigation row
                nav_prev, nav_next, nav_finish = st.columns([1.2, 1.2, 2.6])
                with nav_prev:
                    if st.button(
                        "← Previous",
                        key=f"quiz_prev_{idx}",
                        disabled=(idx == 0),
                        use_container_width=True,
                    ):
                        st.session_state.quiz_index = idx - 1
                        st.rerun()
                with nav_next:
                    if st.button(
                        "Next →",
                        key=f"quiz_next_{idx}",
                        disabled=(idx >= total_q - 1 or not answered),
                        use_container_width=True,
                    ):
                        st.session_state.quiz_index = idx + 1
                        st.rerun()
                with nav_finish:
                    all_answered = len(st.session_state.quiz_answers) == total_q
                    if st.button(
                        "🏁 Finish & See Results",
                        key=f"quiz_finish_{idx}",
                        type="primary" if all_answered else "secondary",
                        disabled=not all_answered,
                        use_container_width=True,
                        help="Answer all questions to unlock the results page." if not all_answered else "",
                    ):
                        st.session_state.quiz_show_results = True
                        st.rerun()

                if not all_answered:
                    remaining = total_q - len(st.session_state.quiz_answers)
                    st.caption(f"ℹ️ {remaining} question{'s' if remaining != 1 else ''} remaining before you can finish.")



    # ── Transcript ─────────────────────────────────────────────────────────
    with tab_transcript:
        st.markdown("##### Raw Transcript")
        st.caption(f"{len(segs)} caption segments · timestamps in seconds")
        lines_html = ""
        for seg in segs:
            t = int(seg.get("start", 0))
            txt = seg.get("text", "")
            mins, secs = divmod(t, 60)
            lines_html += f'<span class="ts-time">[{mins:02d}:{secs:02d}]</span>{txt}<br>'
        st.markdown(f'<div class="transcript-block">{lines_html}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "📥 Download Transcript (.txt)",
            data="\n".join(
                f"[{int(s.get('start',0))//60:02d}:{int(s.get('start',0))%60:02d}] {s.get('text','')}"
                for s in segs
            ),
            file_name="transcript.txt", mime="text/plain",
        )

else:
    # ── Empty state with feature highlights ────────────────────────────────
    st.markdown("""
<div class="empty-state">
  <span class="empty-icon">🎓</span>
  <div class="empty-title">Turn any YouTube lecture into mastery material</div>
  <div class="empty-desc">
    Paste any educational YouTube URL above or click one of the quick samples to generate comprehensive revision notes, semantic topic clusters, active-recall flashcards, and interactive practice quizzes.
  </div>
</div>

<div class="feature-grid">
  <div class="feature-box">
    <div class="feature-icon">📥</div>
    <div class="feature-title">1. Smart Transcription</div>
    <div class="feature-desc">Extracts official or auto-generated captions across 12+ languages with millisecond timestamp synchronization and sliding window chunking.</div>
  </div>
  <div class="feature-box">
    <div class="feature-icon">🧠</div>
    <div class="feature-title">2. AI Synthesis & Topics</div>
    <div class="feature-desc">LangChain map-reduce notes generator with definitions, formulas, and local CPU semantic clustering (Ward linkage).</div>
  </div>
  <div class="feature-box">
    <div class="feature-icon">🃏</div>
    <div class="feature-title">3. Active Recall & Exports</div>
    <div class="feature-desc">Single-objective flashcards with category tags, direct Anki .apkg / CSV export, interactive MCQ quiz, and PDF reports.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin-top:2.5rem;border-color:#E2E8F0;">
<p style="text-align:center;color:#94A3B8;font-size:.78rem;padding-bottom:1rem;">
  PrepGenius · AI-powered study assistant · Built with Streamlit ·
  <span style="color:#6C63FF;">v2.0</span>
</p>
""", unsafe_allow_html=True)
