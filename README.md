# SAN — Study Assistant for Notes 🎓

> **Transform any YouTube university lecture into structured study materials — instantly.**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-600%20passing-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features & Architecture

SAN takes a university lecture video and automatically turns it into a high-yield study suite.

```
YouTube Lecture URL (or Demo Mode)
    │
    ▼
transcript.py          ← Captions extraction (youtube-transcript-api)
    │
    ▼
text_processing.py     ← Filler stripping, normalization & rolling-window chunking
    │
    ├──► topic_clustering.py ← Local Sentence-Transformers embeddings & KMeans
    ├──► summarizer.py       ← LangChain / LLM hierarchical revision notes
    ├──► flashcards.py       ← High-yield Q&A cards with timestamps
    └──► quiz.py             ← MCQ practice quiz with explanations & difficulty levels
    │
    ▼
export.py              ← In-memory builders for PDF, Markdown, Anki CSV & APKG, Quiz JSON
    │
    ▼
app.py                 ← Streamlit interactive UI with caching & safe logging
```

### Module Breakdown

| Feature | Module | Output Description |
|---|---|---|
| 📚 **Revision Notes** | [`summarizer.py`](file:///c:/Users/Aditya/OneDrive/Desktop/SAN/summarizer.py) | Structured notes with Core Thesis, Key Concepts, Technical Details, Formulas, and Revision Checklist. |
| 🔬 **Topic Clustering** | [`topic_clustering.py`](file:///c:/Users/Aditya/OneDrive/Desktop/SAN/topic_clustering.py) | Semantic segmentation using `all-MiniLM-L6-v2` embeddings and optimal KMeans clustering. |
| 🃏 **Flashcards** | [`flashcards.py`](file:///c:/Users/Aditya/OneDrive/Desktop/SAN/flashcards.py) | Active-recall cards with concept tags, timestamp sources, and in-app flippable card viewer. |
| 📝 **Practice Quiz** | [`quiz.py`](file:///c:/Users/Aditya/OneDrive/Desktop/SAN/quiz.py) | Interactive one-question-at-a-time quiz with 4 options, difficulty selection, live scoring, and feedback. |
| ⬇️ **Export Center** | [`export.py`](file:///c:/Users/Aditya/OneDrive/Desktop/SAN/export.py) | Multi-format exports: ReportLab Platypus PDF, Markdown notes, Anki CSV, native Anki `.apkg` deck, and Quiz JSON. |

---

## 🎮 Zero-Config Demo Mode (No API Keys Required)

You do **not** need any API keys to explore and evaluate the application!
SAN includes a full offline demo mode loaded with MIT 6.006 (Algorithms: Peak Finding & Binary Search) lecture material.

1. Launch the app (`streamlit run app.py`).
2. Click **"🎮 Demo Lecture (MIT 6.006)"** on the home screen, or toggle **"Demo Mode (Offline / No API Key)"** in the sidebar.
3. Click **"🚀 Process Lecture"**.
4. The complete pipeline will execute locally: real text preprocessing, real Sentence-Transformers topic clustering, revision notes, flashcards, interactive quiz, and all download exports.

---

## 🚀 Setup Instructions

### 1. Prerequisites
- **Python:** 3.10, 3.11, 3.12, 3.13, or 3.14
- **Git**
- Optional: Virtual environment tool (`venv`)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/SAN.git
cd SAN
```

### 3. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration

For live YouTube processing, copy `.env.example` to `.env`:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and provide your credentials:

```ini
# ─── Summarizer (OpenAI, Groq, Ollama, OpenRouter, Together) ───────────────
OPENAI_API_KEY=sk-your-openai-or-groq-key
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.groq.com/openai/v1   # (Optional custom endpoint)
OPENAI_TEMPERATURE=0.2
SUMMARIZER_FORMAT=detailed

# ─── Flashcards & Quiz Generation (Google Gemini) ─────────────────────────
GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
GEMINI_MODEL=gemini-1.5-flash

# ─── Local Topic Clustering (Runs 100% on CPU) ───────────────────────────
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_TOPICS=8
```

> 🔒 **Security Notice:** All API keys are automatically redacted from console logs and Streamlit outputs by SAN's built-in `SecretMaskingFilter`. Never commit `.env` into version control.

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard:

```bash
# Standard command
streamlit run app.py

# On Windows with py launcher
py -m streamlit run app.py

# Custom port
streamlit run app.py --server.port 8501
```

Once running, open your web browser at:
`http://localhost:8501`

---

## 🧪 Running the Test Suite

SAN includes a comprehensive 600-test suite covering unit functionality, edge cases, error resilience, and integration workflows without making live external API calls:

```bash
# Run all tests
pytest -v

# Or using Python launcher
py -m pytest -v

# Run specific module test suites
pytest tests/test_app_integration.py -v   # End-to-end integration & validation
pytest tests/test_quiz.py -v              # Quiz scoring & validation logic
pytest tests/test_export.py -v            # ReportLab PDF, CSV, Markdown, JSON builders
pytest tests/test_topic_clustering.py -v  # Sentence embeddings & KMeans
pytest tests/test_text_processing.py -v   # Transcript cleaning & chunking
pytest tests/test_summarizer.py -v        # Summarization prompt templates
pytest tests/test_flashcards.py -v        # Flashcards & Anki generation
```

---

## 📥 Supported Export Formats

In the **⬇️ Downloads** tab, you can export generated study materials in five distinct formats:

1. **📄 PDF Study Guide (`.pdf`):**
   - Built with ReportLab Platypus.
   - Clean university typography, running headers/footers with dynamic page counts (`Page X of Y`).
   - Includes full revision notes, formulas callouts, key terms, and a compact revision checklist table.
2. **📝 Markdown Notes (`.md`):**
   - Clean, GitHub-flavored markdown with emojis, collapsible transcript source links, and flashcard tables.
3. **🃏 Anki Flashcards CSV (`.csv`):**
   - RFC 4180 compliant with UTF-8 BOM (`utf-8-sig`) for direct import into Anki desktop.
   - Separated into `Front`, `Back`, and `Tags` columns.
4. **📦 Anki Deck (`.apkg`):**
   - Ready-to-use Anki deck package with custom CSS styling and metadata tags.
5. **📊 Practice Quiz (`.json`):**
   - Full question bank with choices, correct answers, explanations, and user session scoring results.

---

## ⚡ Performance & Caching

- **In-Session Caching:** Processed lectures are cached in `st.session_state.lecture_cache` indexed by `(video_id, language, notes_format, flashcard_count)`. Revisiting previously generated lectures avoids redundant processing and preserves API quota.
- **Bypass Option:** Check *"Force re-process (bypass session cache)"* in the sidebar to regenerate materials anytime.
- **Fast Local Embeddings:** Topic clustering uses `sentence-transformers` running on CPU (utilizing vectorized PyTorch/NumPy without requiring a GPU).

---

## ⚠️ Known Limitations

1. **Captions Required:** The video must have English captions enabled (either creator-uploaded subtitles or YouTube auto-generated captions). Videos with disabled captions cannot be transcribed.
2. **Free API Quota:** When using Google AI Studio free tier keys, rate limits (typically 15 requests per minute) may apply. If you encounter rate limit errors, wait a minute before requesting another lecture.
3. **Very Long Lectures (>3 hours):** Extremely long videos will produce a large volume of text chunks. In such cases, consider increasing chunk word size in the sidebar settings.

---

## 🛠️ Troubleshooting

### 1. `TranscriptsDisabled` or `NoTranscriptFound` Error
- **Cause:** The YouTube video has disabled captions or captions in the requested language are unavailable.
- **Solution:** Verify that the video has the CC button enabled on YouTube. Alternatively, test the pipeline with Demo Mode.

### 2. `ResourceExhausted` (Gemini API 429)
- **Cause:** Free tier Gemini API requests per minute limit was exceeded.
- **Solution:** Wait 60 seconds and click **Process Lecture** again (the app will resume or you can use your session cache).

### 3. Sentence-Transformers First Run Delay
- **Cause:** On the very first run, `all-MiniLM-L6-v2` model weights (~80MB) are downloaded from Hugging Face and cached locally in `~/.cache/huggingface/hub`.
- **Solution:** Subsequent runs load immediately from local disk cache with zero network overhead.

### 4. Windows Long Path or PowerShell Execution Policy
- If `venv\Scripts\Activate.ps1` gives an execution policy error, run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  venv\Scripts\Activate.ps1
  ```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
