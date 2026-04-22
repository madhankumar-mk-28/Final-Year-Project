<div align="center">

#  ML - Based Resume Screening System

### ML - powered candidate ranking using transformer embeddings

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![HuggingFace](https://img.shields.io/badge/SBERT-Transformers-FF6B35?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

[![MPNet](https://img.shields.io/badge/MPNet-768d-7C3AED?style=flat-square)](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)
[![MxBai](https://img.shields.io/badge/MxBai-1024d%20%C2%B7%20MTEB%20%231-EC4899?style=flat-square)](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
[![Arctic](https://img.shields.io/badge/Arctic%20Embed-768d-0EA5E9?style=flat-square)](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)

**Built by [23BCS0163 — Madhan Kumar](https://github.com/madhankumar-mk-28) · VIT Vellore · B.Sc. Computer Science · 2026**

---

*Stop keyword-matching. Start **understanding** resumes.*
Transformer embeddings + configurable weighted scoring + a fully offline React dashboard.

</div>

---

## ✨ What It Does

Upload a pile of PDF resumes. Paste a job description. Click **Run Screening**.

The system parses every resume, extracts structured candidate data, generates sentence embeddings using state-of-the-art transformer models, and produces a **ranked, scored, auditable shortlist** — in under 15 seconds on Apple Silicon, under 60 seconds anywhere else.

No cloud APIs. No data leaves your machine. Fully offline after first model download.

---

## 🖥️ Dashboard Views

> Five dedicated views cover the full recruitment workflow:

| View | What You Get |
|:-----|:-------------|
| **Upload & Configure** | Drag-drop PDFs, paste JD, tag required skills, set skill/semantic weights |
| **Processing** | Live 6-step pipeline progress with per-stage status |
| **Candidates** | Medal-ranked table with circular score rings, filterable by status |
| **Analytics** | Score distribution · Talent quadrant · Skill gap analysis · Model eval metrics |
| **Decisions** | Interview Now / Secondary Review / Archive · Rejection log with root cause |

---

## 🧠 How It Works

```
PDF Resumes ──► PDF Parser ──► Information Extractor ──► Semantic Matcher ──► Scoring Engine ──► Ranked Output
                pdfplumber     regex + 200+ skill DB     SBERT embeddings     weighted formula   React dashboard
                + PyMuPDF      name·email·phone          two-pass strategy    dynamic threshold   + CSV export
                fallback       skills·exp·education      40% full / 60% key   skill + semantic
```

### Two-Pass Semantic Scoring

Rather than embedding the full resume as a single vector (which dilutes key sections with boilerplate like *Declaration* and *Hobbies*), the system uses a two-pass strategy:

| Pass | Weight | What's Embedded |
|:-----|:------:|:----------------|
| Full document | **40%** | All text, chunked into 300-word windows with 50-word stride |
| Key sections only | **60%** | Skills · Experience · Projects · Summary · Certifications |

Chunks are pooled using **mean-max blending** (50% each) before cosine similarity is computed against the job description embedding. The job description embedding is **cached per request** — repeated screenings with the same JD skip re-encoding entirely.

### Weighted Scoring Formula

```
Final Score = (skill_weight × skill_score) + (semantic_weight × semantic_score)

Default:  55% skill  +  45% semantic   (adjustable per session via slider)
```

### Dynamic Shortlisting Threshold

No fixed cutoffs. The threshold is computed live from the **60th percentile of the batch's final scores**, clamped within ±10 pp of the per-model default:

| Model | Default Threshold | Clamp Range |
|:------|:-----------------:|:-----------:|
| MPNet | 0.45 | \[0.35 – 0.55\] |
| MxBai | 0.55 | \[0.45 – 0.65\] |
| Arctic Embed | 0.50 | \[0.40 – 0.60\] |

---

## 🤖 Embedding Models

Three interchangeable models — switch from the topbar dropdown at any time:

<table>
<thead>
<tr>
  <th>Model</th>
  <th align="center">Dimensions</th>
  <th>Best For</th>
  <th align="center">Disk Size</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>MPNet</b><br><sub>multi-qa-mpnet-base-dot-v1</sub></td>
  <td align="center">768</td>
  <td>General job descriptions · most discriminative score distributions · recommended for first use</td>
  <td align="center">~420 MB</td>
</tr>
<tr>
  <td><b>MxBai</b><br><sub>mixedbread-ai/mxbai-embed-large-v1</sub></td>
  <td align="center">1024</td>
  <td>Senior / specialised roles · 2025 MTEB English #1 · highest absolute similarity scores</td>
  <td align="center">~1.3 GB</td>
</tr>
<tr>
  <td><b>Arctic Embed</b><br><sub>Snowflake/snowflake-arctic-embed-m-v1.5</sub></td>
  <td align="center">768</td>
  <td>Large applicant pools (100+ resumes) · tightest score distributions · precision retrieval</td>
  <td align="center">~430 MB</td>
</tr>
</tbody>
</table>

All models run **fully offline** after their initial download from HuggingFace. Up to **2 models** are held in an LRU cache simultaneously — the least-recently-used model is evicted to stay within the ~2 GB memory budget.

---

## ⚡ Key Features

- 📄 **Dual-library PDF parsing** — pdfplumber primary, PyMuPDF automatic fallback for complex/rotated PDFs
- 🔍 **5-tier skill matching** — exact → alias → word-boundary regex → token overlap → semantic embedding
- 🧬 **200+ skill database** — programming languages, ML/AI, cloud, DevOps, databases, soft skills, ITSM
- 🔁 **Duplicate candidate detection** — email-keyed merge with skill union and max-experience selection
- 📊 **Skill gap analysis** — matched vs. missing skills per candidate with semantic fallback matching
- 🔄 **False-negative recovery** — flags borderline rejections where semantic alignment remains high (> 0.55)
- 🛡️ **Overload protection** — concurrent screening cap (3 slots) + per-IP rate limiting (10 req / 60s)
- 💾 **Session persistence** — job configs saved to `config.json`, auto-filled on next upload
- 📤 **CSV export** — shortlisted candidates with rank, name, email, phone; decision-group exports from Analytics
- 🔒 **Security hardened** — SHA-256 dedup, PDF magic-byte validation, 10 MB cap, 500k char limit

---

## 🗂️ Project Structure

```
Resume_screening_system/
│
├── app.py                    # Flask API — session management, orchestration, REST endpoints
├── resume_parser.py          # PDF extraction — pdfplumber primary, PyMuPDF fallback
├── information_extractor.py  # Regex pipeline — name, email, phone, skills, exp, education
├── semantic_matcher.py       # SBERT embeddings — two-pass chunking, LRU model cache, JD cache
├── scoring_engine.py         # Weighted scoring — dynamic threshold, band assignment, FN recovery
├── audit_logger.py           # Thread-safe append-only JSONL audit log + skill normalisation
├── metrics_store.py          # Per-run metrics snapshots + CLI analysis tool
├── cleanup_cache.py          # Post-session cache & upload file cleanup utility
│
├── requirements.txt
├── config.json               # Saved job description, required skills, scoring weights
├── audit.jsonl               # Append-only screening run log (auto-rotated at 500 entries)
├── metrics_log.jsonl         # Detailed per-run metrics with candidate rankings
│
├── uploads/                  # Staged PDFs — isolated per session under uploads/<session_id>/
├── results/                  # Scored JSON output — one file per session
│
└── frontend/                 # Vite + React SPA
    ├── index.html
    ├── package.json
    └── src/
        ├── App.jsx           # Monolithic React SPA — all five dashboard views (~1400 lines)
        └── main.jsx          # React 19 entry point
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- Node.js **18+**
- ~2.5 GB free disk space (all three models)

---

### 1 · Clone the Repository

```bash
git clone https://github.com/madhankumar-mk-28/Final-Year-Project.git
cd Resume_screening_system
```

### 2 · Set Up the Python Backend

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins a CPU-only PyTorch build to save ~1.5 GB on disk. If you have a CUDA GPU, remove the `--extra-index-url` line and install the CUDA wheel manually.

### 3 · Download Embedding Models

Run this once — all subsequent screening runs are fully offline:

```bash
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading MPNet (~420 MB)...')
SentenceTransformer('multi-qa-mpnet-base-dot-v1')
print('Downloading MxBai (~1.3 GB)...')
SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
print('Downloading Arctic Embed (~430 MB)...')
SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v1.5')
print('All models ready.')
"
```

### 4 · Start the Backend

```bash
python app.py
# Flask starts on http://localhost:5001
```

### 5 · Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
# Vite starts on http://localhost:5173
```

### 6 · Open the Dashboard

Navigate to **[http://localhost:5173](http://localhost:5173)** — the sidebar status pill turns **Pipeline Ready** within five seconds once the Flask backend responds to `/api/health`.

---

## 📋 Usage

1. **Upload** — drag and drop PDF resumes onto the drop zone (max 100 files, 10 MB each)
2. **Configure** — paste your job description (minimum 20 characters), add required skills as tags
3. **Tune** — adjust the skill / semantic weight slider and minimum experience floor
4. **Select model** — choose MPNet, MxBai, or Arctic from the topbar dropdown
5. **Run** — click **Run Screening** and watch the 6-step pipeline execute live
6. **Review** — explore the ranked candidate table, click any row for the skill radar and profile drawer
7. **Export** — download shortlisted candidates as CSV from the Candidates view, or export by decision group from Analytics → Decisions

> [!TIP]
> For general roles, start with **MPNet** at the default **55% skill / 45% semantic** split.
> For highly technical roles, increase skill weight to **65–70%**.
> For large pools (100+ resumes), **Arctic Embed** produces the tightest score distributions.

---

## 📊 Analytics Views

The **Analytics** tab has five sections accessible from the segmented tab bar:

| Tab | What You See |
|:----|:-------------|
| **Overview** | Full score distribution bars · Accuracy / Precision / Recall / F1 · Confusion matrix · Soft classification bands |
| **Funnel** | Staged recruitment pipeline with per-stage candidate counts and dropout totals |
| **Talent** | 2×2 quadrant map — Ideal Fit · Skilled · Role-Aware · Weak Match |
| **Skills** | Per-skill coverage bars · Skill gap severity tags (`RARE` / `SCARCE` / `LOW` / `OK`) · Top-3 radar comparison |
| **Decisions** | Interview Now · Secondary Review · Archive · Full borderline list · Per-candidate rejection analysis |

The **soft classification bands** (used in Overview and Decisions) use fixed thresholds:

| Band | Final Score | Action |
|:-----|:-----------:|:-------|
| Strong Fit | ≥ 65% | Interview Now |
| Borderline | 50–64% | Manual Review |
| Weak Fit | < 50% | Archive |

---

## ⚙️ Configuration

| Parameter | Default | Range | Description |
|:----------|:-------:|:-----:|:------------|
| `skill_weight` | **0.55** | 0.0 – 1.0 | Relative importance of skill matching in final score |
| `semantic_weight` | **0.45** | 0.0 – 1.0 | Relative importance of semantic similarity in final score |
| `min_experience_years` | **0** | 0 – 20 | Hard experience floor; 0 = freshers eligible |
| `top_n` | **100** | 1 – ∞ | Maximum candidates returned in results |
| Max file size | **10 MB** | — | Per-PDF upload limit (`MAX_FILE_BYTES`) |
| Max files per session | **200** | — | Per-upload batch limit |
| Max concurrent screenings | **3** | — | Returns HTTP 503 when exceeded |
| Rate limit | **10 req / 60s** | — | Per-IP sliding window; returns HTTP 429 on breach |
| Min free disk | **200 MB** | — | Upload rejected if free disk falls below this |

---

## 📈 Performance

Benchmarks measured on real batches from the audit log (wall-clock time, CPU only):

| Batch Size | Hardware | Model | Time |
|:----------:|:---------|:------|:----:|
| 20 resumes | Apple M1 (MPS) | MPNet | ~10s |
| 34 resumes | Apple M1 (MPS) | MPNet | ~11s |
| 66 resumes | Apple M1 (MPS) | MPNet | ~19–28s |
| 66 resumes | Apple M1 (MPS) | Arctic Embed | ~24s |

> Batch encoding (all chunks in two `model.encode()` calls per run) is the primary reason throughput stays this high — per-resume encoding would be 10–20× slower.

---

## 🔒 Security Notes

- All uploaded PDFs are **SHA-256 hashed** before storage — identical files within a session are silently deduplicated
- **PDF magic-byte validation** (`%PDF` header check) rejects non-PDF files regardless of extension
- Extracted text is **capped at 500,000 characters** — the decompression-bomb defence is applied page-by-page during extraction, not after
- **Session isolation** — each upload creates an independent directory under `uploads/<session_id>/`; no shared file state between users
- **Session ID validation** — all `session_id` values are validated against a strict 32-character hex regex before any file I/O
- No candidate PII is transmitted externally — all processing is local and fully offline after model download

---

## 🛠️ Metrics & Audit CLI

Every screening run writes to two JSONL logs. Both are automatically trimmed to the 500 most recent records.

```bash
# View summary of all past runs
python metrics_store.py

# Full detail for the most recent run (scores, rankings, confusion matrix)
python metrics_store.py --run latest

# Full detail for a specific run by ID
python metrics_store.py --run 2026-04-05T03:19:44.791563+00:00

# Cross-run model comparison and Spearman correlations
python metrics_store.py --compare

# Export all runs to a flat CSV (one row per candidate per run)
python metrics_store.py --export csv

# Delete all stored metrics
python metrics_store.py --clear

# Clean Python cache + uploaded PDFs + session results
python cleanup_cache.py

# Clean Python cache only (keep uploads and results)
python cleanup_cache.py --keep-pdfs
```

---

## 📚 Academic References

| # | Authors | Year | Title |
|:--|:--------|:----:|:------|
| 1 | Devlin et al. | 2019 | BERT: Pre-training of Deep Bidirectional Transformers — *NAACL* |
| 2 | Reimers & Gurevych | 2019 | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks — *EMNLP* |
| 3 | Wang et al. | 2020 | MiniLM: Deep Self-Attention Distillation — *NeurIPS* |
| 4 | Raghavan et al. | 2020 | Mitigating Bias in Algorithmic Hiring — *FAccT* |
| 5 | Gao et al. | 2021 | SimCSE: Simple Contrastive Learning of Sentence Embeddings — *EMNLP* |
| 6 | Muennighoff et al. | 2023 | MTEB: Massive Text Embedding Benchmark — *EACL* |
| 7 | Merrick et al. | 2024 | Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models — *arXiv* |

---

## 🗺️ Roadmap

**Completed**
- [x] PDF parsing with dual-library fallback (pdfplumber + PyMuPDF)
- [x] 200+ entry curated skill database with 150+ alias mappings
- [x] Three interchangeable SBERT embedding models (MPNet · MxBai · Arctic)
- [x] Two-pass semantic scoring (40% full text / 60% key sections)
- [x] Dynamic shortlisting threshold (60th percentile, per-model clamped)
- [x] Configurable skill / semantic weight slider per session
- [x] Skill gap analysis — matched vs. missing skills per candidate
- [x] Duplicate candidate detection and merging (email-keyed)
- [x] False-negative recovery flagging (low score but high semantic)
- [x] Per-IP rate limiting + concurrent screening cap + disk space guard
- [x] Session isolation with SHA-256 dedup and magic-byte PDF validation
- [x] React 19 analytics dashboard — 5 views, 4 analytics sub-tabs
- [x] Confusion matrix + Accuracy / Precision / Recall / F1 in Analytics
- [x] Audit logging (`audit.jsonl`) + metrics store CLI (`metrics_store.py`)
- [x] CSV export — shortlisted candidates + per-decision-group exports
- [x] Background cleanup worker with hourly stale-session eviction

**Planned**
- [ ] OCR support for scanned PDFs (Tesseract integration)
- [ ] Live skill taxonomy integration (EMSI Burning Glass)
- [ ] Multi-recruiter authentication + role-based session isolation
- [ ] Interview scheduling API integration
- [ ] Fine-tuned domain-specific HR embedding model

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with Claude, late night grinding sessions, and way too many tokens as fee😭

**23BCS0163 · Madhan Kumar · VIT Vellore · April 2026**


</div>
