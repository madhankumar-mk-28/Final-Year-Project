<h1 align="center">
  <br>
  <img src="https://img.icons8.com/?size=160&id=114844&format=png&color=4F46E5" alt="Resume Screener" width="130">
  <br>
  <b>ML-Based Resume Screening System</b>
  <br>
  <sub><sup>AI-powered candidate ranking using transformer embeddings</sup></sub>
  <br>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/SBERT-Transformers-FF6B35?style=flat-square&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MPNet-768d-7C3AED?style=flat-square" />
  <img src="https://img.shields.io/badge/MxBai-1024d%20·%20MTEB%20%231-EC4899?style=flat-square" />
  <img src="https://img.shields.io/badge/Arctic%20Embed-768d-0EA5E9?style=flat-square" />
</p>

<p align="center">
  <b>Built by <a href="https://github.com/madhankumar-mk-28">23BCS0163 — Madhan Kumar</a> · VIT Vellore · B.Sc. Computer Science · 2026</b>
</p>

---

<p align="center">
  Stop keyword-matching. Start <i>understanding</i> resumes.
  <br>
  Transformer embeddings + configurable weighted scoring + a fully offline React dashboard.
</p>

---

## ✨ What It Does

Upload a pile of PDF resumes. Paste a job description. Click **Run Screening**.

The system parses every resume, extracts structured candidate data, generates sentence embeddings using state-of-the-art transformer models, and produces a **ranked, scored, auditable shortlist** — in under 15 seconds on Apple Silicon, under 60 seconds anywhere else.

No cloud APIs. No data leaves your machine. Fully offline after first model download.

---

## 🖥️ Dashboard Preview

> The React dashboard across all five views:

| Upload & Configure | Processing | Ranked Candidates |
|:------------------:|:----------:|:-----------------:|
| Drag-drop PDFs, paste JD, set weights | Live 6-step pipeline progress | Medal-ranked table with score rings |

| Analytics | Decisions |
|:---------:|:---------:|
| Score distribution, talent quadrant, skill gaps | Interview Now / Secondary Review / Archive |

---

## 🧠 How It Works

```
PDF Resumes ──► PDF Parser ──► Information Extractor ──► Semantic Matcher ──► Scoring Engine ──► Ranked Output
                pdfplumber     regex + 235-skill DB       SBERT embeddings     weighted formula    React dashboard
                + PyMuPDF      name·email·phone            two-pass strategy   dynamic threshold   + CSV export
                  fallback      skills·exp·education        40% full / 60% key  skill + semantic
```

### Two-Pass Semantic Scoring

Rather than embedding the full resume as a single vector (which dilutes key sections with boilerplate like *Declaration* and *Hobbies*), the system uses a two-pass strategy:

| Pass | Weight | What's Embedded |
|------|--------|-----------------|
| Full document | **40%** | All text, chunked into 300-word windows with 50-word stride |
| Key sections only | **60%** | Skills · Experience · Projects · Summary · Certifications |

Chunks are pooled using **mean-max blending** (50% each) before cosine similarity is computed against the job description embedding.

### Weighted Scoring Formula

```
Final Score = (skill_weight × skill_score) + (semantic_weight × semantic_score)

Default:  55% skill  +  45% semantic   (adjustable via slider)
```

### Dynamic Shortlisting Threshold

No fixed cutoffs. The threshold is computed live from the **60th percentile of eligible candidates' final scores**, clamped within ±10pp of the per-model default:

| Model | Default Threshold | Clamp Range |
|-------|:-----------------:|:-----------:|
| MPNet | 0.45 | [0.35 – 0.55] |
| MxBai | 0.55 | [0.45 – 0.65] |
| Arctic Embed | 0.50 | [0.40 – 0.60] |

---

## 🤖 Embedding Models

Three interchangeable models — switch from the topbar at any time:

<table>
<tr>
  <th>Model</th>
  <th>Dimensions</th>
  <th>Best For</th>
  <th>Disk Size</th>
</tr>
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
</table>

All models run **fully offline** after their initial download from HuggingFace.

---

## ⚡ Key Features

- 📄 **Dual-library PDF parsing** — pdfplumber primary, PyMuPDF automatic fallback
- 🔍 **5-tier skill matching** — exact → alias → substring → token overlap → semantic
- 🧬 **235-entry skill database** — programming, ML/AI, cloud, DevOps, databases, soft skills
- 🔁 **Duplicate detection** — email-keyed merge with skill union + max experience
- ⚠️ **Negation detection** — "no experience in Python" → skill excluded, not penalised
- 📊 **Skill gap analysis** — matched / missing / semantically-matched / negated per candidate
- 🔄 **False-negative recovery** — flags borderline rejections with high semantic alignment
- 💾 **Session persistence** — job configs saved, auto-filled next session
- 📤 **CSV export** — shortlisted candidates with rank, name, email, phone
- 🔒 **Security** — SHA-256 dedup, 10 MB file cap, 500k char limit, per-IP rate limiting

---

## 🗂️ Project Structure

```
Resume_screening_system/
│
├── app.py                    # Flask API — orchestration & REST endpoints
├── resume_parser.py          # PDF extraction — pdfplumber + PyMuPDF fallback
├── information_extractor.py  # Regex pipeline — name, email, phone, skills, exp, edu
├── semantic_matcher.py       # SBERT embeddings — two-pass chunking + LRU cache
├── scoring_engine.py         # Weighted scoring — dynamic threshold + band assignment
├── audit_logger.py           # Thread-safe append-only audit log (JSONL)
├── metrics_store.py          # Per-run metrics snapshots + CLI analysis tool
├── cleanup_cache.py          # Post-session cache & file cleanup utility
│
├── requirements.txt
├── README.md
│
├── uploads/                  # Staged PDFs (SHA-256 deduplicated)
├── results/                  # Scored JSON output per session
│
└── resume_ui/
    └── frontend/
        ├── src/
        │   ├── App.jsx       # React SPA — all five dashboard views
        │   └── index.css
        └── package.json
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

### 3 · Download Embedding Models

Run this once — all subsequent screening runs are fully offline:

```bash
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading MPNet...')
SentenceTransformer('multi-qa-mpnet-base-dot-v1')
print('Downloading MxBai...')
SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
print('Downloading Arctic Embed...')
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
cd resume_ui/frontend
npm install
npm run dev
# Vite starts on http://localhost:5173
```

### 6 · Open the Dashboard

Navigate to **[http://localhost:5173](http://localhost:5173)** — the sidebar status pill turns **Pipeline Ready** within five seconds.

---

## 📋 Usage

1. **Upload** — drag and drop PDF resumes onto the drop zone
2. **Configure** — paste your job description, add required skills as tags
3. **Tune** — adjust the skill / semantic weight slider and experience floor
4. **Select model** — choose MPNet, MxBai, or Arctic from the topbar
5. **Run** — click **Run Screening** and watch the 6-step pipeline execute
6. **Review** — explore the ranked table, click candidates for skill gap detail
7. **Export** — download shortlisted candidates as CSV

> [!TIP]
> For general roles, start with **MPNet** at the default **55% / 45%** split.
> For highly technical roles, increase skill weight to **65–70%**.
> For large pools (100+ resumes), switch to **Arctic Embed**.

---

## 📊 Analytics Views

The **Analytics** tab has five sections:

| Tab | What You See |
|-----|-------------|
| **Overview** | Score distribution bands · Accuracy / Precision / Recall / F1 · Confusion matrix |
| **Funnel** | Staged recruitment pipeline · dropout counts at each stage |
| **Talent** | 2×2 quadrant — Ideal Fit · Skilled · Role-Aware · Weak Match |
| **Skills** | Per-skill coverage bars · Gap severity tags (RARE / SCARCE / LOW / OK) |
| **Decisions** | Interview Now · Secondary Review · Archive · Borderline review list · Rejection log |

---

## ⚙️ Configuration

| Parameter | Default | Range | Description |
|-----------|:-------:|:-----:|-------------|
| Skill weight | **0.55** | 0.0 – 1.0 | Relative importance of skill matching |
| Semantic weight | **0.45** | 0.0 – 1.0 | Relative importance of semantic similarity |
| Min experience | **0** years | 0 – 20 | Hard floor; 0 = freshers eligible |
| Max file size | **10 MB** | — | Per PDF upload limit |
| Max concurrent jobs | **3** | — | HTTP 503 returned on overflow |
| Rate limit | **30 req / 60s** | — | Per-IP; HTTP 429 on breach |

---

## 📈 Performance

| Batch Size | Hardware | Model | Time |
|:----------:|----------|-------|:----:|
| 20 resumes | Apple M4 (no GPU) | MPNet | ~8s |
| 20 resumes | Intel i5 (no GPU) | MPNet | ~55s |
| 100 resumes | Apple M4 (no GPU) | MPNet | ~25s |
| 100 resumes | Apple M4 (no GPU) | MxBai | ~45s |

---

## 🔒 Security Notes

- All uploaded PDFs are **SHA-256 hashed** before storage — identical files are silently deduplicated
- Magic-bytes validation confirms every upload is a genuine PDF regardless of file extension
- Extracted text is **capped at 500,000 characters per file during page extraction** (not after) — this is the actual decompression-bomb defence
- No candidate PII is transmitted externally — all processing is local

---

## 🛠️ Metrics & Audit CLI

Every screening run is logged. Analyse past runs from the terminal:

```bash
# View the most recent run in full
python metrics_store.py --run latest

# Export all runs to CSV
python metrics_store.py --export csv

# Clear all stored metrics
python metrics_store.py --clear

# Clean Python cache and session files
python cleanup_cache.py

# Clean cache only (keep uploaded PDFs and results)
python cleanup_cache.py --keep-pdfs
```

---

## 📚 Academic References

This project builds on the following key works:

| # | Authors | Year | Title |
|---|---------|:----:|-------|
| 1 | Devlin et al. | 2019 | BERT: Pre-training of Deep Bidirectional Transformers — *NAACL* |
| 2 | Reimers & Gurevych | 2019 | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks — *EMNLP* |
| 3 | Wang et al. | 2020 | MiniLM: Deep Self-Attention Distillation — *NeurIPS* |
| 4 | Raghavan et al. | 2020 | Mitigating Bias in Algorithmic Hiring — *FAccT* |
| 5 | Gao et al. | 2021 | SimCSE: Simple Contrastive Learning of Sentence Embeddings — *EMNLP* |
| 6 | Muennighoff et al. | 2023 | MTEB: Massive Text Embedding Benchmark — *EACL* |
| 7 | Merrick et al. | 2024 | Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models — *arXiv* |

---

## 🗺️ Roadmap

- [x] PDF parsing with dual-library fallback
- [x] 235-entry curated skill database
- [x] Three interchangeable SBERT embedding models
- [x] Two-pass semantic scoring (40% full / 60% key sections)
- [x] Dynamic threshold (60th percentile of eligible candidates)
- [x] Configurable skill / semantic weight slider
- [x] Skill gap analysis (matched / missing / semantic / negated)
- [x] Duplicate candidate detection and merging
- [x] Negation context detection
- [x] False-negative recovery flagging
- [x] React analytics dashboard (5 views)
- [x] Audit logging + metrics store CLI
- [x] CSV export of shortlisted candidates
- [ ] OCR support for scanned PDFs (Tesseract)
- [ ] Live skill taxonomy integration (EMSI Burning Glass)
- [ ] Multi-recruiter authentication + session isolation
- [ ] Interview scheduling API integration
- [ ] Fine-tuned domain-specific HR embedding model

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with 🤖 transformers, ☕ late nights, and way too many PDF resumes<br>
  <b>23BCS0163 · Madhan Kumar · VIT Vellore · April 2026</b><br>
  <sub>Under the guidance of Dr. Suba Shanthini S · SCORE · School of Computer Science Engineering and Information Systems</sub>
</p>
