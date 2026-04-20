# FINAL VIVA PPT — CORRECTED SLIDE CONTENT
# ==========================================
# Copy-paste ready. All errors from the 2nd review PPT fixed.
# New slides added for completeness.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 1 — Title
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Capstone Project
ML-Based Resume Screening System
Guide: Dr. Suba Shanthini S
Department: School of Computer Science Engineering and Information Systems (VIT - Vellore)
By MADHAN KUMAR (23BCS0163)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 2 — Introduction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Introduction

- Context: Organizations receive hundreds of applications per opening; manual screening takes 4+ minutes per resume
- Core Problem: Keyword-based ATS fails to capture semantic equivalence ("Python development" vs "Python programming")
- Proposal: A full-stack ML pipeline combining transformer embeddings with configurable weighted scoring
- Key Innovation: Three interchangeable embedding models, fully offline after initial download


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 3 — Project Rationale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project Rationale

- Efficiency: Batch of 20 resumes processed in under 60 seconds on consumer hardware
- Accuracy: Transformer embeddings capture contextual meaning, not just keyword overlap
- Transparency: Skill gap analysis shows exactly why each candidate scored as they did
- Recruiter Control: Configurable skill/semantic weight slider (default 55/45) — not a black box


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 4 — Literature Survey (keep existing table)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Literature Survey
[Keep existing Table 2.1 — no changes needed]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 5 — Research Gaps
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Research Gaps Identified

- Gap 1: Semantic Blindness — keyword ATS cannot recognize "machine learning" ≈ "scikit-learn"
- Gap 2: Lack of Explainability — recruiters don't know why a candidate scored high or low
- Gap 3: Cloud API Dependency — most systems require internet, raising privacy concerns
- Gap 4: Format Fragility — rule-based parsers break on multi-column/non-standard PDF layouts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 6 — Objectives (CORRECTED — NO SPACY)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Objectives

O1: Implement robust PDF parsing using pdfplumber with PyMuPDF fallback for diverse formats

O2: Extract structured candidate information (name, email, phone, skills, experience, education) using pre-compiled regex patterns and a curated 234-entry skill database — no NLP library dependency

O3: Integrate three transformer embedding models (MPNet, MxBai, Arctic Embed) for semantic similarity computation between resumes and job descriptions

O4: Design a dynamic scoring engine with configurable skill/semantic weights, per-model thresholds, and a 60th-percentile dynamic eligibility cutoff

O5: Build an interactive React dashboard with ranked tables, skill gap analysis, 5-tab analytics, and CSV export

O6: Perform duplicate candidate detection by email matching, merging name, phone, skills, and experience across duplicate submissions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 7 — Project Timeline (CORRECTED — ALL COMPLETED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project Timeline

- Phase 1-2: Literature Review and Architecture Design (Completed)
- Phase 3-4: PDF Parsing and Regex Information Extraction (Completed)
- Phase 5-6: Semantic Embedding Modules and Scoring Engine (Completed)
- Phase 7: Flask API Integration and Async Task Pipeline (Completed)
- Phase 8: React Dashboard and 5-Tab Analytics UI (Completed)
- Phase 9-10: System Testing, Bug Fixes, and Documentation (Completed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 8 — System Architecture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

System Architecture — Four-Layer Design

Layer 1 — User Interface:
  React dashboard (App.jsx) on localhost:5173
  5 views: Upload, Processing, Dashboard, Candidates, Analytics

Layer 2 — API Communication:
  Flask REST API (app.py) on localhost:5001
  8 endpoints, async task execution, CORS enabled

Layer 3 — ML Processing Pipeline:
  resume_parser.py → information_extractor.py → semantic_matcher.py → scoring_engine.py

Layer 4 — Storage:
  uploads/, results/<session_id>.json, config.json, metrics_log.jsonl, audit.jsonl

[Include Figure 4.1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 9 — Pipeline Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Methodology: Pipeline Overview

1. Data Ingestion — Batch PDF upload with SHA-256 deduplication
2. Resume Parsing — pdfplumber with PyMuPDF fallback, 500k char cap per file
3. Information Extraction — Regex-based extraction of 6 structured fields, 234-entry skill DB
4. Semantic Vectorization — Two-pass embedding: 40% full text + 60% key sections
5. Scoring and Ranking — Weighted formula + dynamic threshold + band classification
6. Results Output — Atomic JSON write, audit logging, metrics snapshot


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 10 — Module 1: Resume Parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 1: Resume Parser (resume_parser.py)

Dual-Library Parsing Strategy:
- Primary: pdfplumber with x_tolerance=3, y_tolerance=3
- Fallback: PyMuPDF (fitz) if pdfplumber yields < 100 chars
- Safety: 500,000 character cap enforced per-page during extraction

Preprocessing:
- CRLF → LF normalization
- Horizontal whitespace collapse
- 3+ consecutive blank lines → 2

Output: dict[filename → cleaned_text], list[failed_filenames]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 11 — Module 2: Information Extraction (CORRECTED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 2: Information Extraction (information_extractor.py)

Fully regex-based — NO spaCy or NLP library dependency

Six Structured Fields:
- Name: Derived from PDF filename (more reliable than body-text NER for Indian names)
- Email: Standard regex, returned in lowercase
- Phone: 6 regex patterns covering Indian mobile + international formats
- Skills: 234-entry curated database
    Multi-word: longest-first substring matching
    Single-word: single batched word-boundary regex pass
    Hyphenated variants normalized ("problem-solving" → "problem solving")
- Experience: 5 regex patterns + date-range calculation fallback, clamped to 15 years
- Education: Degree pattern regex with full-line extraction (up to 120 chars)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 12 — Module 3: Truth Engine (Dedup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 3: Truth Engine — Duplicate Detection

Problem: Candidates upload multiple resume versions (v1, updated, final)

Deduplication Key: Email address only
  (Phone not used — may be shared across family or formatted inconsistently)

Merge Strategy:
- Name → longer of the two variants
- Email/Phone → non-empty value from either profile
- Skills → set union
- Experience → maximum of both values
- Education → deduplicated union

Result: Single consolidated profile per candidate in ranked output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 13 — Module 4: Semantic Matcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 4: Semantic Matcher (semantic_matcher.py)

Two-Pass Embedding Strategy:
  Pass 1 — Full text (40% weight): 300-word overlapping chunks, stride 50
  Pass 2 — Key sections (60% weight): Skills, Experience, Projects, Education

Pooling: Mean-max blend for MPNet/MxBai, mean-only for Arctic (CLS-trained)

Batched Encoding:
  All chunks from all resumes encoded in ONE model.encode() call
  OOM recovery: auto-retry at batch_size=8 if primary batch fails

Model Cache: LRU, max 2 models in memory simultaneously
JD Cache: SHA-256 keyed FIFO, max 32 entries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 14 — Module 5: Transformer Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 5: Supported Embedding Models

MPNet (multi-qa-mpnet-base-dot-v1):
  768-dim, trained on 215M QA pairs
  Skill threshold: 0.63 | Best for balanced semantic matching

MxBai (mixedbread-ai/mxbai-embed-large-v1):
  1024-dim, 335M parameters, MTEB #1
  Skill threshold: 0.60 | Highest absolute similarity scores

Arctic Embed (Snowflake/snowflake-arctic-embed-m-v1.5):
  768-dim, asymmetric query-document retrieval
  Skill threshold: 0.66 | Tight score distribution, good for large pools

All three run fully offline after initial HuggingFace download


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 15 — Module 6: Scoring Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 6: Scoring Engine (scoring_engine.py)

Three-Pass Pipeline:
  Pass 1 — Component scores + hard eligibility checks
  Pass 2 — Dynamic threshold from eligible candidates' scores (60th percentile)
  Pass 3 — Band assignment + false-negative recovery

Scoring Formula:
  Final Score = (skill_weight x skill_score) + (semantic_weight x semantic_score)
  Default weights: 55% skill / 45% semantic (adjustable via slider)

Skill Matching Hierarchy:
  1. Exact normalized match
  2. Alias dictionary (41 alias groups)
  3. Reverse-alias lookup
  4. Word-boundary substring
  5. Token overlap (>= 2/3)

Rejection Codes: no_contact, skill_below_min, all_skills_negated,
  experience_below_min, missing_education, score_below_threshold

Band Labels:
  Strong Fit — final >= threshold + 0.10
  Borderline — final >= threshold
  Weak Fit — not eligible


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 16 — Negation Detection and FN Recovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Advanced Scoring Features

Negation Context Detection:
  Scans 80-char window before each skill mention
  Patterns: "no experience in", "not skilled in", "lacking", "without", "never"
  Negated skills excluded from both numerator and denominator

False-Negative Recovery:
  Flags ineligible candidates with high semantic alignment
  Condition: final_score < dynamic_threshold AND semantic_score > per-model floor
  Model floors: MPNet 0.52, MxBai 0.62, Arctic 0.55
  Marked as fn_recovered=True for manual recruiter review


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 17 — React Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Module 7: React Dashboard (App.jsx)

Five Named Views:
  1. Upload and Configure — drag-drop PDFs, JD input, skill tags, weight slider
  2. Processing — real-time pipeline progress with 6-step visualization
  3. Dashboard — KPI tiles, score distribution, top candidate gauge
  4. Candidates — ranked table with medal icons, expandable detail drawer, radar chart
  5. Analytics — 5-tab view (Overview, Funnel, Talent Quadrant, Skills, Decisions)

Key Features:
  - State managed with React hooks (useState, useEffect) — no external libraries
  - Real-time health-check polling for pipeline status
  - CSV export of shortlisted candidates
  - Saved job configuration auto-restores on next session

[Include screenshots]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 18 — Results and Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Results and Analysis

Processing Performance:
  20 resumes: < 60 seconds on consumer hardware (no GPU)
  100 resumes on Apple M4: 25-45 seconds (model-dependent)

Output per Candidate: 20+ structured fields including
  skill/semantic/final scores, matched/missing/negated skills,
  band label, dynamic threshold, confidence, near_threshold flag

Model Characteristics:
  MPNet — Fast encoding, balanced accuracy, ideal default
  MxBai — Highest similarity scores, larger memory footprint
  Arctic — Tightest candidate separation, best for large pools

[Include Figure 7.1 and 7.2]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 19 — Analytics Views
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analytics Dashboard — 5-Tab View

1. Overview: Score distribution bands, model evaluation metrics (Accuracy, Precision, Recall, F1), confusion matrix
2. Funnel: Recruitment pipeline visualization with dropout rates per stage
3. Talent: 2x2 quadrant map (Ideal Fit, Skilled, Role-Aware, Weak Match) based on skill vs semantic scores
4. Skills: Per-skill coverage bars + gap severity tags (RARE, SCARCE, LOW, OK)
5. Decisions: Hiring recommendations — Interview Now, Secondary Review, Archive (with CSV export)

[Include Figures A.7-A.10]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 20 — Audit and Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Audit and Metrics Logging

Audit Logger (audit_logger.py):
  Thread-safe append-only logging to audit.jsonl
  Per-run: model, candidate count, timestamp, JD excerpt
  Rotated at 500 entries

Metrics Store (metrics_store.py):
  Full statistical snapshot per run (mean, std, min, max for all score types)
  CLI: --run latest | --compare | --export csv | --clear
  Supports post-session analysis and cross-run model comparison


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 21 — Project Status (CORRECTED — ALL COMPLETED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project Status — All Phases Completed

Completed:
  - Backend API with 8 endpoints, async task execution, rate limiting
  - Resume parsing pipeline (pdfplumber + PyMuPDF fallback)
  - ML embedding integration (3 models, LRU cache, OOM recovery)
  - Candidate deduplication engine (email-based Truth Engine)
  - Scoring engine with dynamic thresholds, negation detection, FN recovery
  - React dashboard with 5 views and 5-tab analytics
  - CSV export, audit logging, metrics store with CLI
  - Full system testing and documentation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 22 — Key Achievements
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key Achievements

- Semantic matching correctly identifies "deep learning" ≈ "neural networks", "PostgreSQL" ≈ "SQL"
- Dynamic threshold adapts to batch quality — rises for strong pools, falls for weaker ones
- Negation detection prevents "no SQL experience" from counting as a skill match
- All-skills-negated rejection code distinguishes genuine skill gaps from negation artifacts
- System operates 100% offline after initial model download
- 41-entry skill alias dictionary covers common abbreviations and related terms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 23 — Limitations and Future Work
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Limitations:
  - Scanned PDFs without OCR produce empty extraction
  - Skill database (234 entries) requires manual updates for new technologies
  - Dynamic threshold falls back to model default for batches < 5 candidates

Future Work:
  - OCR integration (Tesseract) for scanned PDFs
  - Live skill taxonomy updates (EMSI Burning Glass integration)
  - Multi-user support with authentication and role-based access
  - Calendar API integration for automated interview scheduling


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 24 — Conclusion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Conclusion

- Developed a full-stack ML-based resume screening system integrating PDF parsing, regex extraction, transformer embedding, and configurable scoring
- Three embedding models (MPNet, MxBai, Arctic) improve job description-resume alignment beyond keyword matching
- Dynamic thresholds, negation detection, and false-negative recovery provide robust and fair candidate evaluation
- Transparent skill gap analysis and structured rejection codes make every ranking decision auditable
- Automated pipeline reduces manual screening effort from hours to seconds
- System operates entirely offline, making it suitable for privacy-sensitive and restricted-network environments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 25-26 — References (keep existing — no changes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Keep existing references — all 15 are correct]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLIDE 27 — Screenshots (include key figures)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Live Demo / Screenshots

[Include: Upload View, Processing View, Dashboard, Candidates, Analytics tabs]
