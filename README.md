# ML-Based Resume Screening System

A Resume Screening System that automatically ranks job candidates by analyzing resumes against a job description using transformer-based semantic similarity models.

Unlike traditional ATS systems that rely only on keyword matching, this system performs true semantic understanding of resumes using modern embedding models.

This project was developed as a B.Sc Computer Science Final Year Project.

--------------------------------------------------

PROJECT OVERVIEW

The system processes multiple PDF resumes and evaluates how well each candidate matches a given job description.

Pipeline steps:

1. Resume parsing
2. Candidate information extraction
3. Semantic embedding generation
4. Skill matching
5. Weighted ranking

The final output is a ranked list of candidates with skill gap analysis displayed through an interactive dashboard.

--------------------------------------------------

TECHNOLOGY STACK

Frontend: React  
Backend: Flask (Python)  
ML Models: Sentence Transformers (HuggingFace)  
NLP: spaCy  
PDF Parsing: pdfplumber with PyMuPDF fallback  
Charts: Recharts  
Icons: Lucide React  

--------------------------------------------------

SYSTEM ARCHITECTURE

User uploads resumes  
↓  
Flask API server  
↓  
ML Processing Pipeline

1. Resume Parser
2. Information Extractor
3. Semantic Matcher
4. Scoring Engine

↓  

Ranked candidate results → React Dashboard

--------------------------------------------------

PROJECT STRUCTURE

resume_screening_system/

app.py  
main.py  
resume_parser.py  
information_extractor.py  
semantic_matcher.py  
scoring_engine.py  
cleanup_cache.py  
config.json  

resume_ui/
frontend/
src/
App.jsx

--------------------------------------------------

HOW THE PIPELINE WORKS

STEP 1 — Resume Upload

Users upload PDF resumes and provide:

- Job description
- Required skills
- Minimum experience
- Skill vs semantic ranking weight

--------------------------------------------------

STEP 2 — Resume Parsing

resume_parser.py extracts text from PDF files using:

- pdfplumber
- PyMuPDF fallback

--------------------------------------------------

STEP 3 — Information Extraction

information_extractor.py extracts:

- Name (from filename)
- Email
- Phone number
- Skills
- Experience
- Education

Extraction uses regex patterns and spaCy Named Entity Recognition.

--------------------------------------------------

STEP 4 — Semantic Embedding

semantic_matcher.py converts resumes and job descriptions into embeddings using transformer models.

Similarity is computed using cosine similarity.

Two passes are used:

1. Full resume embedding
2. Key sections embedding (skills, experience, projects)

Combined semantic score:

Final Semantic Score =
0.40 × full_resume_similarity +
0.60 × key_section_similarity

--------------------------------------------------

CANDIDATE RANKING ALGORITHM

Final Score =
(skill_weight × skill_score) +
(semantic_weight × semantic_score)

Skill Score =
ratio of required skills found in resume

Semantic Score =
cosine similarity between resume and job description embeddings

Weights are controlled by the user through the dashboard slider.

--------------------------------------------------

EMBEDDING MODELS USED

MPNet
Type: SBERT
768 dimensions
Strong semantic understanding

MxBai
Contrastive embedding model
Ranked #1 on MTEB English leaderboard

Arctic
Retrieval embedding model
Enterprise-grade ranking

Important:

Only MPNet is a true SBERT model.

MxBai and Arctic are separate embedding architectures.

All models run fully offline after first download.

--------------------------------------------------

REACT DASHBOARD FEATURES

Dashboard

- Screening statistics
- Top candidate visualization
- Score distribution charts

Upload & Configure

- Drag-and-drop resume upload
- Job description input
- Skill requirement input
- Experience slider
- Ranking weight slider
- Model selector

Candidates

- Ranked candidate table
- Skill match visualization
- Candidate detail drawer
- CSV export of shortlisted candidates

Analytics

- Candidate score distribution
- KPI metrics
- Top candidate radar chart
- Score range analysis

--------------------------------------------------

RUNNING THE PROJECT

Start backend

python app.py

Flask API runs at

http://localhost:5001


Start frontend

cd resume_ui/frontend
npm install
npm start

React dashboard runs at

http://localhost:3000

--------------------------------------------------

MODEL DOWNLOAD (ONE TIME)

python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('multi-qa-mpnet-base-dot-v1')"

python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')"

python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v1.5')"

After downloading once, models run fully offline.

--------------------------------------------------

OUTPUT

The system produces:

- Ranked candidate list
- Skill match score
- Semantic similarity score
- Final ranking score
- Skill gap analysis

Shortlisted candidates can be exported to CSV containing:

Rank, Name, Email, Phone

--------------------------------------------------

AUTHOR

Madhan Kumar  
B.Sc Computer Science  
Final Year Project
