ML-Based Resume Screening System

This project implements an ML-based Resume Screening System that analyzes and ranks candidate resumes against a given job description. The system combines rule-based extraction with semantic matching using transformer models to provide accurate and meaningful candidate evaluation.

⸻

Overview:

The system processes PDF resumes, extracts relevant information, and evaluates candidates based on:

	•	Skill matching
	•	Semantic similarity with job description
	•	Experience level

Unlike traditional keyword-based systems, this approach uses embeddings to understand the contextual meaning of text.

⸻

Features:

	•	Semantic matching using SentenceTransformers
	•	Robust PDF text extraction with fallback handling
	•	Automatic extraction of candidate details (name, email, phone, skills, experience, education)
	•	Weighted scoring and ranking system
	•	Candidate categorization (Strong Fit, Borderline, Weak Fit, Not Eligible)
	•	Thread-safe backend with support for concurrent processing
	•	Simple React-based frontend for interaction and visualization

⸻

System Architecture:

       Frontend (React)
             ↓
      Flask API (Backend)
             ↓
         ML Pipeline:
        resume_parser
     information_extractor
       semantic_matcher
        scoring_engine
             ↓
   Storage (files and JSON logs)

⸻

Technology Stack:

Backend:
	•	Python
	•	Flask
	•	Gunicorn

Machine Learning:
	•	PyTorch (CPU)
	•	SentenceTransformers
	•	NumPy

Document Processing:
	•	pdfplumber
	•	PyMuPDF

Frontend:
	•	React
	•	Vite

⸻

Setup Instructions:

Prerequisites
	•	Python 3.9 or above
	•	Node.js

⸻

Backend Setup:

git clone https://github.com/madhankumar-mk-28/Final-Year-Project.git
cd Final-Year-Project

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt


⸻

Frontend Setup:

cd resume_ui/frontend

npm install
npm run dev


⸻

Run Backend:

python app.py

- The backend will run on: http://127.0.0.1:5001

⸻

Key Files: 

	•	app.py – Main Flask application and API endpoints
	•	resume_parser.py – Extracts text from PDF resumes
	•	information_extractor.py – Extracts structured candidate information
	•	semantic_matcher.py – Computes semantic similarity using embeddings
	•	scoring_engine.py – Calculates scores and ranks candidates
	•	metrics_store.py – Stores evaluation metrics
	•	audit_logger.py – Logs system activity and errors
	•	cleanup_cache.py – Utility for cleaning temporary files

⸻

Scoring Method

The final score is calculated using a weighted combination:

Final Score =(Skill Score × Weight) +(Semantic Score × Weight) +(Experience Score × Weight)

Candidates are ranked and categorized based on dynamic thresholds.

⸻

Key Parameters:

	•	Maximum resume size: 10 MB
	•	Maximum text length: 500,000 characters
	•	Chunk size: 300 words
	•	Overlap: 50 words
	•	Dynamic threshold: 60th percentile
	•	Maximum concurrent screenings: 3

⸻

Use Cases:

	•	Resume filtering for recruitment
	•	Campus placement screening
	•	Candidate ranking and analysis
	•	HR automation tools

⸻

Author

Madhankumar 23BCS0163
B.Sc Computer Science
VIT, Vellore

⸻

Notes

This project is developed for academic purposes and demonstrates the use of machine learning techniques in recruitment systems.
