"""
main.py
-------
AI/ML Resume Screening System — Main Pipeline Orchestrator

Usage:
    python main.py                    # uses config.json
    python main.py my_config.json     # uses custom config
"""

import os
import sys
import json
import csv
import time

from resume_parser       import load_resumes_from_folder
from information_extractor import extract_all
from semantic_matcher    import rank_resumes_by_similarity
from scoring_engine      import ScoringConfig, score_candidates, print_results


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "resumes_folder": "resumes",
    "job_description": (
        "We are looking for a Python developer with experience in machine learning, "
        "deep learning, and NLP. Skills in scikit-learn, PyTorch or TensorFlow, "
        "pandas, numpy, and SQL are required. Flask or FastAPI for deployment is a plus."
    ),
    "required_skills": [
        "python", "machine learning", "deep learning", "nlp",
        "scikit-learn", "pytorch", "tensorflow", "sql",
        "pandas", "numpy", "flask", "data science"
    ],
    "scoring": {
        "skill_weight": 0.55,
        "semantic_weight": 0.45,
        "min_experience_years": 0.0,
        "required_education": [],
        "top_n": 50,
        "min_final_score": 0.0,
    },
    "output_csv": "results.csv",
}


def load_config(path: str = "config.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        print(f"[Config] Loaded from {path}\n")
        return cfg
    else:
        print(f"[Config] '{path}' not found — using defaults.\n")
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG


def save_results_to_csv(results: list, output_path: str):
    if not results:
        return
    fieldnames = [
        "rank", "name", "filename", "email", "phone",
        "experience_years", "total_skills", "skills_matched_count",
        "required_skills_matched", "skills_missing",
        "skill_score_pct", "semantic_score_pct", "final_score_pct",
        "eligible", "rejection_reason", "education",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({
                "rank":                    i,
                "name":                    r["name"],
                "filename":                r["filename"],
                "email":                   r["email"],
                "phone":                   r["phone"],
                "experience_years":        r["experience_years"],
                "total_skills":            len(r["skills"]),
                "skills_matched_count":    len(r["skills_matched"]),
                "required_skills_matched": ", ".join(r["skills_matched"]),
                "skills_missing":          ", ".join(r["skills_missing"]),
                "skill_score_pct":         f"{r['skill_score']*100:.1f}%",
                "semantic_score_pct":      f"{r['semantic_score']*100:.1f}%",
                "final_score_pct":         f"{r['final_score']*100:.1f}%",
                "eligible":                r["eligible"],
                "rejection_reason":        r["rejection_reason"],
                "education":               " | ".join(r["education"]),
            })
    print(f"[Output] Results saved to: {output_path}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(config_path: str = "config.json"):
    start = time.time()
    config = load_config(config_path)

    resumes_folder   = config.get("resumes_folder", "resumes")
    job_description  = config.get("job_description", "")
    required_skills  = config.get("required_skills", [])
    scoring_params   = config.get("scoring", {})
    output_csv       = config.get("output_csv", "results.csv")

    scoring_cfg = ScoringConfig(**scoring_params)

    print("=" * 65)
    print("   AI/ML RESUME SCREENING SYSTEM")
    print("=" * 65)
    print(f"  Job: {job_description[:80]}...")
    print(f"  Required skills ({len(required_skills)}): {', '.join(required_skills)}")
    print(f"  Scoring: {scoring_cfg.skill_weight*100:.0f}% skills / {scoring_cfg.semantic_weight*100:.0f}% semantic")
    print(f"  Showing top: {scoring_cfg.top_n} candidates")
    print("=" * 65 + "\n")

    # ── Step 1: Parse PDFs ────────────────────────────────────────────────
    print("[Step 1] Parsing resumes from:", resumes_folder)
    resume_texts = load_resumes_from_folder(resumes_folder)

    # ── Step 2: Extract info ──────────────────────────────────────────────
    print("[Step 2] Extracting candidate information...")
    extracted = {}
    for filename, text in resume_texts.items():
        info = extract_all(text, filename)
        extracted[filename] = info
        print(f"  → {filename}")
        print(f"       Name: {info['name']} | Email: {info['email'] or 'N/A'} | Skills: {len(info['skills'])} | Exp: {info['experience_years']}y")

    print()

    # ── Step 3: Semantic similarity ───────────────────────────────────────
    print("[Step 3] Computing semantic similarity...")
    similarity_results = rank_resumes_by_similarity(resume_texts, job_description)
    similarity_map = {r["filename"]: r["similarity_score"] for r in similarity_results}

    # ── Step 4: Score & rank ──────────────────────────────────────────────
    print("[Step 4] Scoring and ranking all candidates...")
    candidates = [
        {
            "filename":      fname,
            "info":          info,
            "semantic_score": similarity_map.get(fname, 0.0),
        }
        for fname, info in extracted.items()
    ]
    results = score_candidates(candidates, required_skills, scoring_cfg)

    # ── Display ───────────────────────────────────────────────────────────
    print_results(results)

    # ── Save CSV ──────────────────────────────────────────────────────────
    save_results_to_csv(results, output_csv)

    elapsed = time.time() - start
    print(f"\n[Done] Pipeline completed in {elapsed:.1f}s")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    run_pipeline(config_path)