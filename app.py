"""
app.py
------
Flask backend for the AI Resume Screening System.
Connects the React frontend to the Python ML pipeline.

Endpoints:
    POST /api/upload-resumes  → save uploaded PDFs to uploads/
    POST /api/screen          → run the full ML pipeline
    GET  /api/results         → return latest ranked results
    GET  /api/candidate/<id>  → return a single candidate by ID
    GET  /api/config          → return current config.json
    POST /api/config          → save updated config.json
    GET  /api/model           → return currently active model
    POST /api/model           → switch active model
    POST /api/clear           → clear all uploads and results

Run:
    python app.py
    → http://localhost:5001
"""

import os
import json
import logging
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── ML pipeline modules ───────────────────────────────────────────────────────
from resume_parser         import load_resumes_from_folder
from information_extractor import extract_all
from semantic_matcher      import (
    rank_resumes_by_similarity,
    MPNET_MODEL, MXBAI_MODEL, ARCTIC_MODEL,
)
from scoring_engine        import ScoringConfig, score_candidates

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend/build", static_url_path="/")
CORS(app)   # Allow React dev server (port 3000) to call this API (port 5001)

# ── File paths ────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = Path("uploads")
RESULTS_FILE  = Path("last_results.json")
CONFIG_FILE   = Path("config.json")

UPLOAD_FOLDER.mkdir(exist_ok=True)

# ── Supported embedding models ────────────────────────────────────────────────
# Maps short frontend key → full HuggingFace model ID (used by sentence-transformers)
VALID_MODELS = {
    "mpnet":  MPNET_MODEL,    # multi-qa-mpnet-base-dot-v1
    "mxbai":  MXBAI_MODEL,    # mixedbread-ai/mxbai-embed-large-v1
    "arctic": ARCTIC_MODEL,   # Snowflake/snowflake-arctic-embed-m-v1.5
}

# Active model key for the current session
_active_model_key = "mpnet"

# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "job_description": "",
    "required_skills": [],
    "model": "mpnet",
    "scoring": {
        "skill_weight":         0.55,
        "semantic_weight":      0.45,
        "min_experience_years": 0.0,
        "required_education":   [],
        "top_n":                20,
        "min_final_score":      0.0,
    },
}


def load_config() -> dict:
    """Load config.json from disk, or return defaults if the file is missing."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return DEFAULT_CONFIG


def save_config(cfg: dict):
    """Write config dict to config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def allowed_file(filename: str) -> bool:
    """Only PDF files are accepted."""
    return filename.lower().endswith(".pdf")


def serialize_results(results: list) -> list:
    """
    Convert the scoring engine output into JSON-safe dicts
    that match the shape the React frontend expects.
    """
    serialized = []
    rank = 1
    for r in results:
        serialized.append({
            "id":               rank,
            "rank":             rank if r.get("eligible") else None,
            "name":             r.get("name", "Unknown"),
            "email":            r.get("email", ""),
            "phone":            r.get("phone", ""),
            "filename":         r.get("filename", ""),
            "experience":       r.get("experience_years", 0.0),
            "education":        r.get("education", []),
            "skills":           r.get("skills", []),
            "matched_skills":   r.get("skills_matched", []),
            "missing_skills":   r.get("skills_missing", []),
            "skillScore":       r.get("skill_score", 0.0),
            "semanticScore":    r.get("semantic_score", 0.0),
            "expScore":         r.get("exp_score", 0.0),
            "finalScore":       r.get("final_score", 0.0),
            "eligible":         r.get("eligible", False),
            "rejection_reason": r.get("rejection_reason", ""),
            "status":           "Shortlisted" if r.get("eligible") else "Rejected",
        })
        rank += 1
    return serialized


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/upload-resumes", methods=["POST"])
def upload_resumes():
    """
    Accept one or more PDF files via multipart form-data.
    Saves them to uploads/ and returns the saved filenames.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files    = request.files.getlist("files")
    saved    = []
    rejected = []

    for file in files:
        if not file.filename:
            continue
        if not allowed_file(file.filename):
            rejected.append(file.filename)
            continue
        filename = secure_filename(file.filename)
        file.save(str(UPLOAD_FOLDER / filename))
        saved.append(filename)
        logger.info("[Upload] Saved: %s", filename)

    if not saved:
        return jsonify({"error": "No valid PDF files uploaded", "rejected": rejected}), 400

    return jsonify({
        "message":  f"{len(saved)} resume(s) uploaded.",
        "saved":    saved,
        "rejected": rejected,
    }), 200


@app.route("/api/screen", methods=["POST"])
def screen():
    """
    Run the full 4-step ML pipeline on all uploaded PDFs.

    Request JSON:
    {
        "job_description": "...",
        "required_skills": ["python", "ml", ...],
        "model": "mpnet" | "mxbai" | "arctic",
        "config": {
            "min_experience_years": 0,
            "skill_weight": 0.55,
            "semantic_weight": 0.45,
            "top_n": 20
        }
    }
    """
    global _active_model_key

    body = request.get_json(silent=True) or {}
    cfg  = load_config()

    job_description = body.get("job_description") or cfg.get("job_description", "")
    required_skills = body.get("required_skills") or cfg.get("required_skills", [])
    cfg_override    = body.get("config", {})

    # Model: prefer request body, then session default
    model_key = body.get("model", _active_model_key)
    if model_key not in VALID_MODELS:
        return jsonify({
            "error": f"Unknown model '{model_key}'. Valid options: {list(VALID_MODELS)}"
        }), 400

    model_name        = VALID_MODELS[model_key]
    _active_model_key = model_key

    # Build scoring config
    base_scoring   = cfg.get("scoring", DEFAULT_CONFIG["scoring"])
    scoring_params = {**base_scoring, **cfg_override}
    try:
        scoring_cfg = ScoringConfig(**scoring_params)
    except Exception as e:
        return jsonify({"error": f"Invalid scoring config: {e}"}), 400

    # Check for uploaded PDFs
    pdf_files = list(UPLOAD_FOLDER.glob("*.pdf"))
    if not pdf_files:
        return jsonify({"error": "No resumes found. Upload PDFs first."}), 400

    try:
        # Step 1 — Parse PDFs
        logger.info("[Pipeline] Step 1 — Parsing %d PDF(s)...", len(pdf_files))
        resume_texts = load_resumes_from_folder(str(UPLOAD_FOLDER))

        # Step 2 — Extract candidate information
        logger.info("[Pipeline] Step 2 — Extracting candidate information...")
        extracted = {
            filename: extract_all(text, filename)
            for filename, text in resume_texts.items()
        }

        # Step 3 — Compute semantic similarity
        logger.info("[Pipeline] Step 3 — Computing semantic similarity (%s)...", model_name)
        similarity_results = rank_resumes_by_similarity(
            resume_texts, job_description, model_name=model_name
        )
        similarity_map = {r["filename"]: r["similarity_score"] for r in similarity_results}

        # Step 4 — Score and rank
        logger.info("[Pipeline] Step 4 — Scoring and ranking...")
        candidates = [
            {
                "filename":       fname,
                "info":           info,
                "semantic_score": similarity_map.get(fname, 0.0),
            }
            for fname, info in extracted.items()
        ]
        results    = score_candidates(candidates, required_skills, scoring_cfg)
        serialized = serialize_results(results)

        # Cache to disk
        with open(RESULTS_FILE, "w") as f:
            json.dump(serialized, f, indent=2)

        logger.info(
            "[Pipeline] Done — %d ranked | %d shortlisted | model: %s",
            len(serialized),
            sum(1 for r in serialized if r["eligible"]),
            model_name,
        )

        return jsonify({
            "message":     "Screening complete.",
            "model_used":  model_name,
            "total":       len(serialized),
            "shortlisted": sum(1 for r in serialized if r["eligible"]),
            "results":     serialized,
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/results", methods=["GET"])
def get_results():
    """Return the most recent screening results from disk."""
    if not RESULTS_FILE.exists():
        return jsonify({"error": "No results yet. Run /api/screen first."}), 404
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    return jsonify({"results": results, "total": len(results)}), 200


@app.route("/api/candidate/<int:candidate_id>", methods=["GET"])
def get_candidate(candidate_id: int):
    """Return full detail for a single candidate by their ID."""
    if not RESULTS_FILE.exists():
        return jsonify({"error": "No results found."}), 404
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    match = next((r for r in results if r["id"] == candidate_id), None)
    if not match:
        return jsonify({"error": f"Candidate {candidate_id} not found."}), 404
    return jsonify(match), 200


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return the current job configuration."""
    return jsonify(load_config()), 200


@app.route("/api/config", methods=["POST"])
def post_config():
    """Save job config from the frontend Job Config panel."""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "No JSON body provided"}), 400
    save_config(body)
    logger.info("[Config] config.json updated.")
    return jsonify({"message": "config.json saved."}), 200


@app.route("/api/model", methods=["GET"])
def get_model():
    """Return the currently active embedding model."""
    return jsonify({
        "active":    _active_model_key,
        "full_name": VALID_MODELS[_active_model_key],
        "available": list(VALID_MODELS.keys()),
    }), 200


@app.route("/api/model", methods=["POST"])
def set_model():
    """
    Switch the active embedding model for this session.
    Body: { "model": "mpnet" | "mxbai" | "arctic" }
    """
    global _active_model_key
    body      = request.get_json(silent=True) or {}
    model_key = body.get("model", "")

    if model_key not in VALID_MODELS:
        return jsonify({
            "error": f"Unknown model '{model_key}'. Valid: {list(VALID_MODELS)}"
        }), 400

    _active_model_key = model_key
    logger.info("[Model] Switched to %s", VALID_MODELS[model_key])
    return jsonify({
        "message":   f"Switched to {VALID_MODELS[model_key]}",
        "active":    model_key,
        "full_name": VALID_MODELS[model_key],
    }), 200


@app.route("/api/clear", methods=["POST"])
def clear_uploads():
    """Delete all uploaded PDFs and clear results. Called on 'New Screening'."""
    for f in UPLOAD_FOLDER.glob("*.pdf"):
        f.unlink()
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    logger.info("[Clear] All uploads and results cleared.")
    return jsonify({"message": "Uploads and results cleared."}), 200


# ── Serve React build (production) ────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    """
    Serves the built React app in production mode.
    In development, React runs on port 3000 — this route won't be hit.
    """
    build_dir = Path("frontend/build")
    if build_dir.exists():
        if path and (build_dir / path).exists():
            return send_from_directory(str(build_dir), path)
        return send_from_directory(str(build_dir), "index.html")
    return jsonify({"message": "Flask API running. Start React dev server on port 3000."}), 200


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))

    print("=" * 58)
    print("  ML-Based Resume Screening System — Flask API")
    print(f"  Running on port {port}")
    print(f"  Default model   : {VALID_MODELS['mpnet']}")
    print(f"  Available models: {', '.join(VALID_MODELS)}")
    print("=" * 58)

    app.run(host="0.0.0.0", port=port)