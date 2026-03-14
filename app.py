"""
app.py
------
Flask backend for the AI Resume Screening System.
Connects the React frontend to the Python ML pipeline.

Endpoints:
    GET  /api/health          → server health check
    POST /api/upload-resumes  → save uploaded PDFs to uploads/
    POST /api/screen          → run the full ML pipeline
    GET  /api/results         → return latest ranked results
    GET  /api/candidate/<id>  → return a single candidate by ID
    GET  /api/config          → return current config.json
    POST /api/config          → save updated config.json
    GET  /api/model           → return currently active model
    POST /api/model           → switch active model
    POST /api/clear           → clear all uploads and results
    GET  /api/export          → download latest results as CSV
    GET  /api/audit           → return recent screening audit log

Run:
    python app.py
    → http://localhost:5001
"""

import csv
import hashlib
import io
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, Response
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
AUDIT_FILE    = Path("audit.jsonl")

UPLOAD_FOLDER.mkdir(exist_ok=True)

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_FILE_BYTES          = 10 * 1024 * 1024   # 10 MB per file
MAX_AUDIT_JD_CHARS      = 120                # truncate long job descriptions in the audit log

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
    """Load config.json from disk, or return defaults if the file is missing or malformed."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[Config] Could not read config.json (%s) — using defaults.", e)
    return DEFAULT_CONFIG


def save_config(cfg: dict):
    """Write config dict to config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def allowed_file(filename: str) -> bool:
    """Only PDF files are accepted."""
    return filename.lower().endswith(".pdf")


def _file_sha256(file_storage) -> str:
    """Return the SHA-256 hex digest of an uploaded file's contents."""
    h = hashlib.sha256()
    file_storage.stream.seek(0)
    for chunk in iter(lambda: file_storage.stream.read(8192), b""):
        h.update(chunk)
    file_storage.stream.seek(0)
    return h.hexdigest()


def _existing_hashes() -> dict:
    """
    Build a map of { sha256_hex: filename } for all PDFs already in uploads/.
    Used for content-based deduplication on each new upload.
    """
    hashes = {}
    for pdf in UPLOAD_FOLDER.glob("*.pdf"):
        try:
            h = hashlib.sha256(pdf.read_bytes()).hexdigest()
            hashes[h] = pdf.name
        except OSError:
            pass
    return hashes


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


def _write_audit(entry: dict):
    """Append a single screening audit record as a JSON line to audit.jsonl."""
    try:
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.warning("[Audit] Could not write audit log: %s", e)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """
    Lightweight health-check endpoint.
    Returns server status, active model, and pipeline readiness.
    """
    pdf_count = len(list(UPLOAD_FOLDER.glob("*.pdf")))
    return jsonify({
        "status":          "ok",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "active_model":    _active_model_key,
        "pdfs_uploaded":   pdf_count,
        "results_ready":   RESULTS_FILE.exists(),
    }), 200


@app.route("/api/upload-resumes", methods=["POST"])
def upload_resumes():
    """
    Accept one or more PDF files via multipart form-data.
    Enforces a 10 MB per-file limit and skips exact duplicate files
    (content-hash based deduplication).
    Saves accepted files to uploads/ and returns a detailed status.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files     = request.files.getlist("files")
    saved     = []
    rejected  = []
    duplicates = []

    existing_hashes = _existing_hashes()

    for file in files:
        if not file.filename:
            continue

        # ── Type check ───────────────────────────────────────────────────────
        if not allowed_file(file.filename):
            rejected.append({"filename": file.filename, "reason": "Not a PDF file"})
            continue

        # ── Size check (read up to MAX+1 bytes to detect oversized files) ────
        file.stream.seek(0, 2)          # seek to end
        size = file.stream.tell()
        file.stream.seek(0)
        if size > MAX_FILE_BYTES:
            mb = size / (1024 * 1024)
            rejected.append({
                "filename": file.filename,
                "reason": f"File too large ({mb:.1f} MB > 10 MB limit)",
            })
            logger.warning("[Upload] Rejected oversized file: %s (%.1f MB)", file.filename, mb)
            continue

        # ── Deduplication check ──────────────────────────────────────────────
        digest = _file_sha256(file)
        if digest in existing_hashes:
            duplicates.append({
                "filename":   file.filename,
                "matches":    existing_hashes[digest],
            })
            logger.info("[Upload] Duplicate skipped: %s (matches %s)", file.filename, existing_hashes[digest])
            continue

        # ── Save ─────────────────────────────────────────────────────────────
        filename = secure_filename(file.filename)
        dest = UPLOAD_FOLDER / filename
        file.save(str(dest))
        existing_hashes[digest] = filename   # update local cache for this batch
        saved.append(filename)
        logger.info("[Upload] Saved: %s", filename)

    if not saved and not duplicates:
        return jsonify({
            "error":    "No valid PDF files uploaded",
            "rejected": rejected,
        }), 400

    return jsonify({
        "message":    f"{len(saved)} resume(s) uploaded.",
        "saved":      saved,
        "duplicates": duplicates,
        "rejected":   rejected,
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

    started_at = datetime.now(timezone.utc)

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

        shortlisted = sum(1 for r in serialized if r["eligible"])
        elapsed_s   = (datetime.now(timezone.utc) - started_at).total_seconds()

        logger.info(
            "[Pipeline] Done — %d ranked | %d shortlisted | model: %s | %.1fs",
            len(serialized), shortlisted, model_name, elapsed_s,
        )

        # ── Audit log ────────────────────────────────────────────────────────
        _write_audit({
            "timestamp":        started_at.isoformat(),
            "model":            model_key,
            "total_resumes":    len(serialized),
            "shortlisted":      shortlisted,
            "rejected":         len(serialized) - shortlisted,
            "elapsed_seconds":  round(elapsed_s, 2),
            "job_description":  (job_description or "")[:MAX_AUDIT_JD_CHARS],
            "required_skills":  required_skills,
            "skill_weight":     round(scoring_cfg.skill_weight, 2),
            "semantic_weight":  round(scoring_cfg.semantic_weight, 2),
        })

        return jsonify({
            "message":     "Screening complete.",
            "model_used":  model_name,
            "total":       len(serialized),
            "shortlisted": shortlisted,
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
    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": f"Could not read results: {e}"}), 500
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


@app.route("/api/export", methods=["GET"])
def export_results():
    """
    Download the latest screening results as a CSV file.
    Only shortlisted (eligible) candidates are included.
    Returns 404 if no results have been generated yet.
    """
    if not RESULTS_FILE.exists():
        return jsonify({"error": "No results available. Run screening first."}), 404

    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": f"Could not read results: {e}"}), 500

    shortlisted = [r for r in results if r.get("eligible")]
    if not shortlisted:
        return jsonify({"error": "No shortlisted candidates to export."}), 404

    output = io.StringIO()
    fieldnames = ["rank", "name", "email", "phone", "experience", "skill_score_pct",
                  "semantic_score_pct", "final_score_pct", "matched_skills", "missing_skills", "filename"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in shortlisted:
        writer.writerow({
            "rank":               r.get("rank", ""),
            "name":               r.get("name", ""),
            "email":              r.get("email", ""),
            "phone":              r.get("phone", ""),
            "experience":         r.get("experience", 0),
            "skill_score_pct":    f"{r.get('skillScore', 0) * 100:.1f}%",
            "semantic_score_pct": f"{r.get('semanticScore', 0) * 100:.1f}%",
            "final_score_pct":    f"{r.get('finalScore', 0) * 100:.1f}%",
            "matched_skills":     "; ".join(r.get("matched_skills", [])),
            "missing_skills":     "; ".join(r.get("missing_skills", [])),
            "filename":           r.get("filename", ""),
        })

    csv_bytes = output.getvalue().encode("utf-8")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=shortlisted_{timestamp}.csv"},
    )


@app.route("/api/audit", methods=["GET"])
def get_audit():
    """
    Return the last N screening audit log entries.
    Query param: ?limit=20 (default 20, max 100)
    """
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except (ValueError, TypeError):
        limit = 20

    if not AUDIT_FILE.exists():
        return jsonify({"entries": [], "total": 0}), 200

    try:
        lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        # Most-recent first
        entries.reverse()
        return jsonify({"entries": entries[:limit], "total": len(entries)}), 200
    except OSError as e:
        return jsonify({"error": f"Could not read audit log: {e}"}), 500


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