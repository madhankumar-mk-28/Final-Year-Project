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
import numpy as np

# ── ML pipeline modules ───────────────────────────────────────────────────────
from resume_parser         import load_resumes_from_folder
from information_extractor import extract_all
import semantic_matcher
from scoring_engine        import ScoringConfig, score_candidates
from metrics_store         import record_run, load_latest_run

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
# "ensemble" is a special key that runs all three models and averages the results.
VALID_MODELS = {
    "mpnet":    semantic_matcher.MPNET_MODEL,    # multi-qa-mpnet-base-dot-v1
    "mxbai":    semantic_matcher.MXBAI_MODEL,    # mixedbread-ai/mxbai-embed-large-v1
    "arctic":   semantic_matcher.ARCTIC_MODEL,   # Snowflake/snowflake-arctic-embed-m-v1.5
    "ensemble": "ensemble",     # weighted average of all three models
}

# Active model key for the current session
_active_model_key = "ensemble"

# Ensemble weights for combining multiple embedding models
ENSEMBLE_WEIGHTS = {
    "mpnet": 0.4,    # multi-qa-mpnet-base-dot-v1
    "mxbai": 0.3,    # mixedbread-ai/mxbai-embed-large-v1
    "arctic": 0.3,   # Snowflake/snowflake-arctic-embed-m-v1.5
}

# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "job_description": "",
    "required_skills": [],
    "model": "ensemble",
    "scoring": {
        "skill_weight":         0.55,
        "semantic_weight":      0.45,
        "min_experience_years": 0.0,
        "required_education":   [],
        "top_n":                100,
        "min_final_score":      0.0,
    },
}


def load_config() -> dict:
    """Load config.json from disk, or return defaults if missing or malformed."""
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
    """Build a map of { sha256_hex: filename } for all PDFs already in uploads/."""
    hashes = {}
    for pdf in UPLOAD_FOLDER.glob("*.pdf"):
        try:
            h = hashlib.sha256(pdf.read_bytes()).hexdigest()
            hashes[h] = pdf.name
        except OSError:
            pass
    return hashes


def serialize_results(results: list) -> list:
    """Convert scoring engine output into JSON-safe dicts matching the React frontend shape."""
    serialized = []
    rank = 1
    for r in results:
        serialized.append({
            "id":               rank,
            "rank":             rank,  # always set — UI uses eligible flag for Rejected badge
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

def _merge_candidate_profiles(profile_a: dict, profile_b: dict) -> dict:
    """Merge two profiles for the same person, keeping the most detailed value for each field."""
    merged = {}

    # Name: pick the longer (more complete) variant
    name_a = (profile_a.get("name") or "").strip()
    name_b = (profile_b.get("name") or "").strip()
    merged["name"] = name_a if len(name_a) >= len(name_b) else name_b

    # Email / phone: prefer non-empty, then longer
    for key in ("email", "phone"):
        val_a = (profile_a.get(key) or "").strip()
        val_b = (profile_b.get(key) or "").strip()
        merged[key] = val_a if len(val_a) >= len(val_b) else val_b

    # Skills: union
    skills_a = set(profile_a.get("skills") or [])
    skills_b = set(profile_b.get("skills") or [])
    merged["skills"] = sorted(skills_a | skills_b)

    # Experience: take the maximum
    merged["experience_years"] = max(
        profile_a.get("experience_years", 0.0),
        profile_b.get("experience_years", 0.0),
    )

    # Education: union of unique entries
    edu_a = profile_a.get("education") or []
    edu_b = profile_b.get("education") or []
    seen = set()
    merged_edu = []
    for entry in edu_a + edu_b:
        key = entry.strip().lower()
        if key not in seen:
            seen.add(key)
            merged_edu.append(entry)
    merged["education"] = merged_edu

    return merged


def merge_duplicate_candidates(extracted: dict) -> dict:
    """Truth Engine: merge multiple PDFs for the same person (matched by email or phone)."""
    groups = []
    assigned = set()

    items = list(extracted.items())

    # Build groups by matching email or phone
    for i, (fname_a, info_a) in enumerate(items):
        if fname_a in assigned:
            continue
        group = [(fname_a, info_a)]
        assigned.add(fname_a)
        email_a = (info_a.get("email") or "").strip().lower()

        for j in range(i + 1, len(items)):
            fname_b, info_b = items[j]
            if fname_b in assigned:
                continue
            email_b = (info_b.get("email") or "").strip().lower()

            match = False
            # Only merge on email — email is a reliable unique identifier.
            # Phone numbers can be shared (family), misread, or partially matched.
            if email_a and email_b and email_a == email_b:
                match = True

            if match:
                group.append((fname_b, info_b))
                assigned.add(fname_b)

        groups.append(group)

    # Merge each group
    merged = {}
    for group in groups:
        if len(group) == 1:
            merged[group[0][0]] = group[0][1]
        else:
            # Progressive merge: fold each profile into the accumulator
            acc = group[0][1]
            filenames = [group[0][0]]
            for fname, info in group[1:]:
                acc = _merge_candidate_profiles(acc, info)
                filenames.append(fname)
            primary_key = filenames[0]
            acc["_merged_from"] = filenames
            merged[primary_key] = acc
            logger.info(
                "[TruthEngine] Merged %d profiles → %s (files: %s)",
                len(filenames), acc.get("name", "?"), filenames,
            )

    return merged


def compare_candidates(profile_a: dict, profile_b: dict) -> dict:
    """Compute the delta (differences) between two candidate profiles."""
    delta = {}

    # Name
    name_a = (profile_a.get("name") or "").strip()
    name_b = (profile_b.get("name") or "").strip()
    if name_a != name_b:
        delta["name"] = {"before": name_a, "after": name_b}

    # Email
    email_a = (profile_a.get("email") or "").strip()
    email_b = (profile_b.get("email") or "").strip()
    if email_a != email_b:
        delta["email"] = {"before": email_a, "after": email_b}

    # Phone
    phone_a = (profile_a.get("phone") or "").strip()
    phone_b = (profile_b.get("phone") or "").strip()
    if phone_a != phone_b:
        delta["phone"] = {"before": phone_a, "after": phone_b}

    # Experience
    exp_a = profile_a.get("experience_years", 0.0)
    exp_b = profile_b.get("experience_years", 0.0)
    if exp_a != exp_b:
        delta["experience_years"] = {"before": exp_a, "after": exp_b}

    # Skills
    skills_a = set(profile_a.get("skills") or [])
    skills_b = set(profile_b.get("skills") or [])
    added   = sorted(skills_b - skills_a)
    removed = sorted(skills_a - skills_b)
    if added or removed:
        delta["skills"] = {"added": added, "removed": removed}

    # Education
    edu_a = set(e.strip().lower() for e in (profile_a.get("education") or []))
    edu_b = set(e.strip().lower() for e in (profile_b.get("education") or []))
    edu_added   = sorted(edu_b - edu_a)
    edu_removed = sorted(edu_a - edu_b)
    if edu_added or edu_removed:
        delta["education"] = {"added": edu_added, "removed": edu_removed}

    return delta


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Lightweight health-check endpoint returning server status and pipeline readiness."""
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
    """Accept PDF files via multipart form-data, deduplicate by content hash, save to uploads/."""
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
    """Run the full 4-step ML pipeline on all uploaded PDFs and return ranked results."""
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
        resume_texts, parse_failures = load_resumes_from_folder(str(UPLOAD_FOLDER))
        if parse_failures:
            logger.warning("[Pipeline] %d PDF(s) unparseable: %s", len(parse_failures), parse_failures)
        if not resume_texts:
            return jsonify({"error": "No text could be extracted. Check files are not scanned images or password-protected."}), 422

        # Step 2 — Extract candidate information
        logger.info("[Pipeline] Step 2 — Extracting candidate information...")
        extracted = {}
        for filename, text in resume_texts.items():
            try:
                extracted[filename] = extract_all(text, filename)
            except Exception as ex:
                logger.warning("[Pipeline] extract_all failed for %s: %s", filename, ex)
                parse_failures.append(filename)

        # Step 2b — Truth Engine: merge duplicate candidates (same email/phone)
        logger.info("[Pipeline] Step 2b — Running Truth Engine (duplicate merge)...")
        count_before_merge = len(extracted)
        extracted = merge_duplicate_candidates(extracted)
        merged_count = count_before_merge - len(extracted)

        # Step 3 — Compute document-level semantic similarity
        # Use ensemble (all 3 models) when model_key == "ensemble", otherwise
        # fall back to the single selected model.
        # We run each model individually (instead of rank_resumes_ensemble) so
        # that we can capture per_model_similarities for metrics recording and
        # Spearman rank-correlation analysis.
        per_model_similarities: dict = {}
        if model_key == "ensemble":
            logger.info("[Pipeline] Step 3 — Ensemble semantic similarity (all 3 models)...")
            for _m in ENSEMBLE_WEIGHTS:
                per_model_similarities[_m] = semantic_matcher.rank_resumes_by_similarity(
                    resume_texts, job_description, model_name=_m
                )
            # Weighted average (mirrors rank_resumes_ensemble logic)
            total_weight = sum(ENSEMBLE_WEIGHTS.values())
            accumulated: dict = {fname: 0.0 for fname in resume_texts}
            for _m, _w in ENSEMBLE_WEIGHTS.items():
                for r in per_model_similarities[_m]:
                    accumulated[r["filename"]] += _w * r["similarity_score"]
            similarity_results = [
                {"filename": f, "similarity_score": round(float(np.clip(s / total_weight, 0.0, 1.0)), 4)}
                for f, s in accumulated.items()
            ]
            similarity_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        else:
            logger.info("[Pipeline] Step 3 — Computing semantic similarity (%s)...", model_name)
            similarity_results = semantic_matcher.rank_resumes_by_similarity(
                resume_texts, job_description, model_name=model_name
            )
        similarity_map = {r["filename"]: r["similarity_score"] for r in similarity_results}

        # Step 3b — Skill-level semantic matching
        # Use embeddings to find required skills that semantically match
        # each candidate's extracted skills (catches near-synonyms not in the
        # static alias table, e.g. "collaboration" → "adaptability").
        skill_model = semantic_matcher.MPNET_MODEL if model_key == "ensemble" else model_name
        logger.info("[Pipeline] Step 3b — Semantic skill matching (%s)...", skill_model)
        semantic_skill_map: dict = {}
        for fname, info in extracted.items():
            semantic_skill_map[fname] = semantic_matcher.compute_skill_semantic_matches(
                info.get("skills", []),
                required_skills,
                model_name=skill_model,
            )

        # Step 4 — Score and rank
        logger.info("[Pipeline] Step 4 — Scoring and ranking...")
        candidates = [
            {
                "filename":              fname,
                "info":                  info,
                "semantic_score":        similarity_map.get(fname, 0.0),
                "semantic_skill_matches": semantic_skill_map.get(fname, set()),
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
            len(serialized), shortlisted, model_key, elapsed_s,
        )

        # ── Metrics store — record cosine similarities, stats, and accuracy ──
        try:
            record_run(
                run_id=started_at.isoformat(),
                model_key=model_key,
                job_description=job_description or "",
                similarity_results=similarity_results,
                per_model_similarities=per_model_similarities or None,
                final_scores=serialized,
            )
        except Exception as _me:
            logger.warning("[MetricsStore] Could not record run metrics: %s", _me)

        # ── Audit log ────────────────────────────────────────────────────────
        _write_audit({
            "timestamp":        started_at.isoformat(),
            "model":            model_key,
            "total_resumes":    len(serialized),
            "shortlisted":      shortlisted,
            "rejected":         len(serialized) - shortlisted,
            "parse_failures":   len(parse_failures),
            "merged_count":     merged_count,
            "elapsed_seconds":  round(elapsed_s, 2),
            "job_description":  (job_description or "")[:MAX_AUDIT_JD_CHARS],
            "required_skills":  required_skills,
            "skill_weight":     round(scoring_cfg.skill_weight, 2),
            "semantic_weight":  round(scoring_cfg.semantic_weight, 2),
        })

        return jsonify({
            "message":       "Screening complete.",
            "model_used":    model_key,
            "total":         len(serialized),
            "shortlisted":   shortlisted,
            "parse_failures": parse_failures,
            "merged_count":  merged_count,
            "results":       serialized,
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
    """Switch the active embedding model for this session."""
    global _active_model_key
    body      = request.get_json(silent=True) or {}
    model_key = body.get("model", "")

    if model_key not in VALID_MODELS:
        return jsonify({
            "error": f"Unknown model '{model_key}'. Valid: {list(VALID_MODELS)}"
        }), 400

    _active_model_key = model_key
    label = "all models (ensemble)" if model_key == "ensemble" else VALID_MODELS[model_key]
    logger.info("[Model] Switched to %s", label)
    return jsonify({
        "message":   f"Switched to {label}",
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
    """Download the latest screening results (shortlisted candidates only) as a CSV file."""
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
    """Return the last N screening audit log entries (default 20, max 100)."""
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


@app.route("/api/compare", methods=["POST"])
def compare():
    """Compare two candidates by their IDs and return a detailed delta."""
    if not RESULTS_FILE.exists():
        return jsonify({"error": "No results found. Run screening first."}), 404

    body = request.get_json(silent=True) or {}
    id_a = body.get("id_a")
    id_b = body.get("id_b")
    if not id_a or not id_b:
        return jsonify({"error": "Provide id_a and id_b in the request body."}), 400

    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": f"Could not read results: {e}"}), 500

    cand_a = next((r for r in results if r["id"] == id_a), None)
    cand_b = next((r for r in results if r["id"] == id_b), None)
    if not cand_a:
        return jsonify({"error": f"Candidate {id_a} not found."}), 404
    if not cand_b:
        return jsonify({"error": f"Candidate {id_b} not found."}), 404

    # Build profile dicts from serialized results
    profile_a = {
        "name": cand_a.get("name", ""),
        "email": cand_a.get("email", ""),
        "phone": cand_a.get("phone", ""),
        "skills": cand_a.get("skills", []),
        "experience_years": cand_a.get("experience", 0.0),
        "education": cand_a.get("education", []),
    }
    profile_b = {
        "name": cand_b.get("name", ""),
        "email": cand_b.get("email", ""),
        "phone": cand_b.get("phone", ""),
        "skills": cand_b.get("skills", []),
        "experience_years": cand_b.get("experience", 0.0),
        "education": cand_b.get("education", []),
    }

    delta = compare_candidates(profile_a, profile_b)

    # score_delta: positive means candidate A scores higher than candidate B
    score_delta = {
        "skillScore":    round(cand_a.get("skillScore", 0) - cand_b.get("skillScore", 0), 4),
        "semanticScore": round(cand_a.get("semanticScore", 0) - cand_b.get("semanticScore", 0), 4),
        "finalScore":    round(cand_a.get("finalScore", 0) - cand_b.get("finalScore", 0), 4),
    }

    return jsonify({
        "candidate_a": cand_a,
        "candidate_b": cand_b,
        "profile_delta": delta,
        "score_delta": score_delta,
    }), 200


@app.route("/metrics/latest", methods=["GET"])
def get_metrics_latest():
    """Return evaluation metrics for the most recent screening run."""
    run = load_latest_run()
    if not run:
        return jsonify({"available": False}), 200

    ev    = run.get("evaluation") or {}
    stats = run.get("stats")      or {}

    return jsonify({
        "available":        True,
        "model":            run.get("model_key"),
        "candidate_count":  run.get("resume_count"),
        "threshold":        ev.get("threshold"),
        "accuracy":         ev.get("accuracy"),
        "precision":        ev.get("precision"),
        "recall":           ev.get("recall"),
        "f1":               ev.get("f1"),
        "mean_similarity":  stats.get("mean"),
        "std_similarity":   stats.get("std"),
        "timestamp":        run.get("timestamp"),
    }), 200


# ── Serve React build (production) ────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    """Serves the built React app in production; in dev, React runs on port 3000."""
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