"""
app.py — Flask backend for the AI Resume Screening System.
Endpoints: /api/health, /api/upload-resumes, /api/screen, /api/result/<id>, /api/config, /api/model, /api/clear,
           /api/stats, /api/result/<task_id>/export, /api/validate-jd
Run: python app.py  →  http://localhost:5001
"""
from __future__ import annotations

__version__ = "1.1.0"



import atexit
import hashlib
import json
import re
import shutil
import time
import uuid
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import csv
import io

import numpy as np
from flask import Flask, Response, g, request, jsonify, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename

from resume_parser         import load_resumes_from_folder
from information_extractor import extract_all
from semantic_matcher      import (
    rank_resumes_by_similarity,
    MPNET_MODEL, MXBAI_MODEL, ARCTIC_MODEL,
    SKILL_THRESHOLD_BY_MODEL, SKILL_SEMANTIC_THRESHOLD,
    _get_model as _sm_get_model,
)
from scoring_engine        import ScoringConfig, score_candidates
import metrics_store
from audit_logger          import write_audit as _write_audit, log_failure as _log_failure, normalize_skill

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# Suppress werkzeug's per-request access log — keeps terminal readable during ML pipeline
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Flask app + CORS
app = Flask(__name__, static_folder="frontend/build", static_url_path="/")

_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
CORS(
    app,
    origins=_cors_origins,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Requested-With"],
    supports_credentials=False,
    max_age=600,
)

# Directory paths
UPLOAD_BASE   = Path("uploads")
RESULTS_DIR   = Path("results")
CONFIG_FILE   = Path("config.json")

# Frontend URL — where the Vite / React app is served.
# Override via the FRONTEND_URL environment variable for production deployments.
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

UPLOAD_BASE.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Input validation
_SESSION_ID_RE = re.compile(r"^[a-f0-9]{32}$")

# Hard limits
MAX_FILE_BYTES          = 10 * 1024 * 1024   # 10 MB per PDF
MAX_AUDIT_JD_CHARS      = 120                # truncate JD in audit entries
MAX_JSONL_ENTRIES       = 500                # max lines per JSONL log
MAX_CONFIG_BYTES        = 8192               # max POST /api/config payload
MAX_JD_CHARS            = 4000               # max job description length
MAX_FILES_PER_SESSION   = 100                # max PDFs per upload session (must match frontend MAX_FILES)
MAX_REQUIRED_SKILLS     = 50                 # max skills list length
ALLOWED_CONFIG_KEYS     = {"job_description", "required_skills", "model", "scoring"}

# Concurrency and rate-limiting tunables
MAX_CONCURRENT_SCREENINGS = 3      # beyond this POST /api/screen returns 503
MIN_FREE_DISK_MB          = 200    # below this POST /api/upload-resumes returns 503
RATE_LIMIT_WINDOW_S       = 60     # sliding window for per-IP rate limiting
RATE_LIMIT_MAX_REQUESTS   = 30     # max requests per IP per window
SLOW_REQUEST_THRESHOLD_S  = 20.0   # log WARNING if request exceeds this duration
TASK_TTL_S                = 3600   # async task expiry (1 hour)
TASK_STORE_LIMIT          = 60     # max in-memory task entries (evict oldest beyond this)

# Structured API error codes — machine-readable companion to human-friendly error messages
API_ERROR = {
    "no_files":         "NO_FILES_PROVIDED",
    "invalid_session":  "INVALID_SESSION_ID",
    "session_missing":  "SESSION_NOT_FOUND",
    "jd_too_short":     "JD_TOO_SHORT",
    "rate_limited":     "RATE_LIMITED",
    "overloaded":       "SERVER_OVERLOADED",
    "disk_full":        "DISK_FULL",
    "bad_model":        "UNKNOWN_MODEL",
    "bad_config":       "INVALID_CONFIG",
    "bad_skills":       "INVALID_SKILLS",
    "too_many_files":   "TOO_MANY_FILES",
    "no_valid_pdfs":    "NO_VALID_PDFS",
    "no_resumes":       "NO_RESUMES_IN_SESSION",
    "task_expired":     "TASK_EXPIRED",
    "task_not_found":   "TASK_NOT_FOUND",
    "payload_too_large":"PAYLOAD_TOO_LARGE",
    "bad_task_id":      "INVALID_TASK_ID",
    "not_found":        "NOT_FOUND",
    "internal_error":   "INTERNAL_ERROR",
    "pipeline_error":   "PIPELINE_ERROR",
}

# Allowed embedding models
VALID_MODELS = {
    "mpnet":  MPNET_MODEL,
    "mxbai":  MXBAI_MODEL,
    "arctic": ARCTIC_MODEL,
}

# Trusted reverse-proxy IPs (only these may set X-Forwarded-For)
_env_proxies = os.environ.get("TRUSTED_PROXIES", "127.0.0.1,::1,10.0.0.0/8")
TRUSTED_PROXIES: set = {p.strip() for p in _env_proxies.split(",") if p.strip()}

# Scoring fields accepted from the request body
_SCORING_FIELDS = frozenset({
    "skill_weight", "semantic_weight", "exp_weight",
    "min_experience_years", "required_education",
    "top_n", "min_final_score",
})

# Default config (used when config.json is absent or corrupt)
DEFAULT_CONFIG = {
    "job_description": "",
    "required_skills": [],
    "model": "mpnet",
    "scoring": {
        "skill_weight":         0.55,
        "semantic_weight":      0.45,
        "min_experience_years": 0.0,
        "required_education":   [],
        "top_n":                100,
        "min_final_score":      0.0,
    },
}


# Input sanitisation — strip HTML tags and control chars from user-supplied text
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    """Strip HTML tags, control characters, and collapse excessive whitespace."""
    text = _HTML_TAG_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse 3+ newlines
    text = re.sub(r"[ \t]{4,}", "  ", text)     # collapse excessive spaces/tabs
    return text.strip()


# Request timing hooks — log slow requests (>20s) as WARNING, others as DEBUG

@app.before_request
def _record_request_start():
    g.req_start = time.time()


@app.after_request
def _log_request_duration(response):
    duration = time.time() - getattr(g, "req_start", time.time())
    if duration >= SLOW_REQUEST_THRESHOLD_S:
        logger.warning(
            "[Timing] SLOW %s %s → %d — %.2fs",
            request.method, request.path, response.status_code, duration,
        )
    else:
        logger.debug(
            "[Timing] %s %s → %d — %.3fs",
            request.method, request.path, response.status_code, duration,
        )
    return response


# Per-IP rate limiting — sliding window, in-memory, no external dependencies

_rate_store: dict[str, list[float]] = {}
_rate_lock  = threading.Lock()


def _get_client_ip() -> str:
    """Return real client IP; only trust X-Forwarded-For from TRUSTED_PROXIES."""
    remote = request.remote_addr or "unknown"
    if remote in TRUSTED_PROXIES:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return remote


def _is_rate_limited(ip: str) -> bool:
    """Return True if this IP has exceeded the request quota in the current window."""
    now    = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_S
    with _rate_lock:
        ts = [t for t in _rate_store.get(ip, []) if t > cutoff]
        if len(ts) >= RATE_LIMIT_MAX_REQUESTS:
            _rate_store[ip] = ts
            return True
        ts.append(now)
        _rate_store[ip] = ts
        return False


def _prune_rate_store():
    """Remove inactive IPs from the rate-limit store — called hourly by cleanup worker."""
    cutoff = time.time() - RATE_LIMIT_WINDOW_S
    with _rate_lock:
        stale = [ip for ip, ts in _rate_store.items() if not any(t > cutoff for t in ts)]
        for ip in stale:
            del _rate_store[ip]
    if stale:
        logger.debug("[RateLimit] Pruned %d stale IP entries.", len(stale))


# Async task store — each /api/screen call creates a task_id and returns 202

_task_store: dict[str, dict] = {}
_task_lock  = threading.Lock()
_screening_executor = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_SCREENINGS,
    thread_name_prefix="screener",
)


def _task_set(task_id: str, data: dict):
    """Insert a new task, evicting the oldest entry if the store is at capacity.

    Cap rationale: each completed task holds a full serialised result JSON (up to ~200 KB).
    Without a cap, 1000 req/hr × 1hr TTL = 1000 entries × 200 KB = 200 MB of in-memory growth
    between hourly prune cycles.  We bound this to TASK_STORE_LIMIT entries instead.
    """
    with _task_lock:
        if len(_task_store) >= TASK_STORE_LIMIT:
            # Evict the oldest task by created_at (not necessarily FIFO if clocks drift)
            oldest_id = min(_task_store, key=lambda tid: _task_store[tid].get("created_at", 0))
            del _task_store[oldest_id]
            logger.debug("[Tasks] Task store cap (%d) reached — evicted oldest: %s", TASK_STORE_LIMIT, oldest_id)
        _task_store[task_id] = data


def _task_get(task_id: str) -> dict | None:
    with _task_lock:
        return _task_store.get(task_id)


def _task_update(task_id: str, patch: dict) -> None:
    """Atomic read-modify-write — prevents TOCTOU race between cleanup and worker."""
    with _task_lock:
        existing = _task_store.get(task_id, {})
        _task_store[task_id] = {**existing, **patch}


def _prune_tasks():
    """Remove tasks older than TASK_TTL_S — called hourly by cleanup worker."""
    cutoff = time.time() - TASK_TTL_S
    with _task_lock:
        stale = [tid for tid, t in _task_store.items()
                 if t.get("created_at", 0) < cutoff]
        for tid in stale:
            del _task_store[tid]
    if stale:
        logger.debug("[Tasks] Pruned %d stale task(s).", len(stale))


# Concurrent-screening counter — guards against burst overload

_active_screenings      = 0
_active_screenings_lock = threading.Lock()


# Thread-safe in-memory metrics — exposed via GET /api/health

_metrics: dict[str, int] = {
    "total_upload_requests":    0,
    "total_files_saved":        0,
    "total_files_rejected":     0,
    "total_screen_requests":    0,
    "total_resumes_processed":  0,
    "total_parse_failures":     0,
    "total_screen_errors":      0,
    "total_rate_limited":       0,
    "total_overload_rejected":  0,
}
_metrics_lock = threading.Lock()


def _inc(key: str, amount: int = 1):
    with _metrics_lock:
        _metrics[key] = _metrics.get(key, 0) + amount


def _get_metrics() -> dict:
    with _metrics_lock:
        return dict(_metrics)


# Disk space guard — blocks uploads when free disk falls below MIN_FREE_DISK_MB

def _disk_space_ok() -> bool:
    """Return False when free disk < MIN_FREE_DISK_MB. Fails open on stat error."""
    try:
        return shutil.disk_usage(UPLOAD_BASE).free / (1024 * 1024) >= MIN_FREE_DISK_MB
    except OSError:
        return True


# In-memory PDF count — kept accurate by upload / cleanup / clear operations

_pdf_count      = 0
_pdf_count_lock = threading.Lock()


def _pdf_count_init():
    global _pdf_count
    with _pdf_count_lock:
        _pdf_count = sum(1 for _ in UPLOAD_BASE.rglob("*.pdf"))


def _pdf_count_add(n: int = 1):
    global _pdf_count
    with _pdf_count_lock:
        _pdf_count += n


def _pdf_count_sub(n: int = 1):
    global _pdf_count
    with _pdf_count_lock:
        _pdf_count = max(0, _pdf_count - n)


# Stale session cleanup — removes upload folders older than max_age_seconds

# Defined here — called by cleanup_stale_sessions which runs at module init
def _session_results_file(session_id: str) -> Path:
    return RESULTS_DIR / f"{session_id}.json"


def cleanup_stale_sessions(max_age_seconds: int = 3600) -> int:
    """Remove session folders and result files older than max_age_seconds. Returns count removed."""
    now, cleaned, errors = time.time(), 0, 0
    logger.info("[Cleanup] Scanning for stale sessions (max_age=%ds).", max_age_seconds)

    try:
        entries = list(UPLOAD_BASE.iterdir())
    except OSError as e:
        logger.error("[Cleanup] Cannot scan uploads directory: %s", e)
        return 0

    for d in entries:
        if not d.is_dir():
            continue
        try:
            # Use newest mtime across dir + all files inside — keeps recently-screened sessions alive
            dir_mtime   = d.stat().st_mtime
            file_times  = [f.stat().st_mtime for f in d.iterdir() if f.is_file()]
            effective   = max([dir_mtime] + file_times) if file_times else dir_mtime
            age         = now - effective
        except OSError:
            continue

        if age <= max_age_seconds:
            continue

        session_id = d.name
        try:
            pdf_count = sum(1 for _ in d.glob("*.pdf"))
            shutil.rmtree(d, ignore_errors=True)
            _pdf_count_sub(pdf_count)
            results_file = _session_results_file(session_id)
            if results_file.exists():
                results_file.unlink(missing_ok=True)
            cleaned += 1
        except Exception as e:
            errors += 1
            logger.warning("[Cleanup] Failed to remove session %s: %s", session_id, e)

    logger.info("[Cleanup] Done — %d session(s) removed, %d error(s).", cleaned, errors)
    return cleaned


def _cleanup_worker():
    """Background daemon — hourly cleanup, rate-store pruning, task pruning. Logs heartbeat each cycle."""
    logger.info("[Cleanup] Background worker started (pid=%d).", os.getpid())
    while True:
        time.sleep(3600)
        logger.info("[Cleanup] Worker heartbeat — running hourly tasks.")
        try:
            _prune_rate_store()
            _prune_tasks()
            cleanup_stale_sessions(max_age_seconds=7200)
        except Exception as exc:
            logger.error("[Cleanup] Unexpected error in worker: %s", exc, exc_info=True)


# Startup init — guarded by Event so it runs exactly once per process

_startup_done = threading.Event()


def _startup_init():
    if _startup_done.is_set():
        return
    _startup_done.set()

    logger.info("[Startup] Initialising (pid=%d).", os.getpid())

    _pdf_count_init()
    logger.info("[Startup] PDF count from disk: %d.", _pdf_count)

    removed = cleanup_stale_sessions(max_age_seconds=7200)
    logger.info("[Startup] Startup cleanup — %d session(s) removed.", removed)

    # Lazy model loading — model is loaded on first /api/screen call instead of at
    # startup, allowing the server to accept health-check / upload requests immediately.
    logger.info("[Startup] Model loading deferred to first /api/screen call (lazy).")

    # Register executor shutdown on SIGTERM/atexit — prevents stranded _active_screenings counter
    atexit.register(lambda: _screening_executor.shutdown(wait=False, cancel_futures=True))

    t = threading.Thread(target=_cleanup_worker, daemon=True, name="cleanup-worker")
    t.start()
    logger.info("[Startup] Background cleanup thread started.")


# Module-level call — runs for Gunicorn workers and direct python invocations
_startup_init()


# Core helpers

def _validate_session_id(session_id: str) -> bool:
    return bool(_SESSION_ID_RE.fullmatch(session_id))


def _safe_iterdir(path: Path):
    """Yield directory entries, ignoring OSError if the directory is deleted mid-iteration."""
    try:
        yield from path.iterdir()
    except OSError:
        return


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[Config] Could not read config.json (%s) — using defaults.", e)
    return DEFAULT_CONFIG


def save_config(cfg: dict):
    """Atomic write via temp file — readers always see a complete JSON file."""
    tmp = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    tmp.replace(CONFIG_FILE)


def allowed_file(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def _file_sha256(file_storage) -> str:
    """Compute SHA-256 of an uploaded file stream; resets stream position."""
    h = hashlib.sha256()
    file_storage.stream.seek(0)
    for chunk in iter(lambda: file_storage.stream.read(8192), b""):
        h.update(chunk)
    file_storage.stream.seek(0)
    return h.hexdigest()


def _existing_hashes(folder: Path) -> dict:
    """Build a {sha256: filename} map for all PDFs already in the session folder."""
    hashes = {}
    for pdf in folder.glob("*.pdf"):
        try:
            h = hashlib.sha256()
            with open(pdf, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            hashes[h.hexdigest()] = pdf.name
        except OSError:
            pass
    return hashes


def serialize_results(results: list, skills_configured: bool = True) -> list:
    """Convert scoring engine output to JSON-safe dicts for the React frontend."""
    serialized = []
    for rank, r in enumerate(results, start=1):
        serialized.append({
            "id":               rank,
            "rank":             rank,
            "name":             r.get("name", "Unknown"),
            "email":            r.get("email") or "N/A",
            "phone":            r.get("phone") or "N/A",
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
            "rejection_code":   r.get("rejection_code", ""),   # stable enum for frontend switch
            "status":           "Shortlisted" if r.get("eligible") else "Rejected",
            "band":             r.get("band", ""),
            "dynamic_threshold": r.get("dynamic_threshold", 0.5),
            "fn_recovered":      r.get("fn_recovered", False),
            "skills_configured": skills_configured,
            "links":            {k: (v or "") for k, v in r.get("links", {"linkedin": "", "github": "", "portfolio": ""}).items()},
        })
    return serialized


def _write_results(results_file: Path, serialized: list):
    """Atomic results write — readers always see a complete file."""
    tmp = results_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(serialized, f, indent=2)
    tmp.replace(results_file)


# Candidate deduplication — merges profiles that share the same email address

def _merge_candidate_profiles(profile_a: dict, profile_b: dict) -> dict:
    """Merge two candidate profiles — keeps longest name/contact, unions skills, takes max exp."""
    merged = {}
    name_a = (profile_a.get("name") or "").strip()
    name_b = (profile_b.get("name") or "").strip()
    merged["name"] = name_a if len(name_a) >= len(name_b) else name_b

    for key in ("email", "phone"):
        val_a = (profile_a.get(key) or "").strip()
        val_b = (profile_b.get(key) or "").strip()
        merged[key] = val_a if len(val_a) >= len(val_b) else val_b

    merged["skills"] = sorted(
        set(profile_a.get("skills") or []) | set(profile_b.get("skills") or [])
    )
    merged["experience_years"] = max(
        profile_a.get("experience_years", 0.0),
        profile_b.get("experience_years", 0.0),
    )

    seen, edu = set(), []
    for entry in (profile_a.get("education") or []) + (profile_b.get("education") or []):
        k = entry.strip().lower()
        if k not in seen:
            seen.add(k)
            edu.append(entry)
    merged["education"] = edu
    return merged


def merge_duplicate_candidates(extracted: dict) -> dict:
    """Group extracted profiles by shared email and merge each group into one."""
    groups, assigned = [], set()
    items = list(extracted.items())

    for i, (fname_a, info_a) in enumerate(items):
        if fname_a in assigned:
            continue
        group   = [(fname_a, info_a)]
        assigned.add(fname_a)
        email_a = (info_a.get("email") or "").strip().lower()

        for j in range(i + 1, len(items)):
            fname_b, info_b = items[j]
            if fname_b in assigned:
                continue
            email_b = (info_b.get("email") or "").strip().lower()
            if email_a and email_b and email_a == email_b:
                group.append((fname_b, info_b))
                assigned.add(fname_b)
        groups.append(group)

    merged = {}
    for group in groups:
        if len(group) == 1:
            merged[group[0][0]] = group[0][1]
        else:
            acc, filenames = group[0][1], [group[0][0]]
            for fname, info in group[1:]:
                acc = _merge_candidate_profiles(acc, info)
                filenames.append(fname)
            acc["_merged_from"] = filenames
            merged[filenames[0]] = acc
            logger.info("[TruthEngine] Merged %d profiles → %s", len(filenames), acc.get("name", "?"))

    return merged


def _compute_similarities(model_key: str, resume_texts: dict, job_description: str) -> tuple[list, dict]:
    model_name = VALID_MODELS[model_key]
    logger.info("[Pipeline] Step 3 — Computing semantic similarity (%s)...", model_name)
    return rank_resumes_by_similarity(resume_texts, job_description, model_name=model_name), {}


# Core ML pipeline — submitted to ThreadPoolExecutor by POST /api/screen

def _run_screening_pipeline(
    req_id: str,
    session_id: str,
    session_folder: Path,
    pdf_files: list,
    job_description: str,
    required_skills: list,
    model_key: str,
    model_name: str,
    scoring_cfg: ScoringConfig,
    started_at: datetime,
) -> dict:
    t_step = time.time()
    parse_failures: list[str] = []

    try:
        # Step 1: Parse PDFs
        logger.info("[%s] Step 1 — Parsing %d PDF(s)...", req_id, len(pdf_files))
        try:
            resume_texts, parse_failures = load_resumes_from_folder(str(session_folder))
        except Exception as parse_exc:
            logger.error("[%s] Parsing stage failed: %s", req_id, parse_exc)
            _inc("total_parse_failures", len(pdf_files))
            return {"error": f"PDF parsing failed: {parse_exc}", "candidates": []}

        logger.info("[%s] Step 1 done — %.2fs", req_id, time.time() - t_step)
        t_step = time.time()

        if parse_failures:
            logger.warning("[%s] %d PDF(s) unparseable: %s", req_id, len(parse_failures), parse_failures)
            _inc("total_parse_failures", len(parse_failures))

        if not resume_texts:
            _inc("total_parse_failures", len(pdf_files))
            return {
                "error": "No text could be extracted. Check files are not scanned images or password-protected.",
                "candidates": [],
            }

        # Step 2: Extract candidate info
        logger.info("[%s] Step 2 — Extracting candidate information...", req_id)
        extracted: dict = {}
        for filename, text in resume_texts.items():
            try:
                extracted[filename] = extract_all(text, filename)
            except Exception as ex:
                logger.warning("[%s] extract_all failed for %s: %s", req_id, filename, ex)
                parse_failures.append(filename)
                _inc("total_parse_failures")

        logger.info("[%s] Step 2 done — %.2fs", req_id, time.time() - t_step)
        t_step = time.time()

        # Step 2b: Deduplicate by shared email
        count_before = len(extracted)
        extracted    = merge_duplicate_candidates(extracted)
        merged_count = count_before - len(extracted)

        # Step 3: Semantic similarity
        try:
            similarity_results, per_model_similarities = _compute_similarities(
                model_key, resume_texts, job_description
            )
            semantic_ok = True
        except Exception as sem_exc:
            # Fall back to skill-only scoring if the semantic model fails
            logger.error("[%s] Semantic similarity failed: %s — skill-only fallback", req_id, sem_exc)
            similarity_results     = [{"filename": f, "similarity_score": 0.0} for f in resume_texts]
            per_model_similarities = {}
            semantic_ok            = False

        similarity_map     = {r["filename"]: r["similarity_score"] for r in similarity_results}
        # Propagate OOM fallback flag — candidates whose zero vectors indicate _safe_encode retry
        # exhaustion will bypass sigmoid calibration and be forced to semantic_score=0.0 in scoring.
        oom_fallback_map   = {r["filename"]: r.get("oom_fallback", False) for r in similarity_results}

        # Best similarity across merged source files
        for fname, info in extracted.items():
            merged_from = info.get("_merged_from", [fname])
            if len(merged_from) > 1:
                similarity_map[fname] = max(similarity_map.get(src, 0.0) for src in merged_from)

        logger.info("[%s] Step 3 done — %.2fs", req_id, time.time() - t_step)
        t_step = time.time()

        # Step 3b: Semantic skill matching
        semantic_skill_map: dict = {}
        if required_skills and semantic_ok:
            try:
                _skill_model = _sm_get_model(model_name)

                # Arctic is asymmetric: required skills are the "query" side, candidate
                # skills are the "document" side. Without the query prefix, both land in
                # document space and soft-skill cosines inflate to ~0.80+ for everything.
                _skill_req_kwargs = dict(
                    convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
                )
                if model_name == ARCTIC_MODEL:
                    _skill_req_kwargs["prompt_name"] = "query"

                # Encode required skills once; batch-encode all unique candidate skills
                req_skill_embs = _skill_model.encode(required_skills, **_skill_req_kwargs)
                unique_skills = list(dict.fromkeys(
                    skill
                    for info in extracted.values()
                    for skill in info.get("skills", [])
                ))
                skill_to_idx = {s: i for i, s in enumerate(unique_skills)}
                cand_indices = [
                    (fname, [skill_to_idx[s] for s in info.get("skills", []) if s in skill_to_idx])
                    for fname, info in extracted.items()
                ]

                if unique_skills:
                    all_cand_embs  = _skill_model.encode(
                        unique_skills,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    skill_threshold = SKILL_THRESHOLD_BY_MODEL.get(model_name, SKILL_SEMANTIC_THRESHOLD)
                    for fname, idxs in cand_indices:
                        if not idxs:
                            semantic_skill_map[fname] = set()
                            continue
                        cand_embs  = all_cand_embs[idxs]
                        sim_matrix = np.dot(req_skill_embs, cand_embs.T)
                        matched    = {
                            req_skill
                            for i, req_skill in enumerate(required_skills)
                            if float(np.max(sim_matrix[i])) >= skill_threshold
                        }
                        semantic_skill_map[fname] = matched
                else:
                    for fname in extracted:
                        semantic_skill_map[fname] = set()

            except Exception as skil_exc:
                # Fall back to empty semantic skill matches; exact-match scoring still runs
                logger.warning("[%s] Semantic skill matching failed: %s — using exact-only", req_id, skil_exc)
                for fname in extracted:
                    semantic_skill_map[fname] = set()
        else:
            for fname in extracted:
                semantic_skill_map[fname] = set()

        logger.info("[%s] Step 3b done — %.2fs", req_id, time.time() - t_step)
        t_step = time.time()

        # Step 4: Score and rank
        logger.info("[%s] Step 4 — Scoring and ranking...", req_id)
        candidates = [
            {
                "filename":               fname,
                "info":                   info,
                "semantic_score":         similarity_map.get(fname, 0.0),
                "oom_fallback":           oom_fallback_map.get(fname, False),
                "semantic_skill_matches": semantic_skill_map.get(fname, set()),
                "resume_text":            resume_texts.get(fname, ""),
            }
            for fname, info in extracted.items()
        ]
        results    = score_candidates(candidates, required_skills, scoring_cfg, model_key=model_key)
        serialized = serialize_results(results, skills_configured=bool(required_skills))

        _write_results(_session_results_file(session_id), serialized)

        shortlisted = sum(1 for r in serialized if r["eligible"])
        elapsed_s   = (datetime.now(timezone.utc) - started_at).total_seconds()
        _inc("total_resumes_processed", len(serialized))

        logger.info(
            "[%s] Done — %d ranked | %d shortlisted | model: %s | %.1fs",
            req_id, len(serialized), shortlisted, model_key, elapsed_s,
        )

        # Record metrics (non-blocking, swallows errors)
        try:
            record_run = getattr(metrics_store, "record_run", None)
            if callable(record_run):
                record_run(
                    run_id=started_at.isoformat(),
                    model_key=model_key,
                    job_description=job_description or "",
                    similarity_results=similarity_results,
                    per_model_similarities=per_model_similarities or None,
                    final_scores=serialized,
                )
        except Exception as exc:
            logger.warning("[MetricsStore] Could not record run metrics: %s", exc)

        _write_audit({
            "timestamp":             started_at.isoformat(),
            "model":                 model_key,
            "total_resumes":         len(serialized),
            "shortlisted":           shortlisted,
            "rejected":              len(serialized) - shortlisted,
            "parse_failures":        len(parse_failures),
            "merged_count":          merged_count,
            "elapsed_seconds":       round(elapsed_s, 2),
            "job_description":       (job_description or "")[:MAX_AUDIT_JD_CHARS],
            "job_description_hash":  hashlib.md5((job_description or "").encode(), usedforsecurity=False).hexdigest()[:12],
            "required_skills":       required_skills,
            "skill_weight":          round(scoring_cfg.skill_weight, 2),
            "semantic_weight":       round(scoring_cfg.semantic_weight, 2),
            "semantic_fallback":     not semantic_ok,
        })

        return {
            "message":        "Screening complete.",
            "model_used":     model_key,
            "total":          len(serialized),
            "shortlisted":    shortlisted,
            "parse_failures": parse_failures,
            "merged_count":   merged_count,
            "results":        serialized,
        }

    except Exception as e:
        # Catch-all: pipeline never propagates; always returns a structured error dict
        _inc("total_screen_errors")
        _log_failure({
            "request_id": req_id,
            "session_id": session_id,
            "model":      model_key,
            "num_files":  len(pdf_files),
        }, e)
        logger.error("[%s] Pipeline failed: %s", req_id, e, exc_info=True)
        return {"error": str(e), "candidates": []}


# Routes


@app.route("/", methods=["GET"])
def index():
    """
    Root redirect — visiting the Flask backend URL in a browser sends the user
    straight to the React frontend.

    Target URL is read from the FRONTEND_URL env-var (default: http://localhost:5173)
    so the redirect works for both local development and production deployments.
    """
    return redirect(FRONTEND_URL, code=302)


# Allowed IPs for /api/health — empty means allow all (default for local dev)
_HEALTH_ALLOWED_IPS: set = {
    ip.strip()
    for ip in os.environ.get("HEALTH_ALLOWED_IPS", "").split(",")
    if ip.strip()
}


@app.route("/api/health", methods=["GET"])
def health():
    """Health check + live metrics. Not rate-limited — monitoring must always reach this."""
    if _HEALTH_ALLOWED_IPS:
        client_ip = request.remote_addr or ""
        if client_ip not in _HEALTH_ALLOWED_IPS:
            return jsonify({"status": "ok"}), 200  # minimal response for untrusted callers

    with _active_screenings_lock:
        active = _active_screenings
    with _pdf_count_lock:
        pdfs = _pdf_count

    return jsonify({
        "status":            "ok",
        "version":           __version__,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "available_models":  list(VALID_MODELS.keys()),
        "pdfs_uploaded":     pdfs,
        "active_sessions":   sum(1 for d in _safe_iterdir(UPLOAD_BASE) if d.is_dir()),
        "active_screenings": active,
        "pending_tasks":     sum(1 for t in _task_store.values() if t.get("status") == "pending"),
        "metrics":           _get_metrics(),
    }), 200


@app.route("/api/upload-resumes", methods=["POST"])
def upload_resumes():
    """Accept PDF uploads for a new session. Returns session_id used by /api/screen."""
    ip = _get_client_ip()
    if _is_rate_limited(ip):
        _inc("total_rate_limited")
        logger.warning("[RateLimit] Upload blocked for %s", ip)
        return jsonify({"error": "Too many requests. Please wait a minute.", "code": API_ERROR["rate_limited"]}), 429

    if not _disk_space_ok():
        logger.error("[Overload] Upload rejected — insufficient disk space.")
        return jsonify({"error": "Server storage is temporarily full. Try again later.", "code": API_ERROR["disk_full"]}), 503

    _inc("total_upload_requests")

    if "files" not in request.files:
        return jsonify({"error": "No files provided", "code": API_ERROR["no_files"]}), 400

    session_id     = uuid.uuid4().hex
    session_folder = UPLOAD_BASE / session_id
    session_folder.mkdir(parents=True, exist_ok=True)

    files = request.files.getlist("files")
    if len(files) > MAX_FILES_PER_SESSION:
        shutil.rmtree(session_folder, ignore_errors=True)
        return jsonify({"error": f"Too many files. Maximum {MAX_FILES_PER_SESSION} per session.", "code": API_ERROR["too_many_files"]}), 400

    saved, rejected, duplicates = [], [], []
    existing_hashes_map = _existing_hashes(session_folder)

    for file in files:
        if not file.filename:
            continue

        if not allowed_file(file.filename):
            rejected.append({"filename": file.filename, "reason": "Not a PDF file"})
            _inc("total_files_rejected")
            continue

        # Size check
        file.stream.seek(0, 2)
        size = file.stream.tell()
        file.stream.seek(0)
        if size > MAX_FILE_BYTES:
            mb = size / (1024 * 1024)
            rejected.append({"filename": file.filename, "reason": f"File too large ({mb:.1f} MB > 10 MB limit)"})
            _inc("total_files_rejected")
            logger.warning("[Upload] Rejected oversized file: %s (%.1f MB)", file.filename, mb)
            continue

        # Magic bytes check — confirms the file is actually a PDF
        file.stream.seek(0)
        header = file.stream.read(5)
        file.stream.seek(0)
        if header[:4] != b"%PDF":
            rejected.append({"filename": file.filename, "reason": "File is not a valid PDF"})
            _inc("total_files_rejected")
            continue

        # Deduplication by SHA-256
        digest = _file_sha256(file)
        if digest in existing_hashes_map:
            duplicates.append({"filename": file.filename, "matches": existing_hashes_map[digest]})
            logger.info("[Upload] Duplicate skipped: %s", file.filename)
            continue

        filename = secure_filename(file.filename)
        if not filename:
            rejected.append({"filename": file.filename, "reason": "Invalid filename after sanitization"})
            _inc("total_files_rejected")
            continue

        dest = session_folder / filename
        file.save(str(dest))
        existing_hashes_map[digest] = filename
        saved.append(filename)
        _pdf_count_add()
        _inc("total_files_saved")
        logger.info("[Upload] Saved to session %s: %s", session_id, filename)

    if not saved and not duplicates:
        shutil.rmtree(session_folder, ignore_errors=True)
        return jsonify({"error": "No valid PDF files uploaded", "rejected": rejected, "code": API_ERROR["no_valid_pdfs"]}), 400

    logger.info("[Upload] Session %s created with %d file(s).", session_id, len(saved))
    return jsonify({
        "message":    f"{len(saved)} resume(s) uploaded.",
        "session_id": session_id,
        "saved":      saved,
        "duplicates": duplicates,
        "rejected":   rejected,
    }), 200


@app.route("/api/screen", methods=["POST"])
def screen():
    """
    Async screening — returns task_id immediately (HTTP 202).
    Poll GET /api/result/<task_id> until status is 'done' or 'error'.
    Pass {"sync": true} in the body for synchronous mode (CLI/test use only).
    """
    global _active_screenings

    ip = _get_client_ip()
    if _is_rate_limited(ip):
        _inc("total_rate_limited")
        logger.warning("[RateLimit] Screen blocked for %s", ip)
        return jsonify({"error": "Too many requests. Please wait a minute.", "code": API_ERROR["rate_limited"]}), 429

    _inc("total_screen_requests")

    body = request.get_json(silent=True) or {}
    cfg  = load_config()

    session_id = body.get("session_id", "")
    if not session_id:
        return jsonify({"error": "Missing session_id. Upload resumes first.", "code": API_ERROR["invalid_session"]}), 400
    if not _validate_session_id(session_id):
        return jsonify({"error": "Invalid session_id format.", "code": API_ERROR["invalid_session"]}), 400

    session_folder = UPLOAD_BASE / session_id
    if not session_folder.is_dir():
        return jsonify({"error": f"Session '{session_id}' not found. Upload resumes first.", "code": API_ERROR["session_missing"]}), 404

    raw_jd = (body.get("job_description") or cfg.get("job_description", ""))[:MAX_JD_CHARS]
    job_description = _sanitize_text(raw_jd)
    if not job_description or len(job_description.strip()) < 20:
        return jsonify({"error": "Job description is required (minimum 20 characters).", "code": API_ERROR["jd_too_short"]}), 400

    required_skills = body.get("required_skills") or cfg.get("required_skills", [])
    if isinstance(required_skills, str):
        required_skills = [s.strip() for s in required_skills.split(",") if s.strip()]
    elif not isinstance(required_skills, list):
        return jsonify({"error": "required_skills must be a list of strings.", "code": API_ERROR["bad_skills"]}), 400

    # Normalize, deduplicate, and cap the skills list
    required_skills = list(dict.fromkeys(
        normalize_skill(s)[:100]
        for s in required_skills
        if isinstance(s, str) and s.strip()
    ))[:MAX_REQUIRED_SKILLS]

    model_key = body.get("model") or cfg.get("model", "mpnet")
    if model_key not in VALID_MODELS:
        return jsonify({"error": f"Unknown model '{model_key}'. Valid options: {list(VALID_MODELS)}", "code": API_ERROR["bad_model"]}), 400
    model_name = VALID_MODELS[model_key]

    cfg_override   = body.get("config", {})
    base_scoring   = cfg.get("scoring", DEFAULT_CONFIG["scoring"])
    scoring_params = {k: v for k, v in {**base_scoring, **cfg_override}.items() if k in _SCORING_FIELDS}
    try:
        scoring_cfg = ScoringConfig(**scoring_params)
    except Exception as e:
        return jsonify({"error": f"Invalid scoring config: {e}", "code": API_ERROR["bad_config"]}), 400

    pdf_files = list(session_folder.glob("*.pdf"))
    if not pdf_files:
        return jsonify({"error": "No resumes found in session. Upload PDFs first.", "code": API_ERROR["no_resumes"]}), 400

    # Concurrency guard — increment before submitting to prevent burst overload
    with _active_screenings_lock:
        if _active_screenings >= MAX_CONCURRENT_SCREENINGS:
            _inc("total_overload_rejected")
            logger.warning(
                "[Overload] Screen rejected — %d/%d slots in use.",
                _active_screenings, MAX_CONCURRENT_SCREENINGS,
            )
            return jsonify({"error": "Server is busy. Please try again shortly.", "code": API_ERROR["overloaded"]}), 503
        _active_screenings += 1

    started_at = datetime.now(timezone.utc)
    req_id     = uuid.uuid4().hex[:8]
    task_id    = uuid.uuid4().hex

    # Register task before submitting — prevents polling races before the worker sets "running"
    _task_set(task_id, {
        "status":     "pending",
        "session_id": session_id,
        "created_at": time.time(),
        "req_id":     req_id,
    })

    def _worker():
        try:
            _task_update(task_id, {"status": "running"})
            result = _run_screening_pipeline(
                req_id=req_id,
                session_id=session_id,
                session_folder=session_folder,
                pdf_files=pdf_files,
                job_description=job_description,
                required_skills=required_skills,
                model_key=model_key,
                model_name=model_name,
                scoring_cfg=scoring_cfg,
                started_at=started_at,
            )
            _task_update(task_id, {
                "status":      "error" if "error" in result else "done",
                "result":      result,
                "finished_at": time.time(),
            })
        finally:
            # Always decrement, even on unexpected exception
            with _active_screenings_lock:
                global _active_screenings
                _active_screenings = max(0, _active_screenings - 1)

    # Synchronous mode ({"sync": true}) — blocks the request thread; useful for CLI/tests
    if body.get("sync", False):
        _worker()
        task = _task_get(task_id)
        if task is None:
            logger.error("[%s] Task %s missing after synchronous execution.", req_id, task_id)
            return jsonify({"error": "Task result unavailable."}), 500
        result = task.get("result", {})
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result), 200

    # Async mode — submit to ThreadPoolExecutor
    # Wrap in try/except: if submit() itself raises (e.g. executor is shut down),
    # _worker's finally block never runs and _active_screenings permanently leaks.
    try:
        _screening_executor.submit(_worker)
    except Exception as submit_exc:
        with _active_screenings_lock:
            _active_screenings = max(0, _active_screenings - 1)
        _task_update(task_id, {
            "status": "error",
            "result": {"error": f"Failed to queue screening task: {submit_exc}", "candidates": []},
            "finished_at": time.time(),
        })
        logger.error("[%s] Executor submit failed: %s", req_id, submit_exc)
        return jsonify({"error": "Server is busy or shutting down. Please retry."}), 503

    logger.info("[%s] Task %s submitted (session %s, model %s)", req_id, task_id, session_id, model_key)

    return jsonify({
        "task_id":  task_id,
        "status":   "pending",
        "poll_url": f"/api/result/{task_id}",
        "message":  f"Screening started. Poll {task_id} for results.",
    }), 202


@app.route("/api/result/<task_id>", methods=["GET"])
def get_result(task_id: str):
    """Poll endpoint for async screening results. Returns 202 while pending/running, 200 when done."""
    if not re.fullmatch(r"[a-f0-9]{32}", task_id):
        return jsonify({"error": "Invalid task_id format.", "code": API_ERROR["bad_task_id"]}), 400

    task = _task_get(task_id)
    if task is None:
        return jsonify({"error": "Task not found. It may have expired (TTL 1hr).", "code": API_ERROR["task_expired"]}), 404

    status = task.get("status", "unknown")
    if status in ("pending", "running"):
        return jsonify({"status": status, "task_id": task_id}), 202

    result = task.get("result", {})
    if status == "error" or ("error" in result and not result.get("results")):
        return jsonify({
            "status":     "error",
            "error":      result.get("error", "Unknown error"),
            "candidates": [],
        }), 500

    return jsonify({"status": "done", "task_id": task_id, **result}), 200


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(load_config()), 200


@app.route("/api/config", methods=["POST"])
def post_config():
    """Validate and persist config updates. Only ALLOWED_CONFIG_KEYS are accepted."""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "No JSON body provided"}), 400
    try:
        body_size = len(json.dumps(body))
    except (TypeError, ValueError):
        return jsonify({"error": "Config payload contains non-serializable data."}), 400
    if body_size > MAX_CONFIG_BYTES:
        return jsonify({"error": "Config payload too large."}), 413

    sanitized = {k: v for k, v in body.items() if k in ALLOWED_CONFIG_KEYS}
    if not sanitized:
        return jsonify({"error": "No valid config keys provided."}), 400

    if "job_description" in sanitized and not isinstance(sanitized["job_description"], str):
        return jsonify({"error": "job_description must be a string."}), 400

    if "required_skills" in sanitized:
        if not isinstance(sanitized["required_skills"], list):
            return jsonify({"error": "required_skills must be a list."}), 400
        sanitized["required_skills"] = [
            normalize_skill(s)[:100]
            for s in sanitized["required_skills"]
            if isinstance(s, str) and s.strip()
        ]

    existing = load_config()
    existing.update(sanitized)
    save_config(existing)
    logger.info("[Config] config.json updated (keys: %s).", list(sanitized.keys()))
    return jsonify({"message": "config.json saved."}), 200


@app.route("/api/model", methods=["POST"])
def set_model():
    """Validate and confirm a model selection (does not persist to config)."""
    body      = request.get_json(silent=True) or {}
    model_key = body.get("model", "")
    if model_key not in VALID_MODELS:
        return jsonify({"error": f"Unknown model '{model_key}'. Valid: {list(VALID_MODELS)}"}), 400
    label = VALID_MODELS[model_key]
    logger.info("[Model] Client switched to %s", label)
    return jsonify({"message": f"Switched to {label}", "active": model_key, "full_name": label}), 200


@app.route("/api/clear", methods=["POST"])
def clear_uploads():
    """Delete all uploads and results for a given session_id."""
    body       = request.get_json(silent=True) or {}
    session_id = body.get("session_id", "")

    if not session_id:
        return jsonify({"error": "No session_id provided."}), 400
    if not _validate_session_id(session_id):
        return jsonify({"error": "Invalid session_id format."}), 400

    session_folder = UPLOAD_BASE / session_id
    if session_folder.is_dir():
        pdf_count = sum(1 for _ in session_folder.glob("*.pdf"))
        shutil.rmtree(session_folder, ignore_errors=True)
        _pdf_count_sub(pdf_count)
        logger.info("[Clear] Session uploads cleared: %s", session_id)

    results_file = _session_results_file(session_id)
    if results_file.exists():
        results_file.unlink(missing_ok=True)  # FIX: race with cleanup_worker
        logger.info("[Clear] Session results cleared: %s", session_id)

    return jsonify({"message": "Session cleared."}), 200


# ── Global error handlers — return JSON for all error types ────────────────

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Endpoint not found", "code": API_ERROR["not_found"]}), 404


@app.errorhandler(500)
def handle_500(e):
    logger.error("[ErrorHandler] 500 Internal Server Error: %s", e)
    return jsonify({"error": "Internal server error", "code": API_ERROR["internal_error"]}), 500


@app.errorhandler(413)
def handle_413(e):
    return jsonify({"error": "Request payload too large", "code": API_ERROR["payload_too_large"]}), 413


# ── GET /api/stats — aggregate historical screening statistics ─────────────

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return aggregate screening statistics from metrics_log.jsonl."""
    try:
        load_all_runs = getattr(metrics_store, "load_all_runs", None)
        if not callable(load_all_runs):
            return jsonify({"error": "metrics_store.load_all_runs not available"}), 501

        runs = load_all_runs()
        if not runs:
            return jsonify({
                "total_runs": 0,
                "total_resumes_screened": 0,
                "models_used": {},
                "avg_score_by_model": {},
            }), 200

        total_runs = len(runs)
        total_resumes = 0
        models_used = {}
        score_sums = {}
        score_counts = {}

        for run in runs:
            model = run.get("model_key", "unknown")
            models_used[model] = models_used.get(model, 0) + 1

            finals = run.get("final_scores", [])
            n = len(finals) if isinstance(finals, list) else 0
            total_resumes += n

            if isinstance(finals, list):
                for c in finals:
                    fs = c.get("finalScore", 0) if isinstance(c, dict) else 0
                    score_sums[model] = score_sums.get(model, 0) + fs
                    score_counts[model] = score_counts.get(model, 0) + 1

        avg_by_model = {
            m: round(score_sums.get(m, 0) / score_counts[m] * 100)
            for m in score_counts if score_counts[m] > 0
        }

        return jsonify({
            "total_runs":             total_runs,
            "total_resumes_screened": total_resumes,
            "models_used":           models_used,
            "avg_score_by_model":    avg_by_model,
        }), 200

    except Exception as exc:
        logger.error("[Stats] Failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ── GET /api/result/<task_id>/export — CSV export of screening results ──────

@app.route("/api/result/<task_id>/export", methods=["GET"])
def export_result(task_id: str):
    """Download screening results as a CSV file."""
    if not re.fullmatch(r"[a-f0-9]{32}", task_id):
        return jsonify({"error": "Invalid task_id format.", "code": API_ERROR["bad_task_id"]}), 400

    task = _task_get(task_id)
    if task is None:
        return jsonify({"error": "Task not found.", "code": API_ERROR["task_expired"]}), 404

    result = task.get("result", {})
    candidates = result.get("results", [])
    if not candidates:
        return jsonify({"error": "No results available for export."}), 404

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rank", "Name", "Email", "Phone", "Final Score", "Skill Score",
                     "Semantic Score", "Status", "Band", "LinkedIn", "GitHub"])

    for c in candidates:
        links = c.get("links", {})
        writer.writerow([
            c.get("rank", ""),
            c.get("name", "Unknown"),
            c.get("email", ""),
            c.get("phone", ""),
            f"{round((c.get('finalScore', 0)) * 100)}%",
            f"{round((c.get('skillScore', 0)) * 100)}%",
            f"{round((c.get('semanticScore', 0)) * 100)}%",
            c.get("status", ""),
            c.get("band", ""),
            links.get("linkedin", ""),
            links.get("github", ""),
        ])

    csv_content = output.getvalue()
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=screening_{task_id[:8]}.csv"},
    )


# ── POST /api/validate-jd — job description quality scorer ─────────────────

@app.route("/api/validate-jd", methods=["POST"])
def validate_jd():
    """Score a job description for quality and provide improvement suggestions."""
    body = request.get_json(silent=True) or {}
    jd = _sanitize_text(body.get("job_description", ""))

    if not jd:
        return jsonify({"quality_score": 0, "suggestions": ["Provide a job description."]}), 200

    suggestions = []
    score = 0.0

    # Length scoring
    word_count = len(jd.split())
    if word_count >= 80:
        score += 0.25
    elif word_count >= 40:
        score += 0.15
        suggestions.append("Add more detail — aim for at least 80 words for best results.")
    else:
        score += 0.05
        suggestions.append("Job description is very short. Add responsibilities and requirements.")

    # Keyword diversity
    jd_lower = jd.lower()
    role_keywords = [
        "responsible", "requirements", "qualifications", "experience",
        "proficiency", "skills", "ability", "knowledge", "team",
        "develop", "design", "manage", "implement", "analyze",
    ]
    matched_keywords = sum(1 for kw in role_keywords if kw in jd_lower)
    kw_ratio = matched_keywords / len(role_keywords)
    score += kw_ratio * 0.30
    if matched_keywords < 3:
        suggestions.append("Include more role-specific language (e.g. responsibilities, qualifications, requirements).")

    # Section detection
    sections = ["responsibilities", "requirements", "qualifications", "experience", "education", "skills"]
    found_sections = sum(1 for s in sections if s in jd_lower)
    if found_sections >= 2:
        score += 0.20
    else:
        suggestions.append("Structure your JD with clear sections (e.g. Responsibilities, Requirements).")
    
    # Specificity — presence of technical terms or tools
    has_technical = bool(re.search(
        r"\b(python|java|sql|aws|react|node|docker|kubernetes|machine learning|api|database|cloud|git|agile)\b",
        jd_lower,
    ))
    if has_technical:
        score += 0.15
    else:
        suggestions.append("Mention specific technologies, tools, or methodologies for better matching.")

    # Bonus for mentioning experience level
    if re.search(r"\d+\s*\+?\s*years?", jd_lower):
        score += 0.10
    else:
        suggestions.append("Specify required years of experience for more accurate screening.")

    quality_score = round(min(score, 1.0), 2)

    return jsonify({
        "quality_score": quality_score,
        "word_count":    word_count,
        "suggestions":   suggestions if quality_score < 0.85 else [],
        "rating":        "Excellent" if quality_score >= 0.80 else "Good" if quality_score >= 0.60 else "Needs Improvement",
    }), 200


# Entry point — _startup_init() already ran at module level above

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 58)
    print(f"  ML-Based Resume Screening System v{__version__} — Flask API")
    print(f"  Running on port {port}")
    print(f"  Default model   : {VALID_MODELS['mpnet']}")
    print(f"  Available models: {', '.join(VALID_MODELS)}")
    print("=" * 58)
    app.run(host="0.0.0.0", port=port)
