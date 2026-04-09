import json
import logging
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("audit_logger")

AUDIT_FILE        = Path("audit.jsonl")
FAILURES_FILE     = Path("failures.jsonl")
MAX_JSONL_ENTRIES = 500

_audit_lock = threading.Lock()


def normalize_skill(s: str) -> str:
    """Lowercase, strip trailing/leading punctuation, and collapse whitespace — single canonical form."""
    return re.sub(r"\s+", " ", s.lower().strip().strip(".,;:!?")).strip()


def _rotate_jsonl(filepath: Path, max_entries: int = MAX_JSONL_ENTRIES):
    """Keep only the latest max_entries lines in a JSONL file — must be called under _audit_lock."""
    try:
        if not filepath.exists():
            return
        lines = filepath.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) > max_entries:
            filepath.write_text("\n".join(lines[-max_entries:]) + "\n", encoding="utf-8")
            logger.info("[Rotate] Trimmed %s from %d to %d entries.", filepath.name, len(lines), max_entries)
    except OSError as e:
        logger.warning("[Rotate] Could not rotate %s: %s", filepath.name, e)


def write_audit(entry: dict):
    """Append a single screening audit record to audit.jsonl (thread-safe)."""
    try:
        line = json.dumps(entry) + "\n"
        with _audit_lock:
            with open(AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(line)
            _rotate_jsonl(AUDIT_FILE)
    except OSError as e:
        logger.warning("[Audit] Could not write audit log: %s", e)


def log_failure(context: dict, error: Exception):
    """Write a structured failure snapshot to failures.jsonl (thread-safe)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error":     str(error),
        "traceback": traceback.format_exc(),
        "context":   context,
    }
    try:
        with _audit_lock:
            with open(FAILURES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            _rotate_jsonl(FAILURES_FILE)
    except OSError:
        pass  # last-resort — must not crash the crash handler