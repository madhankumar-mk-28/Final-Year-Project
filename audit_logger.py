"""audit_logger.py — Append-only audit and failure log writer.

write_audit()  — called after every screening run to log a structured summary.
log_failure()  — called on pipeline errors to capture traceback context.
normalize_skill() — shared skill normalisation used by app.py.

Both JSONL files are rotated (trimmed to MAX_JSONL_ENTRIES most-recent lines)
only when they actually exceed that size, keeping I/O overhead near zero.
"""

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
    """Return a canonical skill string: lowercased, punctuation-stripped, whitespace-collapsed."""
    return re.sub(r"\s+", " ", s.lower().strip().strip(".,;:!?")).strip()


def _rotate_jsonl(filepath: Path, max_entries: int = MAX_JSONL_ENTRIES) -> None:
    """Trim filepath to the most-recent max_entries lines.

    Only performs I/O when the file actually needs trimming — skips files
    with fewer than max_entries lines.  Must be called while holding
    _audit_lock so concurrent writers cannot interleave with the trim.
    """
    try:
        if not filepath.exists():
            return

        lines = filepath.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) <= max_entries:
            return     # file is within limit — nothing to do

        filepath.write_text("\n".join(lines[-max_entries:]) + "\n", encoding="utf-8")
        logger.info(
            "[Audit] Rotated %s: %d → %d entries.",
            filepath.name, len(lines), max_entries,
        )
    except OSError as e:
        logger.warning("[Audit] Could not rotate %s: %s", filepath.name, e)


def write_audit(entry: dict) -> None:
    """Append one screening run record to audit.jsonl (thread-safe).

    Rotation is performed inside the same lock as the write, ensuring readers
    always see a consistent JSONL file (no partial trims interleaved with writes).
    """
    try:
        line = json.dumps(entry) + "\n"
        with _audit_lock:
            with open(AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(line)
            _rotate_jsonl(AUDIT_FILE)
    except OSError as e:
        logger.warning("[Audit] Could not write audit log: %s", e)


def log_failure(context: dict, error: Exception) -> None:
    """Write a structured failure record to failures.jsonl (thread-safe).

    Silently swallows all exceptions — this function is the last safety net
    and must not raise.
    """
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
        pass   # must not crash the crash handler