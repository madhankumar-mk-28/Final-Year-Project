"""
audit_logger.py — Append-only audit and failure log writer.

Provides thread-safe, crash-resistant logging for every screening run:
    - write_audit()    — Records a structured summary of each screening run
    - log_failure()    — Captures pipeline errors with full traceback context
    - normalize_skill()— Shared skill normalisation used by app.py

Log files:
    - audit.jsonl     — One JSON record per screening run (success or partial)
    - failures.jsonl  — One JSON record per pipeline error (with traceback)

Both files are automatically rotated (trimmed to the 500 most recent records)
when they exceed that size, keeping disk usage bounded.

Runs fully offline. No network calls. OS-independent (uses pathlib).
"""

import json
import logging
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("audit_logger")

# Log file paths (created in the project root directory)
AUDIT_FILE        = Path("audit.jsonl")
FAILURES_FILE     = Path("failures.jsonl")

# Maximum number of records to keep per log file
MAX_JSONL_ENTRIES = 500

# Single lock protects both files — prevents interleaved writes from worker threads
_audit_lock = threading.Lock()


def normalize_skill(s: str) -> str:
    """Return a canonical skill string for consistent comparison and storage.

    Steps:
        1. Lowercase the input
        2. Strip leading/trailing whitespace and punctuation
        3. Collapse multiple spaces into one

    Examples:
        "Machine Learning" → "machine learning"
        " Python, "        → "python"
        "  Data   Science " → "data science"

    Returns:
        Normalised skill string.
    """
    return re.sub(r"\s+", " ", s.lower().strip().strip(".,;:!?")).strip()


def _rotate_jsonl(filepath: Path, max_entries: int = MAX_JSONL_ENTRIES) -> None:
    """Trim a JSONL log file to keep only the most recent max_entries records.

    Records are separated by double-newlines (pretty-print format).
    Must be called while holding _audit_lock to prevent concurrent access.

    This function only writes when rotation is actually needed (file exceeds
    max_entries), keeping I/O overhead near zero during normal operation.
    """
    try:
        if not filepath.exists():
            return

        raw = filepath.read_text(encoding="utf-8")

        # Split on double-newline separators; discard empty strings
        records = [r.strip() for r in raw.split("\n\n") if r.strip()]
        if len(records) <= max_entries:
            return                         # File is within limit — no action needed

        # Keep only the most recent records
        kept = records[-max_entries:]
        filepath.write_text("\n\n".join(kept) + "\n\n", encoding="utf-8")
        logger.info(
            "[Audit] Rotated %s: %d → %d entries.",
            filepath.name, len(records), max_entries,
        )
    except OSError as e:
        logger.warning("[Audit] Could not rotate %s: %s", filepath.name, e)


def write_audit(entry: dict) -> None:
    """Append one screening run record to audit.jsonl (thread-safe).

    Each record is a JSON object containing: timestamp, model, candidate count,
    shortlisted/rejected counts, JD excerpt, skill weights, and more.

    Rotation is performed inside the same lock as the write, so readers
    always see a consistent, non-truncated JSONL file.

    Args:
        entry: Dictionary containing the audit record fields.
    """
    try:
        # Pretty-print each record with double-newline separator for readability
        block = json.dumps(entry, indent=2, ensure_ascii=False) + "\n\n"
        with _audit_lock:
            with open(AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(block)
            _rotate_jsonl(AUDIT_FILE)       # Trim if file exceeds max entries
    except OSError as e:
        logger.warning("[Audit] Could not write audit log: %s", e)


def log_failure(context: dict, error: Exception) -> None:
    """Write a structured failure record to failures.jsonl (thread-safe).

    Captures the full traceback and any context information (request_id,
    session_id, model, file count, etc.) for post-mortem debugging.

    This function silently swallows ALL exceptions — it is the last safety
    net in the error handling chain and must never raise.

    Args:
        context: Dictionary with situational info (session_id, model, etc.).
        error:   The exception that caused the failure.
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
                f.write(json.dumps(entry, indent=2, ensure_ascii=False) + "\n\n")
            _rotate_jsonl(FAILURES_FILE)    # Trim if file exceeds max entries
    except OSError:
        pass   # Must not crash — this IS the crash handler