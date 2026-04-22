"""
cleanup_cache.py — Post-screening cleanup utility for the ML Resume Screening System.

Removes temporary files generated during screening sessions without disturbing
the running Flask server.  Safe to run at any time.

What it cleans:
    1. Python bytecode  (__pycache__ directories and .pyc files)
    2. Uploaded PDFs    (uploads/<session_id>/*.pdf)
    3. Session results  (results/<session_id>.json)
    4. Metrics logs     (metrics_log.jsonl, embeddings_store.json) — opt-in via --metrics

Usage:
    python cleanup_cache.py                  — Full clean (cache + uploads + results)
    python cleanup_cache.py --keep-pdfs      — Python cache only
    python cleanup_cache.py --metrics        — Also wipe metrics_log.jsonl
    python cleanup_cache.py --session <id>   — Clean one specific session only
    python cleanup_cache.py --dry-run        — Preview what would be deleted (no changes)

Runs fully offline.  No network calls.  OS-independent (uses pathlib / os).
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path


# ─── ANSI colour helpers ─────────────────────────────────────────────────────
# These fall back to empty strings on terminals that don't support ANSI codes
# (e.g. Windows CMD without VT processing).

def _ansi(code: str) -> str:
    """Return an ANSI escape sequence, or '' if stdout is not a tty."""
    return f"\033[{code}m" if sys.stdout.isatty() else ""

_RESET  = _ansi("0")
_BOLD   = _ansi("1")
_GREEN  = _ansi("32")
_YELLOW = _ansi("33")
_RED    = _ansi("31")
_CYAN   = _ansi("36")
_DIM    = _ansi("2")
_BLUE   = _ansi("34")


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (KB / MB)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _print_header(title: str) -> None:
    """Print a styled section header."""
    line = "─" * 52
    print(f"\n{_BOLD}{_CYAN}{line}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{line}{_RESET}")


def _print_step(label: str) -> None:
    """Print a step label before listing its items."""
    print(f"\n{_BOLD}{_BLUE}→ {label}{_RESET}")


def _print_item(tag: str, path: str, size_bytes: int = 0, dry: bool = False) -> None:
    """Print a single file/directory action line."""
    prefix = f"{_YELLOW}[dry-run]{_RESET} " if dry else f"{_GREEN}[removed]{_RESET} "
    size   = f" {_DIM}({_fmt_bytes(size_bytes)}){_RESET}" if size_bytes else ""
    print(f"  {prefix}{_DIM}{tag:<12}{_RESET} {path}{size}")


def _print_error(path: str, exc: Exception) -> None:
    print(f"  {_RED}[error]   {_RESET} {path}: {exc}")


# ─── Cleanup functions ────────────────────────────────────────────────────────

def cleanup_python_cache(root_dir: str = ".", dry: bool = False) -> tuple[int, int, int]:
    """Recursively remove __pycache__ directories and .pyc files.

    Processing order:
        1. Walk directory tree top-down
        2. Remove __pycache__/ directories (entire subtree)
        3. Remove individual .pyc files found elsewhere

    Returns:
        Tuple of (dirs_removed, files_removed, bytes_freed).
    """
    removed_dirs  = 0
    removed_files = 0
    bytes_freed   = 0

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # Skip hidden dirs and the virtualenv to avoid unnecessary scanning
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "node_modules"]

        if "__pycache__" in dirnames:
            cache_path = Path(dirpath) / "__pycache__"
            # Measure size before deleting so we can report bytes freed
            try:
                size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            except OSError:
                size = 0
            if not dry:
                shutil.rmtree(cache_path, ignore_errors=True)
            _print_item("cache dir", str(cache_path), size, dry)
            removed_dirs += 1
            bytes_freed  += size
            dirnames.remove("__pycache__")     # Don't descend into (now-deleted) dir

        for filename in filenames:
            if filename.endswith(".pyc"):
                file_path = Path(dirpath) / filename
                try:
                    size = file_path.stat().st_size
                    if not dry:
                        file_path.unlink()
                    _print_item(".pyc file", str(file_path), size, dry)
                    removed_files += 1
                    bytes_freed   += size
                except OSError as e:
                    _print_error(str(file_path), e)

    return removed_dirs, removed_files, bytes_freed


def cleanup_uploads(uploads_dir: str = "uploads",
                    session_id: str | None = None,
                    dry: bool = False) -> tuple[int, int]:
    """Delete PDF files from uploads/<session_id>/ directories.

    When session_id is provided, only that session folder is cleaned.
    Otherwise all sessions are processed.

    Also removes empty session directories after PDF deletion.

    Returns:
        Tuple of (pdfs_removed, bytes_freed).
    """
    uploads_path = Path(uploads_dir)
    if not uploads_path.is_dir():
        print(f"  {_DIM}uploads/ directory not found — skipping.{_RESET}")
        return 0, 0

    removed    = 0
    bytes_freed = 0

    # Determine which session folders to scan
    if session_id:
        target_dirs = [uploads_path / session_id]
    else:
        target_dirs = [d for d in uploads_path.iterdir() if d.is_dir()]

    for session_dir in target_dirs:
        if not session_dir.is_dir():
            print(f"  {_RED}Session '{session_id}' not found in uploads/.{_RESET}")
            continue
        for pdf_path in session_dir.rglob("*.pdf"):
            try:
                size = pdf_path.stat().st_size
                if not dry:
                    pdf_path.unlink()
                _print_item("upload pdf", str(pdf_path), size, dry)
                removed     += 1
                bytes_freed += size
            except OSError as e:
                _print_error(str(pdf_path), e)

    # Remove empty session directories after PDF deletion
    for session_dir in (target_dirs if session_id else list(uploads_path.iterdir())):
        if not isinstance(session_dir, Path) or not session_dir.is_dir():
            continue
        try:
            if not dry:
                session_dir.rmdir()          # Only succeeds when directory is empty
            _print_item("session dir", str(session_dir), dry=dry)
        except OSError:
            pass                             # Non-empty session — leave it

    return removed, bytes_freed


def cleanup_results(results_dir: str = "results",
                    session_id: str | None = None,
                    dry: bool = False) -> tuple[int, int]:
    """Delete JSON result files from the results/ directory.

    When session_id is provided, only that session's result file is removed.

    Returns:
        Tuple of (files_removed, bytes_freed).
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"  {_DIM}results/ directory not found — skipping.{_RESET}")
        return 0, 0

    removed    = 0
    bytes_freed = 0

    if session_id:
        targets = [results_path / f"{session_id}.json"]
    else:
        targets = list(results_path.glob("*.json"))

    for json_file in targets:
        if not json_file.exists():
            if session_id:
                print(f"  {_DIM}No result file found for session '{session_id}'.{_RESET}")
            continue
        try:
            size = json_file.stat().st_size
            if not dry:
                json_file.unlink()
            _print_item("result json", str(json_file), size, dry)
            removed     += 1
            bytes_freed += size
        except OSError as e:
            _print_error(str(json_file), e)

    return removed, bytes_freed


def cleanup_metrics(dry: bool = False) -> tuple[int, int]:
    """Delete metrics_log.jsonl and embeddings_store.json.

    This resets the historical screening statistics visible in the Dashboard.
    Use only when starting a fresh demo or presentation session.

    Returns:
        Tuple of (files_removed, bytes_freed).
    """
    removed    = 0
    bytes_freed = 0

    targets = [
        Path("metrics_log.jsonl"),
        Path("embeddings_store.json"),
        Path("audit.jsonl"),
    ]

    for f in targets:
        if f.exists():
            try:
                size = f.stat().st_size
                if not dry:
                    f.unlink()
                _print_item("metrics", str(f), size, dry)
                removed     += 1
                bytes_freed += size
            except OSError as e:
                _print_error(str(f), e)
        else:
            print(f"  {_DIM}{f} — not found{_RESET}")

    return removed, bytes_freed


# ─── Main orchestrator ────────────────────────────────────────────────────────

def run_cleanup(
    keep_pdfs:  bool = False,
    with_metrics: bool = False,
    session_id: str | None = None,
    dry:        bool = False,
    root_dir:   str  = ".",
) -> None:
    """Run the full cleanup sequence and print a formatted summary.

    Sequence:
        1. Python bytecode cache  (always)
        2. Uploaded PDFs          (unless --keep-pdfs)
        3. Session results        (unless --keep-pdfs)
        4. Metrics + audit logs   (only if --metrics)

    Args:
        keep_pdfs:    If True, skip upload and result cleanup.
        with_metrics: If True, also wipe metrics_log.jsonl and audit.jsonl.
        session_id:   If set, restrict upload/result cleanup to this session.
        dry:          If True, print what would be deleted without deleting.
        root_dir:     Root directory to scan for Python cache files.
    """
    _print_header(
        f"ML Resume Screening — {'DRY RUN ' if dry else ''}Post-Session Cleanup"
    )
    if dry:
        print(f"\n  {_YELLOW}⚠  Dry-run mode — no files will be deleted.{_RESET}")
    if session_id:
        print(f"\n  {_CYAN}ℹ  Targeting session: {session_id}{_RESET}")

    total_files = 0
    total_bytes = 0

    # ── Step 1: Python cache ─────────────────────────────────────────────────
    _print_step("Scanning Python bytecode cache…")
    dirs_rm, files_rm, b = cleanup_python_cache(root_dir, dry=dry)
    total_files += files_rm
    total_bytes += b
    verb = "Would remove" if dry else "Removed"
    print(f"\n  {verb} {_BOLD}{dirs_rm}{_RESET} __pycache__ folder(s), "
          f"{_BOLD}{files_rm}{_RESET} .pyc file(s) — {_fmt_bytes(b)} freed")

    # ── Step 2 & 3: Uploads and results ─────────────────────────────────────
    if not keep_pdfs:
        _print_step("Clearing uploaded resumes from uploads/…")
        pdfs_rm, b = cleanup_uploads("uploads", session_id=session_id, dry=dry)
        total_files += pdfs_rm
        total_bytes += b
        print(f"\n  {verb} {_BOLD}{pdfs_rm}{_RESET} uploaded PDF file(s) — {_fmt_bytes(b)} freed")

        _print_step("Clearing session results from results/…")
        res_rm, b = cleanup_results("results", session_id=session_id, dry=dry)
        total_files += res_rm
        total_bytes += b
        print(f"\n  {verb} {_BOLD}{res_rm}{_RESET} result file(s) — {_fmt_bytes(b)} freed")
    else:
        print(f"\n  {_DIM}→ Skipping uploads and results (--keep-pdfs).{_RESET}")

    # ── Step 4: Metrics logs (opt-in) ────────────────────────────────────────
    if with_metrics:
        _print_step("Wiping metrics and audit logs…")
        met_rm, b = cleanup_metrics(dry=dry)
        total_files += met_rm
        total_bytes += b
        print(f"\n  {verb} {_BOLD}{met_rm}{_RESET} metrics/audit file(s) — {_fmt_bytes(b)} freed")
    else:
        print(f"\n  {_DIM}→ Metrics logs kept (pass --metrics to wipe them).{_RESET}")

    # ── Summary ──────────────────────────────────────────────────────────────
    line = "─" * 52
    print(f"\n{_BOLD}{_GREEN}{line}{_RESET}")
    if dry:
        print(f"{_BOLD}{_YELLOW}  DRY RUN complete — {total_files} file(s) / {_fmt_bytes(total_bytes)} "
              f"would be freed.{_RESET}")
    else:
        print(f"{_BOLD}{_GREEN}  ✓ Cleanup complete — {total_files} file(s) removed, "
              f"{_fmt_bytes(total_bytes)} freed.{_RESET}")
    print(f"{_BOLD}{_GREEN}    Project ready for the next screening session.{_RESET}")
    print(f"{_BOLD}{_GREEN}{line}{_RESET}\n")


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse simple flags from sys.argv (no argparse dependency to keep the
    # script fully self-contained and importable without side-effects).
    argv        = sys.argv[1:]
    keep_pdfs   = "--keep-pdfs"  in argv
    with_metrics = "--metrics"   in argv
    dry         = "--dry-run"    in argv

    # --session <id>: restrict to a single session
    session_id: str | None = None
    if "--session" in argv:
        idx = argv.index("--session")
        if idx + 1 < len(argv):
            session_id = argv[idx + 1]
        else:
            print(f"{_RED}Error: --session requires a session ID argument.{_RESET}")
            sys.exit(1)

    # --help
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        sys.exit(0)

    run_cleanup(
        keep_pdfs=keep_pdfs,
        with_metrics=with_metrics,
        session_id=session_id,
        dry=dry,
    )