"""
cleanup_cache.py — Post-screening cleanup utility.

Removes temporary files generated during screening sessions:
    1. Python cache files (__pycache__ directories and .pyc files)
    2. Uploaded PDF resumes from uploads/<session_id>/ directories
    3. JSON result files from results/ directory

Usage:
    python cleanup_cache.py              — Clean everything (cache + uploads + results)
    python cleanup_cache.py --keep-pdfs  — Clean Python cache only (keep uploads and results)

This is a standalone script — safe to run at any time, even while the server
is running (it won't delete files that are currently being processed).

Runs fully offline. No network calls. OS-independent.
"""

import os
import sys
import shutil
from pathlib import Path


def cleanup_python_cache(root_dir: str = ".") -> tuple[int, int]:
    """Recursively remove all __pycache__ directories and .pyc files.

    Walks the directory tree starting from root_dir and removes:
        - __pycache__/ directories (and all contents)
        - Individual .pyc files found anywhere in the tree

    Returns:
        Tuple of (directories_removed, files_removed).
    """
    removed_dirs  = 0
    removed_files = 0

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # Remove __pycache__ directories
        if "__pycache__" in dirnames:
            cache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(cache_path, ignore_errors=True)
            print(f"  [cache dir]  {cache_path}")
            removed_dirs += 1
            dirnames.remove("__pycache__")     # Don't descend into removed dir

        # Remove individual .pyc files
        for filename in filenames:
            if filename.endswith(".pyc"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"  [.pyc file]  {file_path}")
                    removed_files += 1
                except OSError as e:
                    print(f"  [error]      Could not remove {file_path}: {e}")

    return removed_dirs, removed_files


def cleanup_uploads(uploads_dir: str = "uploads") -> int:
    """Delete all PDF files from uploads/<session_id>/ directories.

    Also removes any empty session directories after PDF deletion.
    Non-empty directories (e.g. with non-PDF files) are left intact.

    Returns:
        Number of PDF files removed.
    """
    uploads_path = Path(uploads_dir)
    if not uploads_path.is_dir():
        return 0

    removed = 0

    # Delete all PDF files recursively
    for pdf_path in uploads_path.rglob("*.pdf"):
        try:
            pdf_path.unlink()
            print(f"  [upload pdf]  {pdf_path}")
            removed += 1
        except OSError as e:
            print(f"  [error]       Could not remove {pdf_path}: {e}")

    # Remove empty session directories
    for entry in uploads_path.iterdir():
        if not entry.is_dir():
            continue                               # Skip stray files (e.g. .DS_Store)
        try:
            entry.rmdir()                          # Only succeeds when directory is empty
            print(f"  [session dir] removed empty {entry}")
        except OSError:
            pass                                   # Non-empty session — leave it

    return removed


def cleanup_results(results_dir: str = "results") -> int:
    """Delete all JSON result files from the results/ directory.

    Returns:
        Number of result files removed.
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        return 0

    removed = 0
    for json_file in results_path.glob("*.json"):
        try:
            json_file.unlink()
            print(f"  [result json] {json_file}")
            removed += 1
        except OSError as e:
            print(f"  [error]       Could not remove {json_file}: {e}")

    return removed


def run_cleanup(keep_pdfs: bool = False, root_dir: str = ".") -> None:
    """Run the full cleanup sequence: Python cache → uploads → results.

    Args:
        keep_pdfs: If True, skip upload and result cleanup (cache-only mode).
        root_dir:  Root directory to scan for Python cache files.
    """
    print("=" * 52)
    print("  ML Resume Screening — Post-Session Cleanup")
    print("=" * 52)

    # Step 1: Always clean Python cache files
    print("\n→ Scanning for Python cache files…")
    dirs_removed, files_removed = cleanup_python_cache(root_dir)
    print(f"  Removed {dirs_removed} __pycache__ folder(s), {files_removed} .pyc file(s).")

    # Step 2: Optionally clean uploads and results
    if not keep_pdfs:
        print("\n→ Clearing uploaded resumes from uploads/…")
        pdfs_removed = cleanup_uploads(os.path.join(root_dir, "uploads"))
        print(f"  Removed {pdfs_removed} uploaded PDF file(s).")

        print("\n→ Clearing session results from results/…")
        results_removed = cleanup_results(os.path.join(root_dir, "results"))
        print(f"  Removed {results_removed} result file(s).")
    else:
        print("\n→ Skipping uploads and results (--keep-pdfs).")

    print("\n✓ Cleanup complete. Project ready for the next screening session.\n")


if __name__ == "__main__":
    keep_pdfs = "--keep-pdfs" in sys.argv
    run_cleanup(keep_pdfs=keep_pdfs)