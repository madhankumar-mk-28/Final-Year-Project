"""
cleanup_cache.py — Post-screening cleanup utility.
Usage:
    python cleanup_cache.py              # cleans everything
    python cleanup_cache.py --keep-pdfs  # only cleans Python cache
"""

import os
import sys
import shutil
from pathlib import Path


def cleanup_python_cache(root_dir="."):
    """Recursively remove all __pycache__ folders and .pyc files under root_dir."""
    removed_dirs  = 0
    removed_files = 0
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        if "__pycache__" in dirnames:
            cache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(cache_path, ignore_errors=True)
            print(f"  [cache dir]  {cache_path}")
            removed_dirs += 1
            dirnames.remove("__pycache__")
        for filename in filenames:
            if filename.endswith(".pyc"):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"  [.pyc file]  {file_path}")
                removed_files += 1
    return removed_dirs, removed_files


def cleanup_uploads(uploads_dir="uploads"):
    """Delete all uploaded PDFs from uploads/<session_id>/ and remove empty session directories."""
    uploads_path = Path(uploads_dir)
    if not uploads_path.is_dir():
        return 0
    removed = 0
    for pdf_path in uploads_path.rglob("*.pdf"):  # rglob required — PDFs are nested under session subdirs
        try:
            pdf_path.unlink()
            print(f"  [upload pdf] {pdf_path}")
            removed += 1
        except OSError as e:
            print(f"  [error]      Could not remove {pdf_path}: {e}")

    for session_dir in uploads_path.iterdir():
        if session_dir.is_dir():
            try:
                session_dir.rmdir()  # only succeeds if the directory is already empty
                print(f"  [session dir] removed empty {session_dir}")
            except OSError:
                pass  # non-empty — leave it

    return removed


def cleanup_results(results_dir="results"):
    """Delete all JSON result files from the results/ directory."""
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


def run_cleanup(keep_pdfs=False, root_dir="."):
    """Run the full cleanup sequence: Python cache, uploads, results."""
    print("=" * 52)
    print("  ML Resume Screening — Post-Session Cleanup")
    print("=" * 52)

    print("\n→ Scanning for Python cache files…")
    dirs_removed, files_removed = cleanup_python_cache(root_dir)
    print(f"  Removed {dirs_removed} __pycache__ folder(s), {files_removed} .pyc file(s).")

    if not keep_pdfs:
        print("\n→ Clearing uploaded resumes from uploads/…")
        pdfs_removed = cleanup_uploads(os.path.join(root_dir, "uploads"))
        print(f"  Removed {pdfs_removed} uploaded PDF file(s).")

        print("\n→ Clearing session results from results/…")
        results_removed = cleanup_results(os.path.join(root_dir, "results"))
        print(f"  Removed {results_removed} result file(s).")
    else:
        print("\n→ Skipping uploads and results folders (--keep-pdfs flag).")

    print("\n✓ Cleanup complete. Project ready for next screening session.\n")


if __name__ == "__main__":
    keep_pdfs = "--keep-pdfs" in sys.argv
    run_cleanup(keep_pdfs=keep_pdfs)