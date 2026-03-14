"""
cleanup_cache.py
----------------
Post-screening cleanup utility for the ML Resume Screening System.

What gets deleted:
  1. All __pycache__ folders (recursively)
  2. All .pyc compiled bytecode files
  3. All uploaded PDF resumes from the uploads/ folder

Usage:
    python cleanup_cache.py              # cleans everything
    python cleanup_cache.py --keep-pdfs  # keep uploads, only clean Python cache
"""

import os
import sys
import shutil
import glob


def cleanup_python_cache(root_dir="."):
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
    if not os.path.isdir(uploads_dir):
        return 0
    removed = 0
    for pdf_path in glob.glob(os.path.join(uploads_dir, "*.pdf")):
        try:
            os.remove(pdf_path)
            print(f"  [upload pdf] {pdf_path}")
            removed += 1
        except OSError as e:
            print(f"  [error]      Could not remove {pdf_path}: {e}")
    return removed


def run_cleanup(keep_pdfs=False, root_dir="."):
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
    else:
        print("\n→ Skipping uploads folder (--keep-pdfs flag).")

    print("\n✓ Cleanup complete. Project ready for next screening session.\n")


if __name__ == "__main__":
    keep_pdfs = "--keep-pdfs" in sys.argv
    run_cleanup(keep_pdfs=keep_pdfs)