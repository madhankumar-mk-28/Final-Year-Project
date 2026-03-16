"""
resume_parser.py
----------------
Extracts clean text from PDF resumes.

Strategy:
  1. pdfplumber  — best for text-layer PDFs (primary)
  2. PyMuPDF     — fallback for scanned or complex-layout PDFs
"""

import os
import re
import logging

import pdfplumber

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("resume_parser")


def _clean(raw: str) -> str:
    """Normalise whitespace and line endings."""
    raw = re.sub(r"\r\n", "\n", raw)
    raw = re.sub(r"\r", "\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)          # collapse horizontal whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)        # max 2 blank lines
    return raw.strip()


def _extract_pdfplumber(pdf_path: str) -> str:
    """Primary extractor: pdfplumber (best for text-layer PDFs)."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                parts.append(text)
    return _clean("\n".join(parts))


def _extract_pymupdf(pdf_path: str) -> str:
    """Fallback extractor: PyMuPDF (handles tables, scanned, rotated pages)."""
    parts = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text("text")
        if text:
            parts.append(text)
    doc.close()
    return _clean("\n".join(parts))


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and clean all text from a PDF, falling back to PyMuPDF if pdfplumber fails."""
    fname = os.path.basename(pdf_path)

    if not os.path.isfile(pdf_path):
        logger.error("[resume_parser] File not found: %s", pdf_path)
        return ""

    text = ""

    # ── Primary: pdfplumber ───────────────────────────────────────────────────
    try:
        text = _extract_pdfplumber(pdf_path)
        logger.info("[resume_parser] pdfplumber → %s (%d chars)", fname, len(text))
    except PermissionError as e:
        logger.error("[resume_parser] Permission denied reading %s: %s", fname, e)
        return ""
    except Exception as e:
        logger.warning("[resume_parser] pdfplumber failed for %s: %s", fname, e)

    # ── Fallback: PyMuPDF (if primary gave nothing or <100 chars) ─────────────
    if len(text) < 100 and _FITZ_AVAILABLE:
        logger.info("[resume_parser] Falling back to PyMuPDF for %s", fname)
        try:
            fitz_text = _extract_pymupdf(pdf_path)
            if len(fitz_text) > len(text):
                text = fitz_text
                logger.info("[resume_parser] PyMuPDF → %s (%d chars)", fname, len(text))
        except Exception as e:
            logger.warning("[resume_parser] PyMuPDF also failed for %s: %s", fname, e)

    if not text:
        logger.error("[resume_parser] Could not extract any text from %s", fname)

    return text


def load_resumes_from_folder(folder_path: str) -> tuple[dict, list]:
    """Load all PDFs from a folder; return ({filename: text}, [failed_filenames])."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {folder_path}")

    logger.info("[resume_parser] Parsing %d PDF(s) from %s", len(pdf_files), folder_path)
    results, failed = {}, []
    for fname in pdf_files:
        full_path = os.path.join(folder_path, fname)
        try:
            text = extract_text_from_pdf(full_path)
        except Exception as e:
            logger.error("[resume_parser] Exception parsing %s: %s", fname, e)
            text = ""
        if text.strip():
            results[fname] = text
            logger.info("[resume_parser] ✓ Loaded: %s", fname)
        else:
            failed.append(fname)
            logger.warning("[resume_parser] ✗ Empty/unreadable: %s", fname)

    logger.info("[resume_parser] %d loaded, %d failed.\n", len(results), len(failed))
    return results, failed