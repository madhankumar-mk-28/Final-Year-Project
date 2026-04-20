"""
resume_parser.py — PDF-to-text extraction module for the Resume Screening System.

This module reads PDF files and extracts their text content using a dual-library
strategy: pdfplumber (primary) with PyMuPDF (fallback). A hard character cap
prevents memory exhaustion from decompression-bomb PDFs.

Public API:
    extract_text_from_pdf(pdf_path)       → str
    load_resumes_from_folder(folder_path) → (dict[filename, text], list[failed])

Dependencies:
    - pdfplumber  (required — primary PDF parser)
    - PyMuPDF     (optional — fallback for complex layouts)

Runs fully offline. No network calls. OS-independent (uses os.path / pathlib).
"""

import os
import re
import logging

import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError

# PyMuPDF is optional — system works without it, but loses fallback capability
try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

logger = logging.getLogger("resume_parser")

# Hard character cap applied per-page during extraction.
# Prevents decompression-bomb PDFs (e.g. 100-page scans) from exhausting RAM.
# 500,000 chars ≈ 250 pages of dense text — well above any normal resume.
_MAX_CHARS = 500_000


def _clean(raw: str) -> str:
    """Normalise whitespace and line endings in extracted PDF text.

    Steps:
        1. Convert Windows (\\r\\n) and old Mac (\\r) line endings to Unix (\\n)
        2. Collapse runs of spaces/tabs into a single space
        3. Reduce 3+ consecutive blank lines down to 2
        4. Strip leading/trailing whitespace
    """
    raw = re.sub(r"\r\n", "\n", raw)           # Windows line endings → Unix
    raw = re.sub(r"\r", "\n", raw)             # Old Mac line endings → Unix
    raw = re.sub(r"[ \t]+", " ", raw)          # Collapse horizontal whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)       # Max 2 consecutive blank lines
    return raw.strip()


def _extract_pdfplumber(pdf_path: str) -> str:
    """Extract text page-by-page using pdfplumber (primary parser).

    Uses x_tolerance=3 and y_tolerance=3 to handle multi-column layouts
    where characters aren't perfectly grid-aligned. Stops extracting
    once the cumulative character count reaches _MAX_CHARS.

    Returns:
        Cleaned text string (may be empty if PDF has no extractable text).
    """
    parts = []
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # x/y tolerance helps merge split tokens in multi-column PDFs
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            remaining = _MAX_CHARS - total
            if remaining <= 0:
                break                          # Character budget exhausted
            chunk = text[:remaining]           # Trim this page to fit budget
            if chunk:
                parts.append(chunk)
            total += len(chunk)
    return _clean("\n".join(parts))


def _extract_pymupdf(pdf_path: str) -> str:
    """Extract text using PyMuPDF (fallback parser).

    PyMuPDF handles scanned, rotated, and complex-layout PDFs that
    pdfplumber misses. Same per-page character cap applies.

    Returns:
        Cleaned text string (may be empty).
    """
    parts = []
    total = 0
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""     # "text" mode = plain UTF-8
            remaining = _MAX_CHARS - total
            if remaining <= 0:
                break                              # Character budget exhausted
            chunk = text[:remaining]
            if chunk:
                parts.append(chunk)
            total += len(chunk)
    finally:
        doc.close()                                # Always close the document handle
    return _clean("\n".join(parts))


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and clean all text from a single PDF file.

    Strategy:
        1. Try pdfplumber first (handles most standard PDFs reliably)
        2. If pdfplumber yields < 100 characters, fall back to PyMuPDF
        3. Keep whichever parser produced more text
        4. Apply the 500k character safety cap at the end

    Edge cases handled:
        - File not found → returns empty string
        - Permission denied → returns empty string (logged)
        - Corrupted PDF → falls back to PyMuPDF, then returns empty
        - Scanned image PDF → returns empty (no OCR support)
        - Decompression bomb → capped at _MAX_CHARS

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Cleaned text content, or empty string if extraction failed.
    """
    fname = os.path.basename(pdf_path)

    # Guard: verify the file actually exists before attempting to parse
    if not os.path.isfile(pdf_path):
        logger.error("[resume_parser] File not found: %s", pdf_path)
        return ""

    text = ""

    # Attempt 1: pdfplumber (primary parser)
    try:
        text = _extract_pdfplumber(pdf_path)
        logger.info("[resume_parser] pdfplumber → %s (%d chars)", fname, len(text))
    except PermissionError as e:
        logger.error("[resume_parser] Permission denied reading %s: %s", fname, e)
        return ""
    except PDFSyntaxError as e:
        logger.warning("[resume_parser] PDF syntax error (possibly corrupted) for %s: %s", fname, e)
    except Exception as e:
        logger.warning("[resume_parser] pdfplumber failed for %s: %s", fname, e)

    # Attempt 2: PyMuPDF fallback (only if pdfplumber yielded < 100 chars)
    if len(text) < 100 and _FITZ_AVAILABLE:
        logger.info("[resume_parser] Falling back to PyMuPDF for %s", fname)
        try:
            fitz_text = _extract_pymupdf(pdf_path)
            if len(fitz_text) > len(text):         # Keep the better result
                text = fitz_text
                logger.info("[resume_parser] PyMuPDF → %s (%d chars)", fname, len(text))
        except Exception as e:
            err_type = type(e).__name__
            is_corrupted = err_type == "FileDataError"
            logger.warning(
                "[resume_parser] PyMuPDF failed for %s [%s%s]: %s",
                fname, err_type, " (corrupted PDF)" if is_corrupted else "", e,
            )

    # No text extracted from either parser
    if not text:
        logger.error("[resume_parser] Could not extract any text from %s", fname)
        return ""

    # Safety net: enforce the hard character cap even if a page somehow bypassed it
    if len(text) > _MAX_CHARS:
        logger.warning("[resume_parser] Post-parse cap hit for %s (%d → %d chars)",
                       fname, len(text), _MAX_CHARS)
        text = text[:_MAX_CHARS]

    return text


def load_resumes_from_folder(folder_path: str) -> tuple[dict, list]:
    """Load and parse all PDF files from a folder.

    Scans the given folder for .pdf files (case-insensitive), parses each one,
    and returns the successfully extracted texts alongside a list of failures.

    Args:
        folder_path: Path to the directory containing PDF resumes.

    Returns:
        Tuple of:
            - dict mapping filename → extracted text (successful parses only)
            - list of filenames that failed to parse or yielded empty text

    Raises:
        NotADirectoryError: If folder_path does not exist or is not a directory.
        FileNotFoundError: If no PDF files are found in the folder.
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    # Sort for deterministic processing order across OS platforms
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