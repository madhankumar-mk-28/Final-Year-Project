"""
information_extractor.py — Structured data extraction from resume plain text.

This module takes the raw text output from resume_parser.py and extracts six
structured fields per candidate:
    1. Name       — derived from PDF filename (primary) or text heuristic (fallback)
    2. Email      — first valid email address found
    3. Phone      — first Indian mobile or international phone number
    4. Skills     — matched against a curated 234-entry skill database
    5. Experience — years of professional work experience (clamped to 15)
    6. Education  — degree/diploma mentions with surrounding context

Design decisions:
    - Purely regex-based — NO dependency on spaCy, NLTK, or any NLP library
    - Name extraction from filename is more reliable for Indian names than NER
    - Skills matched via substring (multi-word) and batched regex (single-word)
    - Experience extraction excludes academic context lines to avoid false matches
    - Hyphenated skill variants normalised automatically ("problem-solving" → "problem solving")

Public API:
    extract_all(text, filename) → dict with all six fields + links

Runs fully offline. No network calls. OS-independent.
"""

import os
import re
import logging
from datetime import datetime

logger = logging.getLogger("information_extractor")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SKILLS DATABASE — 234 technical and soft skills                        ║
# ║  Single canonical set. Additions/removals should be made here only.     ║
# ║  Note: bare "c" and "r" are excluded because word-boundary regex        ║
# ║  matches them inside unrelated words ("Grade C", "R&D").                ║
# ║  Use "c programming" and "r programming" instead.                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

SKILLS_DB = {
    # ── Programming languages ────────────────────────────────────────────
    "c programming", "r programming",
    "python", "java", "javascript", "typescript", "c++", "c#",
    "go", "golang", "rust", "kotlin", "swift", "php", "ruby", "scala",
    "matlab", "bash", "shell", "perl", "haskell", "lua", "dart",

    # ── Machine learning and AI ──────────────────────────────────────────
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "computer vision",
    "reinforcement learning", "neural networks", "neural network",
    "data science", "data analysis", "data analytics", "data engineering",
    "feature engineering", "model deployment", "mlops",
    "statistical analysis", "statistics", "predictive modeling",
    "time series", "anomaly detection", "recommendation system",

    # ── ML/AI frameworks and libraries ───────────────────────────────────
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost", "hugging face", "huggingface",
    "sentence-transformers", "spacy", "nltk", "gensim",
    "pillow", "transformers", "bert", "gpt", "llm",
    "langchain", "llamaindex", "diffusers", "stable diffusion",
    "fastai", "jax", "flax",

    # ── Data and visualisation ───────────────────────────────────────────
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "bokeh", "streamlit", "gradio", "jupyter", "notebook",
    "tableau", "power bi", "excel", "google sheets",

    # ── Databases ────────────────────────────────────────────────────────
    "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
    "mongodb", "redis", "cassandra", "elasticsearch", "firebase",
    "dynamodb", "neo4j", "influxdb",

    # ── Big data and pipelines ───────────────────────────────────────────
    "hadoop", "spark", "apache spark", "kafka", "airflow",
    "hive", "pig", "flink", "databricks", "snowflake",

    # ── Backend frameworks ───────────────────────────────────────────────
    "flask", "django", "fastapi", "node", "nodejs", "express",
    "spring", "springboot", "laravel", "rails", "asp.net",
    "graphql", "rest api", "restful", "microservices", "grpc",

    # ── Frontend frameworks ──────────────────────────────────────────────
    "react", "angular", "vue", "html", "css", "tailwind",
    "bootstrap", "next.js", "nextjs", "redux", "webpack",

    # ── Cloud and DevOps ─────────────────────────────────────────────────
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "k8s", "terraform", "ansible", "jenkins", "github actions",
    "ci/cd", "devops", "linux", "unix", "nginx", "apache",
    "serverless", "lambda",

    # ── Version control ──────────────────────────────────────────────────
    "git", "github", "gitlab", "bitbucket",

    # ── Tools and methodologies ──────────────────────────────────────────
    "agile", "scrum", "jira", "figma", "postman",

    # ── Specialised AI/ML tasks ──────────────────────────────────────────
    "object detection", "image classification", "text classification",
    "sentiment analysis", "named entity recognition", "ner",
    "transfer learning", "fine-tuning", "rag",
    "generative ai", "llms", "prompt engineering",

    # ── Mobile development ───────────────────────────────────────────────
    "android", "ios", "flutter", "react native",

    # ── Computer vision and misc ─────────────────────────────────────────
    "opencv", "image processing", "speech recognition",
    "data visualization", "etl", "web scraping", "selenium", "beautifulsoup",
    "pyspark", "airbyte", "dbt", "duckdb",

    # ── Vector databases and LLM tools ───────────────────────────────────
    "vector database", "pinecone", "weaviate", "faiss", "milvus",
    "prompt tuning", "llama", "mistral", "vllm", "ray", "ray serve",
    "prefect", "kedro", "mlflow", "dvc",

    # ── Database operations ──────────────────────────────────────────────
    "crud", "stored procedures", "stored procedure", "itsm",
    "it service management", "sql queries", "query optimization",
    "database design", "database management",

    # ── Soft skills ──────────────────────────────────────────────────────
    "time management", "written communication", "verbal communication",
    "teamwork", "collaboration", "problem solving", "analytical skills",
    "critical thinking", "communication", "leadership", "presentation",
    "project management", "attention to detail", "multitasking",
    "adaptability", "creativity", "innovation",
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PRE-COMPILED PATTERNS — built once at import, reused on every resume   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Multi-word skills sorted longest-first so "machine learning" matches
# before "machine" or "learning" individually (prevents partial matches)
_MULTI_WORD_SKILLS  = sorted([s for s in SKILLS_DB if " " in s], key=len, reverse=True)
_SINGLE_WORD_SKILLS = sorted([s for s in SKILLS_DB if " " not in s], key=len, reverse=True)

# Single combined regex for ALL single-word skills — one pass instead of 100+ loops
_SINGLE_SKILL_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _SINGLE_WORD_SKILLS) + r")\b"
)

# Degree pattern — locates degree mentions in text; we then extract the
# surrounding line (up to 120 chars) rather than just the regex match
DEGREE_PATTERN = re.compile(
    r"(b\.?sc|b\.?tech|b\.?e\.?|bca|bba|bachelor|m\.?sc|m\.?tech|m\.?e\.?|mca|mba|master|ph\.?d|diploma)"
    r"[\w\s,.()-]{0,80}",
    re.IGNORECASE,
)

# Normalises hyphenated skill variants to their spaced form
# Example: "problem-solving" → "problem solving", "time-management" → "time management"
_HYPHEN_SKILL_RE = re.compile(
    r"\b(problem|decision|critical|multi|self|cross|result|goal|forward|detail|team|time|data|risk|cost)"
    r"-(\w+)\b",
    re.IGNORECASE,
)

# Minimum character length for an education entry to be kept (filters noise)
_EDU_MIN_LEN = 15

# Strips lowercase-only prefix fragments from PDF text-join artefacts
# Example: "ment Systems..." → removes "ment " (mid-word join from multi-column PDF)
_EDU_LEADING_FRAGMENT_RE = re.compile(r"^[a-z]+\s+")


# ── Name extraction constants ───────────────────────────────────────────────

# Section headings that should NOT be mistaken for candidate names
_NAME_NOISE_RE = re.compile(
    r"^(resume|curriculum\s*vitae|cv|profile|summary|objective|personal\s+info"
    r"|contact|about\s+me|career|experience|education|skills|projects|cover\s+letter"
    r"|references|declaration|portfolio)$",
    re.IGNORECASE,
)

# A plausible name line: 2-5 words, letters/spaces/hyphens/periods/apostrophes only
_NAME_VALID_RE = re.compile(
    r"^[A-Za-z][A-Za-z .\-']{1,60}$"
)


# ── Link extraction patterns ────────────────────────────────────────────────

_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/([\w\-%.]+)/?",
    re.IGNORECASE,
)
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([\w\-]+)/?",
    re.IGNORECASE,
)
# Portfolio URL — any https:// URL that isn't a major social media platform
_PORTFOLIO_RE = re.compile(
    r"(https?://(?!(?:www\.)?(?:linkedin|github|google|facebook|twitter|instagram|youtube)\.com)"
    r"[\w\-]+\.[\w\-.]+(?:/[\w\-./]*)?)",
    re.IGNORECASE,
)


# ── Experience extraction constants ─────────────────────────────────────────

# Patterns that include the word "experience" to avoid matching academic durations
# like "4-year B.Tech programme"
_EXP_WORK_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+(?:professional\s+|work\s+|industry\s+|total\s+)?experience",
    r"experience\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",
    r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s+(?:of\s+)?(?:professional\s+|work\s+)?experience",
    r"worked\s+for\s+(\d+(?:\.\d+)?)\s*\+?\s*years?",
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?(?:work\s+)?experience",
]

# Date-range pattern for employment history sections:
# Matches formats like: "Jan 2020 – Dec 2023", "06/2019 - 12/2019", "2020 - Present"
_MONTH_NAMES = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_RANGE_RE = re.compile(
    r"(?:" + _MONTH_NAMES + r"[.,]?\s*)?(?:\d{1,2}[/\-.])?\\s*(\\d{4})"
    r"\s*(?:–|—|\-|to)\s*"
    r"(?:(?:" + _MONTH_NAMES + r"[.,]?\s*)?(?:\d{1,2}[/\-.])?\\s*(\\d{4}|present|current|ongoing|till\\s+date|now))",
    re.IGNORECASE,
)

# Academic context keywords — lines containing these are deprioritised for experience
# to prevent "4-year B.Tech programme" counting as 4 years of work
_EXP_ACADEMIC_RE = re.compile(
    r"\b(degree|b\.?tech|m\.?tech|b\.?e\.?|bca|b\.?sc|m\.?sc|mba|engineering\s+program"
    r"|college|university|academic|curriculum|course\s*work|semester|fresher|pursuing"
    r"|programme?|graduation|undergraduate|postgraduate|study|studies|institution)\b",
    re.IGNORECASE,
)

# Hard cap: no entry-level or mid-level candidate plausibly has more than 15 years.
# Values above this are clamped (not discarded), so "20 years" → 15.0, not 0.0.
_EXP_MAX_YEARS = 15.0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PUBLIC API                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def extract_all(text: str, filename: str = "") -> dict:
    """Extract all structured fields from resume text into a single dict.

    Extraction order:
        1. Name from filename (reliable for Indian names) → text-based fallback
        2. Email, phone, skills, experience, education, links (each independent)

    Each sub-extractor is wrapped in a try/except so a failure in one
    (e.g. a regex crash on malformed text) doesn't lose all other fields.

    Args:
        text:     Raw text content extracted from a PDF resume.
        filename: Original PDF filename (used to derive candidate name).

    Returns:
        dict with keys: name, email, phone, skills, experience_years, education, links
    """
    # Step 1: Derive name from filename (more reliable than text-based NER for Indian names)
    file_name = name_from_filename(filename) if filename else "Unknown"
    if file_name == "Unknown":
        # Fallback: try to extract name from the first few lines of the resume text
        text_name = extract_name_from_text(text)
        name = text_name if text_name and text_name != "Unknown" else file_name
    else:
        name = file_name

    # Step 2: Extract each field independently — failure in one won't affect others
    def _safe(fn, *args, default=None):
        """Run an extractor function, returning default on any exception."""
        try:
            return fn(*args)
        except Exception as exc:
            logger.warning("[extractor] %s failed for %s: %s", fn.__name__, filename or "?", exc)
            return default

    result = {
        "name":             name,
        "email":            _safe(extract_email, text, default=""),
        "phone":            _safe(extract_phone, text, default=""),
        "skills":           _safe(extract_skills, text, default=[]),
        "experience_years": _safe(extract_experience_years, text, default=0.0),
        "education":        _safe(extract_education, text, default=[]),
        "links":            _safe(extract_links, text, default={"linkedin": "", "github": "", "portfolio": ""}),
    }

    logger.info(
        "[extractor] %s → name=%s | email=%s | skills=%d | exp=%.1f yrs | edu=%d | links=%d",
        filename or "?", result["name"], result["email"] or "—",
        len(result["skills"]), result["experience_years"], len(result["education"]),
        sum(1 for v in result["links"].values() if v),
    )
    return result


def name_from_filename(filename: str) -> str:
    """Derive a human-readable candidate name from the PDF filename.

    Processing steps:
        1. Strip file extensions (.pdf, .doc, .docx)
        2. Replace delimiters (underscores, hyphens, dots) with spaces
        3. Remove noise words (resume, cv, latest, final, updated, etc.)
        4. Remove numbering suffixes like (1), [2], -3
        5. Remove non-letter characters
        6. Title-case the result, preserving short all-caps tokens (e.g. "ML")

    Examples:
        "Madhan_Kumar_Resume.pdf"     → "Madhan Kumar"
        "John-Smith-CV-Final(2).pdf"  → "John Smith"
        "resume.pdf"                  → "Unknown"

    Returns:
        Cleaned name string, or "Unknown" if the filename is a noise word.
    """
    base = filename

    # Strip known file extensions (handles double extensions like .pdf.pdf)
    while True:
        stem, ext = os.path.splitext(base)
        if ext.lower() in (".pdf", ".doc", ".docx"):
            base = stem
        else:
            break

    # Convert delimiters to spaces so word-boundary matching works correctly
    base = re.sub(r"[_\-\.]", " ", base)

    # Remove common noise words that aren't part of a person's name
    noise_words = (
        r"\b(resume|cv|curriculum|vitae|new|latest|final|updated|"
        r"deloitte|profile|portfolio|theme|engineeringresumes|rendercv)\b"
    )
    base = re.sub(noise_words, "", base, flags=re.IGNORECASE)

    # Remove numbering suffixes: (1), [2], -3
    base = re.sub(r"\s*[\(\[]\d+[\)\]]", "", base)
    base = re.sub(r"\-\d+$", "", base)

    # Keep only letters and spaces
    base = re.sub(r"[^A-Za-z\s]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()

    # If nothing meaningful remains, try a simpler fallback
    if not base or len(base) < 2:
        fallback = os.path.splitext(filename)[0].replace("_", " ").strip()
        _noise = {"resume", "cv", "final", "new", "latest", "updated", "profile",
                  "curriculum vitae", "final updated resume", "final resume"}
        if fallback.lower() in _noise or not fallback or len(fallback) < 2:
            return "Unknown"
        return fallback.title()

    # Title-case words, but preserve short all-caps tokens like "ML", "AI"
    words = base.split()
    titled = []
    for w in words:
        if w.isupper() and len(w) <= 3:   # Keep "ML", "AI", "DL" as-is
            titled.append(w)
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


def extract_email(text: str) -> str:
    """Return the first valid email address found in the resume text.

    Uses a standard email regex pattern. Returns lowercase to ensure
    consistent deduplication in the Truth Engine.

    Returns:
        Email string in lowercase, or empty string if none found.
    """
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group(0).lower() if m else ""


def _normalize_phone(raw: str) -> str:
    """Normalise a matched phone string to a consistent format.

    Formats:
        - Indian with country code: +91-9876543210
        - Bare 10-digit Indian:     9876543210
        - International:            +1-2125551234
        - Fallback:                 digits only
    """
    has_plus = raw.startswith("+") or raw.startswith("(+")
    digits   = re.sub(r"[^\d]", "", raw)

    if digits.startswith("91") and len(digits) == 12:   # Indian with +91
        return f"+91-{digits[2:]}"
    if len(digits) == 10 and digits[0] in "6789":       # Bare Indian mobile
        return digits
    if has_plus and len(digits) >= 11:                   # International format
        if len(digits) <= 13:
            cc_len = len(digits) - 10
            return f"+{digits[:cc_len]}-{digits[cc_len:]}"

    return digits if digits else ""


def extract_phone(text: str) -> str:
    """Return the first Indian mobile or international phone number found.

    Tries six regex patterns in order of specificity (most specific first).
    The first match wins and is normalised to a consistent format.

    Returns:
        Normalised phone string, or empty string if none found.
    """
    patterns = [
        r"\+91[\s\-]?[6-9]\d{9}",                           # +91 followed by 10-digit Indian mobile
        r"\+91[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{4}",        # +91 with grouped digits
        r"\b[6-9]\d{4}[\s\-]?\d{5}\b",                      # 10-digit with mid-space
        r"\b[6-9]\d{9}\b",                                   # Bare 10-digit Indian mobile
        r"\(\+91\)[\s]?[6-9]\d{9}",                          # (+91) format
        r"\+\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",  # International
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return _normalize_phone(m.group(0).strip())
    return ""


def extract_skills(text: str) -> list:
    """Match skills from the 234-entry SKILLS_DB against resume text.

    Matching strategy (executed in this order):
        1. Normalise hyphenated variants ("problem-solving" → "problem solving")
        2. Multi-word skills: longest-first substring search in lowercased text
        3. Single-word skills: single pre-compiled regex pass with word boundaries

    The longest-first ordering prevents partial matches — "machine learning"
    is matched before "machine" or "learning" individually.

    Returns:
        Sorted list of matched skill strings (alphabetical).
    """
    # Normalise hyphenated variants to their spaced form for DB matching
    normalised = _HYPHEN_SKILL_RE.sub(lambda m: m.group(1) + " " + m.group(2), text)
    text_lower = normalised.lower()
    found = set()

    # Pass 1: Multi-word skills via substring search (longest-first priority)
    for skill in _MULTI_WORD_SKILLS:
        if skill in text_lower:
            found.add(skill)

    # Pass 2: Single-word skills via one batched regex (fast — single scan)
    found.update(_SINGLE_SKILL_RE.findall(text_lower))

    return sorted(found)


def extract_experience_years(text: str) -> float:
    """Extract years of professional work experience from resume text.

    Strategy:
        1. Scan each line for experience-related patterns ("X years of experience")
        2. Lines containing academic keywords are deprioritised (used as fallback only)
        3. If no explicit mentions found, calculate from employment date ranges
        4. All values clamped to 15 years maximum

    The academic filter prevents "4-year B.Tech programme" from being counted
    as 4 years of professional work experience.

    Returns:
        Float representing years of experience (0.0 if none found, max 15.0).
    """
    work_matches:     list[float] = []   # From non-academic lines (high confidence)
    fallback_matches: list[float] = []   # From academic lines (low confidence)

    for line in text.splitlines():
        is_academic = bool(_EXP_ACADEMIC_RE.search(line))

        for pat in _EXP_WORK_PATTERNS:
            for m in re.finditer(pat, line, re.IGNORECASE):
                val = min(float(m.group(1)), _EXP_MAX_YEARS)   # Clamp to 15 years
                if is_academic:
                    fallback_matches.append(val)   # Academic context — lower priority
                else:
                    work_matches.append(val)       # Work context — high priority

    # Return the highest value found, preferring work context over academic
    if work_matches:
        return max(work_matches)
    if fallback_matches:
        return max(fallback_matches)

    # Last-resort: calculate experience from employment date ranges
    date_exp = _extract_date_range_experience(text)
    if date_exp > 0.0:
        return min(date_exp, _EXP_MAX_YEARS)

    return 0.0


def _clean_edu_entry(raw: str) -> str:
    """Clean a raw education string extracted from PDF text.

    Handles artefacts from multi-column PDF layouts where pdfplumber rejoins
    hyphenated line-breaks, producing fragments like "ment Systems...".

    Steps:
        1. Collapse embedded newlines/tabs into single space
        2. Strip leading lowercase-only fragments (mid-word join artefacts)
        3. Remove non-printable control characters
    """
    raw = re.sub(r"[\n\r\t]+", " ", raw)              # Collapse embedded newlines
    raw = re.sub(r" {2,}", " ", raw).strip()           # Collapse multiple spaces
    raw = _EDU_LEADING_FRAGMENT_RE.sub("", raw).strip()  # Strip mid-word fragments
    raw = re.sub(r"[\x00-\x1f\x7f]", " ", raw).strip()   # Remove control chars
    return raw


def extract_education(text: str) -> list:
    """Extract education qualifications (degrees, diplomas) from resume text.

    Strategy:
        1. Use DEGREE_PATTERN to locate degree mentions (B.Tech, M.Sc, MBA, etc.)
        2. For each match, extract the complete surrounding line (up to 120 chars)
        3. Clean each entry to remove PDF artefacts
        4. Deduplicate by lowercase comparison

    This approach avoids the common artefact where the regex match starts mid-word
    because pdfplumber rejoined a hyphenated line-break from a multi-column layout.

    Returns:
        List of unique education strings (cleaned and deduplicated).
    """
    found: list[str] = []
    seen: set[str] = set()

    for m in DEGREE_PATTERN.finditer(text):
        match_start = m.start()

        # Find the start of the line containing this match
        line_start = text.rfind("\n", 0, match_start)
        line_start = line_start + 1 if line_start >= 0 else 0

        # Find the end of the line containing this match
        line_end = text.find("\n", match_start)
        if line_end == -1:
            line_end = len(text)

        # Extract up to 120 chars from the line start
        raw_line = text[line_start:min(line_end, line_start + 120)]
        entry = _clean_edu_entry(raw_line)

        # Filter out short fragments and uppercase noise (section headers)
        if len(entry) < _EDU_MIN_LEN:
            continue
        if entry.isupper() and len(entry) < 30:
            continue

        # Deduplicate by lowercase key
        key = entry.lower()
        if key not in seen:
            seen.add(key)
            found.append(entry)

    return found


def extract_name_from_text(text: str) -> str:
    """Extract the candidate's name from the top of the resume text (fallback method).

    This is used only when the filename doesn't yield a usable name (e.g. "resume.pdf").

    Heuristic: the name is almost always in the first 5 non-empty lines.
    A valid name line must be:
        - 2 to 5 words long
        - Only letters, spaces, hyphens, periods, and apostrophes
        - Not a section heading (RESUME, CURRICULUM VITAE, etc.)
        - Not contain digits, emails, phone numbers, or URLs

    Preference: title-case or ALL-CAPS lines are prioritised (typical for names).

    Returns:
        Best candidate name string, or "Unknown" if no plausible name found.
    """
    lines = text.strip().splitlines()
    candidates = []

    for line in lines[:15]:              # Scan first 15 raw lines
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue

        # Skip lines containing emails, long digit sequences, or URLs
        if "@" in stripped or re.search(r"\d{5,}", stripped):
            continue
        if re.search(r"https?://|www\.", stripped, re.IGNORECASE):
            continue

        # Skip known section headings
        if _NAME_NOISE_RE.fullmatch(stripped.strip("  \t:-•|/")):
            continue

        # Keep only letters, spaces, hyphens, periods, apostrophes
        clean = re.sub(r"[^A-Za-z .\-']", "", stripped).strip()
        if not clean or len(clean) < 3:
            continue

        # Name should be 2-5 words
        words = clean.split()
        if len(words) < 2 or len(words) > 5:
            continue

        # Must match the valid name pattern
        if not _NAME_VALID_RE.match(clean):
            continue

        # Prefer title-case or ALL-CAPS lines (typical name formatting)
        if (clean.istitle() or clean.isupper()) and not candidates:
            candidates.insert(0, clean)    # High confidence → front of list
        else:
            candidates.append(clean)

        if len(candidates) >= 3:
            break                          # Enough candidates to choose from

    if not candidates:
        return "Unknown"

    # Return the best candidate, title-cased if it was all-caps
    best = candidates[0]
    return best.title() if best.isupper() else best


def extract_links(text: str) -> dict:
    """Extract LinkedIn, GitHub, and portfolio URLs from resume text.

    Portfolio URL: any https:// link that isn't a major social media platform.
    GitHub: excludes common false positives (github.com/topics, /features, etc.).

    Returns:
        dict with keys: linkedin, github, portfolio (each a URL string or "").
    """
    links = {"linkedin": "", "github": "", "portfolio": ""}

    # LinkedIn profile URL
    m = _LINKEDIN_RE.search(text)
    if m:
        raw = m.group(0)
        links["linkedin"] = raw if raw.startswith("http") else f"https://{raw}"

    # GitHub profile URL (excluding common non-user pages)
    m = _GITHUB_RE.search(text)
    if m:
        username = m.group(1).lower()
        if username not in {"topics", "features", "explore", "settings", "login", "signup", "about"}:
            raw = m.group(0)
            links["github"] = raw if raw.startswith("http") else f"https://{raw}"

    # Portfolio / personal website URL
    m = _PORTFOLIO_RE.search(text)
    if m:
        links["portfolio"] = m.group(1)

    return links


def _extract_date_range_experience(text: str) -> float:
    """Calculate total years of experience from employment date ranges.

    Scans for date range patterns like "Jan 2020 – Dec 2023" or "2019 - Present"
    in non-academic lines, then sums non-overlapping employment periods.

    Overlapping periods are automatically merged to prevent double-counting
    (e.g. if a candidate lists concurrent positions).

    Returns:
        Float representing total years from date ranges (0.0 if none found).
    """
    periods = []
    now = datetime.now()

    for line in text.splitlines():
        # Skip lines with academic context to avoid counting education duration
        if _EXP_ACADEMIC_RE.search(line):
            continue

        for m in _DATE_RANGE_RE.finditer(line):
            start_year_str = m.group(1)
            end_str = m.group(2).strip().lower()

            try:
                start_year = int(start_year_str)
            except (ValueError, TypeError):
                continue

            # "Present", "Current", "Ongoing" → use current year
            if end_str in {"present", "current", "ongoing", "now"} or end_str.startswith("till"):
                end_year = now.year
            else:
                try:
                    end_year = int(end_str)
                except (ValueError, TypeError):
                    continue

            # Sanity checks: reject implausible date ranges
            if start_year < 1970 or start_year > now.year:
                continue
            if end_year < start_year or end_year > now.year + 1:
                continue

            periods.append((start_year, end_year))

    if not periods:
        return 0.0

    # Merge overlapping periods to avoid double-counting concurrent positions
    periods.sort()
    merged = [periods[0]]
    for start, end in periods[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:                          # Overlapping → extend
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))                # Non-overlapping → new period

    # Sum all merged period durations
    total = sum(end - start for start, end in merged)
    return round(float(total), 1)