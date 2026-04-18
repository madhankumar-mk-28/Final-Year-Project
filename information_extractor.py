import os
import re
import logging
from datetime import datetime

logger = logging.getLogger("information_extractor")

# ---------------------------------------------------------------------------
# Skills database
# ---------------------------------------------------------------------------
# Single canonical list of recognised skills.  "c" and "r" (single chars) are
# intentionally absent — word-boundary regex matches them inside words like
# "Grade C" or "R&D", which produces false positives.

SKILLS_DB = {
    "c programming", "r programming",
    "python", "java", "javascript", "typescript", "c++", "c#",
    "go", "golang", "rust", "kotlin", "swift", "php", "ruby", "scala",
    "matlab", "bash", "shell", "perl", "haskell", "lua", "dart",

    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "computer vision",
    "reinforcement learning", "neural networks", "neural network",
    "data science", "data analysis", "data analytics", "data engineering",
    "feature engineering", "model deployment", "mlops",
    "statistical analysis", "statistics", "predictive modeling",
    "time series", "anomaly detection", "recommendation system",

    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost", "hugging face", "huggingface",
    "sentence-transformers", "spacy", "nltk", "gensim",
    "pillow", "transformers", "bert", "gpt", "llm",
    "langchain", "llamaindex", "diffusers", "stable diffusion",
    "fastai", "jax", "flax",

    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "bokeh", "streamlit", "gradio", "jupyter", "notebook",
    "tableau", "power bi", "excel", "google sheets",

    "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
    "mongodb", "redis", "cassandra", "elasticsearch", "firebase",
    "dynamodb", "neo4j", "influxdb",

    "hadoop", "spark", "apache spark", "kafka", "airflow",
    "hive", "pig", "flink", "databricks", "snowflake",

    "flask", "django", "fastapi", "node", "nodejs", "express",
    "spring", "springboot", "laravel", "rails", "asp.net",
    "graphql", "rest api", "restful", "microservices", "grpc",

    "react", "angular", "vue", "html", "css", "tailwind",
    "bootstrap", "next.js", "nextjs", "redux", "webpack",

    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "k8s", "terraform", "ansible", "jenkins", "github actions",
    "ci/cd", "devops", "linux", "unix", "nginx", "apache",
    "serverless", "lambda",

    "git", "github", "gitlab", "bitbucket",

    "agile", "scrum", "jira", "figma", "postman",
    "object detection", "image classification", "text classification",
    "sentiment analysis", "named entity recognition", "ner",
    "transfer learning", "fine-tuning", "rag",
    "generative ai", "llms", "prompt engineering",
    "android", "ios", "flutter", "react native",
    "opencv", "image processing", "speech recognition",
    "data visualization", "etl", "web scraping", "selenium", "beautifulsoup",
    "pyspark", "airbyte", "dbt", "duckdb",
    "vector database", "pinecone", "weaviate", "faiss", "milvus",
    "prompt tuning", "llama", "mistral", "vllm", "ray", "ray serve",
    "prefect", "kedro", "mlflow", "dvc",

    "crud", "stored procedures", "stored procedure", "itsm",
    "it service management", "sql queries", "query optimization",
    "database design", "database management",

    "time management", "written communication", "verbal communication",
    "teamwork", "collaboration", "problem solving", "analytical skills",
    "critical thinking", "communication", "leadership", "presentation",
    "project management", "attention to detail", "multitasking",
    "adaptability", "creativity", "innovation",
}

# ---------------------------------------------------------------------------
# Pre-compiled patterns — built once at import, reused on every resume
# ---------------------------------------------------------------------------

# Multi-word skills sorted longest-first so "machine learning" is matched
# before "machine" or "learning" individually.
_MULTI_WORD_SKILLS  = sorted([s for s in SKILLS_DB if " " in s], key=len, reverse=True)
_SINGLE_WORD_SKILLS = sorted([s for s in SKILLS_DB if " " not in s], key=len, reverse=True)

# One combined regex for all single-word skills — faster than iterating them
_SINGLE_SKILL_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _SINGLE_WORD_SKILLS) + r")\b"
)

# Degree pattern — used to LOCATE degree mentions; we then extract the
# surrounding line rather than the raw regex group (which starts mid-word
# when pdfplumber rejoins hyphenated line-breaks from multi-column PDFs).
DEGREE_PATTERN = re.compile(
    r"(b\.?sc|b\.?tech|b\.?e\.?|bca|bba|bachelor|m\.?sc|m\.?tech|m\.?e\.?|mca|mba|master|ph\.?d|diploma)"
    r"[\w\s,.()-]{0,80}",
    re.IGNORECASE,
)

# Hyphenated skills normalisation — converts e.g. "problem-solving" → "problem solving"
# so they are recognised as entries in SKILLS_DB.
_HYPHEN_SKILL_RE = re.compile(
    r"\b(problem|decision|critical|multi|self|cross|result|goal|forward|detail|team|time|data|risk|cost)"
    r"-(\w+)\b",
    re.IGNORECASE,
)

# Minimum length for an education entry to be kept (avoids single-word noise)
_EDU_MIN_LEN = 15

# Leading fragment pattern — strips lowercase-only prefix chars that result from
# mid-word text joins in multi-column PDF layouts (e.g. "ment Systems" → "Systems").
_EDU_LEADING_FRAGMENT_RE = re.compile(r"^[a-z]+\s+")

# ---------------------------------------------------------------------------
# Name extraction constants
# ---------------------------------------------------------------------------

# Section headings that should NOT be treated as candidate names.
_NAME_NOISE_RE = re.compile(
    r"^(resume|curriculum\s*vitae|cv|profile|summary|objective|personal\s+info"
    r"|contact|about\s+me|career|experience|education|skills|projects|cover\s+letter"
    r"|references|declaration|portfolio)$",
    re.IGNORECASE,
)

# A plausible name line: 2-5 words, only letters/spaces/hyphens/periods/apostrophes,
# no digits, no email-like '@' or URL-like patterns.
_NAME_VALID_RE = re.compile(
    r"^[A-Za-z][A-Za-z .\-']{1,60}$"
)

# ---------------------------------------------------------------------------
# Link extraction patterns
# ---------------------------------------------------------------------------

_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/([\w\-%.]+)/?",
    re.IGNORECASE,
)
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([\w\-]+)/?",
    re.IGNORECASE,
)
_PORTFOLIO_RE = re.compile(
    r"(https?://(?!(?:www\.)?(?:linkedin|github|google|facebook|twitter|instagram|youtube)\.com)"
    r"[\w\-]+\.[\w\-.]+(?:/[\w\-./]*)?)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Experience extraction constants — at module level so they compile once
# ---------------------------------------------------------------------------

# Patterns that require the word "experience" — much less likely to fire on
# educational duration phrases like "4-year B.Tech programme".
_EXP_WORK_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+(?:professional\s+|work\s+|industry\s+|total\s+)?experience",
    r"experience\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",
    r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s+(?:of\s+)?(?:professional\s+|work\s+)?experience",
    r"worked\s+for\s+(\d+(?:\.\d+)?)\s*\+?\s*years?",
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?(?:work\s+)?experience",
]

# Date-range pattern for employment history:
#   Jan 2020 – Dec 2023, 06/2019 - 12/2019, 2020 - Present, etc.
_MONTH_NAMES = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_RANGE_RE = re.compile(
    r"(?:" + _MONTH_NAMES + r"[.,]?\s*)?(?:\d{1,2}[/\-.])?\s*(\d{4})"
    r"\s*(?:–|—|\-|to)\s*"
    r"(?:(?:" + _MONTH_NAMES + r"[.,]?\s*)?(?:\d{1,2}[/\-.])?\s*(\d{4}|present|current|ongoing|till\s+date|now))",
    re.IGNORECASE,
)

# Lines containing any of these words are treated as academic context.
# Year figures on such lines are used only as a last-resort fallback,
# preventing "4-year B.Tech programme" from being counted as 4 years of work.
_EXP_ACADEMIC_RE = re.compile(
    r"\b(degree|b\.?tech|m\.?tech|b\.?e\.?|bca|b\.?sc|m\.?sc|mba|engineering\s+program"
    r"|college|university|academic|curriculum|course\s*work|semester|fresher|pursuing"
    r"|programme?|graduation|undergraduate|postgraduate|study|studies|institution)\b",
    re.IGNORECASE,
)

# Hard cap: no plausible entry-level or mid-level candidate has more than 15
# years of experience.  Values above this are clamped, not discarded, so
# "20 years of experience" still yields 15.0 rather than 0.0.
_EXP_MAX_YEARS = 15.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_all(text: str, filename: str = "") -> dict:
    """Extract all structured fields from resume text.

    Uses filename-derived name as the primary source (more reliable for Indian
    names). Falls back to text-based name extraction only when the filename
    yields 'Unknown' (e.g. files named 'resume.pdf' or 'document.pdf').
    """
    # Name: filename first (reliable for Indian names), text-based as fallback
    file_name = name_from_filename(filename) if filename else "Unknown"
    if file_name == "Unknown":
        text_name = extract_name_from_text(text)
        name = text_name if text_name and text_name != "Unknown" else file_name
    else:
        name = file_name

    # IMPROVED: Each extractor is wrapped individually so a failure in one
    # (e.g. a regex crash on malformed text) doesn't lose all other fields.
    def _safe(fn, *args, default=None):
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

    Strips file extensions, common noise words (resume, cv, latest, etc.),
    numbering suffixes, and special characters, then returns a title-cased name.
    Returns 'Unknown' when the filename itself is a noise word.
    """
    base = filename
    while True:
        stem, ext = os.path.splitext(base)
        if ext.lower() in (".pdf", ".doc", ".docx"):
            base = stem
        else:
            break

    # Convert delimiters to spaces FIRST so word-boundary regex \b works
    # correctly on tokens like "_Resume" (underscore is a word char in regex).
    base = re.sub(r"[_\-\.]", " ", base)

    noise_words = (
        r"\b(resume|cv|curriculum|vitae|new|latest|final|updated|"
        r"deloitte|profile|portfolio|theme|engineeringresumes|rendercv)\b"
    )
    base = re.sub(noise_words, "", base, flags=re.IGNORECASE)
    base = re.sub(r"\s*[\(\[]\d+[\)\]]", "", base)   # strip trailing (1), [2], etc.
    base = re.sub(r"\-\d+$", "", base)                # strip trailing dash-number suffix
    base = re.sub(r"[^A-Za-z\s]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()

    if not base or len(base) < 2:
        fallback = os.path.splitext(filename)[0].replace("_", " ").strip()
        _noise = {"resume", "cv", "final", "new", "latest", "updated", "profile",
                  "curriculum vitae", "final updated resume", "final resume"}
        if fallback.lower() in _noise or not fallback or len(fallback) < 2:
            return "Unknown"
        return fallback.title()

    words = base.split()
    titled = []
    for w in words:
        if w.isupper() and len(w) <= 3:   # keep short all-caps like "ML" intact
            titled.append(w)
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


def extract_email(text: str) -> str:
    """Return the first valid email address found in the resume, or an empty string."""
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group(0).lower() if m else ""


def _normalize_phone(raw: str) -> str:
    """Normalise a matched phone string to +CC-XXXXXXXXXX or a bare 10-digit number."""
    has_plus = raw.startswith("+") or raw.startswith("(+")
    digits   = re.sub(r"[^\d]", "", raw)

    if digits.startswith("91") and len(digits) == 12:   # Indian with country code
        return f"+91-{digits[2:]}"
    if len(digits) == 10 and digits[0] in "6789":        # bare 10-digit Indian mobile
        return digits
    if has_plus and len(digits) >= 11:                   # international
        if len(digits) <= 13:
            cc_len = len(digits) - 10
            return f"+{digits[:cc_len]}-{digits[cc_len:]}"

    return digits if digits else ""


def extract_phone(text: str) -> str:
    """Return the first Indian mobile or international phone number found, or an empty string."""
    patterns = [
        r"\+91[\s\-]?[6-9]\d{9}",
        r"\+91[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{4}",
        r"\b[6-9]\d{4}[\s\-]?\d{5}\b",
        r"\b[6-9]\d{9}\b",
        r"\(\+91\)[\s]?[6-9]\d{9}",
        r"\+\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return _normalize_phone(m.group(0).strip())
    return ""


def extract_skills(text: str) -> list:
    """Match skills from SKILLS_DB against resume text.

    Multi-word skills use substring search (longest-first to prevent partial
    matches).  Single-word skills use a single pre-compiled regex pass.
    Hyphenated variants (e.g. "problem-solving", "time-management") are
    normalised to their spaced form before matching.
    """
    # Normalise hyphenated skill variants → spaced form ("problem-solving" → "problem solving")
    normalised = _HYPHEN_SKILL_RE.sub(lambda m: m.group(1) + " " + m.group(2), text)
    text_lower = normalised.lower()
    found = set()

    for skill in _MULTI_WORD_SKILLS:
        if skill in text_lower:
            found.add(skill)

    found.update(_SINGLE_SKILL_RE.findall(text_lower))
    return sorted(found)


def extract_experience_years(text: str) -> float:
    """Return the candidate's years of professional work experience.

    Processes the resume line-by-line.  Lines containing academic keywords
    (degree, college, university, etc.) are deprioritised: any year figures on
    those lines are used only as a last-resort fallback.  This prevents a
    '4-year B.Tech programme' from being counted as 4 years of work experience.

    Values are clamped to _EXP_MAX_YEARS (15) so that template phrases like
    '20+ years of industry experience' don't wildly inflate a junior candidate's
    score.
    """
    work_matches:     list[float] = []
    fallback_matches: list[float] = []

    for line in text.splitlines():
        is_academic = bool(_EXP_ACADEMIC_RE.search(line))
        for pat in _EXP_WORK_PATTERNS:
            for m in re.finditer(pat, line, re.IGNORECASE):
                val = min(float(m.group(1)), _EXP_MAX_YEARS)   # clamp in place
                if is_academic:
                    fallback_matches.append(val)
                else:
                    work_matches.append(val)

    if work_matches:
        return max(work_matches)
    if fallback_matches:
        return max(fallback_matches)
    # Last-resort fallback: compute from date ranges in employment history
    date_exp = _extract_date_range_experience(text)
    if date_exp > 0.0:
        return min(date_exp, _EXP_MAX_YEARS)
    return 0.0


def _clean_edu_entry(raw: str) -> str:
    """Clean a raw education string extracted from PDF text.

    Removes leading lowercase word-fragments produced when pdfplumber rejoins
    a hyphenated line-break from a multi-column layout.  Example:
      "ment Systems..." → remove leading "ment "
      "MENTS" → discard entirely (too short / uppercase noise)
    Also collapses embedded newlines and excess whitespace.
    """
    # Collapse any embedded newlines / tabs into a single space
    raw = re.sub(r"[\n\r\t]+", " ", raw)
    raw = re.sub(r" {2,}", " ", raw).strip()

    # Strip leading lowercase-only fragments (mid-word artefact from PDF joins)
    raw = _EDU_LEADING_FRAGMENT_RE.sub("", raw).strip()

    # Remove any remaining non-printable / control characters
    raw = re.sub(r"[\x00-\x1f\x7f]", " ", raw).strip()

    return raw


def extract_education(text: str) -> list:
    """Extract education qualifications (degrees, diplomas) from resume text.

    Strategy: use DEGREE_PATTERN to *locate* degree mentions, then extract the
    **complete line** (up to 120 chars) that surrounds each match.  This avoids
    the common artefact where the regex group starts mid-word because pdfplumber
    has rejoined a hyphenated line-break from a multi-column PDF layout.

    Each extracted line is cleaned to remove leading word-fragments, embedded
    newlines, and short noise strings before being returned.
    """
    found: list[str] = []
    seen: set[str] = set()

    for m in DEGREE_PATTERN.finditer(text):
        match_start = m.start()

        # Walk backwards in the text to find the start of the current line
        line_start = text.rfind("\n", 0, match_start)
        line_start = line_start + 1 if line_start >= 0 else 0

        # Walk forwards to find the end of the current line
        line_end = text.find("\n", match_start)
        if line_end == -1:
            line_end = len(text)

        # Cap the extracted segment to 120 chars around the match start
        raw_line = text[line_start:min(line_end, line_start + 120)]

        entry = _clean_edu_entry(raw_line)

        # Discard fragments that are too short or are pure-uppercase noise like "MENTS"
        if len(entry) < _EDU_MIN_LEN:
            continue
        # Discard entries that are ALL uppercase and very short (section-header artefacts)
        if entry.isupper() and len(entry) < 30:
            continue

        key = entry.lower()
        if key not in seen:
            seen.add(key)
            found.append(entry)

    return found


def extract_name_from_text(text: str) -> str:
    """Extract the candidate's name from the top of the resume text.

    Heuristic: the name is almost always in the first 5 non-empty lines.
    A valid name line must be:
      - 2 to 5 words long
      - Only letters, spaces, hyphens, periods, and apostrophes
      - Not a section heading (RESUME, CURRICULUM VITAE, etc.)
      - Not contain digits, emails, phone numbers, or URLs

    Returns 'Unknown' if no plausible name is found.
    """
    lines = text.strip().splitlines()
    candidates = []

    for line in lines[:15]:  # scan first 15 raw lines
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue

        # Skip lines with emails, phone numbers, or URLs
        if "@" in stripped or re.search(r"\d{5,}", stripped):
            continue
        if re.search(r"https?://|www\.", stripped, re.IGNORECASE):
            continue

        # Skip known section headings
        if _NAME_NOISE_RE.fullmatch(stripped.strip("  \t:-•|/")):
            continue

        # Check if it looks like a name: letters/spaces/hyphens only, 2-5 words
        clean = re.sub(r"[^A-Za-z .\-']", "", stripped).strip()
        if not clean or len(clean) < 3:
            continue

        words = clean.split()
        if len(words) < 2 or len(words) > 5:
            continue

        # Must match valid name pattern
        if not _NAME_VALID_RE.match(clean):
            continue

        # Prefer title-case or ALL-CAPS lines (typical for names)
        # The first high-confidence match wins (earliest line = most likely the name)
        if (clean.istitle() or clean.isupper()) and not candidates:
            candidates.insert(0, clean)  # first high confidence → front
        else:
            candidates.append(clean)

        if len(candidates) >= 3:
            break

    if not candidates:
        return "Unknown"

    # Return the best candidate, title-cased
    best = candidates[0]
    return best.title() if best.isupper() else best


def extract_links(text: str) -> dict:
    """Extract LinkedIn, GitHub, and portfolio URLs from resume text.

    Returns a dict with keys 'linkedin', 'github', 'portfolio'.
    Each value is the full URL string, or an empty string if not found.
    """
    links = {"linkedin": "", "github": "", "portfolio": ""}

    m = _LINKEDIN_RE.search(text)
    if m:
        raw = m.group(0)
        links["linkedin"] = raw if raw.startswith("http") else f"https://{raw}"

    m = _GITHUB_RE.search(text)
    if m:
        # Exclude common false positives (github.com/topics, github.com/features, etc.)
        username = m.group(1).lower()
        if username not in {"topics", "features", "explore", "settings", "login", "signup", "about"}:
            raw = m.group(0)
            links["github"] = raw if raw.startswith("http") else f"https://{raw}"

    m = _PORTFOLIO_RE.search(text)
    if m:
        links["portfolio"] = m.group(1)

    return links


def _extract_date_range_experience(text: str) -> float:
    """Calculate total years of professional experience from date ranges.

    Scans for patterns like 'Jan 2020 – Dec 2023', '2019 - Present' and
    sums non-overlapping employment periods.  Academic lines are excluded.

    Returns 0.0 if no valid date ranges are found.
    """
    periods = []
    now = datetime.now()

    for line in text.splitlines():
        # Skip academic context lines
        if _EXP_ACADEMIC_RE.search(line):
            continue

        for m in _DATE_RANGE_RE.finditer(line):
            start_year_str = m.group(1)
            end_str = m.group(2).strip().lower()

            try:
                start_year = int(start_year_str)
            except (ValueError, TypeError):
                continue

            if end_str in {"present", "current", "ongoing", "now"} or end_str.startswith("till"):
                end_year = now.year
            else:
                try:
                    end_year = int(end_str)
                except (ValueError, TypeError):
                    continue

            # Sanity checks
            if start_year < 1970 or start_year > now.year:
                continue
            if end_year < start_year or end_year > now.year + 1:
                continue

            periods.append((start_year, end_year))

    if not periods:
        return 0.0

    # Merge overlapping periods to avoid double-counting
    periods.sort()
    merged = [periods[0]]
    for start, end in periods[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    total = sum(end - start for start, end in merged)
    return round(float(total), 1)