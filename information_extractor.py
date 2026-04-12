import os
import re
import logging

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

# Degree pattern — 80-char cap prevents it from consuming entire paragraphs
DEGREE_PATTERN = re.compile(
    r"(b\.?sc|b\.?tech|b\.?e\.?|bca|bba|bachelor|m\.?sc|m\.?tech|m\.?e\.?|mca|mba|master|ph\.?d|diploma)"
    r"[\w\s,.()-]{0,80}",
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

    Uses the filename as the candidate's name when the text itself does not
    contain a clear name at the top of the document.
    """
    result = {
        "name":             name_from_filename(filename) if filename else "Unknown",
        "email":            extract_email(text),
        "phone":            extract_phone(text),
        "skills":           extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education":        extract_education(text),
    }
    logger.info(
        "[extractor] %s → name=%s | email=%s | skills=%d | exp=%.1f yrs | edu=%d",
        filename or "?", result["name"], result["email"] or "—",
        len(result["skills"]), result["experience_years"], len(result["education"]),
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
    """
    text_lower = text.lower()
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
    return 0.0


def extract_education(text: str) -> list:
    """Extract education qualifications (degrees, diplomas) from resume text."""
    found = []
    for m in DEGREE_PATTERN.finditer(text):
        entry = m.group(0).strip()[:80]
        if entry not in found:
            found.append(entry)
    return found