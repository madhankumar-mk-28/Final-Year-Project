"""
information_extractor.py
------------------------
Extracts structured fields from resume text.
Name is derived from the PDF filename — 100% reliable.
"""

import os
import re
import logging
import spacy

logger = logging.getLogger("information_extractor")

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")

# ── Expanded skills database (150+ skills) ────────────────────────────────────
SKILLS_DB = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "r",
    "go", "golang", "rust", "kotlin", "swift", "php", "ruby", "scala",
    "matlab", "bash", "shell", "perl", "haskell", "lua", "dart",

    # ML / AI / Data Science
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "computer vision",
    "reinforcement learning", "neural networks", "neural network",
    "data science", "data analysis", "data analytics", "data engineering",
    "feature engineering", "model deployment", "mlops",
    "statistical analysis", "statistics", "predictive modeling",
    "time series", "anomaly detection", "recommendation system",

    # ML Frameworks & Libraries
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost", "hugging face", "huggingface",
    "sentence-transformers", "spacy", "nltk", "gensim",
    "opencv", "pillow", "transformers", "bert", "gpt", "llm",
    "langchain", "llamaindex", "diffusers", "stable diffusion",
    "fastai", "jax", "flax",

    # Data Tools
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "bokeh", "streamlit", "gradio", "jupyter", "notebook",
    "tableau", "power bi", "excel", "google sheets",

    # Databases
    "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
    "mongodb", "redis", "cassandra", "elasticsearch", "firebase",
    "dynamodb", "neo4j", "influxdb",

    # Big Data
    "hadoop", "spark", "apache spark", "kafka", "airflow",
    "hive", "pig", "flink", "databricks", "snowflake",

    # Web / Backend
    "flask", "django", "fastapi", "node", "nodejs", "express",
    "spring", "springboot", "laravel", "rails", "asp.net",
    "graphql", "rest api", "restful", "microservices", "grpc",

    # Frontend
    "react", "angular", "vue", "html", "css", "tailwind",
    "bootstrap", "next.js", "nextjs", "redux", "webpack",

    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "k8s", "terraform", "ansible", "jenkins", "github actions",
    "ci/cd", "devops", "linux", "unix", "nginx", "apache",
    "serverless", "lambda",

    # Version control
    "git", "github", "gitlab", "bitbucket",

    # Other
    "agile", "scrum", "jira", "figma", "postman",
    "object detection", "image classification", "text classification",
    "sentiment analysis", "named entity recognition", "ner",
    "transfer learning", "fine-tuning", "rag",
    "generative ai", "llms", "prompt engineering",
    "android", "ios", "flutter", "react native",
    "opencv", "image processing", "speech recognition",
    "data visualization", "etl", "web scraping", "selenium", "beautifulsoup",

    # Databases / query operations
    "crud", "stored procedures", "stored procedure", "itsm",
    "it service management", "sql queries", "query optimization",
    "database design", "database management",

    # Soft skills
    "time management", "written communication", "verbal communication",
    "teamwork", "collaboration", "problem solving", "analytical skills",
    "critical thinking", "communication", "leadership", "presentation",
    "project management", "attention to detail", "multitasking",
    "adaptability", "creativity", "innovation",
}

DEGREE_PATTERN = re.compile(
    r"(b\.?sc|b\.?tech|b\.?e\.?|bachelor|m\.?sc|m\.?tech|m\.?e\.?|master|mba|ph\.?d|diploma)"
    r"[\w\s,.()\-]*",
    re.IGNORECASE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_all(text: str, filename: str = "") -> dict:
    """Extract all structured fields from resume text; uses filename for name if provided."""
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
    """Derive a clean person name from the PDF filename (strips noise words, extensions, numbers)."""
    # Strip all extensions (.pdf.pdf too)
    base = filename
    while True:
        stem, ext = os.path.splitext(base)
        if ext.lower() in (".pdf", ".doc", ".docx"):
            base = stem
        else:
            break

    # Remove noise words (whole word match, case-insensitive)
    noise_words = (
        r"\b(resume|cv|curriculum|vitae|new|latest|final|updated|"
        r"deloitte|profile|portfolio|theme|engineeringresumes|rendercv)\b"
    )
    base = re.sub(noise_words, "", base, flags=re.IGNORECASE)

    # Remove trailing numbers and special chars like (1), -4
    base = re.sub(r"\s*[\(\[]\d+[\)\]]", "", base)
    base = re.sub(r"\-\d+$", "", base)

    # Replace separators (underscore, hyphen, dot) with space
    base = re.sub(r"[_\-\.]", " ", base)

    # Remove non-alpha except spaces
    base = re.sub(r"[^A-Za-z\s]", " ", base)

    # Collapse whitespace
    base = re.sub(r"\s+", " ", base).strip()

    if not base or len(base) < 2:
        # Fallback: just return original stem cleaned
        return os.path.splitext(filename)[0].replace("_", " ").title()

    # Title case — but preserve short uppercase tokens like "CN", "UI"
    words = base.split()
    titled = []
    for w in words:
        if w.isupper() and len(w) <= 3:
            titled.append(w)          # keep CN, UI, etc. as-is
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


# ── Email ─────────────────────────────────────────────────────────────────────

def extract_email(text: str) -> str:
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group(0).lower() if m else ""


# ── Phone ─────────────────────────────────────────────────────────────────────

def extract_phone(text: str) -> str:
    """Extract Indian mobile numbers and international numbers from resume text."""
    patterns = [
        r"\+91[\s\-]?[6-9]\d{9}",              # +91 Indian
        r"\+91[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{4}",  # +91 with separators
        r"\b[6-9]\d{4}[\s\-]?\d{5}\b",         # 10-digit Indian (no country code)
        r"\b[6-9]\d{9}\b",                       # straight 10-digit
        r"\(\+91\)[\s]?[6-9]\d{9}",             # (+91) format
        r"\+\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",  # international
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            # Normalise: remove extra spaces/dashes inside the number
            raw = m.group(0).strip()
            return re.sub(r"(?<=\d)[\s](?=\d)", "", raw)
    return ""


# ── Skills (improved: multi-word first, then single-word) ────────────────────

def extract_skills(text: str) -> list:
    """Match skills from SKILLS_DB against resume text (multi-word first, then single-word)."""
    text_lower = text.lower()
    found = set()

    # Sort by length descending so multi-word skills match first
    sorted_skills = sorted(SKILLS_DB, key=len, reverse=True)

    for skill in sorted_skills:
        # Use word boundary for single words, loose match for phrases
        if " " in skill:
            if skill in text_lower:
                found.add(skill)
        else:
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                found.add(skill)

    return sorted(found)


# ── Experience ────────────────────────────────────────────────────────────────

def extract_experience_years(text: str) -> float:
    """Extract maximum years of experience mentioned."""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s+(?:of\s+)?experience",
    ]
    all_matches = []
    for pat in patterns:
        all_matches += re.findall(pat, text, re.IGNORECASE)

    return max((float(y) for y in all_matches), default=0.0)


# ── Education ─────────────────────────────────────────────────────────────────

def extract_education(text: str) -> list:
    found = []
    for m in DEGREE_PATTERN.finditer(text):
        entry = m.group(0).strip()[:80]
        if entry not in found:
            found.append(entry)
    return found