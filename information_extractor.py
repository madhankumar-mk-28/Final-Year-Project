import os
import re
import logging

logger = logging.getLogger("information_extractor")

SKILLS_DB = {
    # "c" and "r" removed — single-char \b matches produce false positives ("Grade C", "R&D")
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

# Pre-compiled at import — replaces 100+ per-resume regex calls with a single pass
_MULTI_WORD_SKILLS = sorted([s for s in SKILLS_DB if " " in s], key=len, reverse=True)
_SINGLE_WORD_SKILLS = sorted([s for s in SKILLS_DB if " " not in s], key=len, reverse=True)
_SINGLE_SKILL_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _SINGLE_WORD_SKILLS) + r")\b"
)
DEGREE_PATTERN = re.compile(
    r"(b\.?sc|b\.?tech|b\.?e\.?|bca|bba|bachelor|m\.?sc|m\.?tech|m\.?e\.?|mca|mba|master|ph\.?d|diploma)"
    r"[\w\s,.()-]{0,80}",  # Hard cap at 80 chars — prevents consuming entire paragraphs
    re.IGNORECASE,
)


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
    """Derive a clean candidate name from the PDF filename by stripping extensions, noise words, and numbers."""
    base = filename
    while True:
        stem, ext = os.path.splitext(base)
        if ext.lower() in (".pdf", ".doc", ".docx"):
            base = stem
        else:
            break

    noise_words = (
        r"\b(resume|cv|curriculum|vitae|new|latest|final|updated|"
        r"deloitte|profile|portfolio|theme|engineeringresumes|rendercv)\b"
    )
    base = re.sub(noise_words, "", base, flags=re.IGNORECASE)
    base = re.sub(r"\s*[\(\[]\d+[\)\]]", "", base)   # remove trailing (1), [2], etc.
    base = re.sub(r"\-\d+$", "", base)                # remove trailing dash-number
    base = re.sub(r"[_\-\.]", " ", base)
    base = re.sub(r"[^A-Za-z\s]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()

    if not base or len(base) < 2:
        # Raw filename is itself a noise word (e.g. "resume.pdf") — return "Unknown" instead
        fallback = os.path.splitext(filename)[0].replace("_", " ").strip()
        _noise = {"resume", "cv", "final", "new", "latest", "updated", "profile",
                  "curriculum vitae", "final updated resume", "final resume"}
        if fallback.lower() in _noise or not fallback or len(fallback) < 2:
            return "Unknown"
        return fallback.title()

    words = base.split()
    titled = []
    for w in words:
        if w.isupper() and len(w) <= 3:  # preserve short all-caps abbreviations (e.g. "ML")
            titled.append(w)
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


def extract_email(text: str) -> str:
    """Extract the first valid email address found in the resume text."""
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group(0).lower() if m else ""


def _normalize_phone(raw: str) -> str:
    """Normalise a matched phone string into +CC-XXXXXXXXXX or bare 10-digit format."""
    has_plus = raw.startswith("+") or raw.startswith("(+")
    digits = re.sub(r"[^\d]", "", raw)

    if digits.startswith("91") and len(digits) == 12:  # Indian with country code
        return f"+91-{digits[2:]}"
    if len(digits) == 10 and digits[0] in "6789":       # Bare 10-digit Indian mobile
        return digits
    if has_plus and len(digits) >= 11:                  # International
        if len(digits) <= 13:
            cc_len = len(digits) - 10
            return f"+{digits[:cc_len]}-{digits[cc_len:]}"

    return digits if digits else ""


def extract_phone(text: str) -> str:
    """Extract Indian mobile numbers and international numbers from resume text."""
    patterns = [
        r"\+91[\s\-]?[6-9]\d{9}",                                      # +91 Indian
        r"\+91[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{4}",                  # +91 with separators
        r"\b[6-9]\d{4}[\s\-]?\d{5}\b",                                  # 10-digit Indian
        r"\b[6-9]\d{9}\b",                                               # straight 10-digit
        r"\(\+91\)[\s]?[6-9]\d{9}",                                     # (+91) format
        r"\+\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",       # international
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return _normalize_phone(m.group(0).strip())
    return ""


def extract_skills(text: str) -> list:
    """Match skills from SKILLS_DB against resume text using substring for multi-word and regex for single-word."""
    text_lower = text.lower()
    found = set()

    for skill in _MULTI_WORD_SKILLS:   # longest first to prevent partial matches
        if skill in text_lower:
            found.add(skill)

    found.update(_SINGLE_SKILL_RE.findall(text_lower))  # single combined regex pass

    return sorted(found)


def extract_experience_years(text: str) -> float:
    """Extract the maximum years-of-experience figure mentioned in the resume text."""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s+(?:of\s+)?experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?(?:work\s+)?experience",
        r"worked\s+for\s+(\d+(?:\.\d+)?)\s*\+?\s*years?",
    ]
    all_matches: set[str] = set()
    for pat in patterns:
        all_matches.update(re.findall(pat, text, re.IGNORECASE))  # set deduplicates overlapping pattern matches

    return max((float(y) for y in all_matches), default=0.0)


def extract_education(text: str) -> list:
    """Extract education qualifications from resume text using DEGREE_PATTERN."""
    found = []
    for m in DEGREE_PATTERN.finditer(text):
        entry = m.group(0).strip()[:80]
        if entry not in found:
            found.append(entry)
    return found