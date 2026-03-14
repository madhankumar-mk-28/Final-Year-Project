import re
import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("semantic_matcher")

# ── Model IDs (HuggingFace model names) ──────────────────────────────────────
MPNET_MODEL  = "multi-qa-mpnet-base-dot-v1"
MXBAI_MODEL  = "mixedbread-ai/mxbai-embed-large-v1"
ARCTIC_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"

# Default model used when no model is specified by the caller
DEFAULT_MODEL = MPNET_MODEL

# Ensemble model weights: all three models contribute, favouring stronger ones
ENSEMBLE_WEIGHTS = {
    MPNET_MODEL:  0.35,
    MXBAI_MODEL:  0.40,
    ARCTIC_MODEL: 0.25,
}

# Minimum cosine similarity between two skill phrases to count as a semantic match
SKILL_SEMANTIC_THRESHOLD = 0.60

# Text chunking settings (for handling long resumes)
CHUNK_SIZE   = 300   # words per chunk
CHUNK_STRIDE = 100   # overlap between consecutive chunks

# In-memory model cache: { model_id_string: SentenceTransformer }
# Models are loaded once and reused — never reloaded during a session.
_model_cache: dict = {}


def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load a SentenceTransformer from local cache, or load it the first time.

    Device priority:
      1. Apple MPS (Mac GPU) — best for local Mac development
      2. CUDA  (NVIDIA GPU)  — if on Linux/Windows with GPU
      3. CPU                 — universal fallback
    """
    if model_name not in _model_cache:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.info("[SemanticMatcher] Loading model '%s' on %s ...", model_name, device)
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
        logger.info("[SemanticMatcher] Model '%s' ready.", model_name)

    return _model_cache[model_name]


# ── Text preprocessing helpers ────────────────────────────────────────────────

def _preprocess(text: str) -> str:
    """Clean text — collapse whitespace, strip non-ASCII noise."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def _extract_key_sections(text: str) -> str:
    """
    Extract the most job-relevant sections from a resume.
    (Skills, Experience, Projects, Summary, Certifications)
    These carry more signal than contact info or headers.
    Falls back to full text if no known sections are found.
    """
    header_pattern = re.compile(
        r"(skills|technical skills|experience|work experience|projects?|"
        r"summary|objective|internship|achievements?|certifications?)",
        re.IGNORECASE,
    )
    lines    = text.splitlines()
    capturing = False
    key_lines = []

    for line in lines:
        if header_pattern.search(line) and len(line.strip()) < 60:
            capturing = True
        if capturing:
            key_lines.append(line)

    result = " ".join(key_lines).strip()
    return result if len(result) > 100 else text


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, stride: int = CHUNK_STRIDE) -> list:
    """
    Split text into overlapping word-level chunks.
    Most embedding models have a 512-token limit.
    Chunking + pooling avoids information loss on long resumes.
    """
    words  = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - stride):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks if chunks else [text]


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_text(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a long document using max-pooling over chunks.
    Max-pooling keeps the strongest signal from each part of the document.
    """
    model      = _get_model(model_name)
    text       = _preprocess(text)
    chunks     = _chunk_text(text)
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,   # unit-length vectors — needed for cosine similarity
    )
    return np.max(embeddings, axis=0)


def embed_job_description(jd: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a job description string.
    JDs are usually short enough to fit in one pass (no chunking needed).
    """
    model = _get_model(model_name)
    jd    = _preprocess(jd)
    return model.encode(
        [jd],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors. Returns value in [0, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))


# ── Main ranking function ─────────────────────────────────────────────────────

def rank_resumes_by_similarity(
    resume_texts: dict,
    job_description: str,
    model_name: str = DEFAULT_MODEL,
) -> list:
    """
    Rank resumes by semantic similarity to the job description.

    Two-pass scoring:
      • Full resume vs JD  — 40% weight (broad context)
      • Key sections vs JD — 60% weight (skills / experience are most relevant)

    Args:
        resume_texts:    dict of { filename: extracted_text }
        job_description: Job description string
        model_name:      One of MPNET_MODEL, MXBAI_MODEL, ARCTIC_MODEL

    Returns:
        List of { filename, similarity_score } sorted by score descending.
    """
    logger.info(
        "[SemanticMatcher] Ranking %d resumes using '%s'",
        len(resume_texts), model_name,
    )

    # Embed the job description once and reuse for every resume
    jd_emb = embed_job_description(job_description, model_name)

    results = []

    for filename, text in resume_texts.items():
        clean_text = _preprocess(text)

        # Score 1: full resume text
        full_emb   = embed_text(clean_text, model_name)
        full_score = cosine_similarity(full_emb, jd_emb)

        # Score 2: key resume sections (skills, experience, projects)
        key_text   = _extract_key_sections(clean_text)
        key_emb    = embed_text(key_text, model_name)
        key_score  = cosine_similarity(key_emb, jd_emb)

        # Weighted combination
        combined = float(np.clip(0.40 * full_score + 0.60 * key_score, 0.0, 1.0))

        results.append({
            "filename":         filename,
            "similarity_score": round(combined, 4),
        })

        logger.debug(
            "[SemanticMatcher] %s — full=%.3f key=%.3f final=%.3f",
            filename, full_score, key_score, combined,
        )

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    logger.info("[SemanticMatcher] Ranking complete.")
    return results


def rank_resumes_ensemble(
    resume_texts: dict,
    job_description: str,
    weights: dict | None = None,
) -> list:
    """
    Rank resumes using an ensemble of all three embedding models.

    Each model independently scores every resume.  The final similarity score
    is the weighted average across models, producing a more robust signal than
    any single model alone.

    Args:
        resume_texts:    dict of { filename: extracted_text }
        job_description: Job description string
        weights:         Optional dict { model_id: float }.
                         Defaults to ENSEMBLE_WEIGHTS.

    Returns:
        List of { filename, similarity_score } sorted by score descending.
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    total_weight = sum(weights.values())
    models = list(weights.keys())

    logger.info(
        "[SemanticMatcher] Ensemble ranking %d resumes using %d models: %s",
        len(resume_texts), len(models), models,
    )

    # Accumulate weighted scores per filename
    accumulated: dict = {fname: 0.0 for fname in resume_texts}

    for model_name, weight in weights.items():
        model_results = rank_resumes_by_similarity(resume_texts, job_description, model_name)
        for r in model_results:
            accumulated[r["filename"]] += weight * r["similarity_score"]

    # Normalise and build output list
    results = [
        {
            "filename":         fname,
            "similarity_score": round(float(np.clip(score / total_weight, 0.0, 1.0)), 4),
        }
        for fname, score in accumulated.items()
    ]
    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    logger.info("[SemanticMatcher] Ensemble ranking complete.")
    return results


# ── Skill-level semantic matching ─────────────────────────────────────────────

def compute_skill_semantic_matches(
    candidate_skills: list,
    required_skills: list,
    model_name: str = DEFAULT_MODEL,
    threshold: float = SKILL_SEMANTIC_THRESHOLD,
) -> set:
    """
    Use embedding similarity to find required skills that semantically match
    at least one of the candidate's skills.

    This complements alias-based matching by capturing near-synonyms and
    related concepts that are not listed in the static SKILL_ALIASES table.
    For example:
        • "verbal communication" → semantically matches "communication"
        • "data analytics"       → semantically matches "analytics" / "business intelligence"
        • "problem solving"      → semantically matches "critical thinking"
        • "mysql"                → semantically matches "sql"

    Note: The threshold (default 0.60) is intentionally conservative to avoid
    matching conceptually distinct skills.  Increase it to tighten matching.

    Args:
        candidate_skills: list of skill strings extracted from the resume
        required_skills:  list of skill strings the job requires
        model_name:       embedding model to use for comparison
        threshold:        cosine similarity cut-off (default 0.60)

    Returns:
        Set of required skill strings that have a semantic match.
    """
    if not candidate_skills or not required_skills:
        return set()

    model = _get_model(model_name)

    # Embed all skill phrases in one batch for efficiency
    req_embs  = model.encode(
        required_skills,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cand_embs = model.encode(
        candidate_skills,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Similarity matrix: shape [n_required, n_candidate]
    sim_matrix = np.dot(req_embs, cand_embs.T)

    matched = set()
    for i, req_skill in enumerate(required_skills):
        max_sim       = float(np.max(sim_matrix[i]))
        best_cand_idx = int(np.argmax(sim_matrix[i]))
        best_cand     = candidate_skills[best_cand_idx]
        if max_sim >= threshold:
            matched.add(req_skill)
            logger.debug(
                "[SemanticMatcher] Skill match  '%s' ↔ '%s'  sim=%.3f",
                req_skill, best_cand, max_sim,
            )
        else:
            logger.debug(
                "[SemanticMatcher] Skill no-match '%s' (best=%.3f via '%s')",
                req_skill, max_sim, best_cand,
            )

    return matched