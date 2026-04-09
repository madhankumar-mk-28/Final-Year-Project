from __future__ import annotations
import gc
import hashlib
import re
import logging
import threading
from collections import OrderedDict
from typing import Dict, List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("semantic_matcher")


def _flush_mps_cache():
    """Release MPS Metal compute buffers after batch encode — no-op on CPU/CUDA."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


MPNET_MODEL  = "multi-qa-mpnet-base-dot-v1"
MXBAI_MODEL  = "mixedbread-ai/mxbai-embed-large-v1"
ARCTIC_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"

DEFAULT_MODEL = MPNET_MODEL

SKILL_SEMANTIC_THRESHOLD = 0.65  # fallback only — prefer per-model values below

# Arctic's embedding space places soft skills unusually close; raised from 0.55 to prevent degenerate 1.0 scores
SKILL_THRESHOLD_BY_MODEL = {
    MPNET_MODEL:  0.70,
    MXBAI_MODEL:  0.65,
    ARCTIC_MODEL: 0.72,
}

CHUNK_SIZE   = 300
CHUNK_STRIDE = 50

MAX_CACHED_MODELS = 2
_model_cache: OrderedDict[str, SentenceTransformer] = OrderedDict()
_cache_lock = threading.Lock()

# Per-model loading locks prevent two threads from simultaneously loading the same heavy model
_loading_locks: Dict[str, threading.Lock] = {}
_loading_locks_meta = threading.Lock()

# JD embedding cache — md5(jd + model_name) key, FIFO eviction at 32 entries
_jd_cache: Dict[str, np.ndarray] = {}
_jd_cache_lock = threading.Lock()
_JD_CACHE_MAX = 32


def _get_loading_lock(model_name: str) -> threading.Lock:
    """Return the per-model loading lock, creating it if absent."""
    with _loading_locks_meta:
        if model_name not in _loading_locks:
            _loading_locks[model_name] = threading.Lock()
        return _loading_locks[model_name]


def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Return a cached SentenceTransformer, loading from local cache first to avoid network calls."""
    # Fast path — cache hit
    with _cache_lock:
        if model_name in _model_cache:
            _model_cache.move_to_end(model_name)
            return _model_cache[model_name]

    # Slow path — serialise same-model loads; different models may load in parallel
    with _get_loading_lock(model_name):
        with _cache_lock:  # double-check after acquiring lock
            if model_name in _model_cache:
                _model_cache.move_to_end(model_name)
                return _model_cache[model_name]

        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[SemanticMatcher] Loading model '%s' on %s ...", model_name, device)

        # Offline-first: try local cache before hitting HuggingFace network
        try:
            model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            logger.info(
                "[SemanticMatcher] Model '%s' not in local cache — downloading now...", model_name
            )
            try:
                model = SentenceTransformer(model_name, device=device, local_files_only=False)
            except Exception as exc:
                logger.error("[SemanticMatcher] Failed to load model '%s': %s", model_name, exc)
                raise RuntimeError(
                    f"Could not load model '{model_name}'. "
                    f"Download it first: python -m sentence_transformers download '{model_name}'. "
                    f"Detail: {exc}"
                ) from exc

        with _cache_lock:
            while len(_model_cache) >= MAX_CACHED_MODELS:
                evicted_name, evicted_model = _model_cache.popitem(last=False)
                del evicted_model
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("[SemanticMatcher] Evicted model '%s' from cache.", evicted_name)

            _model_cache[model_name] = model
            logger.info(
                "[SemanticMatcher] Model '%s' ready. Cache: %d/%d",
                model_name, len(_model_cache), MAX_CACHED_MODELS,
            )

        return model


def _preprocess(text: str) -> str:
    """Strip non-printable control characters and collapse whitespace."""
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", text)  # keep printable Unicode intact
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_key_sections(text: str) -> str:
    """Extract job-relevant sections (skills, experience, projects) from resume text."""
    # Line must begin with the keyword and be short enough to be a heading — prevents prose false-triggers
    header_pattern = re.compile(
        r"^[\s\-\*•#\d\.]*"
        r"(skills|technical skills|experience|work experience|projects?|"
        r"summary|objective|internship|achievements?|certifications?)"
        r"[\s:]*$",
        re.IGNORECASE,
    )
    stop_pattern = re.compile(
        r"^(references?|hobbies|interests|declaration|personal\s+info)\s*$",
        re.IGNORECASE,
    )
    lines     = text.splitlines()
    capturing = False
    key_lines = []

    for line in lines:
        stripped = line.strip()
        if capturing and stop_pattern.match(stripped):
            capturing = False
        if header_pattern.match(stripped) and len(stripped) < 60:
            capturing = True
        if capturing:
            key_lines.append(line)

    result = " ".join(key_lines).strip()
    return result if len(result) > 100 else text


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, stride: int = CHUNK_STRIDE) -> list:
    """Split text into sentence-aware overlapping chunks for embedding."""
    sentence_re  = re.compile(r"(?<=[.!?])\s+|\n")
    raw_sentences = sentence_re.split(text)
    sentences    = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    word_counts = [len(s.split()) for s in sentences]
    total_words = sum(word_counts)

    if total_words <= chunk_size:
        return [text]

    chunks: List[str] = []
    current_sents: List[str] = []
    current_words = 0
    stride_text   = ""

    for sent, wc in zip(sentences, word_counts):
        if current_words + wc > chunk_size and current_sents:
            chunk_text = (stride_text + " " + " ".join(current_sents)).strip()
            if chunk_text:
                chunks.append(chunk_text)
            all_words   = chunk_text.split()
            stride_text = " ".join(all_words[-stride:]) if len(all_words) > stride else chunk_text
            current_sents = []
            current_words = 0
        current_sents.append(sent)
        current_words += wc

    if current_sents:
        chunk_text = (stride_text + " " + " ".join(current_sents)).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks if chunks else [text]


def _pool_chunks(embeddings: np.ndarray) -> np.ndarray:
    """Mean-max pool a batch of chunk embeddings into a single normalised vector."""
    if embeddings.ndim == 2 and embeddings.shape[0] == 0:
        return np.zeros(embeddings.shape[1])
    if embeddings.ndim < 2:
        dim = embeddings.shape[-1] if embeddings.size > 0 else 768
        return embeddings.flatten() if embeddings.size > 0 else np.zeros(dim)
    mean_pool = np.mean(embeddings, axis=0)
    max_pool  = np.max(embeddings, axis=0)
    combined  = 0.5 * mean_pool + 0.5 * max_pool  # equal mean-max blend
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


def embed_text(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a single resume text using chunked mean-max pooling."""
    model  = _get_model(model_name)
    text   = _preprocess(text)
    chunks = _chunk_text(text)

    max_seq     = getattr(model, "max_seq_length", 512)
    long_chunks = sum(1 for c in chunks if len(c.split()) > max_seq)
    if long_chunks:
        logger.warning(
            "[SemanticMatcher] %d/%d chunk(s) exceed max_seq_length=%d for '%s' — tail tokens truncated.",
            long_chunks, len(chunks), max_seq, model_name,
        )

    try:
        embeddings = model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    except Exception as exc:
        logger.error("[SemanticMatcher] Encoding failed: %s", exc)
        raise RuntimeError(f"Embedding encoding error: {exc}") from exc

    return _pool_chunks(embeddings)


def embed_job_description(jd: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a job description with FIFO-capped caching to avoid redundant inference."""
    cache_key = hashlib.md5((jd + model_name).encode()).hexdigest()

    with _jd_cache_lock:
        if cache_key in _jd_cache:
            logger.debug("[SemanticMatcher] JD cache hit for model '%s'.", model_name)
            return _jd_cache[cache_key]

    # Encode outside the lock — ~300ms inference must not block other threads
    model    = _get_model(model_name)
    jd_clean = _preprocess(jd)
    try:
        emb = model.encode(
            [jd_clean],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
    except Exception as exc:
        logger.error("[SemanticMatcher] JD encoding failed: %s", exc)
        raise RuntimeError(f"Job description encoding error: {exc}") from exc

    with _jd_cache_lock:
        if len(_jd_cache) >= _JD_CACHE_MAX:
            del _jd_cache[next(iter(_jd_cache))]  # FIFO eviction — lock held
        _jd_cache[cache_key] = emb

    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity clamped to [0, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))


def rank_resumes_by_similarity(
    resume_texts: dict,
    job_description: str,
    model_name: str = DEFAULT_MODEL,
) -> list:
    """Rank resumes by two-pass semantic similarity — 40% full text + 60% key sections, batched."""
    logger.info("[SemanticMatcher] Ranking %d resumes using '%s'", len(resume_texts), model_name)
    model  = _get_model(model_name)
    jd_emb = embed_job_description(job_description, model_name)

    filenames    = []
    full_chunks  = []
    full_offsets = []
    key_chunks   = []
    key_offsets  = []

    for filename, text in resume_texts.items():
        filenames.append(filename)
        clean_text = _preprocess(text)

        # Empty-resume guard: empty text would encode to a non-zero vector (~0.25–0.40 cosine) — assign 0.0 instead
        if not clean_text.strip():
            full_offsets.append((len(full_chunks), len(full_chunks)))
            key_offsets.append((len(key_chunks), len(key_chunks)))
            continue

        fc = _chunk_text(clean_text)
        full_offsets.append((len(full_chunks), len(full_chunks) + len(fc)))
        full_chunks.extend(fc)

        key_text = _extract_key_sections(clean_text)
        kc = _chunk_text(key_text)
        key_offsets.append((len(key_chunks), len(key_chunks) + len(kc)))
        key_chunks.extend(kc)

    logger.info(
        "[SemanticMatcher] Batch encoding: %d full-text chunks + %d key-section chunks",
        len(full_chunks), len(key_chunks),
    )

    # MxBai is 2× heavier than MPNet — use smaller batches to prevent OOM
    _batch_size = (16 if len(full_chunks) > 200 else 32) if model_name == MXBAI_MODEL else (32 if len(full_chunks) > 500 else 64)

    def _safe_encode(chunks: list, label: str) -> np.ndarray:
        """OOM-resilient encode — retries at batch_size=8, falls back to zero vectors."""
        if not chunks:
            dim = model.get_sentence_embedding_dimension() or 768
            return np.zeros((0, dim), dtype=np.float32)
        try:
            return model.encode(
                chunks, convert_to_numpy=True, show_progress_bar=False,
                normalize_embeddings=True, batch_size=_batch_size,
            )
        except (MemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower() or isinstance(exc, MemoryError):
                logger.warning(
                    "[SemanticMatcher] OOM on %s (%d chunks, bs=%d) — retrying bs=8",
                    label, len(chunks), _batch_size,
                )
                gc.collect()
                _flush_mps_cache()
                try:
                    return model.encode(
                        chunks, convert_to_numpy=True, show_progress_bar=False,
                        normalize_embeddings=True, batch_size=8,
                    )
                except Exception as retry_exc:
                    logger.error("[SemanticMatcher] Retry failed for %s: %s — zero vecs", label, retry_exc)
            else:
                logger.error("[SemanticMatcher] Encode error for %s: %s", label, exc)
            dim = model.get_sentence_embedding_dimension() or 768
            return np.zeros((len(chunks), dim), dtype=np.float32)

    try:
        all_full_embs = _safe_encode(full_chunks, "full-text")
        _flush_mps_cache()
        all_key_embs  = _safe_encode(key_chunks,  "key-sections")
        _flush_mps_cache()
    except Exception as exc:
        logger.error("[SemanticMatcher] Batch encoding failed: %s", exc)
        raise RuntimeError(f"Batch embedding error: {exc}") from exc

    results = []

    for i, filename in enumerate(filenames):
        f_start, f_end = full_offsets[i]
        full_score     = cosine_similarity(_pool_chunks(all_full_embs[f_start:f_end]), jd_emb)

        k_start, k_end = key_offsets[i]
        key_score      = cosine_similarity(_pool_chunks(all_key_embs[k_start:k_end]), jd_emb)

        combined = float(np.clip(0.40 * full_score + 0.60 * key_score, 0.0, 1.0))  # 40% full + 60% key sections

        results.append({"filename": filename, "similarity_score": round(combined, 4)})
        logger.debug(
            "[SemanticMatcher] %s — full=%.3f key=%.3f final=%.3f",
            filename, full_score, key_score, combined,
        )

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    logger.info("[SemanticMatcher] Ranking complete.")
    return results


def compute_skill_semantic_matches(
    candidate_skills: list,
    required_skills: list,
    model_name: str = DEFAULT_MODEL,
    threshold: Optional[float] = None,
    req_embs: Optional[np.ndarray] = None,
) -> set:
    """Return the set of required skills semantically matched by the candidate's skill list."""
    if not candidate_skills or not required_skills:
        return set()

    if threshold is None:
        threshold = SKILL_THRESHOLD_BY_MODEL.get(model_name, SKILL_SEMANTIC_THRESHOLD)

    model = _get_model(model_name)

    if req_embs is None:
        req_embs = model.encode(
            required_skills, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
    cand_embs  = model.encode(
        candidate_skills, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    sim_matrix = np.dot(req_embs, cand_embs.T)

    matched = set()
    for i, req_skill in enumerate(required_skills):
        max_sim       = float(np.max(sim_matrix[i]))
        best_cand_idx = int(np.argmax(sim_matrix[i]))
        best_cand     = candidate_skills[best_cand_idx]
        if max_sim >= threshold:
            matched.add(req_skill)
            logger.debug("[SemanticMatcher] Skill match  '%s' ↔ '%s'  sim=%.3f", req_skill, best_cand, max_sim)
        else:
            logger.debug("[SemanticMatcher] Skill no-match '%s' (best=%.3f via '%s')", req_skill, max_sim, best_cand)

    return matched