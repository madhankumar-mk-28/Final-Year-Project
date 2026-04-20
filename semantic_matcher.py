"""
semantic_matcher.py — Transformer-based semantic similarity between resumes and job descriptions.

This module uses sentence-transformer embedding models to compute how semantically
similar each resume is to a given job description. It supports three interchangeable
models, all running fully offline after initial download.

Supported Models:
    1. MPNet  (multi-qa-mpnet-base-dot-v1)              — 768-dim, balanced
    2. MxBai  (mixedbread-ai/mxbai-embed-large-v1)      — 1024-dim, highest absolute scores
    3. Arctic (Snowflake/snowflake-arctic-embed-m-v1.5)  — 768-dim, asymmetric retrieval

Two-Pass Embedding Strategy:
    Pass 1 — Full text (40% weight): 300-word overlapping chunks, stride 50
    Pass 2 — Key sections (60% weight): Skills, Experience, Projects, Education

Performance Features:
    - Batched encoding: all chunks from all resumes in ONE model.encode() call
    - LRU model cache: max 2 models in memory simultaneously
    - JD embedding cache: SHA-256 keyed FIFO, max 32 entries
    - OOM recovery: auto-retry at batch_size=8 if primary batch fails
    - MPS/CUDA/CPU auto-detection for optimal hardware utilisation

Public API:
    rank_resumes_by_similarity(resume_texts, jd, model_name) → list[dict]
    compute_skill_semantic_matches(cand_skills, req_skills, model_name) → set
    embed_text(text, model_name) → np.ndarray
    embed_job_description(jd, model_name) → np.ndarray

Runs fully offline after initial model download. OS-independent.
"""

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
    """Release Apple Metal GPU compute buffers after batch operations.

    Only runs on macOS with Apple Silicon (MPS backend). No-op on CPU/CUDA.
    Prevents memory accumulation during sequential model.encode() calls.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL IDENTIFIERS                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

MPNET_MODEL  = "multi-qa-mpnet-base-dot-v1"             # 768-dim, trained on 215M QA pairs
MXBAI_MODEL  = "mixedbread-ai/mxbai-embed-large-v1"     # 1024-dim, 335M params, MTEB #1
ARCTIC_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5" # 768-dim, asymmetric retrieval

DEFAULT_MODEL = MPNET_MODEL

# Generic fallback threshold used when model key is not in SKILL_THRESHOLD_BY_MODEL
SKILL_SEMANTIC_THRESHOLD = 0.62


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PER-MODEL SKILL-MATCHING THRESHOLDS                                   ║
# ║                                                                        ║
# ║  Derived from embedding-space distribution analysis across 69 resumes. ║
# ║  Each threshold sits at the natural valley between "close variants"     ║
# ║  (e.g. "communication" ↔ "verbal communication") and "near-misses"     ║
# ║  (e.g. "leadership" ↔ "management") in that model's cosine range.      ║
# ║                                                                        ║
# ║  Model    Threshold  Cosine Range   Rationale                          ║
# ║  ─────    ─────────  ────────────   ─────────                          ║
# ║  MPNet    0.63       0.30–0.95      Valley at 0.62–0.63                ║
# ║  MxBai    0.60       0.25–0.90      Runs ~0.03 lower than MPNet        ║
# ║  Arctic   0.66       0.15–0.80      Compressed range, higher valley    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

SKILL_THRESHOLD_BY_MODEL = {
    MPNET_MODEL:  0.63,
    MXBAI_MODEL:  0.60,
    ARCTIC_MODEL: 0.66,
}

# Models using asymmetric retrieval training — JD must use a "query" prompt prefix.
# Resumes are "documents" — no prefix needed. Omitting the prefix on the JD side
# causes similarity scores to drop 40–60% below their correct values.
_QUERY_PROMPT_MODELS: frozenset = frozenset({ARCTIC_MODEL})

# Chunking parameters for long text segmentation
CHUNK_SIZE   = 300    # Maximum words per chunk
CHUNK_STRIDE = 50     # Overlap between consecutive chunks (words)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL CACHE — LRU with max 2 models in memory                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

MAX_CACHED_MODELS = 2
_model_cache: OrderedDict[str, SentenceTransformer] = OrderedDict()
_cache_lock = threading.Lock()

# Per-model loading locks prevent two threads from simultaneously downloading/loading
_loading_locks: Dict[str, threading.Lock] = {}
_loading_locks_meta = threading.Lock()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  JD EMBEDDING CACHE — SHA-256 keyed, FIFO eviction at 32 entries       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_jd_cache: Dict[str, np.ndarray] = {}
_jd_cache_lock = threading.Lock()
_JD_CACHE_MAX = 32


def _get_loading_lock(model_name: str) -> threading.Lock:
    """Return the per-model loading lock, creating it on first request."""
    with _loading_locks_meta:
        if model_name not in _loading_locks:
            _loading_locks[model_name] = threading.Lock()
        return _loading_locks[model_name]


def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load and return a SentenceTransformer model, with LRU caching.

    Loading strategy (offline-first):
        1. Check in-memory LRU cache → instant return if found
        2. Try loading from local HuggingFace cache (no network)
        3. If local cache miss, download from HuggingFace Hub

    Hardware auto-detection:
        - Apple Silicon Mac → MPS (Metal Performance Shaders)
        - NVIDIA GPU → CUDA
        - Fallback → CPU

    Cache eviction: when cache is full (2 models), the least-recently-used
    model is evicted and garbage collected to free GPU/MPS memory.

    Args:
        model_name: HuggingFace model identifier string.

    Returns:
        Loaded SentenceTransformer model ready for encoding.

    Raises:
        RuntimeError: If the model cannot be loaded (provides download instructions).
    """
    # Fast path: model already in cache
    with _cache_lock:
        if model_name in _model_cache:
            _model_cache.move_to_end(model_name)     # Mark as most recently used
            return _model_cache[model_name]

    # Slow path: load model (serialised per-model to prevent duplicate downloads)
    with _get_loading_lock(model_name):
        # Double-check: another thread may have loaded it while we waited
        with _cache_lock:
            if model_name in _model_cache:
                _model_cache.move_to_end(model_name)
                return _model_cache[model_name]

        # Auto-detect the best available hardware
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[SemanticMatcher] Loading model '%s' on %s ...", model_name, device)

        # Try 1: Load from local HuggingFace cache (fully offline)
        try:
            model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            # Try 2: Download from HuggingFace Hub (requires internet)
            logger.info(
                "[SemanticMatcher] Model '%s' not in local cache — downloading now...", model_name
            )
            try:
                model = SentenceTransformer(model_name, device=device, local_files_only=False)
            except Exception as exc:
                logger.error("[SemanticMatcher] Failed to load model '%s': %s", model_name, exc)
                raise RuntimeError(
                    f"Could not load model '{model_name}'. "
                    f"Pre-download: python3 -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\" "
                    f"Detail: {exc}"
                ) from exc

        # Add to cache, evicting the oldest model if cache is full
        with _cache_lock:
            while len(_model_cache) >= MAX_CACHED_MODELS:
                evicted_name, evicted_model = _model_cache.popitem(last=False)
                del evicted_model
                gc.collect()                              # Free Python objects
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()                # Free Metal GPU memory
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()               # Free CUDA GPU memory
                logger.info("[SemanticMatcher] Evicted model '%s' from cache.", evicted_name)
                # Clean up the loading lock for the evicted model
                with _loading_locks_meta:
                    _loading_locks.pop(evicted_name, None)

            _model_cache[model_name] = model
            logger.info(
                "[SemanticMatcher] Model '%s' ready. Cache: %d/%d",
                model_name, len(_model_cache), MAX_CACHED_MODELS,
            )

        return model


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEXT PREPROCESSING AND CHUNKING                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _preprocess(text: str) -> str:
    """Strip non-printable control characters and collapse whitespace.

    Keeps all printable Unicode characters (accented letters, CJK, etc.)
    but removes ASCII control codes that can confuse tokenizers.
    """
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_key_sections(text: str) -> str:
    """Extract job-relevant sections from resume text for focused embedding.

    Identifies sections by heading keywords (Skills, Experience, Projects,
    Education, Certifications, etc.) and captures all text until a noise
    section is reached (Hobbies, References, Declaration, etc.).

    This produces a focused text that weights the most job-relevant content
    more heavily in the semantic similarity computation (60% weight).

    Returns:
        Concatenated key sections text, or the full text if key sections
        are too short (< 100 chars) to be meaningful.
    """
    # Heading detection — intentionally permissive on line length (< 80 chars)
    # to handle multi-column PDF layouts like "SKILLS Python | React | Node"
    header_pattern = re.compile(
        r"^[\s\-\*•#\d\.]*"
        r"(skills?|technical skills?|core competenc|experience|work experience|projects?|"
        r"summary|profile|objective|internship|achievements?|certifications?|"
        r"education|academic|qualifications?|courses?|coursework)"
        r"[\s:\-|·]*",
        re.IGNORECASE,
    )

    # Stop capturing at noise sections that add no semantic signal
    stop_pattern = re.compile(
        r"^[\s\-\*•#]*"
        r"(references?|hobbies|interests|declaration|personal\s+info|"
        r"languages?|extracurricular|activities|awards?\s+and\s+honours?)"
        r"[\s:\-]*$",
        re.IGNORECASE,
    )

    lines     = text.splitlines()
    capturing = False
    key_lines = []

    for line in lines:
        stripped = line.strip()

        # Stop capturing when we hit a noise section heading
        if capturing and stop_pattern.match(stripped):
            capturing = False

        # Start capturing when we find a relevant section heading
        if header_pattern.match(stripped) and len(stripped) < 80:
            capturing = True

        if capturing:
            key_lines.append(line)

    result = " ".join(key_lines).strip()

    # If key sections are too short, use the full text instead
    return result if len(result) > 100 else text


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, stride: int = CHUNK_STRIDE) -> list:
    """Split text into sentence-aware overlapping chunks for embedding.

    Sentences are kept intact within chunks (no mid-sentence splits).
    Adjacent chunks overlap by `stride` words for context continuity.

    This prevents long resumes from being truncated at the model's
    max_seq_length (typically 512 tokens).

    Args:
        text:       Input text to chunk.
        chunk_size: Maximum words per chunk (default 300).
        stride:     Overlap between consecutive chunks in words (default 50).

    Returns:
        List of chunk strings. Returns [text] if text fits in one chunk.
    """
    # Split text into sentences at sentence boundaries or newlines
    sentence_re  = re.compile(r"(?<=[.!?])\s+|\n")
    raw_sentences = sentence_re.split(text)
    sentences    = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    word_counts = [len(s.split()) for s in sentences]
    total_words = sum(word_counts)

    # Short text: no chunking needed
    if total_words <= chunk_size:
        return [text]

    chunks: List[str] = []
    current_sents: List[str] = []
    current_words = 0
    stride_text   = ""                  # Overlap carryover from previous chunk

    for sent, wc in zip(sentences, word_counts):
        # Start a new chunk if adding this sentence would exceed chunk_size
        if current_words + wc > chunk_size and current_sents:
            chunk_text = (stride_text + " " + " ".join(current_sents)).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # Save the last `stride` words for overlap with the next chunk
            all_words   = chunk_text.split()
            stride_text = " ".join(all_words[-stride:]) if len(all_words) > stride else chunk_text
            current_sents = []
            current_words = 0

        current_sents.append(sent)
        current_words += wc

    # Don't forget the final chunk
    if current_sents:
        chunk_text = (stride_text + " " + " ".join(current_sents)).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks if chunks else [text]


def _pool_chunks(embeddings: np.ndarray, use_max: bool = True) -> np.ndarray:
    """Pool multiple chunk embeddings into a single normalised document vector.

    Pooling strategies:
        use_max=True  — Mean-max blend (50/50): default for MPNet and MxBai.
                        Max-pooling captures the strongest signal per dimension.
        use_max=False — Mean-only: used for Arctic (CLS-pool trained).
                        Max-pooling across chunks misaligns with Arctic's CLS space.

    The result is L2-normalised so dot product equals cosine similarity.

    Args:
        embeddings: 2D numpy array of shape (num_chunks, embedding_dim).
        use_max:    Whether to include max-pooling in the blend.

    Returns:
        1D numpy array: unit-normalised pooled embedding vector.
    """
    # Edge case: empty embeddings array
    if embeddings.ndim == 2 and embeddings.shape[0] == 0:
        return np.zeros(embeddings.shape[1])

    # Edge case: 1D or scalar input
    if embeddings.ndim < 2:
        dim = embeddings.shape[-1] if embeddings.size > 0 else 768
        return embeddings.flatten() if embeddings.size > 0 else np.zeros(dim)

    # Compute pooled vector based on strategy
    mean_pool = np.mean(embeddings, axis=0)
    if use_max:
        max_pool = np.max(embeddings, axis=0)
        combined = 0.5 * mean_pool + 0.5 * max_pool      # Equal mean-max blend
    else:
        combined = mean_pool                               # Mean-only for CLS models

    # L2-normalise so dot product == cosine similarity
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EMBEDDING FUNCTIONS                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def embed_text(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a single resume text using chunked pooling.

    Steps:
        1. Preprocess text (strip control chars, collapse whitespace)
        2. Split into sentence-aware overlapping chunks
        3. Encode all chunks with the selected model
        4. Pool chunks into a single document vector (mean-max or mean-only)

    Args:
        text:       Raw resume text to embed.
        model_name: HuggingFace model identifier.

    Returns:
        1D numpy array: unit-normalised embedding vector.
    """
    model  = _get_model(model_name)
    text   = _preprocess(text)
    chunks = _chunk_text(text)

    # Warn if any chunks exceed the model's maximum sequence length
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

    # Arctic uses CLS-pooling — mean-only to match its training approach
    use_max = model_name not in _QUERY_PROMPT_MODELS
    return _pool_chunks(embeddings, use_max=use_max)


def embed_job_description(jd: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a job description with caching and asymmetric model support.

    For asymmetric retrieval models (Arctic), the JD is encoded with a "query"
    prompt prefix. Resumes (documents) never use this prefix.

    Results are cached using a SHA-256 key of (JD text + model name) to avoid
    re-encoding the same JD across multiple screening runs.

    Args:
        jd:         Job description text.
        model_name: HuggingFace model identifier.

    Returns:
        1D numpy array: unit-normalised JD embedding vector (cached).
    """
    # Check cache first (SHA-256 key prevents collisions)
    cache_key = hashlib.sha256((jd + model_name).encode()).hexdigest()
    with _jd_cache_lock:
        if cache_key in _jd_cache:
            logger.debug("[SemanticMatcher] JD cache hit for model '%s'.", model_name)
            return _jd_cache[cache_key]

    # Encode outside the lock so inference doesn't block other threads
    model    = _get_model(model_name)
    jd_clean = _preprocess(jd)
    chunks   = _chunk_text(jd_clean)

    # Asymmetric models (Arctic) require a query prompt prefix on the JD side
    is_query_model = model_name in _QUERY_PROMPT_MODELS
    encode_kwargs  = dict(convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    if is_query_model:
        encode_kwargs["prompt_name"] = "query"
        logger.debug(
            "[SemanticMatcher] Arctic asymmetric mode — encoding JD (%d chunk(s)) with query prefix.",
            len(chunks),
        )
    else:
        logger.debug(
            "[SemanticMatcher] Encoding JD for '%s' — %d chunk(s).", model_name, len(chunks)
        )

    try:
        chunk_embs = model.encode(chunks, **encode_kwargs)
    except Exception as exc:
        logger.error("[SemanticMatcher] JD encoding failed: %s", exc)
        raise RuntimeError(f"Job description encoding error: {exc}") from exc

    # Pool using the same strategy as resume embeddings to keep spaces aligned
    use_max = model_name not in _QUERY_PROMPT_MODELS
    emb = _pool_chunks(chunk_embs, use_max=use_max)

    # Store in cache with FIFO eviction
    with _jd_cache_lock:
        if len(_jd_cache) >= _JD_CACHE_MAX:
            del _jd_cache[next(iter(_jd_cache))]       # Evict oldest entry
        _jd_cache[cache_key] = emb

    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, clamped to [0, 1].

    Handles zero-vector edge cases (returns 0.0 if either vector is zero).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BATCH RANKING — main pipeline entry point                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def rank_resumes_by_similarity(
    resume_texts: dict,
    job_description: str,
    model_name: str = DEFAULT_MODEL,
) -> list:
    """Rank all resumes by two-pass semantic similarity to the job description.

    Two-Pass Strategy:
        Pass 1 — Full text embedding (40% weight): captures broad context
        Pass 2 — Key sections embedding (60% weight): focuses on skills/experience

    All chunks from ALL resumes are batched into a single model.encode() call
    for efficiency (vs. encoding one resume at a time).

    OOM Recovery:
        If a batch encoding fails with an out-of-memory error, the function
        automatically retries with batch_size=8. If that also fails, zero
        vectors are returned and the candidate is flagged with oom_fallback=True.

    Args:
        resume_texts:    dict mapping filename → resume text string.
        job_description: Full job description text.
        model_name:      HuggingFace model identifier.

    Returns:
        List of dicts with keys: filename, similarity_score, oom_fallback.
        Sorted by similarity_score descending.
    """
    logger.info("[SemanticMatcher] Ranking %d resumes using '%s'", len(resume_texts), model_name)
    model  = _get_model(model_name)
    jd_emb = embed_job_description(job_description, model_name)

    # Early return for empty input
    if not resume_texts:
        logger.warning("[SemanticMatcher] No resume texts provided — returning empty.")
        return []

    # Prepare chunk lists for batched encoding
    filenames    = []
    full_chunks  = []           # All chunks from full-text pass
    full_offsets = []           # (start, end) index range per resume in full_chunks
    key_chunks   = []           # All chunks from key-sections pass
    key_offsets  = []           # (start, end) index range per resume in key_chunks

    for filename, text in resume_texts.items():
        filenames.append(filename)
        clean_text = _preprocess(text)

        # Edge case: empty resume → don't encode (would produce misleading non-zero cosine)
        if not clean_text.strip():
            full_offsets.append((len(full_chunks), len(full_chunks)))
            key_offsets.append((len(key_chunks), len(key_chunks)))
            continue

        # Full-text chunks
        fc = _chunk_text(clean_text)
        full_offsets.append((len(full_chunks), len(full_chunks) + len(fc)))
        full_chunks.extend(fc)

        # Key-sections chunks (skills, experience, projects, education)
        key_text = _extract_key_sections(clean_text)
        kc = _chunk_text(key_text)
        key_offsets.append((len(key_chunks), len(key_chunks) + len(kc)))
        key_chunks.extend(kc)

    logger.info(
        "[SemanticMatcher] Batch encoding: %d full-text chunks + %d key-section chunks",
        len(full_chunks), len(key_chunks),
    )

    # Pooling strategy: mean-max for MPNet/MxBai, mean-only for Arctic (CLS-trained)
    _use_max_pool = model_name not in _QUERY_PROMPT_MODELS

    # Adaptive batch size: MxBai is 2× heavier than MPNet, needs smaller batches
    _batch_size = (16 if len(full_chunks) > 200 else 32) if model_name == MXBAI_MODEL else (32 if len(full_chunks) > 500 else 64)

    def _safe_encode(chunks: list, label: str) -> np.ndarray:
        """OOM-resilient encode — retries with smaller batch, falls back to zero vectors."""
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

    # Batch-encode all chunks in two passes
    try:
        all_full_embs = _safe_encode(full_chunks, "full-text")
        _flush_mps_cache()                             # Free GPU memory between passes
        all_key_embs  = _safe_encode(key_chunks,  "key-sections")
        _flush_mps_cache()
    except Exception as exc:
        logger.error("[SemanticMatcher] Batch encoding failed: %s", exc)
        raise RuntimeError(f"Batch embedding error: {exc}") from exc

    # Compute per-resume similarity scores
    results = []
    for i, filename in enumerate(filenames):
        f_start, f_end = full_offsets[i]
        k_start, k_end = key_offsets[i]

        # Empty resumes have equal start/end offsets (no chunks were added)
        is_empty = (f_start == f_end)
        if is_empty:
            results.append({"filename": filename, "similarity_score": 0.0, "oom_fallback": False})
            continue

        # Pool per-resume chunks into single vectors
        full_emb = _pool_chunks(all_full_embs[f_start:f_end], use_max=_use_max_pool)
        key_emb  = _pool_chunks(all_key_embs[k_start:k_end],  use_max=_use_max_pool)

        # Cosine similarity via dot product (both vectors are unit-normalised)
        full_score = float(np.clip(np.dot(full_emb, jd_emb), 0.0, 1.0))
        key_score  = float(np.clip(np.dot(key_emb,  jd_emb), 0.0, 1.0))

        # Weighted combination: 40% full text + 60% key sections
        combined = float(np.clip(0.40 * full_score + 0.60 * key_score, 0.0, 1.0))

        # OOM detection: non-empty resume producing 0.0 score indicates zero vectors
        oom_suspect = combined == 0.0

        results.append({
            "filename":         filename,
            "similarity_score": round(combined, 4),
            "oom_fallback":     oom_suspect,
        })
        logger.debug(
            "[SemanticMatcher] %s — full=%.3f key=%.3f final=%.3f%s",
            filename, full_score, key_score, combined,
            " [OOM?]" if oom_suspect else "",
        )

    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    logger.info("[SemanticMatcher] Ranking complete.")
    return results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SKILL-LEVEL SEMANTIC MATCHING                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_skill_semantic_matches(
    candidate_skills: list,
    required_skills: list,
    model_name: str = DEFAULT_MODEL,
    threshold: Optional[float] = None,
    req_embs: Optional[np.ndarray] = None,
) -> set:
    """Return the set of required skills semantically matched by a candidate's skill list.

    For each required skill, computes cosine similarity against all candidate
    skills using the embedding model. If the highest similarity exceeds the
    per-model threshold, the required skill is considered matched.

    This is a standalone utility — the main pipeline (app.py) uses an inline
    batched version for better performance across all candidates.

    Args:
        candidate_skills: List of skills extracted from the candidate's resume.
        required_skills:  List of skills required by the job description.
        model_name:       Embedding model to use for similarity computation.
        threshold:        Custom match threshold (defaults to per-model value).
        req_embs:         Pre-computed required skill embeddings (optional optimisation).

    Returns:
        Set of required skill strings that were semantically matched.
    """
    if not candidate_skills or not required_skills:
        return set()

    # Use per-model threshold if not explicitly provided
    if threshold is None:
        threshold = SKILL_THRESHOLD_BY_MODEL.get(model_name, SKILL_SEMANTIC_THRESHOLD)

    model = _get_model(model_name)

    # Encode required skills (skip if pre-computed embeddings provided)
    if req_embs is None:
        req_embs = model.encode(
            required_skills, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

    # Encode candidate skills
    cand_embs  = model.encode(
        candidate_skills, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )

    # Compute similarity matrix: required_skills × candidate_skills
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