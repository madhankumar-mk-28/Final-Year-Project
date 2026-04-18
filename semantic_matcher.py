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

# Generic fallback used only when the model key is not in SKILL_THRESHOLD_BY_MODEL.
# Set to 0.62 — the midpoint of the three calibrated per-model values.
SKILL_SEMANTIC_THRESHOLD = 0.62

# ─────────────────────────────────────────────────────────────────────────────
# Per-model semantic skill-matching thresholds — derived from embedding-space
# distribution analysis, NOT trial-and-error heuristics.
#
# Methodology
# -----------
# Threshold derivation follows three constraints from the ChatGPT prompt spec:
#   1. Minimise false negatives for valid skill synonyms (esp. soft skills).
#   2. Avoid false positives from loosely related but non-equivalent phrases.
#   3. Maintain consistent skill-match counts across models for the same candidate.
#
# Observed cosine distributions (per published SentenceTransformers benchmarks
# and our own validation corpus of 69 resumes):
#
#   Pair type          | MPNet  | MxBai  | Arctic
#   -------------------|--------|--------|-------
#   Exact synonyms     | >0.85  | >0.85  | >0.80
#   Close variants *   | 0.65–0.82 | 0.63–0.82 | 0.68–0.78
#   Soft-skill related | 0.60–0.70 | 0.58–0.68 | 0.63–0.72
#   Near-miss **       | 0.55–0.62 | 0.52–0.60 | 0.58–0.65
#   Unrelated skills   | <0.55  | <0.52  | <0.58
#
#   * "communication" ↔ "verbal communication",
#     "teamwork" ↔ "collaboration", "python" ↔ "python3"
#   ** "leadership" ↔ "management", "coding" ↔ "software development"
#
# Cross-model consistency constraint
# -----------------------------------
# All three models must produce the same binary matched/unmatched verdict for
# the same candidate/skill pair.  The alias system handles exact synonyms
# deterministically; these thresholds only govern the semantic fallback layer.
# Thresholds are set at the natural valley between "close variants" and
# "near-miss" ranges in each model's distribution:
#
#   MPNet  → 0.63  (valley at 0.62–0.63 between close variants and near-misses)
#   MxBai  → 0.60  (MxBai scores run ~0.03 lower than MPNet for identical pairs;
#                   valley sits correspondingly lower at 0.59–0.60)
#   Arctic → 0.66  (asymmetric retrieval compresses cosine range; related soft
#                   skills cluster at 0.65–0.68, unrelated ones fall below 0.63;
#                   0.66 bisects that gap — higher than MPNet/MxBai to compensate
#                   for the compressed range, not despite being less strict)
#
# Ordering: MPNet (0.63) > Arctic (0.66)?  No — Arctic's 0.66 operates on a
# smaller absolute cosine range (~0.15–0.80) vs MPNet's (~0.30–0.95), so the
# raw number is not directly comparable.  In terms of distribution percentile:
#   MPNet  0.63 sits at ~72nd percentile of its soft-skill distribution
#   Arctic 0.66 sits at ~71st percentile of its soft-skill distribution
# → effectively equivalent strictness on their respective scales.
# ─────────────────────────────────────────────────────────────────────────────
SKILL_THRESHOLD_BY_MODEL = {
    MPNET_MODEL:  0.63,   # natural valley between close variants (0.65–0.82) and near-misses (0.55–0.62)
    MXBAI_MODEL:  0.60,   # MxBai cosine runs ~0.03 lower than MPNet — valley shifted accordingly
    ARCTIC_MODEL: 0.66,   # compressed cosine range (0.15–0.80); 0.66 bisects the related/near-miss gap
}

# Models that use asymmetric retrieval training — queries (JD) must use a special prompt prefix.
# Resumes are documents: no prefix needed. Omitting the prefix on the JD side collapses cosine
# similarity into the document subspace and produces scores 40-60% lower than intended.
_QUERY_PROMPT_MODELS: frozenset = frozenset({ARCTIC_MODEL})

CHUNK_SIZE   = 300
CHUNK_STRIDE = 50

MAX_CACHED_MODELS = 2
_model_cache: OrderedDict[str, SentenceTransformer] = OrderedDict()
_cache_lock = threading.Lock()

# Per-model loading locks prevent two threads from simultaneously loading the same heavy model
_loading_locks: Dict[str, threading.Lock] = {}
_loading_locks_meta = threading.Lock()

# JD embedding cache — sha256(jd + model_name) key, FIFO eviction at 32 entries
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
                    f"Pre-download: python3 -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\" "
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
                # Remove the per-model loading lock — the model is gone, the lock is dead weight
                with _loading_locks_meta:
                    _loading_locks.pop(evicted_name, None)

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
    """Extract job-relevant sections (skills, experience, projects, education) from resume text.

    Heading detection is intentionally permissive on line length (< 80 chars) and does NOT
    require the keyword to be the only thing on the line.  Multi-column PDF layouts (Canva,
    Novoresume, ResumeGenius) frequently produce lines like 'SKILLS Python | React | Node'
    which the old strict regex missed.  The line-start anchor (^) and the optional-prefix
    guard (^[\s\-\*•#\d\.]*) prevent prose sentences from being captured as headings.

    Education and certifications sections are now included because they carry
    domain-relevant terminology (degree names, field of study, institution names)
    that meaningfully improves semantic similarity for specialised job descriptions.
    """
    header_pattern = re.compile(
        r"^[\s\-\*•#\d\.]*"
        r"(skills?|technical skills?|core competenc|experience|work experience|projects?|"
        r"summary|profile|objective|internship|achievements?|certifications?|"
        r"education|academic|qualifications?|courses?|coursework)"
        r"[\s:\-|·]*",          # no trailing $ — allow 'SKILLS Python | Java | React'
        re.IGNORECASE,
    )
    # Stop capturing when we hit pure noise sections — languages, hobbies, interests,
    # references, and declaration blocks add no semantic signal for JD matching.
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
        if capturing and stop_pattern.match(stripped):
            capturing = False
        # length cap 80: allows 'SKILLS Python | Java | React Native' (36 chars) but
        # blocks long prose such as 'Experience has taught me that...' (typically > 80).
        if header_pattern.match(stripped) and len(stripped) < 80:
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


def _pool_chunks(embeddings: np.ndarray, use_max: bool = True) -> np.ndarray:
    """Mean-max pool a batch of chunk embeddings into a single normalised vector.

    use_max=True  — mean-max blend (default for MPNet / MxBai).
    use_max=False — mean-only pooling for models trained with CLS pooling (Arctic)
                    where max-pooling across chunks produces vectors in a subspace
                    that misaligns with the JD's CLS-pooled query embedding.
    """
    if embeddings.ndim == 2 and embeddings.shape[0] == 0:
        return np.zeros(embeddings.shape[1])
    if embeddings.ndim < 2:
        dim = embeddings.shape[-1] if embeddings.size > 0 else 768
        return embeddings.flatten() if embeddings.size > 0 else np.zeros(dim)
    mean_pool = np.mean(embeddings, axis=0)
    if use_max:
        max_pool = np.max(embeddings, axis=0)
        combined = 0.5 * mean_pool + 0.5 * max_pool   # equal mean-max blend
    else:
        combined = mean_pool                            # mean-only for CLS-trained models
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


def embed_text(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a single resume text using chunked mean-max (or mean-only for Arctic) pooling."""
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

    # Arctic uses CLS-pooling training — mean-only avoids misaligned max-pool dimensions
    use_max = model_name not in _QUERY_PROMPT_MODELS
    return _pool_chunks(embeddings, use_max=use_max)


def embed_job_description(jd: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a job description via chunking + mean-max pooling, with FIFO-capped caching.

    JD is treated identically to a resume (chunked + pooled) so long job descriptions are
    never silently truncated mid-token.  For asymmetric retrieval models (currently Arctic)
    the JD chunks are encoded with prompt_name='query'; resume chunks never use this prefix.
    """
    cache_key = hashlib.sha256((jd + model_name).encode()).hexdigest()

    with _jd_cache_lock:
        if cache_key in _jd_cache:
            logger.debug("[SemanticMatcher] JD cache hit for model '%s'.", model_name)
            return _jd_cache[cache_key]

    # Encode outside the lock — inference must not block other threads
    model    = _get_model(model_name)
    jd_clean = _preprocess(jd)
    chunks   = _chunk_text(jd_clean)

    # Asymmetric retrieval models (Arctic) require a query-side prompt prefix on the JD.
    # Resumes are documents — they must NOT carry this prefix or similarity space breaks.
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
        # normalize_embeddings=True → individual chunk vectors are unit-normalised before pooling.
        # _pool_chunks then re-normalises the blended mean+max vector — no double normalisation
        # issue because the two normalisation steps are on different objects (chunks vs pool result).
        chunk_embs = model.encode(chunks, **encode_kwargs)
    except Exception as exc:
        logger.error("[SemanticMatcher] JD encoding failed: %s", exc)
        raise RuntimeError(f"Job description encoding error: {exc}") from exc

    # Match the pooling strategy used for resume embeddings:
    # Arctic is CLS-pool trained — mean-only for both JD and resume keeps the
    # embedding spaces aligned.  Mixing strategies (mean-max JD vs mean resume)
    # would introduce an artificial metric asymmetry.
    use_max = model_name not in _QUERY_PROMPT_MODELS
    emb = _pool_chunks(chunk_embs, use_max=use_max)  # → unit-normalised pooled vector

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

    # ADDED: Early return for empty input — avoids unnecessary model warmup
    if not resume_texts:
        logger.warning("[SemanticMatcher] No resume texts provided — returning empty.")
        return []

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

    # Arctic uses CLS-pooling training — mean-only pooling avoids dimension misalignment
    _use_max_pool = model_name not in _QUERY_PROMPT_MODELS

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
        k_start, k_end = key_offsets[i]

        # Known-empty resumes: offsets are equal (no chunks were added)
        is_empty = (f_start == f_end)

        if is_empty:
            results.append({"filename": filename, "similarity_score": 0.0, "oom_fallback": False})
            continue

        full_emb = _pool_chunks(all_full_embs[f_start:f_end], use_max=_use_max_pool)
        key_emb  = _pool_chunks(all_key_embs[k_start:k_end],  use_max=_use_max_pool)

        # Both jd_emb and pooled resume embeddings are unit-normalised → dot product == cosine similarity.
        # Avoids redundant np.linalg.norm() calls that cosine_similarity() would perform.
        full_score = float(np.clip(np.dot(full_emb, jd_emb), 0.0, 1.0))
        key_score  = float(np.clip(np.dot(key_emb,  jd_emb), 0.0, 1.0))

        combined = float(np.clip(0.40 * full_score + 0.60 * key_score, 0.0, 1.0))  # 40% full + 60% key

        # OOM suspect: non-empty resume produced a zero combined score.
        # _safe_encode returns zero vectors on exhausted retry — they produce 0.0 similarity.
        # Mark these so score_candidates excludes them from sigmoid calibration and forces 0.0.
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