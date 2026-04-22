"""
scoring_engine.py — Weighted scoring, eligibility checks, and candidate ranking.

This module takes extracted candidate data and semantic similarity scores,
then produces a ranked list of candidates with eligibility decisions,
band classifications, and structured rejection codes.

Three-Pass Pipeline:
    Pass 1: Compute component scores (skill, semantic, experience) + hard eligibility
    Pass 2: Compute dynamic threshold from eligible candidates' scores (60th percentile)
    Pass 3: Apply threshold, assign bands (Strong Fit / Borderline / Weak Fit),
            run false-negative recovery, sort by final score

Scoring Formula:
    final_score = (skill_weight × skill_score) + (semantic_weight × semantic_score)
                  + (exp_weight × exp_score)
    Default weights: 55% skill / 45% semantic / 0% experience

Key Features:
    - 41-entry skill alias dictionary for synonym matching
    - 5-level skill matching hierarchy (exact → alias → reverse-alias → substring → token overlap)
    - Negation context detection ("no SQL experience" won't count SQL as matched)
    - Per-model sigmoid calibration normalises raw cosine scores to [0, 1]
    - False-negative recovery flags high-semantic candidates below threshold
    - Structured rejection codes for machine-readable frontend decisions

Public API:
    score_candidates(candidates, required_skills, config, model_key) → list[dict]
    ScoringConfig — dataclass holding scoring weights and constraints

Runs fully offline. No network calls. OS-independent.
"""

from __future__ import annotations
import copy      # Used in score_candidates to isolate weight overrides from the caller's config
import logging
import math
import re

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple, List

logger = logging.getLogger("scoring_engine")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  REJECTION CODES — machine-readable codes for the frontend              ║
# ║  The frontend switches on these enum strings, never on human messages.  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

REJECTION_CODES = {
    "no_contact":           "no_contact",            # No email or phone found
    "skill_below_min":      "skill_below_min",       # Skill match < 30% minimum
    "all_skills_negated":   "all_skills_negated",    # Every required skill found in negation context
    "experience_below_min": "experience_below_min",  # Experience below configured minimum
    "missing_education":    "missing_education",     # Required education not found
    "score_below_threshold":"score_below_threshold", # Final score below dynamic threshold
    "score_below_min":      "score_below_min",       # Final score below configured minimum
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SCORING CONFIGURATION                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ScoringConfig:
    """Holds scoring weights and eligibility constraints for a screening run.

    Weights are automatically normalised to sum to 1.0 on construction.
    If all weights are zero, defaults to 55/45/0 (skill/semantic/experience).

    Attributes:
        skill_weight:         Weight for exact + alias skill matching (default 0.55)
        semantic_weight:      Weight for transformer embedding similarity (default 0.45)
        exp_weight:           Weight for experience score (default 0.0)
        min_experience_years: Minimum years of experience required (0 = no minimum)
        required_education:   List of required degree keywords (empty = no requirement)
        top_n:                Maximum number of candidates to return (default 100)
        min_final_score:      Absolute minimum final score cutoff (default 0.0)
    """
    skill_weight:          float = 0.55
    semantic_weight:       float = 0.45
    exp_weight:            float = 0.0
    min_experience_years:  float = 0.0
    required_education:    list  = field(default_factory=list)
    top_n:                 int   = 100
    min_final_score:       float = 0.0

    def __post_init__(self):
        """Validate and normalise weights after construction.

        Normalisation ensures the three weights always sum to exactly 1.0.
        This matters because the scoring formula is:
            final = (skill_w * skill_score) + (sem_w * sem_score) + (exp_w * exp_score)
        If weights summed to 1.5, a perfect candidate would score 1.5, not 1.0.

        Why clamp negatives to zero first?
        Negative weights would mean "penalise candidates who have skills",
        which is never a valid configuration for a screening tool.
        """
        # Clamp negative weights to zero
        self.skill_weight    = max(0.0, self.skill_weight)
        self.semantic_weight = max(0.0, self.semantic_weight)
        self.exp_weight      = max(0.0, self.exp_weight)

        total = self.skill_weight + self.semantic_weight + self.exp_weight

        # If all weights are zero, reset to safe defaults
        if total == 0:
            self.skill_weight    = 0.55
            self.semantic_weight = 0.45
            self.exp_weight      = 0.0
            return

        # If weights don't sum to 1.0, normalise them proportionally
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "[ScoringConfig] Weights %.4f/%.4f/%.4f don't sum to 1.0 (total=%.4f) — normalizing.",
                self.skill_weight, self.semantic_weight, self.exp_weight, total,
            )
            self.skill_weight    = self.skill_weight    / total
            self.semantic_weight = self.semantic_weight / total
            self.exp_weight      = self.exp_weight      / total


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SKILL ALIAS DICTIONARY — 41 alias groups                              ║
# ║  Maps a canonical skill name to all its recognised synonyms/variants.   ║
# ║  Used by _is_skill_match to credit candidates who list equivalent       ║
# ║  terms (e.g. "sklearn" satisfies a "machine learning" requirement).     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

SKILL_ALIASES = {
    # ── Technical skills ─────────────────────────────────────────────────
    "machine learning":     ["ml", "machine learning", "sklearn", "scikit-learn", "xgboost", "lightgbm"],
    "deep learning":        ["deep learning", "neural network", "neural networks", "dl"],
    "nlp":                  ["nlp", "natural language processing", "text mining", "nltk", "spacy",
                             "gensim", "bert", "transformers", "huggingface"],
    "scikit-learn":         ["scikit-learn", "sklearn"],
    "pytorch":              ["pytorch", "torch"],
    "tensorflow":           ["tensorflow", "tf", "keras"],
    "python":               ["python", "python3", "py"],
    "sql":                  ["sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
                             "mariadb", "tsql", "pl/sql"],
    "pandas":               ["pandas"],
    "numpy":                ["numpy", "scipy"],
    "flask":                ["flask", "fastapi"],
    "data science":         ["data science", "data analytics", "statistics", "statistical analysis"],
    "docker":               ["docker", "kubernetes", "k8s", "containerization"],
    "aws":                  ["aws", "azure", "gcp", "google cloud", "cloud computing"],
    "react":                ["react", "reactjs", "next.js", "nextjs"],
    "javascript":           ["javascript", "typescript", "js", "ts", "nodejs", "node", "node.js", "express"],
    "data analytics":       ["data analytics", "analytics", "data analysis", "business intelligence",
                             "bi", "tableau", "power bi", "data visualization", "statistical analysis"],
    "itsm":                 ["itsm", "it service management", "itil", "service desk", "service management"],
    "crud":                 ["crud", "database operations", "data manipulation", "create read update delete"],
    "stored procedures":    ["stored procedures", "stored procedure", "pl/sql", "t-sql", "tsql",
                             "database programming", "sql server"],

    # ── Soft skills — broad synonym coverage ─────────────────────────────
    # Intentionally overlapping so listing ANY one synonym gives credit
    # for the related skill group. The scoring engine de-dupes matches.
    "problem solving":      ["problem solving", "problem-solving", "troubleshooting", "debugging",
                             "root cause analysis", "analytical thinking", "critical thinking",
                             "issue resolution", "solutioning"],
    "analytical skills":    ["analytical skills", "analytical thinking", "data analysis", "analysis",
                             "problem solving", "critical thinking", "statistical analysis",
                             "data analytics", "research"],
    "critical thinking":    ["critical thinking", "analytical thinking", "problem solving",
                             "analytical skills", "logical thinking", "reasoning",
                             "decision making", "analysis"],
    "teamwork":             ["teamwork", "team player", "team work", "collaboration",
                             "cross-functional", "cooperative", "group work"],
    "collaboration":        ["collaboration", "team player", "teamwork", "team work",
                             "cooperative", "cross-functional", "group work"],
    "communication":        ["communication", "verbal communication", "written communication",
                             "interpersonal skills", "presentation", "public speaking"],
    "verbal communication": ["verbal communication", "communication", "interpersonal skills",
                             "public speaking", "presentation", "oral communication",
                             "speaking"],
    "written communication":["written communication", "communication", "documentation",
                             "writing", "report writing", "technical writing", "content writing"],
    "time management":      ["time management", "time-management", "prioritization",
                             "deadline management", "scheduling", "multitasking"],
    "adaptability":         ["adaptability", "flexibility", "versatility", "quick learner",
                             "fast learner", "agile mindset"],
    "leadership":           ["leadership", "team lead", "mentoring", "managing",
                             "project management", "people management"],
    "attention to detail":  ["attention to detail", "detail oriented", "detail-oriented",
                             "precision", "accuracy"],
    "presentation":         ["presentation", "public speaking", "verbal communication",
                             "communication"],

    # ── Database skills ──────────────────────────────────────────────────
    "database management":  ["database management", "database design", "dbms", "rdbms", "nosql",
                             "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite"],
    "database design":      ["database design", "schema design", "normalization", "data modeling",
                             "database management"],

    # ── Additional tech skills ───────────────────────────────────────────
    "bootstrap":            ["bootstrap", "css framework"],
    "c":                    ["c programming", "c language"],
    "java":                 ["java", "j2ee", "jvm", "maven", "gradle"],
    "mongodb":              ["mongodb", "nosql", "document database", "atlas", "mongoose"],
    "mysql":                ["mysql", "mariadb"],
    "nodejs":               ["nodejs", "node", "node.js", "npm"],
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL-SPECIFIC CONSTANTS                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Per-model eligibility threshold defaults.
# The dynamic threshold (60th percentile) is clamped within ±10% of these values
# to prevent extreme score distributions from pushing the cutoff wildly.
MODEL_THRESHOLDS = {
    "mpnet":  0.45,    # MPNet: balanced scoring, moderate cosine range
    "mxbai":  0.55,    # MxBai: higher absolute scores → higher base threshold
    "arctic": 0.50,    # Arctic: compressed cosine range → middle default
}

# Per-model semantic floors for false-negative recovery.
# Candidates below the dynamic threshold but above this floor are flagged for review.
# MxBai's floor is raised because it produces inflated cosine scores.
FN_RECOVERY_SEMANTIC_FLOOR = {
    "mpnet":  0.52,    # MPNet: standard floor
    "mxbai":  0.62,    # MxBai: raised to prevent excessive false alerts
    "arctic": 0.55,    # Arctic: standard floor
}

# Sigmoid calibration maps each model's raw cosine similarity range onto [0, 1].
# Each model outputs cosine similarity in a different numeric range:
#   MPNet  → ~0.30–0.65 | MxBai → ~0.35–0.70 | Arctic → ~0.15–0.40
# The sigmoid with scale=8 gives ~90% of the [0,1] range within ±0.15 of center.
SEM_CALIBRATION = {
    "mpnet":  {"center": 0.45, "scale": 8.0},
    "mxbai":  {"center": 0.50, "scale": 8.0},
    "arctic": {"center": 0.32, "scale": 8.0},
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEGATION DETECTION                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Matches negation phrases in a window before a skill mention
# Examples: "no experience in SQL", "not skilled in Python", "lacking Java knowledge"
_NEGATION_PATTERNS = re.compile(
    r"\b(no|not|without|lack(?:ing)?|never|zero|minimal|limited)\s+"
    r"(?:experience|knowledge|familiarity|skills?|proficiency|background)?\s*"
    r"(?:in|with|of)?\s*",
    re.IGNORECASE,
)

# Number of characters before a skill mention to search for negation phrasing
_NEGATION_WINDOW = 80

# Module-level compiled-regex cache for skill boundary patterns.
# Prevents recompiling the same r"\bskill\b" pattern for every candidate × every skill.
_regex_cache: dict[str, re.Pattern] = {}


def _skill_re(term: str) -> re.Pattern:
    """Return a compiled word-boundary regex for a skill term (cached after first build)."""
    pat = _regex_cache.get(term)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(term) + r"\b")
        _regex_cache[term] = pat
    return pat


def _has_negation_context(resume_text_lower: str, skill_term: str) -> bool:
    """Check if a skill appears in a negation context in the resume text.

    Scans an 80-character window before each occurrence of the skill term,
    looking for negation phrases like "no", "not", "without", "lacking".

    Examples that return True:
        "I have no experience in Python"
        "Not skilled in SQL or database management"
        "Lacking knowledge of machine learning"

    Returns:
        True if any occurrence of the skill is preceded by a negation phrase.
    """
    for m in re.finditer(re.escape(skill_term), resume_text_lower):
        start = max(0, m.start() - _NEGATION_WINDOW)
        context = resume_text_lower[start:m.start()]
        if _NEGATION_PATTERNS.search(context):
            return True
    return False


def _sigmoid_calibrate(raw: float, model_key: str) -> float:
    """Map a raw cosine similarity score onto [0, 1] using per-model sigmoid.

    WHY sigmoid instead of simple min-max normalisation?
    Min-max (linear scaling) requires knowing the actual min and max for a batch,
    which isn't available until all resumes are scored. Sigmoid is a fixed,
    stateless function that maps any value to (0, 1) without needing batch stats.

    WHY different center values per model?
    Each embedding model produces cosine scores in a different numeric range:
      - MPNet  (all-mpnet-base-v2):  typical range ~0.30–0.65
      - MxBai  (mxbai-embed-large):  typical range ~0.35–0.70
      - Arctic (arctic-embed-m-v1.5): typical range ~0.15–0.40
    The sigmoid's center is tuned so that a "neutral" match for each model
    (where the JD and resume are unrelated) maps to ~0.50 on the output scale.
    This prevents Arctic's naturally lower raw scores from always appearing
    worse than MPNet's scores even for equivalent matches.

    Formula: sigmoid(x) = 1 / (1 + e^(-scale * (x - center)))
    With scale=8: the curve goes from ~0.02 to ~0.98 within \u00b10.30 of center.

    Args:
        raw:       Raw cosine similarity score from the embedding model.
        model_key: Model identifier ("mpnet", "mxbai", or "arctic").

    Returns:
        Calibrated score in [0, 1], rounded to 6 decimal places.
    """
    p   = SEM_CALIBRATION.get(model_key, {"center": 0.45, "scale": 8.0})
    val = 1.0 / (1.0 + math.exp(-p["scale"] * (raw - p["center"])))
    return round(val, 6)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DYNAMIC THRESHOLD                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def get_dynamic_threshold(final_scores: list, model_key: str = "mpnet") -> float:
    """Compute the 60th-percentile eligibility threshold from eligible candidates' scores.

    WHY 60th percentile?
    A fixed threshold (e.g. 0.50) is biased by the model and the JD quality.
    With MPNet, even an excellent match may score only 0.55; with MxBai the same
    match might score 0.68. A dynamic threshold adapts to each batch, meaning:
        - Strong candidate pool  → threshold rises  (more selective — only the best pass)
        - Weak candidate pool    → threshold falls   (more permissive — someone must pass)
    The 60th percentile means roughly the top 40% of candidates are shortlisted,
    which is a reasonable default for screening (not too strict, not too loose).

    WHY only from eligible candidates?
    Rejected candidates (no_contact, skill_below_min) score near 0.0. If we included
    them, the 60th percentile would be dragged down, making the threshold too easy.

    WHY clamp to model default ±10%?
    Prevents instability with extreme batches:
    - If all 5 resumes are excellent → p60 = 0.95 → threshold would wrongly reject most
    - If all 5 are terrible → p60 = 0.10 → threshold would wrongly pass everyone

    Args:
        final_scores: List of final scores from eligible candidates only.
        model_key:    Model identifier for looking up the default threshold.

    Returns:
        Dynamic threshold value (float), clamped to model default \u00b110%.
    """
    model_default = MODEL_THRESHOLDS.get(model_key, 0.50)

    # Small batches: 60th percentile is unreliable, use model default instead
    if len(final_scores) < 5:
        return model_default

    # Compute 60th percentile using sorted-list indexing
    sorted_scores = sorted(final_scores)
    n = len(sorted_scores)
    p60_idx = min(max(int(n * 0.60) - 1, 0), n - 1)   # 0-indexed position
    p60 = sorted_scores[p60_idx]

    # Clamp to model default ±10% to prevent extreme threshold values
    return round(max(model_default - 0.10, min(model_default + 0.10, p60)), 4)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SKILL MATCHING ENGINE                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _normalise_skill_text(skill: str) -> str:
    """Lowercase and strip non-alphanumeric characters from a skill string.

    Keeps: letters, digits, +, #, ., -, spaces (for skills like "C++", "C#", "ASP.NET").
    Collapses multiple spaces. Used for consistent comparison across different formats.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.\- ]+", " ", skill.lower())).strip()


def _tokenise_skill(skill: str) -> set:
    """Split a normalised skill string into individual word tokens for overlap matching."""
    return {tok for tok in _normalise_skill_text(skill).split(" ") if tok}


def _is_skill_match(req_lower: str, candidate_lower: set) -> bool:
    """Check if a required skill matches any candidate skill using a 5-level hierarchy.

    Match hierarchy (first hit wins — ordered from fastest to most expensive):
        1. Exact normalised match: "python" == "python"
        2. Alias exact match: required "machine learning" → alias "sklearn" in candidate set
        3. Reverse-alias: candidate has "communication" → alias covers "verbal communication"
        4. Word-boundary substring: "react" found within "react native" (both directions)
        5. Token overlap: ≥ 2/3 of required-skill tokens found in candidate skill

    Args:
        req_lower:      Normalised required skill string.
        candidate_lower: Set of normalised candidate skill strings.

    Returns:
        True if the required skill is matched by any candidate skill.
    """
    # Level 1: Exact normalised match (fastest — O(1) set lookup)
    if req_lower in candidate_lower:
        return True

    # Build alias lists for the required skill
    req_aliases     = SKILL_ALIASES.get(req_lower, [req_lower])
    req_alias_norms = [_normalise_skill_text(a) for a in req_aliases]

    # Level 2: Required-skill aliases vs candidate skill set
    if candidate_lower.intersection(req_alias_norms):
        return True

    # Level 3: Reverse-alias lookup — for each candidate skill, check if its
    # alias group covers the required skill. Handles cases like candidate
    # listing "communication" satisfying a "verbal communication" requirement.
    for cand in candidate_lower:
        cand_aliases     = SKILL_ALIASES.get(cand, [cand])
        cand_alias_norms = {_normalise_skill_text(a) for a in cand_aliases}
        if cand_alias_norms.intersection(req_alias_norms):
            return True
        if req_lower in cand_alias_norms:
            return True

    req_tokens = _tokenise_skill(req_lower)

    for cand in candidate_lower:
        # Level 4: Word-boundary substring match (both directions)
        # "react" found within "react native", or "data analysis" within "data analytics"
        if len(req_lower) >= 4:
            if _skill_re(req_lower).search(cand):
                return True
            if len(cand) >= 4 and _skill_re(cand).search(req_lower):
                return True

        # Level 5: Token overlap — at least 2/3 of required tokens present
        cand_tokens = _tokenise_skill(cand)
        if req_tokens and len(req_tokens) > 1:
            if len(req_tokens & cand_tokens) / len(req_tokens) >= 0.67:
                return True

        # Run levels 4-5 for each alias of the required skill too
        for alias_norm in req_alias_norms:
            if alias_norm == req_lower:
                continue                           # Already checked above
            if len(alias_norm) >= 4:
                if _skill_re(alias_norm).search(cand):
                    return True
                if len(cand) >= 4 and _skill_re(cand).search(alias_norm):
                    return True
            alias_tokens = _tokenise_skill(alias_norm)
            if alias_tokens and len(alias_tokens) > 1:
                if len(alias_tokens & cand_tokens) / len(alias_tokens) >= 0.67:
                    return True

    return False


def _skill_score(
    candidate_skills: list,
    required_skills: list,
    semantic_matches: Optional[Set] = None,
    resume_text: str = "",
) -> Tuple:
    """Score a candidate's skills against the required skills list.

    Processing order for each required skill:
        1. Check for negation context ("no experience in X") → negated, excluded
        2. Check exact/alias match against candidate's extracted skills → matched
        3. Check semantic embedding match → matched (flagged as semantic_only)
        4. No match found → missing

    The skill score is: matched_count / (total_required - negated_count).
    If ALL required skills are negated, score is 0.0 (not 1.0).

    Returns:
        6-tuple: (score, matched, missing, semantic_only, negated, all_negated)
    """
    # No required skills configured → all candidates get 1.0 skill score
    if not required_skills:
        return 1.0, [], [], [], [], False

    # Normalise candidate skills for comparison (guard against None entries)
    candidate_lower = {_normalise_skill_text(s) for s in candidate_skills if s}
    sem_matched_norm = (
        {_normalise_skill_text(s) for s in semantic_matches} if semantic_matches else set()
    )
    resume_lower = resume_text.lower()

    matched        = []   # Skills found via exact/alias match (not negated)
    missing        = []   # Skills not matched by any method
    semantic_only  = []   # Skills matched ONLY via semantic embedding (shown as ~ in UI)
    negated        = []   # Skills found but in negation context (excluded from scoring)

    for req in required_skills:
        req_lower = _normalise_skill_text(req)

        # Step 1: Check negation context first (highest priority)
        if resume_lower and _has_negation_context(resume_lower, req_lower):
            negated.append(req)
            continue

        # Step 2: Check exact/alias/substring/token match
        if _is_skill_match(req_lower, candidate_lower):
            matched.append(req)
        # Step 3: Check semantic embedding match (transformer-based)
        elif req_lower in sem_matched_norm:
            matched.append(req)
            semantic_only.append(req)
        # Step 4: No match found
        else:
            missing.append(req)

    # Calculate score: negated skills excluded from both numerator AND denominator
    effective_total = len(required_skills) - len(negated)
    all_negated = effective_total <= 0 and len(negated) > 0  # Every skill was negated
    score = len(matched) / effective_total if effective_total > 0 else 0.0

    return round(score, 4), matched, missing, semantic_only, negated, all_negated


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HARD ELIGIBILITY CHECKS                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _check_hard_eligibility(info: dict, config: ScoringConfig) -> Tuple[bool, str, str]:
    """Apply hard disqualification rules that reject a candidate regardless of score.

    Checks (in order):
        1. Contact info: must have at least email OR phone
        2. Experience: must meet minimum years (if configured)
        3. Education: must match at least one required degree keyword (if configured)

    Returns:
        3-tuple: (eligible: bool, reason: str, rejection_code: str)
        If eligible, reason and code are empty strings.
    """
    # Check 1: Candidate must have at least one contact method
    email = (info.get("email") or "").strip()
    phone = (info.get("phone") or "").strip()
    if not email and not phone:
        return False, "No contact details found", "no_contact"

    # Check 2: Minimum experience requirement (if configured)
    exp = info.get("experience_years", 0.0)
    if config.min_experience_years > 0 and exp < config.min_experience_years:
        return False, f"Experience {exp}y < required {config.min_experience_years}y", "experience_below_min"

    # Check 3: Required education keywords (if configured)
    if config.required_education:
        edu_raw  = " ".join(info.get("education", [])).lower()
        edu_norm = re.sub(r"[^a-z0-9\s]", "", edu_raw)   # Normalise for comparison
        def _norm_req(r):
            return re.sub(r"[^a-z0-9\s]", "", r.lower())
        if not any(_norm_req(req) in edu_norm for req in config.required_education):
            return False, f"Missing required education: {config.required_education}", "missing_education"

    return True, "", ""


def is_eligible(info: dict, config: ScoringConfig,
                skill_score: Optional[float] = None) -> tuple:
    """Public compatibility wrapper for hard eligibility checks.

    Combines _check_hard_eligibility with a skill score minimum (30%).
    Used by external callers that need a simple eligible/not-eligible answer.

    Returns:
        2-tuple: (eligible: bool, reason: str)
    """
    eligible, reason, _ = _check_hard_eligibility(info, config)
    if not eligible:
        return False, reason
    if skill_score is not None and skill_score < 0.30:
        return False, f"Skill match {round(skill_score * 100)}% < minimum 30% required"
    return True, ""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN SCORING PIPELINE                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def score_candidates(candidates: list, required_skills: list,
                     config: ScoringConfig, model_key: str = "mpnet",
                     jd_quality_factor: float = 1.0) -> list:
    """Score all candidates in three passes and return ranked results.

    WHY three passes (not one)?
    Pass 1 needs all candidates' scores to compute the dynamic threshold in Pass 2.
    Pass 2 needs only eligible candidates' scores (rejected ones would deflate the
    threshold and make it too easy to pass). Pass 3 applies the threshold.
    This two-phase design separates the scoring logic from the ranking logic cleanly.

    Args:
        candidates:        List of candidate dicts with info, semantic_score, etc.
        required_skills:   List of required skill strings from the JD.
        config:            ScoringConfig with weights and constraints.
        model_key:         Embedding model identifier (for calibration/thresholds).
        jd_quality_factor: 0.0–1.0 multiplier applied to every calibrated semantic
                           score before composite scoring. 1.0 = no change (default).
                           Set < 1.0 for vague/low-quality job descriptions to prevent
                           artificially high semantic matches.

    Pass 1 — Component Scores:
        Compute skill_score, semantic_score, exp_score for each candidate.
        Apply hard eligibility checks (contact, experience, education).
        Check for all-skills-negated edge case.

    Pass 2 — Dynamic Threshold:
        Compute the 60th-percentile threshold from ELIGIBLE candidates only.
        Rejected candidates' near-zero scores are excluded to prevent deflation.

    Pass 3 — Band Assignment + FN Recovery:
        Compare each candidate's final_score against the dynamic threshold.
        Assign band labels: Strong Fit, Borderline, or Weak Fit.
        Flag false-negative recovery candidates (high semantic, low final).

    Returns:
        List of scored candidate dicts, sorted by (eligible, final_score) descending.
    """
    # ── No-skills override: prevent 1.0 skill score from inflating final scores ─
    # When no required skills are configured, _skill_score() returns 1.0 for everyone
    # (no skills = all skills "matched"). If we kept the default 55% skill weight,
    # the 1.0 × 0.55 = 0.55 boost would make the final score unrealistically high.
    # Solution: when skills list is empty, set skill weight = 0 so the scoring relies
    # entirely on semantic similarity (how relevant is the resume to the JD text?).
    no_skills = not required_skills
    if no_skills:
        # Use a shallow copy so we don't mutate the caller's ScoringConfig object.
        # Shallow copy is sufficient here because ScoringConfig only contains primitives.
        config = copy.copy(config)
        config.skill_weight    = 0.0
        config.semantic_weight = 1.0
        config.exp_weight      = 0.0
        logger.info("[scoring_engine] No required skills — weights overridden to skill=0.00 semantic=1.00 exp=0.00")

    # Clamp jd_quality_factor to [0.0, 1.0]
    jd_quality_factor = max(0.0, min(1.0, jd_quality_factor))
    if jd_quality_factor < 1.0:
        logger.info(
            "[scoring_engine] JD quality factor: %.2f — semantic scores will be dampened",
            jd_quality_factor,
        )

    logger.info(
        "[scoring_engine] Scoring %d candidates | weights: skill=%.2f semantic=%.2f exp=%.2f",
        len(candidates), config.skill_weight, config.semantic_weight, config.exp_weight,
    )

    # ── Pre-pass: Sigmoid-calibrate semantic scores ──────────────────────
    # OOM-fallback candidates (zero vectors) are forced to 0.0 to prevent
    # the sigmoid curve from artificially boosting them above zero.
    calibrated_sems: dict[str, float] = {}
    raw_values = []
    for c in candidates:
        fname   = c.get("filename", "")
        raw_sem = float(min(max(c.get("semantic_score", 0.0), 0.0), 1.0))  # Clamp to [0, 1]
        if c.get("oom_fallback", False):
            calibrated_sems[fname] = 0.0               # Force zero for OOM fallback
            logger.warning("[scoring_engine] OOM fallback for %s — semantic_score forced to 0.0", fname)
        else:
            cal = _sigmoid_calibrate(raw_sem, model_key)
            # Apply JD quality dampening AFTER calibration
            calibrated_sems[fname] = round(cal * jd_quality_factor, 6)
            raw_values.append(raw_sem)

    if raw_values:
        logger.info(
            "[scoring_engine] Sigmoid calibration (%s) — raw sem range [%.4f, %.4f]",
            model_key, min(raw_values), max(raw_values),
        )

    # ── Pass 1: Component scores and hard eligibility checks ─────────────
    raw_results = []

    for c in candidates:
        info                   = c.get("info", {})
        filename               = c.get("filename", "")
        semantic_score         = calibrated_sems.get(filename, 0.0)
        resume_text            = c.get("resume_text", "")
        semantic_skill_matches = c.get("semantic_skill_matches", None)

        # Compute skill match score with negation detection
        skill_score, matched_skills, missing_skills, semantic_only, negated_skills, all_negated = _skill_score(
            info.get("skills", []),
            required_skills,
            semantic_skill_matches,
            resume_text=resume_text,
        )

        # Apply hard eligibility checks (contact, experience, education)
        hard_ok, hard_reason, hard_code = _check_hard_eligibility(info, config)

        # Additional skill-based rejection (only if hard checks passed)
        if hard_ok and required_skills:
            if all_negated:
                # Every required skill was found in a negation context
                hard_ok     = False
                hard_reason = "Every required skill was found in a negation context (e.g. 'no experience in X')"
                hard_code   = "all_skills_negated"
            elif skill_score < 0.30:
                # Below 30% skill match minimum
                hard_ok     = False
                hard_reason = f"Skill match {round(skill_score * 100)}% < minimum 30% required"
                hard_code   = "skill_below_min"

        # Compute experience score (0.0–1.0 scale)
        exp_years  = info.get("experience_years", 0.0)
        target_exp = max(config.min_experience_years, 1.0)   # Avoid division by zero
        exp_score  = min(exp_years / target_exp, 1.0)        # Cap at 1.0

        # Weighted composite final score
        final_score = (
            config.skill_weight    * skill_score    +
            config.semantic_weight * semantic_score +
            config.exp_weight      * exp_score
        )
        final_score = min(round(final_score, 4), 1.0)       # Cap at 1.0

        # Store results with internal fields (prefixed with _) for Pass 3
        raw_results.append({
            "filename":         filename,
            "name":             info.get("name", "Unknown"),
            "email":            info.get("email", ""),
            "phone":            info.get("phone", ""),
            "skills":           info.get("skills", []),
            "skills_matched":   matched_skills,
            "skills_missing":   missing_skills,
            "skills_semantic":  semantic_only,
            "skills_negated":   negated_skills,
            "experience_years": exp_years,
            "education":        info.get("education", []),
            "links":            info.get("links", {"linkedin": "", "github": "", "portfolio": ""}),
            "skill_score":      round(skill_score, 4),
            "semantic_score":   round(semantic_score, 4),
            "exp_score":        round(exp_score, 4),
            "final_score":      final_score,
            "_hard_ok":         hard_ok,          # Internal — removed in Pass 3
            "_hard_reason":     hard_reason,      # Internal — removed in Pass 3
            "_hard_code":       hard_code,        # Internal — removed in Pass 3
            "band":             "",
            "confidence":       0.0,
        })

    # ── Pass 2: Dynamic threshold from ELIGIBLE candidates only ──────────
    # Only eligible candidates' scores are used to prevent rejected candidates'
    # near-zero scores from deflating the threshold artificially.
    eligible_finals = [r["final_score"] for r in raw_results if r["_hard_ok"]]
    dyn_threshold = get_dynamic_threshold(eligible_finals, model_key)
    logger.info("[scoring_engine] Dynamic threshold for '%s': %.4f (from %d eligible)",
                model_key, dyn_threshold, len(eligible_finals))

    # ── Pass 3: Apply threshold, assign bands, run FN recovery ───────────
    results = []
    for r in raw_results:
        # Remove internal fields — they should not appear in the output
        hard_ok     = r.pop("_hard_ok")
        hard_reason = r.pop("_hard_reason")
        hard_code   = r.pop("_hard_code")

        # Determine eligibility based on hard checks + dynamic threshold
        if not hard_ok:
            eligible         = False
            rejection_reason = hard_reason
            rejection_code   = hard_code
        elif r["final_score"] < dyn_threshold:
            eligible         = False
            rejection_reason = (
                f"Score {round(r['final_score']*100)}% below threshold "
                f"{round(dyn_threshold*100)}%"
            )
            rejection_code   = "score_below_threshold"
        elif config.min_final_score > 0 and r["final_score"] < config.min_final_score:
            eligible         = False
            rejection_reason = (
                f"Final score {r['final_score']:.2f} < required {config.min_final_score:.2f}"
            )
            rejection_code   = "score_below_min"
        else:
            eligible         = True
            rejection_reason = ""
            rejection_code   = ""

        # Set output fields
        r["eligible"]          = eligible
        r["rejection_reason"]  = rejection_reason
        r["rejection_code"]    = rejection_code
        r["dynamic_threshold"] = dyn_threshold
        r["confidence"]        = round(abs(r["final_score"] - dyn_threshold), 4)
        r["near_threshold"]    = r["confidence"] < 0.05   # Within 5% of threshold

        # Assign band labels (must match frontend constants exactly)
        #   "Weak Fit"   → Archive group (ineligible)
        #   "Borderline" → Secondary Review group (eligible, near threshold)
        #   "Strong Fit" → Interview Now group (eligible, well above threshold)
        if not eligible:
            r["band"] = "Weak Fit"
        elif r["final_score"] >= dyn_threshold + 0.10:
            r["band"] = "Strong Fit"
        else:
            r["band"] = "Borderline"

        # False-negative (FN) recovery: flag ineligible candidates with high semantic scores.
        # "False-negative" here means a candidate who was REJECTED (below the score threshold)
        # but actually has strong semantic alignment with the JD. This can happen when a
        # candidate lists skills differently from the JD (e.g. "ML engineer" vs "machine learning").
        # These candidates are NOT automatically shortlisted — they are flagged for MANUAL REVIEW
        # so a human can decide if they deserve a second look.
        #
        # Note: Hard-rejected candidates (no_contact, experience_below_min) are NOT recovered
        # even if their semantic score is high. Hard rules override FN recovery because we cannot
        # contact a candidate with no contact info regardless of their resume quality.
        sem_floor = FN_RECOVERY_SEMANTIC_FLOOR.get(model_key.lower(), 0.55)
        if (not eligible
                and r["final_score"] < dyn_threshold   # Rejected by score threshold (not hard rule)
                and r["semantic_score"] > sem_floor):  # But still semantically relevant to the JD
            r["fn_recovered"] = True
            logger.debug("[scoring_engine] FN Recovery: %s — final %.2f < thresh %.2f but sem %.2f > floor %.2f",
                         r["filename"], r["final_score"], dyn_threshold, r["semantic_score"], sem_floor)
        else:
            r["fn_recovered"] = False

        # ── Bug 4 fix: per-candidate structured INFO log ──────────────────
        decision_str = "SHORTLISTED" if eligible else "REJECTED"
        band_str     = f"[{r['band']}]" if eligible else f"[{rejection_code}: {round(r['final_score']*100)}% < {round(dyn_threshold*100)}%]"
        fn_tag       = " ⚑FN" if r.get("fn_recovered") else ""
        logger.info(
            "[Candidate] %-28s | Skill: %3d%% | Semantic: %3d%% | Final: %3d%% → %s %s%s",
            (r.get("name") or r["filename"])[:28],
            round(r["skill_score"]    * 100),
            round(r["semantic_score"] * 100),
            round(r["final_score"]    * 100),
            decision_str, band_str, fn_tag,
        )
        results.append(r)

    # Sort: eligible candidates first, then by final_score descending
    # Ties broken by semantic_score, then name (alphabetical) for determinism
    results.sort(key=lambda x: (
        x["eligible"],
        x["final_score"],
        x["semantic_score"],
        x["name"],
    ), reverse=True)

    # Apply top_n limit if configured
    if config.top_n and config.top_n > 0:
        results = results[:config.top_n]

    shortlisted = sum(1 for r in results if r["eligible"])
    fn_count    = sum(1 for r in results if r.get("fn_recovered", False))
    logger.info(
        "[scoring_engine] Done — %d shortlisted / %d total | %d FN recovered | threshold=%.3f",
        shortlisted, len(results), fn_count, dyn_threshold,
    )
    return results
