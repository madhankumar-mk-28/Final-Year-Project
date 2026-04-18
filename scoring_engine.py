from __future__ import annotations
import logging
import math
import re

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple, List

logger = logging.getLogger("scoring_engine")

# Structured rejection codes — frontend switches on these, never parses the human-readable reason string
REJECTION_CODES = {
    "no_contact":           "no_contact",
    "skill_below_min":      "skill_below_min",
    "experience_below_min": "experience_below_min",
    "missing_education":    "missing_education",
    "score_below_threshold":"score_below_threshold",
    "score_below_min":      "score_below_min",
}


@dataclass
class ScoringConfig:
    """Holds scoring weights and eligibility constraints for a screening run."""
    skill_weight:          float = 0.55
    semantic_weight:       float = 0.45
    exp_weight:            float = 0.0
    min_experience_years:  float = 0.0
    required_education:    list  = field(default_factory=list)
    top_n:                 int   = 100
    min_final_score:       float = 0.0

    def __post_init__(self):
        self.skill_weight    = max(0.0, self.skill_weight)
        self.semantic_weight = max(0.0, self.semantic_weight)
        self.exp_weight      = max(0.0, self.exp_weight)
        total = self.skill_weight + self.semantic_weight + self.exp_weight
        if total == 0:
            self.skill_weight    = 0.55
            self.semantic_weight = 0.45
            self.exp_weight      = 0.0
            return
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "[ScoringConfig] Weights %.4f/%.4f/%.4f don't sum to 1.0 (total=%.4f) — normalizing.",
                self.skill_weight, self.semantic_weight, self.exp_weight, total,
            )
            self.skill_weight    = self.skill_weight    / total
            self.semantic_weight = self.semantic_weight / total
            self.exp_weight      = self.exp_weight      / total


SKILL_ALIASES = {
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

    # ── Soft skills — broad synonym coverage ─────────────────────────────────
    # The aliases below intentionally overlap so that a candidate who lists any
    # ONE of: communication / teamwork / collaboration / problem solving  gets
    # credit for the related required-skills group.  The scoring engine de-dupes
    # so no double-counting occurs.

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
    # communication covers both verbal and written — a candidate who lists
    # "communication" satisfies verbal communication AND written communication
    # requirements because it is the parent/superset concept.
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
    # ── Database skills ───────────────────────────────────────────────────────
    "database management":  ["database management", "database design", "dbms", "rdbms", "nosql",
                             "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite"],
    "database design":      ["database design", "schema design", "normalization", "data modeling",
                             "database management"],
    # ── Tech skills ───────────────────────────────────────────────────────────
    "bootstrap":            ["bootstrap", "css framework"],
    "c":                    ["c programming", "c language"],
    "java":                 ["java", "j2ee", "jvm", "maven", "gradle"],
    "mongodb":              ["mongodb", "nosql", "document database", "atlas", "mongoose"],
    "mysql":                ["mysql", "mariadb"],
    "nodejs":               ["nodejs", "node", "node.js", "npm"],
}

# Per-model eligibility threshold defaults.
# The dynamic threshold (60th percentile) is clamped within ±10% of these values
# to prevent extreme score distributions from pushing the cutoff out of range.
MODEL_THRESHOLDS = {
    "mpnet":  0.45,
    "mxbai":  0.55,
    "arctic": 0.50,
}

# Sigmoid calibration maps each model's raw cosine similarity range onto [0, 1].
# Different models output similarity in different numeric ranges:
#   MPNet  → ~0.30 – 0.65
#   MxBai  → ~0.35 – 0.70
#   Arctic → ~0.15 – 0.40  (smaller because it uses asymmetric retrieval training)
# Min-max normalisation across a single batch inflates noise when all candidates
# score within 0.03 of each other.  Sigmoid with a per-model center preserves
# ranking order while placing scores on a shared probabilistic scale.
# scale=8 gives ~90% of the [0,1] range within ±0.15 of center — sufficient
# discrimination without clipping legitimate outliers.
SEM_CALIBRATION = {
    "mpnet":  {"center": 0.45, "scale": 8.0},
    "mxbai":  {"center": 0.50, "scale": 8.0},
    "arctic": {"center": 0.32, "scale": 8.0},
}

# Matches negation phrases directly before a skill mention ("no experience in X", "not skilled in X")
_NEGATION_PATTERNS = re.compile(
    r"\b(no|not|without|lack(?:ing)?|never|zero|minimal|limited)\s+"
    r"(?:experience|knowledge|familiarity|skills?|proficiency|background)?\s*"
    r"(?:in|with|of)?\s*",
    re.IGNORECASE,
)

# Characters before a skill mention to search for negation phrasing.
_NEGATION_WINDOW = 80

# Module-level compiled-regex cache for skill boundary patterns.
# _is_skill_match builds r"\b<skill>\b" patterns on the fly; without caching
# Python recompiles the same pattern for every candidate × every required skill
# ({cands} × {req_skills} × {aliases} iterations per screening run).
# With a dict cache each unique skill string compiles its pattern exactly once.
_regex_cache: dict[str, re.Pattern] = {}


def _skill_re(term: str) -> re.Pattern:
    """Return a compiled word-boundary regex for term, building it on first use."""
    pat = _regex_cache.get(term)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(term) + r"\b")
        _regex_cache[term] = pat
    return pat


def _has_negation_context(resume_text_lower: str, skill_term: str) -> bool:
    """Return True if the skill appears in a negation context (e.g. 'no SQL experience')."""
    for m in re.finditer(re.escape(skill_term), resume_text_lower):
        start = max(0, m.start() - _NEGATION_WINDOW)
        context = resume_text_lower[start:m.start()]
        if _NEGATION_PATTERNS.search(context):
            return True
    return False


def _sigmoid_calibrate(raw: float, model_key: str) -> float:
    """Map a raw cosine similarity score onto [0, 1] via per-model sigmoid.

    Rationale: raw cosine ranges differ per model (Arctic ~0.15–0.40, MPNet ~0.30–0.65).
    Sigmoid centering on the model's expected neutral-match score maps the natural similarity
    distribution onto a stable [0, 1] interval without amplifying noise in tight batches.
    """
    p   = SEM_CALIBRATION.get(model_key, {"center": 0.45, "scale": 8.0})
    val = 1.0 / (1.0 + math.exp(-p["scale"] * (raw - p["center"])))
    return round(val, 6)


def get_dynamic_threshold(final_scores: list, model_key: str = "mpnet") -> float:
    """Compute the 60th-percentile score as the eligibility threshold, clamped to model default ±10%."""
    model_default = MODEL_THRESHOLDS.get(model_key, 0.50)
    if len(final_scores) < 5:
        return model_default
    sorted_scores = sorted(final_scores)
    n = len(sorted_scores)
    p60_idx = min(max(int(n * 0.60) - 1, 0), n - 1)
    p60 = sorted_scores[p60_idx]  # 60th percentile cutoff
    return round(max(model_default - 0.10, min(model_default + 0.10, p60)), 4)


def _normalise_skill_text(skill: str) -> str:
    """Lowercase and strip non-alphanumeric characters from a skill string for comparison."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.\- ]+", " ", skill.lower())).strip()


def _tokenise_skill(skill: str) -> set:
    """Split a normalised skill string into individual word tokens."""
    return {tok for tok in _normalise_skill_text(skill).split(" ") if tok}


def _is_skill_match(req_lower: str, candidate_lower: set) -> bool:
    """Return True if req_lower matches any candidate skill via exact, alias, substring, or token overlap.

    Match hierarchy (first hit wins):
      1. Exact normalised match.
      2. Required-skill alias exact match against candidate skills.
      3. Reverse-alias: check if ANY alias of EACH candidate skill covers the required skill.
         This handles the case where candidate lists "communication" and required is
         "verbal communication" — "communication" is an alias of "verbal communication".
      4. Word-boundary substring match (both directions).
      5. Token overlap ≥ 2/3 of required-skill tokens.
    Patterns are retrieved from _regex_cache and compiled at most once per unique term.
    """
    if req_lower in candidate_lower:
        return True

    req_aliases     = SKILL_ALIASES.get(req_lower, [req_lower])
    req_alias_norms = [_normalise_skill_text(a) for a in req_aliases]

    # ── Check 2: required-skill aliases against candidate skill set ────────────
    if candidate_lower.intersection(req_alias_norms):
        return True

    # ── Check 3: reverse-alias lookup ─────────────────────────────────────────
    # For each candidate skill, get ITS alias list and check whether req_lower
    # (or any of req's aliases) is covered by that alias list.
    for cand in candidate_lower:
        cand_aliases     = SKILL_ALIASES.get(cand, [cand])
        cand_alias_norms = {_normalise_skill_text(a) for a in cand_aliases}
        # Does any required-skill alias appear in the candidate's alias set?
        if cand_alias_norms.intersection(req_alias_norms):
            return True
        # Does the required skill itself appear in the candidate's alias list?
        if req_lower in cand_alias_norms:
            return True

    req_tokens = _tokenise_skill(req_lower)

    for cand in candidate_lower:
        # ── Check 4: Word-boundary substring match (both directions) ──────────
        if len(req_lower) >= 4:
            if _skill_re(req_lower).search(cand):
                return True
            if len(cand) >= 4 and _skill_re(cand).search(req_lower):
                return True

        # ── Check 5: Token overlap ≥ 2/3 of required-skill tokens ─────────────
        cand_tokens = _tokenise_skill(cand)
        if req_tokens and len(req_tokens) > 1:
            if len(req_tokens & cand_tokens) / len(req_tokens) >= 0.67:
                return True

        # Same substring + token checks for each required-skill alias
        for alias_norm in req_alias_norms:
            if alias_norm == req_lower:
                continue  # already checked above
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
    """Match required skills against candidate skills via exact/alias → semantic → negation; return 5-tuple."""
    if not required_skills:
        return 1.0, [], [], [], []

    candidate_lower = {_normalise_skill_text(s) for s in candidate_skills if s}  # FIX: guard against None
    sem_matched_norm = (
        {_normalise_skill_text(s) for s in semantic_matches} if semantic_matches else set()
    )
    resume_lower = resume_text.lower()

    matched        = []  # exact or alias match, not negated
    missing        = []  # no match found
    semantic_only  = []  # matched only via semantic embedding (shown as ~ in UI)
    negated        = []  # found but explicitly negated in the resume text

    for req in required_skills:
        req_lower = _normalise_skill_text(req)

        if resume_lower and _has_negation_context(resume_lower, req_lower):  # negation checked first
            negated.append(req)
            continue

        if _is_skill_match(req_lower, candidate_lower):
            matched.append(req)
        elif req_lower in sem_matched_norm:
            matched.append(req)
            semantic_only.append(req)
        else:
            missing.append(req)

    effective_total = len(required_skills) - len(negated)  # negated skills excluded from denominator
    score = len(matched) / effective_total if effective_total > 0 else 1.0
    return round(score, 4), matched, missing, semantic_only, negated


def _check_hard_eligibility(info: dict, config: ScoringConfig) -> Tuple[bool, str, str]:
    """Apply hard disqualification rules: no contact info, insufficient experience, missing education."""
    email = (info.get("email") or "").strip()
    phone = (info.get("phone") or "").strip()
    if not email and not phone:
        return False, "No contact details found", "no_contact"

    exp = info.get("experience_years", 0.0)
    if config.min_experience_years > 0 and exp < config.min_experience_years:
        return False, f"Experience {exp}y < required {config.min_experience_years}y", "experience_below_min"

    if config.required_education:
        edu_raw  = " ".join(info.get("education", [])).lower()
        edu_norm = re.sub(r"[^a-z0-9\s]", "", edu_raw)
        def _norm_req(r):
            return re.sub(r"[^a-z0-9\s]", "", r.lower())
        if not any(_norm_req(req) in edu_norm for req in config.required_education):
            return False, f"Missing required education: {config.required_education}", "missing_education"

    return True, "", ""


def is_eligible(info: dict, config: ScoringConfig,
                skill_score: Optional[float] = None) -> tuple:  # pyright: ignore
    """Thin compatibility wrapper around _check_hard_eligibility for external callers."""
    eligible, reason, _ = _check_hard_eligibility(info, config)
    if not eligible:
        return False, reason
    if skill_score is not None and skill_score < 0.30:
        return False, f"Skill match {round(skill_score * 100)}% < minimum 30% required"
    return True, ""


def score_candidates(candidates: list, required_skills: list,
                     config: ScoringConfig, model_key: str = "mpnet") -> list:
    """Score all candidates in three passes: component scores → dynamic threshold → eligibility + bands."""
    logger.info(
        "[scoring_engine] Scoring %d candidates | weights: skill=%.2f semantic=%.2f exp=%.2f",
        len(candidates), config.skill_weight, config.semantic_weight, config.exp_weight,
    )

    # ── Pre-pass: Sigmoid-calibrate semantic scores per model ─────────────────
    # OOM-fallback candidates (zero vectors from _safe_encode retry exhaustion) are
    # forced to semantic_score=0.0 and excluded from calibration to avoid artificially
    # boosting them via the sigmoid curve.
    # All other candidates are mapped through _sigmoid_calibrate which normalises the
    # model-specific cosine range onto [0,1] without amplifying noise in tight batches.
    calibrated_sems: dict[str, float] = {}
    raw_values = []
    for c in candidates:
        fname   = c.get("filename", "")
        raw_sem = float(min(max(c.get("semantic_score", 0.0), 0.0), 1.0))
        if c.get("oom_fallback", False):
            calibrated_sems[fname] = 0.0
            logger.warning("[scoring_engine] OOM fallback for %s — semantic_score forced to 0.0", fname)
        else:
            calibrated_sems[fname] = _sigmoid_calibrate(raw_sem, model_key)
            raw_values.append(raw_sem)

    if raw_values:
        logger.info(
            "[scoring_engine] Sigmoid calibration (%s) — raw sem range [%.4f, %.4f]",
            model_key, min(raw_values), max(raw_values),
        )

    # ── Pass 1: Component scores and hard eligibility ─────────────────────────
    raw_results = []

    for c in candidates:
        info                   = c.get("info", {})
        filename               = c.get("filename", "")
        semantic_score         = calibrated_sems.get(filename, 0.0)
        resume_text            = c.get("resume_text", "")     # for negation detection
        semantic_skill_matches = c.get("semantic_skill_matches", None)

        skill_score, matched_skills, missing_skills, semantic_only, negated_skills = _skill_score(
            info.get("skills", []),
            required_skills,
            semantic_skill_matches,
            resume_text=resume_text,
        )

        hard_ok, hard_reason, hard_code = _check_hard_eligibility(info, config)
        if hard_ok and skill_score < 0.30 and required_skills:  # 30% skill floor is a hard minimum
            hard_ok     = False
            hard_reason = f"Skill match {round(skill_score * 100)}% < minimum 30% required"
            hard_code   = "skill_below_min"

        exp_years  = info.get("experience_years", 0.0)
        target_exp = max(config.min_experience_years, 1.0)
        exp_score  = min(exp_years / target_exp, 1.0)

        final_score = (
            config.skill_weight    * skill_score    +
            config.semantic_weight * semantic_score +
            config.exp_weight      * exp_score       # weighted sum of all three components
        )
        final_score = min(round(final_score, 4), 1.0)

        fn_recovered = False
        if hard_ok and final_score < 0.50 and semantic_score > 0.55:  # rescue strong-semantic candidates
            fn_recovered = True
            logger.debug("[scoring_engine] FN Recovery: %s — low final %.2f but semantic %.2f",
                         filename, final_score, semantic_score)

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
            "_hard_ok":         hard_ok,
            "_hard_reason":     hard_reason,
            "_hard_code":       hard_code,
            "fn_recovered":     fn_recovered,
            "band":             "",
            "confidence":       0.0,
        })

    # ── Pass 2: Dynamic threshold from full score distribution ────────────────
    all_finals = [r["final_score"] for r in raw_results]
    dyn_threshold = get_dynamic_threshold(all_finals, model_key)  # 60th percentile, clamped ±10%
    logger.info("[scoring_engine] Dynamic threshold for '%s': %.4f", model_key, dyn_threshold)

    # ── Pass 3: Apply threshold, assign bands, sort ───────────────────────────
    results = []
    for r in raw_results:
        hard_ok     = r.pop("_hard_ok")
        hard_reason = r.pop("_hard_reason")
        hard_code   = r.pop("_hard_code")

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

        r["eligible"]          = eligible
        r["rejection_reason"]  = rejection_reason
        r["rejection_code"]    = rejection_code
        r["dynamic_threshold"] = dyn_threshold   # included on every candidate — single source of truth
        r["confidence"]        = round(abs(r["final_score"] - dyn_threshold), 4)
        r["near_threshold"]    = r["confidence"] < 0.05

        # Band and eligibility both derived from dyn_threshold — guaranteed to be consistent
        # Band strings MUST match the frontend's checks in AnalyticsView and Decisions tab:
        #   c.band === "Weak Fit"   → Archive group (ineligible)
        #   c.band === "Borderline" → Secondary Review group (eligible, near threshold)
        #   c.band === "Strong Fit" → Interview Now group (eligible, well above threshold)
        if not eligible:
            r["band"] = "Weak Fit"
        elif r["final_score"] >= dyn_threshold + 0.10:
            r["band"] = "Strong Fit"
        else:
            r["band"] = "Borderline"

        logger.debug(
            "[scoring_engine] %s | skill=%.2f sem=%.2f exp=%.2f → final=%.2f | thresh=%.2f | %s",
            r["filename"], r["skill_score"], r["semantic_score"], r["exp_score"],
            r["final_score"], dyn_threshold, "eligible" if eligible else rejection_reason,
        )
        results.append(r)

    results.sort(key=lambda x: (
        x["eligible"],
        x["final_score"],
        x["semantic_score"],
        x["name"],
    ), reverse=True)   # stable deterministic ranking

    if config.top_n and config.top_n > 0:
        results = results[:config.top_n]

    shortlisted = sum(1 for r in results if r["eligible"])
    fn_count    = sum(1 for r in results if r.get("fn_recovered", False))
    logger.info(
        "[scoring_engine] Done — %d shortlisted / %d total | %d FN recovered | threshold=%.3f",
        shortlisted, len(results), fn_count, dyn_threshold,
    )
    return results
