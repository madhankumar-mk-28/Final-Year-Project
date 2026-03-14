"""
scoring_engine.py
-----------------
Two-stage scoring pipeline:
  Stage 1 — Eligibility filter (hard rules)
  Stage 2 — Weighted score (skills + semantic + experience)

Default weights:
  0.55 skill  +  0.45 semantic
  Optional: experience weight can be added if needed.
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger("scoring_engine")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ScoringConfig:
    skill_weight:          float = 0.55
    semantic_weight:       float = 0.45
    exp_weight:            float = 0.0
    min_experience_years:  float = 0.0
    required_education:    list  = field(default_factory=list)
    top_n:                 int   = 50
    min_final_score:       float = 0.0

    def __post_init__(self):
        self.skill_weight = max(0.0, self.skill_weight)
        self.semantic_weight = max(0.0, self.semantic_weight)
        self.exp_weight = max(0.0, self.exp_weight)
        total = self.skill_weight + self.semantic_weight + self.exp_weight
        if total == 0:
            self.skill_weight = 0.55
            self.semantic_weight = 0.45
            self.exp_weight = 0.0
            return
        if abs(total - 1.0) > 0.01:
            # Auto-normalise so weights always sum to 1
            self.skill_weight    = self.skill_weight    / total
            self.semantic_weight = self.semantic_weight / total
            self.exp_weight      = self.exp_weight      / total


# ── Skill aliases — if a candidate has any alias, the required skill is matched
SKILL_ALIASES = {
    "machine learning":     ["ml", "machine learning", "sklearn", "scikit-learn", "xgboost", "lightgbm"],
    "deep learning":        ["deep learning", "neural network", "neural networks", "dl"],
    "nlp":                  ["nlp", "natural language processing", "text mining", "nltk", "spacy", "gensim", "bert", "transformers", "huggingface"],
    "scikit-learn":         ["scikit-learn", "sklearn", "machine learning"],
    "pytorch":              ["pytorch", "torch"],
    "tensorflow":           ["tensorflow", "tf", "keras"],
    "python":               ["python"],
    "sql":                  ["sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "database"],
    "pandas":               ["pandas", "data analysis", "data analytics"],
    "numpy":                ["numpy", "scipy"],
    "flask":                ["flask", "fastapi", "django", "rest api", "restful"],
    "data science":         ["data science", "data analysis", "data analytics", "statistics", "statistical analysis"],
    "docker":               ["docker", "kubernetes", "k8s", "containerization"],
    "aws":                  ["aws", "azure", "gcp", "google cloud", "cloud"],
    "react":                ["react", "reactjs", "next.js", "nextjs"],
    "javascript":           ["javascript", "typescript", "js", "ts", "nodejs", "node"],
}


@lru_cache(maxsize=1024)
def _normalise_skill_text(skill: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.\- ]+", " ", skill.lower())).strip()


@lru_cache(maxsize=1024)
def _tokenise_skill(skill: str) -> frozenset:
    return frozenset(tok for tok in _normalise_skill_text(skill).split(" ") if tok)


def _is_skill_match(req_lower: str, candidate_lower: set) -> bool:
    if req_lower in candidate_lower:
        return True

    aliases = SKILL_ALIASES.get(req_lower, [req_lower])
    alias_norm = {_normalise_skill_text(a) for a in aliases}

    if candidate_lower.intersection(alias_norm):
        return True

    req_tokens = _tokenise_skill(req_lower)
    for cand in candidate_lower:
        if req_lower in cand or cand in req_lower:
            return True

        cand_tokens = _tokenise_skill(cand)
        if req_tokens and len(req_tokens) > 1:
            overlap = len(req_tokens & cand_tokens) / len(req_tokens)
            if overlap >= 0.67:
                return True

        for alias in alias_norm:
            if alias in cand or cand in alias:
                return True
            alias_tokens = _tokenise_skill(alias)
            if alias_tokens and len(alias_tokens) > 1:
                overlap = len(alias_tokens & cand_tokens) / len(alias_tokens)
                if overlap >= 0.67:
                    return True

    return False


def _skill_score(candidate_skills: list, required_skills: list) -> tuple:
    """
    Match required skills against candidate skills using aliases.
    Returns (score 0-1, matched list, missing list).
    """
    if not required_skills:
        return 1.0, [], []

    candidate_lower = {_normalise_skill_text(s) for s in candidate_skills if s}
    matched = []
    missing = []

    for req in required_skills:
        req_lower = _normalise_skill_text(req)
        if _is_skill_match(req_lower, candidate_lower):
            matched.append(req)
            continue
        missing.append(req)

    score = len(matched) / len(required_skills)
    return round(score, 4), matched, missing


# ── Eligibility filter ────────────────────────────────────────────────────────

def is_eligible(info: dict, config: ScoringConfig) -> tuple:
    """Hard filter. Returns (eligible: bool, reason: str)."""
    exp = info.get("experience_years", 0.0)
    if config.min_experience_years > 0 and exp < config.min_experience_years:
        return False, f"Experience {exp}y < required {config.min_experience_years}y"

    if config.required_education:
        edu = " ".join(info.get("education", [])).lower()
        if not any(req.lower() in edu for req in config.required_education):
            return False, f"Missing required education: {config.required_education}"

    return True, ""


# ── Main scoring ──────────────────────────────────────────────────────────────

def score_candidates(candidates: list, required_skills: list, config: ScoringConfig) -> list:
    """
    Score and rank all candidates.

    Each candidate dict must have:
      { "filename", "info": {name, skills, experience_years, ...}, "semantic_score" }

    Returns sorted list with scoring details.
    """
    logger.info("[scoring_engine] Scoring %d candidates | weights: skill=%.2f semantic=%.2f exp=%.2f",
                len(candidates), config.skill_weight, config.semantic_weight, config.exp_weight)

    results = []
    target_exp = config.min_experience_years if config.min_experience_years > 0 else 2.0

    for c in candidates:
        info           = c.get("info", {})
        semantic_score = min(max(c.get("semantic_score", 0.0), 0.0), 1.0)
        filename       = c.get("filename", "")

        # Eligibility
        eligible, rejection_reason = is_eligible(info, config)

        # Skill score with alias matching
        skill_score, matched_skills, missing_skills = _skill_score(
            info.get("skills", []), required_skills
        )

        # Experience score saturates after reaching the target + buffer.
        exp_years  = info.get("experience_years", 0.0)
        exp_score  = min(exp_years / (target_exp + 2.0), 1.0)

        # Weighted final score
        final_score = (
            config.skill_weight    * skill_score    +
            config.semantic_weight * semantic_score +
            config.exp_weight      * exp_score
        )
        final_score = min(round(final_score, 4), 1.0)

        if eligible and config.min_final_score > 0 and final_score < config.min_final_score:
            eligible = False
            rejection_reason = (
                f"Final score {final_score:.2f} < required {config.min_final_score:.2f}"
            )

        logger.debug("[scoring_engine] %s | skill=%.2f sem=%.2f exp=%.2f → final=%.2f | %s",
                     filename, skill_score, semantic_score, exp_score, final_score,
                     "eligible" if eligible else rejection_reason)

        results.append({
            "filename":         filename,
            "name":             info.get("name", "Unknown"),
            "email":            info.get("email", ""),
            "phone":            info.get("phone", ""),
            "skills":           info.get("skills", []),
            "skills_matched":   matched_skills,
            "skills_missing":   missing_skills,
            "experience_years": exp_years,
            "education":        info.get("education", []),
            "skill_score":      round(skill_score, 4),
            "semantic_score":   round(semantic_score, 4),
            "exp_score":        round(exp_score, 4),
            "final_score":      final_score,
            "eligible":         eligible,
            "rejection_reason": rejection_reason,
        })

    # Sort: eligible first, then by final score
    results.sort(key=lambda x: (x["eligible"], x["final_score"]), reverse=True)

    # Apply top_n limit
    if config.top_n:
        results = results[:config.top_n]

    shortlisted = sum(1 for r in results if r["eligible"])
    logger.info("[scoring_engine] Done — %d shortlisted / %d total", shortlisted, len(results))
    return results


# ── Console display ───────────────────────────────────────────────────────────

def print_results(results: list):
    print("\n" + "=" * 80)
    print(f"{'RANK':<5} {'NAME':<28} {'SKILLS':>7} {'MATCHED':>9} {'SEMANTIC':>9} {'FINAL':>7} {'STATUS'}")
    print("=" * 80)

    rank = 1
    for r in results:
        name     = r["name"][:26]
        skills   = len(r["skills"])
        matched  = len(r["skills_matched"])
        req      = matched  # already filtered
        semantic = r["semantic_score"] * 100
        final    = r["final_score"] * 100
        skill_p  = r["skill_score"] * 100
        status   = "✓ Eligible" if r["eligible"] else f"✗ {r['rejection_reason'][:20]}"

        print(f"{rank:<5} {name:<28} {skills:>4} skl  {skill_p:>5.1f}%  {semantic:>6.1f}%  {final:>5.1f}%  {status}")
        rank += 1

    print("=" * 80)
    shortlisted = sum(1 for r in results if r["eligible"])
    print(f"Total: {len(results)} candidates | Shortlisted: {shortlisted} | Rejected: {len(results) - shortlisted}")
    print("=" * 80 + "\n")
