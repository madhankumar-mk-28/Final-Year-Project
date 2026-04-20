# REPORT CORRECTIONS — COPY-PASTE READY
# =======================================
# Each section below shows the FIND text and REPLACE text.
# Use Ctrl+H (Find & Replace) in your document editor.
# Corrections are grouped by location in the report.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBAL: Replace "235" with "234" (skills count)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Appears in: Executive Summary, Section 1.1, Section 5.4,
#             Section 6.2, Section 8.1, Section 8.3

# FIND (exact phrase variations):

#   "235 entries"  →  "234 entries"
#   "two hundred and thirty-five"  →  "two hundred and thirty-four"
#   "over 235"  →  "234 entries"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4.5 — JD Cache Hash Method
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND:
# The job description embedding is separately cached using a FIFO dictionary keyed by the MD5 hash of the job description and model name. This cache holds up to 32 entries and evicts the oldest on overflow, avoiding repeated encoding during repeated runs.

# REPLACE WITH:
# The job description embedding is separately cached using a FIFO dictionary keyed by the SHA-256 hash of the job description concatenated with the model name. This cache holds up to 32 entries and evicts the oldest on overflow, avoiding repeated encoding during repeated runs.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4.6 — Dedup wording
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND:
# When two profiles share a contact identifier, they are merged into a single record.

# REPLACE WITH:
# When two profiles share the same email address, they are merged into a single record.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4.7 — Dynamic Threshold (after FIX 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND:
# Instead of a fixed cutoff, the scoring engine computes a dynamic threshold from the batch score distribution. The threshold is set at the 60th percentile of final scores and clamped within ±10 percentage points of a per-model default (0.45 for MPNet, 0.55 for MxBai, 0.50 for Arctic) to prevent instability in small batches.

# REPLACE WITH:
# Instead of a fixed cutoff, the scoring engine computes a dynamic threshold from the eligible candidates' score distribution. The threshold is set at the 60th percentile of eligible candidates' final scores and clamped within ±10 percentage points of a per-model default (0.45 for MPNet, 0.55 for MxBai, 0.50 for Arctic) to prevent instability in small batches. Rejected candidates' near-zero scores are excluded from the threshold computation to prevent deflation.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5.2 — Embedding Models (ALL THREE THRESHOLDS WRONG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- MPNet paragraph ---
# FIND:
# MPNet (multi-qa-mpnet-base-dot-v1): An MPNet-based sentence embedding model trained on 215 million question-answer pairs using a Siamese network with dot-product similarity objective. Produces 768-dimensional embeddings. The default semantic skill matching threshold for this model is 0.70.

# REPLACE WITH:
# MPNet (multi-qa-mpnet-base-dot-v1): An MPNet-based sentence embedding model trained on 215 million question-answer pairs using a Siamese network with dot-product similarity objective. Produces 768-dimensional embeddings. The default semantic skill matching threshold for this model is 0.63.

# --- MxBai paragraph ---
# FIND:
# MxBai (mixedbread-ai/mxbai-embed-large-v1): A contrastively trained embedding model with 335 million parameters that achieved state-of-the-art performance on the MTEB English benchmark. Produces 1024-dimensional embeddings. The default skill matching threshold for this model is 0.65.

# REPLACE WITH:
# MxBai (mixedbread-ai/mxbai-embed-large-v1): A contrastively trained embedding model with 335 million parameters that achieved state-of-the-art performance on the MTEB English benchmark. Produces 1024-dimensional embeddings. The default skill matching threshold for this model is 0.60.

# --- Arctic paragraph (BROKEN SENTENCE FIX) ---
# FIND:
# Arctic Embed (Snowflake/snowflake-arctic-embed-m-v1.5): An enterprise-grade retrieval model by Snowflake designed for asymmetric query-document ranking at scale. Produces 768 - dimensional embeddings. The default skill matching threshold for this model is 0.72, raised from 0.55 because this model's embedding space places the embedding space exhibits higher semantic proximity for generic or soft skills, increasing false-positive similarity scores, which would otherwise produce inflated skill match scores.

# REPLACE WITH:
# Arctic Embed (Snowflake/snowflake-arctic-embed-m-v1.5): An enterprise-grade retrieval model by Snowflake designed for asymmetric query-document ranking at scale. Produces 768-dimensional embeddings. The default skill matching threshold for this model is 0.66. Arctic's asymmetric retrieval training compresses the cosine similarity range, causing generic and soft skills to exhibit higher semantic proximity. A threshold of 0.66 bisects the gap between related soft-skill clusters (0.65-0.68) and unrelated skills (below 0.63), preventing inflated skill match scores.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6.4 — Scoring Engine (MULTIPLE FIXES)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- Fix band labels (appears twice in 6.4) ---

# FIND:
# a band label is assigned (Strong Fit, Good Fit, or Not Eligible) based on the relationship between the final score and the threshold.

# REPLACE WITH:
# a band label is assigned (Strong Fit, Borderline, or Weak Fit) based on the relationship between the final score and the threshold.


# FIND:
# The band thresholds are: Not Eligible for any candidate failing hard checks or scoring below the dynamic threshold; Strong Fit for eligible candidates whose final score is at least 0.10 above the dynamic threshold; and Good Fit for all remaining eligible candidates.

# REPLACE WITH:
# The band thresholds are: Weak Fit for any candidate failing hard checks or scoring below the dynamic threshold; Strong Fit for eligible candidates whose final score is at least 0.10 above the dynamic threshold; and Borderline for all remaining eligible candidates.


# --- Fix fn_recovery description ---

# FIND:
# A false-negative recovery mechanism flags candidates who pass all hard eligibility checks but whose composite final score falls below 0.50 despite a semantic score exceeding 0.55. These candidates are marked fn_recovered = True, signalling that the embedding model considers them strongly aligned with the job description and that manual recruiter review is warranted before a final rejection decision.

# REPLACE WITH:
# A false-negative recovery mechanism flags ineligible candidates whose composite final score falls below the dynamic threshold despite a semantic score exceeding a per-model floor (0.52 for MPNet, 0.62 for MxBai, 0.55 for Arctic). These candidates are marked fn_recovered = True, signalling that the embedding model considers them strongly aligned with the job description and that manual recruiter review is warranted before a final rejection decision. Already-eligible candidates are not flagged, and the per-model floors prevent MxBai's inflated cosine scores from triggering excessive false alerts.


# --- Fix dynamic threshold source ---

# FIND:
# In the second pass, the dynamic threshold is computed from the full distribution of final scores in the batch.

# REPLACE WITH:
# In the second pass, the dynamic threshold is computed from the distribution of final scores among eligible candidates only. Rejected candidates' near-zero scores are excluded to prevent threshold deflation.


# --- ADD all_skills_negated note (insert after the skill_below_min sentence) ---
# After the sentence ending "...with the skill_below_min code.", ADD:

# If every required skill is found in a negation context (effective denominator drops to zero), the candidate is rejected with the all_skills_negated code and a distinct human-readable message advising manual review, rather than the generic skill_below_min code.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8.1 — Summary of Work Done (band labels)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND:
# Eligible candidates are classified into two bands: Strong Fit (final score at least ten percentage points above the threshold) and Good Fit (final score above the threshold but within ten points). A structured rejection code system allows the frontend to generate accurate, human-readable explanations for every shortlisting decision.

# REPLACE WITH:
# Eligible candidates are classified into two bands: Strong Fit (final score at least ten percentage points above the threshold) and Borderline (final score above the threshold but within ten points). Ineligible candidates receive the Weak Fit band label. A structured rejection code system allows the frontend to generate accurate, human-readable explanations for every shortlisting decision.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APPENDIX B.2 — Skill Extraction Code
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND (the entire code block):
# def extract_skills(text: str) -> list:
#     """Matches text against a curated database of 235 technical and soft skills."""
#     text_lower = " " + text.lower().replace('\n', ' ') + " "
#     found_skills = set()
#     # Match multi-word skills via prioritized substring checking
#     for skill in MULTI_WORD_SKILLS:
#         skill_lower = skill.lower()
#         if f" {skill_lower} " in text_lower or f" {skill_lower}," in text_lower:
#             found_skills.add(skill)
#     # Match single-word skills via a batched Regex boundary operation
#     matches = _SINGLE_WORD_REGEX.findall(text)
#     for match in matches:
#         found_skills.add(_SINGLE_WORD_MAP[match.lower()])
#     return sorted(list(found_skills))

# REPLACE WITH:
# def extract_skills(text: str) -> list:
#     """Matches text against a curated database of 234 technical and soft skills."""
#     # Normalise hyphenated skill variants (e.g. "problem-solving" -> "problem solving")
#     normalised = _HYPHEN_SKILL_RE.sub(lambda m: m.group(1) + " " + m.group(2), text)
#     text_lower = normalised.lower()
#     found = set()
#     # Match multi-word skills via substring checking (longest-first)
#     for skill in _MULTI_WORD_SKILLS:
#         if skill in text_lower:
#             found.add(skill)
#     # Match single-word skills via a single pre-compiled regex pass
#     found.update(_SINGLE_SKILL_RE.findall(text_lower))
#     return sorted(found)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APPENDIX B.4 — Dynamic Threshold Code
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND (entire B.4 code):
# def get_dynamic_threshold(final_scores: list, model_key: str = "mpnet") -> float:
#     """Computes dynamic passing threshold clamped at the 60th percentile of candidates."""
#     if not final_scores:
#         return DEFAULT_THRESHOLDS.get(model_key, 0.45)
#     default = DEFAULT_THRESHOLDS.get(model_key, 0.45)
#     if len(final_scores) < 5:
#         return default
#     # 60th Percentile Evaluation
#     percentile_60 = float(np.percentile(final_scores, 60))
#     # Threshold clamped mathematically between +/- 10% bounds
#     clamped = max(default - 0.10, min(percentile_60, default + 0.10))
#     return float(round(clamped, 2))
# def score_candidates(candidates: list, config: ScoringConfig) -> list:
#        """Process eligibility, resolve weighting, apply dynamic bounds."""
#   """Final Formula blending exact skill matches against embedded semantics"""
#     final_score = (config.skill_weight * skill_score) + (config.semantic_weight * semantic_score)

# REPLACE WITH:
# def get_dynamic_threshold(final_scores: list, model_key: str = "mpnet") -> float:
#     """Compute the 60th-percentile score as the eligibility threshold, clamped to model default +/-10%."""
#     model_default = MODEL_THRESHOLDS.get(model_key, 0.50)
#     if len(final_scores) < 5:
#         return model_default
#     sorted_scores = sorted(final_scores)
#     n = len(sorted_scores)
#     p60_idx = min(max(int(n * 0.60) - 1, 0), n - 1)
#     p60 = sorted_scores[p60_idx]
#     return round(max(model_default - 0.10, min(model_default + 0.10, p60)), 4)
#
# def score_candidates(candidates: list, required_skills: list,
#                      config: ScoringConfig, model_key: str = "mpnet") -> list:
#     """Score all candidates in three passes: component scores, dynamic threshold, eligibility + bands."""
#     # Final Score Formula:
#     final_score = (config.skill_weight * skill_score) + (config.semantic_weight * semantic_score)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APPENDIX C.1 — Sample JSON Note
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND:
# Below is an excerpt showing the data structure of the top-ranked candidate.

# REPLACE WITH:
# Below is an excerpt showing the data structure of a candidate record. Additional fields present in the actual output include rejection_code, dynamic_threshold, band, confidence, near_threshold, fn_recovered, skills_configured, and links, which are omitted here for brevity. Note: this sample was captured from a run using custom skill/semantic weights (approximately 40/60), not the default 55/45 configuration.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APPENDIX C.2 — Evaluation Threshold Note
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ADD this note after the evaluation JSON block:
# Note: The threshold value of 0.60 shown above is the dynamically computed value for this specific batch, not a hardcoded default. The dynamic threshold varies per screening run based on the 60th percentile of eligible candidates' final scores, clamped within +/-10% of the model-specific default.
