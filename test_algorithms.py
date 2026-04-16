"""
test_algorithms.py — Comprehensive algorithm test suite for the ML Resume Screening System.

Tests cover:
  • resume_parser.py      — PDF extraction, fallback logic, edge cases
  • information_extractor.py — name, email, phone, skills, experience, education
  • scoring_engine.py     — ScoringConfig, skill scoring, dynamic threshold, eligibility,
                            bands, weight normalisation, exception handling
  • semantic_matcher.py   — cosine similarity, chunking, pooling, empty-text guards

Run with:
    python test_algorithms.py
or:
    python -m pytest test_algorithms.py -v
"""

import math
import re
import sys
import traceback
import unittest
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m~ SKIP\033[0m"

_results = {"passed": 0, "failed": 0, "skipped": 0}


def _report(name, ok, detail=""):
    if ok:
        _results["passed"] += 1
        print(f"  {PASS}  {name}")
    else:
        _results["failed"] += 1
        print(f"  {FAIL}  {name}")
        if detail:
            print(f"         ↳ {detail}")


# ──────────────────────────────────────────────────────────────────────────────
# 1 · information_extractor.py
# ──────────────────────────────────────────────────────────────────────────────

def test_information_extractor():
    print("\n══ 1. information_extractor ══")
    from information_extractor import (
        extract_email, extract_phone, extract_skills,
        extract_experience_years, extract_education,
        name_from_filename, extract_all, SKILLS_DB,
    )

    # ── email ──
    _report("email: valid address extracted",
            extract_email("Contact: john.doe@example.com for details") == "john.doe@example.com")
    _report("email: subdomain handled",
            extract_email("mail me at user@mail.company.co.uk") == "user@mail.company.co.uk")
    _report("email: empty string on no address",
            extract_email("No email here at all") == "")
    _report("email: empty string on empty input",
            extract_email("") == "")
    _report("email: lowercase normalisation",
            extract_email("Reach out: John.DOE@GMAIL.COM") == "john.doe@gmail.com")

    # ── phone ──
    _report("phone: +91 Indian mobile",
            extract_phone("+91 9876543210 is my number") == "+91-9876543210")
    _report("phone: bare 10-digit Indian mobile",
            extract_phone("Call 9123456789 anytime") == "9123456789")
    _report("phone: no phone → empty string",
            extract_phone("No phone here") == "")
    _report("phone: empty input → empty string",
            extract_phone("") == "")
    _report("phone: short fake number not extracted",
            extract_phone("code 12345") == "")

    # ── skills ──
    resume_with_skills = "Proficient in Python, machine learning, SQL, and Docker. Also do deep learning."
    skills = extract_skills(resume_with_skills)
    _report("skills: python detected", "python" in skills)
    _report("skills: machine learning detected", "machine learning" in skills)
    _report("skills: sql detected", "sql" in skills)
    _report("skills: docker detected", "docker" in skills)
    _report("skills: deep learning detected", "deep learning" in skills)
    _report("skills: empty resume → empty list", extract_skills("") == [])
    _report("skills: no skills → empty list", extract_skills("I am a hard worker and love my job.") == [])
    _report("skills: hyphenated form normalised",
            "time management" in extract_skills("Skilled in time-management and deadline handling."))
    _report("skills: returns sorted list",
            extract_skills("python and sql") == sorted(extract_skills("python and sql")))
    _report("skills: whitespace-only → empty list", extract_skills("   \n\t  ") == [])

    # ── experience ──
    _report("exp: explicit 'X years of experience'",
            extract_experience_years("I have 5 years of experience in software development.") == 5.0)
    _report("exp: 'X+ years'",
            extract_experience_years("3+ years of professional experience") == 3.0)
    _report("exp: no experience → 0.0",
            extract_experience_years("Fresh graduate looking for opportunities.") == 0.0)
    _report("exp: academic year not counted as work",
            extract_experience_years("Completed a 4-year B.Tech programme in Computer Science.") == 0.0)
    _report("exp: clamped at 15.0 max",
            extract_experience_years("20 years of experience in enterprise software.") == 15.0)
    _report("exp: empty string → 0.0",
            extract_experience_years("") == 0.0)
    _report("exp: decimal years handled",
            extract_experience_years("2.5 years of experience in data analysis.") == 2.5)

    # ── education ──
    edu = extract_education("B.Sc Computer Science from XYZ University, 2023.")
    _report("edu: B.Sc detected", any("b.sc" in e.lower() or "bsc" in e.lower() for e in edu))
    edu2 = extract_education("The candidate has ten years of work in finance and marketing.")
    _report("edu: no degree → empty or short list", len(edu2) == 0)
    edu3 = extract_education("")
    _report("edu: empty input → empty list", edu3 == [])

    # ── name_from_filename ──
    _report("name: normal name extracted", name_from_filename("John_Doe_Resume.pdf") == "John Doe")
    _report("name: noise word stripped", name_from_filename("resume.pdf").lower() == "unknown")
    _report("name: underscore separator", name_from_filename("Alice_Smith.pdf") == "Alice Smith")
    _report("name: hyphen separator", name_from_filename("Bob-Jones-CV.pdf") == "Bob Jones")
    # 'new_final_updated.pdf' → strips noise words → falls back to full stem → title-cases it
    # The function returns 'New Final Updated', not 'Unknown', by design.
    _report("name: noise words preserved in fallback when no real name remains",
            name_from_filename("new_final_updated.pdf") == "New Final Updated")
    _report("name: pure noise filename → Unknown",
            name_from_filename("resume.pdf").lower() == "unknown")
    _report("name: empty filename → Unknown",
            name_from_filename("") in ("Unknown", ""))

    # ── extract_all (integration) ──
    full_resume = """
John Smith
john.smith@email.com | +91 9988776655
Skills: Python, Machine Learning, SQL, Docker, Deep Learning
Experience: 3 years of experience in data science.
Education: B.Sc Computer Science from ABC University.
"""
    result = extract_all(full_resume, "John_Smith_Resume.pdf")
    _report("extract_all: returns dict", isinstance(result, dict))
    _report("extract_all: has name", bool(result.get("name")))
    _report("extract_all: email extracted", result.get("email") == "john.smith@email.com")
    _report("extract_all: phone extracted", bool(result.get("phone")))
    _report("extract_all: skills list", isinstance(result.get("skills"), list))
    _report("extract_all: experience_years", result.get("experience_years") == 3.0)
    _report("extract_all: education list", isinstance(result.get("education"), list))


# ──────────────────────────────────────────────────────────────────────────────
# 2 · scoring_engine.py
# ──────────────────────────────────────────────────────────────────────────────

def test_scoring_engine():
    print("\n══ 2. scoring_engine ══")
    from scoring_engine import (
        ScoringConfig, score_candidates, get_dynamic_threshold,
        _skill_score, _sigmoid_calibrate, _check_hard_eligibility,
        _is_skill_match, _normalise_skill_text,
    )

    # ── ScoringConfig ──
    cfg = ScoringConfig(skill_weight=0.55, semantic_weight=0.45)
    _report("config: default weights sum to 1.0",
            abs(cfg.skill_weight + cfg.semantic_weight + cfg.exp_weight - 1.0) < 0.001)

    cfg2 = ScoringConfig(skill_weight=0.6, semantic_weight=0.3, exp_weight=0.1)
    total = cfg2.skill_weight + cfg2.semantic_weight + cfg2.exp_weight
    _report("config: non-1.0 weights normalised", abs(total - 1.0) < 0.001)

    cfg3 = ScoringConfig(skill_weight=-0.1, semantic_weight=0.8)
    _report("config: negative weight clamped to 0", cfg3.skill_weight >= 0.0)

    cfg_zero = ScoringConfig(skill_weight=0, semantic_weight=0, exp_weight=0)
    total_zero = cfg_zero.skill_weight + cfg_zero.semantic_weight + cfg_zero.exp_weight
    _report("config: all-zero falls back to defaults", abs(total_zero - 1.0) < 0.001)

    # ── _sigmoid_calibrate ──
    val_mpnet = _sigmoid_calibrate(0.45, "mpnet")
    _report("sigmoid: center value → ~0.5", abs(val_mpnet - 0.5) < 0.05)
    val_high = _sigmoid_calibrate(0.75, "mpnet")
    val_low  = _sigmoid_calibrate(0.20, "mpnet")
    _report("sigmoid: high raw → score > 0.5", val_high > 0.5)
    _report("sigmoid: low raw → score < 0.5", val_low < 0.5)
    _report("sigmoid: output always [0,1]", 0.0 <= _sigmoid_calibrate(0.99, "mpnet") <= 1.0)
    _report("sigmoid: unknown model → fallback default", 0.0 <= _sigmoid_calibrate(0.5, "unknown") <= 1.0)

    # ── get_dynamic_threshold ──
    scores_small = [0.8, 0.7]   # fewer than 5 → return model default
    thresh_small = get_dynamic_threshold(scores_small, "mpnet")
    _report("dynamic threshold: < 5 scores → model default (0.45)", thresh_small == 0.45)

    scores_norm = [0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    thresh_norm = get_dynamic_threshold(scores_norm, "mpnet")
    _report("dynamic threshold: clamped within ±0.10 of default (0.45)",
            0.35 <= thresh_norm <= 0.55)

    scores_empty = []
    thresh_empty = get_dynamic_threshold(scores_empty, "mpnet")
    _report("dynamic threshold: empty list → model default", thresh_empty == 0.45)

    # ── _is_skill_match ──
    _report("skill match: exact match", _is_skill_match("python", {"python"}))
    _report("skill match: alias match (sklearn → scikit-learn)",
            _is_skill_match("scikit-learn", {"sklearn"}))
    _report("skill match: reverse alias (communication → verbal communication)",
            _is_skill_match("verbal communication", {"communication"}))
    _report("skill match: no match", not _is_skill_match("kubernetes", {"react", "python"}))
    _report("skill match: empty candidate set → no match",
            not _is_skill_match("python", set()))

    # ── _skill_score ──
    sc, matched, missing, sem_only, negated = _skill_score(
        ["python", "sql", "machine learning"],
        ["python", "sql", "java"],
    )
    _report("skill score: correct fraction (2/3)", abs(sc - 2/3) < 0.01)
    _report("skill score: matched list correct", set(matched) == {"python", "sql"})
    _report("skill score: missing list correct", missing == ["java"])
    _report("skill score: empty required skills → 1.0",
            _skill_score([], [], [])[0] == 1.0)
    _report("skill score: no candidate skills → 0.0",
            _skill_score([], ["python", "sql"])[0] == 0.0)

    # negation detection
    negation_text = "I have no SQL experience whatsoever."
    sc_neg, _, _, _, neg_list = _skill_score(
        [],            # candidate skills (empty)
        ["sql"],       # required
        None,
        resume_text=negation_text,
    )
    _report("skill score: negated skill in negated list", "sql" in neg_list)

    # ── _check_hard_eligibility ──
    info_ok = {"email": "a@b.com", "phone": "", "experience_years": 0.0, "education": []}
    ok, _, _ = _check_hard_eligibility(info_ok, ScoringConfig())
    _report("hard eligibility: email present → eligible", ok)

    info_no_contact = {"email": "", "phone": "", "experience_years": 0.0, "education": []}
    ok2, _, code2 = _check_hard_eligibility(info_no_contact, ScoringConfig())
    _report("hard eligibility: no contact → ineligible", not ok2)
    _report("hard eligibility: rejection code = 'no_contact'", code2 == "no_contact")

    info_exp = {"email": "x@y.com", "phone": "", "experience_years": 1.0, "education": []}
    cfg_exp = ScoringConfig(min_experience_years=3.0)
    ok3, _, code3 = _check_hard_eligibility(info_exp, cfg_exp)
    _report("hard eligibility: low experience → ineligible", not ok3)
    _report("hard eligibility: rejection code = 'experience_below_min'", code3 == "experience_below_min")

    info_edu = {"email": "a@b.com", "phone": "", "experience_years": 3.0,
                "education": ["Higher Secondary Certificate"]}
    cfg_edu = ScoringConfig(required_education=["B.Sc"])
    ok4, _, code4 = _check_hard_eligibility(info_edu, cfg_edu)
    _report("hard eligibility: missing education → ineligible", not ok4)
    _report("hard eligibility: rejection code = 'missing_education'", code4 == "missing_education")

    # ── score_candidates (end-to-end) ──
    def _make_candidate(name, email, skill_list, sem_score, filename=None, phone="9876543210"):
        return {
            "filename":     filename or f"{name.replace(' ','_')}.pdf",
            "info":         {
                "name": name, "email": email, "phone": phone,
                "skills": skill_list, "experience_years": 2.0, "education": [],
            },
            "semantic_score": sem_score,
            "resume_text":  f"{name} proficient in {', '.join(skill_list)}",
            "semantic_skill_matches": None,
            "oom_fallback": False,
        }

    required = ["python", "machine learning", "sql"]
    candidates = [
        _make_candidate("Alice",   "alice@x.com", ["python", "machine learning", "sql"], 0.65),
        _make_candidate("Bob",     "bob@x.com",   ["python"],                            0.40),
        _make_candidate("Charlie", "charlie@x.com", [],                                  0.20),
        # Dana: no email AND no phone → must be rejected for missing contact details
        _make_candidate("Dana",    "",             ["python", "machine learning"],        0.55, phone=""),
    ]
    scored = score_candidates(candidates, required, ScoringConfig(), model_key="mpnet")

    alice = next((r for r in scored if r["name"] == "Alice"), None)
    bob   = next((r for r in scored if r["name"] == "Bob"),   None)
    dana  = next((r for r in scored if r["name"] == "Dana"),  None)

    _report("score_candidates: returns list", isinstance(scored, list))
    _report("score_candidates: Alice has highest score",
            alice is not None and all(alice["final_score"] >= r["final_score"] for r in scored if r["name"] != "Alice"))
    _report("score_candidates: Alice skill_score = 1.0", alice is not None and alice["skill_score"] == 1.0)
    _report("score_candidates: scores clamped to [0,1]",
            all(0.0 <= r["final_score"] <= 1.0 for r in scored))
    _report("score_candidates: Dana rejected (no contact)",
            dana is not None and not dana["eligible"])
    _report("score_candidates: dynamic_threshold present on all",
            all("dynamic_threshold" in r for r in scored))
    _report("score_candidates: band assigned to all",
            all(r["band"] in ("Strong Fit","Borderline","Weak Fit") for r in scored))
    _report("score_candidates: band/eligible consistent (Weak Fit→ineligible)",
            all((not r["eligible"]) == (r["band"] == "Weak Fit") for r in scored))
    _report("score_candidates: confidence field present",
            all("confidence" in r for r in scored))
    _report("score_candidates: sorted descending by final_score",
            all(scored[i]["final_score"] >= scored[i+1]["final_score"] for i in range(len(scored)-1)))
    _report("score_candidates: empty candidates → empty list",
            score_candidates([], required, ScoringConfig()) == [])
    _report("score_candidates: empty required → skill_score=1.0 for eligible candidates",
            all(r["skill_score"] == 1.0 for r in
                score_candidates(candidates[:1], [], ScoringConfig(), "mpnet")))

    # top_n limiting
    scored_top2 = score_candidates(candidates, required, ScoringConfig(top_n=2), "mpnet")
    _report("score_candidates: top_n=2 limits output to 2", len(scored_top2) <= 2)


# ──────────────────────────────────────────────────────────────────────────────
# 3 · resume_parser.py
# ──────────────────────────────────────────────────────────────────────────────

def test_resume_parser():
    print("\n══ 3. resume_parser ══")
    import os, tempfile
    from resume_parser import extract_text_from_pdf, load_resumes_from_folder, _clean

    # ── _clean ──
    _report("clean: collapses tabs/spaces",
            "  " not in _clean("Hello\t World\t\t!"))
    _report("clean: normalises CRLF to LF",
            "\r" not in _clean("line1\r\nline2"))
    _report("clean: collapses 3+ blank lines",
            "\n\n\n" not in _clean("a\n\n\n\nb"))
    _report("clean: strips leading/trailing whitespace",
            _clean("  hello  ") == "hello")
    _report("clean: empty string → empty string", _clean("") == "")

    # ── extract_text_from_pdf: file-not-found ──
    result_missing = extract_text_from_pdf("/nonexistent/path/resume.pdf")
    _report("extract_text: missing file → empty string", result_missing == "")

    # ── extract_text_from_pdf: non-PDF binary ──
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(b"\x00\x01\x02\x03 not a real PDF")
        bad_path = tf.name
    try:
        bad_result = extract_text_from_pdf(bad_path)
        _report("extract_text: corrupt PDF → empty string or graceful fallback",
                isinstance(bad_result, str))
    finally:
        os.unlink(bad_path)

    # ── load_resumes_from_folder: missing directory ──
    raised = False
    try:
        load_resumes_from_folder("/no/such/dir")
    except NotADirectoryError:
        raised = True
    _report("load_folder: missing dir → NotADirectoryError", raised)

    # ── load_resumes_from_folder: empty directory (no PDFs) ──
    with tempfile.TemporaryDirectory() as tmpdir:
        raised2 = False
        try:
            load_resumes_from_folder(tmpdir)
        except FileNotFoundError:
            raised2 = True
        _report("load_folder: no PDFs → FileNotFoundError", raised2)

    # ── load_resumes_from_folder: directory with a corrupt PDF ──
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_pdf = os.path.join(tmpdir, "corrupt.pdf")
        with open(bad_pdf, "wb") as f:
            f.write(b"%PDF-1.4 \xff\xfe bad content")
        texts, failed = load_resumes_from_folder(tmpdir)
        _report("load_folder: corrupt PDF → graceful empty result",
                isinstance(texts, dict) and isinstance(failed, list))


# ──────────────────────────────────────────────────────────────────────────────
# 4 · semantic_matcher.py (no model load — tests pure-Python helpers only)
# ──────────────────────────────────────────────────────────────────────────────

def test_semantic_matcher_helpers():
    print("\n══ 4. semantic_matcher (pure-Python helpers) ══")
    from semantic_matcher import (
        _preprocess, _extract_key_sections, _chunk_text, _pool_chunks,
        cosine_similarity,
    )

    # ── _preprocess ──
    _report("preprocess: strips control chars",
            "\x00" not in _preprocess("hello\x00world"))
    _report("preprocess: collapses whitespace",
            "  " not in _preprocess("hello   world"))
    _report("preprocess: strips surrounding whitespace",
            _preprocess("  hi  ") == "hi")
    _report("preprocess: empty string → empty string", _preprocess("") == "")
    _report("preprocess: keeps normal text intact",
            _preprocess("Machine learning is great.") == "Machine learning is great.")

    # ── _extract_key_sections ──
    resume = """
John Smith
john@example.com

SKILLS
Python, Machine Learning, SQL, Docker

EXPERIENCE
3 years at Acme Corp as Data Scientist.

EDUCATION
B.Sc Computer Science, 2020

HOBBIES
Photography, Hiking
"""
    key = _extract_key_sections(resume)
    _report("key_sections: returns non-empty string", bool(key.strip()))
    _report("key_sections: skills section captured", "python" in key.lower() or "Python" in key)
    _report("key_sections: hobbies section excluded (stop pattern)",
            "photography" not in key.lower() and "hiking" not in key.lower())

    # Short/no-section resume falls back to full text
    short = "Alice Smith — Python developer with 2 years experience."
    key_short = _extract_key_sections(short)
    _report("key_sections: short text → falls back to full text", len(key_short) > 0)

    # ── _chunk_text ──
    short_text = "This is a short resume with a few words only."
    chunks_short = _chunk_text(short_text)
    _report("chunk_text: short text → single chunk", len(chunks_short) == 1)

    # _chunk_text splits on sentence boundaries (punctuation).  A single long
    # string with no punctuation is treated as one sentence and stays as one chunk
    # regardless of word count.  Use properly punctuated sentences so the regex
    # splitter activates and produces multiple chunks across the 300-word boundary.
    long_text = " ".join(
        [f"This is sentence number {i} in the resume and it contains relevant information."
         for i in range(60)]   # 60 × ~13 words ≈ 780 words — well above CHUNK_SIZE=300
    )
    chunks_long = _chunk_text(long_text)
    _report("chunk_text: long sentence-structured text → multiple chunks", len(chunks_long) > 1)
    _report("chunk_text: all chunks non-empty", all(c.strip() for c in chunks_long))
    _report("chunk_text: empty string → returns non-empty list or empty",
            isinstance(_chunk_text(""), list))

    # ── _pool_chunks ──
    embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    pooled = _pool_chunks(embs, use_max=True)
    _report("pool_chunks: output is unit-norm", abs(np.linalg.norm(pooled) - 1.0) < 1e-5)
    _report("pool_chunks: output shape matches embedding dim", pooled.shape == (3,))

    # Mean-only (Arctic mode)
    pooled_mean = _pool_chunks(embs, use_max=False)
    _report("pool_chunks: mean-only is unit-norm", abs(np.linalg.norm(pooled_mean) - 1.0) < 1e-5)

    # Single-row edge case
    single = np.array([[0.6, 0.8, 0.0]], dtype=np.float32)
    pooled_single = _pool_chunks(single, use_max=True)
    _report("pool_chunks: single embedding → unit-norm output",
            abs(np.linalg.norm(pooled_single) - 1.0) < 1e-5)

    # Empty array edge case
    empty_emb = np.zeros((0, 3), dtype=np.float32)
    pooled_empty = _pool_chunks(empty_emb)
    _report("pool_chunks: 0-row array → zero vector of correct dim", pooled_empty.shape == (3,))

    # ── cosine_similarity ──
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    _report("cosine_sim: identical vectors → 1.0", cosine_similarity(a, b) == 1.0)

    c = np.array([0.0, 1.0, 0.0])
    _report("cosine_sim: orthogonal vectors → 0.0", cosine_similarity(a, c) == 0.0)

    d = np.array([-1.0, 0.0, 0.0])
    _report("cosine_sim: opposite vectors → clamped to 0.0", cosine_similarity(a, d) == 0.0)

    z = np.array([0.0, 0.0, 0.0])
    _report("cosine_sim: zero vector → 0.0 (no div-by-zero)", cosine_similarity(a, z) == 0.0)
    _report("cosine_sim: both zero → 0.0", cosine_similarity(z, z) == 0.0)

    e = np.array([0.6, 0.8, 0.0])
    f = np.array([0.8, 0.6, 0.0])
    sim_ef = cosine_similarity(e, f)
    _report("cosine_sim: partial overlap → in [0, 1]", 0.0 <= sim_ef <= 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 5 · Score Accuracy / Integration Checks
# ──────────────────────────────────────────────────────────────────────────────

def test_score_accuracy():
    print("\n══ 5. Score accuracy & integration ══")
    from scoring_engine import ScoringConfig, score_candidates, _sigmoid_calibrate

    # Build a fully controlled small batch and verify final_score arithmetic
    def _cand(name, email, skills, sem_raw):
        return {
            "filename": f"{name}.pdf",
            "info": {
                "name": name, "email": email, "phone": "9876543210",
                "skills": skills, "experience_years": 2.0, "education": [],
            },
            "semantic_score": sem_raw,
            "resume_text": " ".join(skills),
            "semantic_skill_matches": None,
            "oom_fallback": False,
        }

    required = ["python", "sql", "machine learning"]
    c1 = _cand("Perfect", "p@x.com", ["python", "sql", "machine learning"], 0.65)  # skill=1.0
    c2 = _cand("Partial", "q@x.com", ["python"],                            0.45)  # skill=1/3
    c3 = _cand("Weak",    "r@x.com", ["javascript"],                       0.20)  # skill=0
    cfg = ScoringConfig(skill_weight=0.55, semantic_weight=0.45, exp_weight=0.0)

    results = score_candidates([c1, c2, c3], required, cfg, model_key="mpnet")
    perfect = next(r for r in results if r["name"] == "Perfect")
    partial = next(r for r in results if r["name"] == "Partial")
    weak    = next(r for r in results if r["name"] == "Weak")

    expected_p_skill  = 1.0
    calibrated_p_sem  = _sigmoid_calibrate(0.65, "mpnet")
    expected_p_final  = round(0.55 * expected_p_skill + 0.45 * calibrated_p_sem, 4)

    _report("accuracy: Perfect skill_score = 1.0", perfect["skill_score"] == 1.0)
    _report("accuracy: Perfect final_score matches formula",
            abs(perfect["final_score"] - expected_p_final) < 0.001,
            f"got {perfect['final_score']}, expected ~{expected_p_final}")

    _report("accuracy: Partial skill_score ≈ 1/3", abs(partial["skill_score"] - 1/3) < 0.01)
    _report("accuracy: Weak skill_score = 0.0", weak["skill_score"] == 0.0)

    # Skill floor: Weak has 0% skill → must be rejected (skill_below_min)
    _report("accuracy: 0% skill → rejected", not weak["eligible"])
    _report("accuracy: Weak rejection code = 'skill_below_min'",
            weak.get("rejection_code") == "skill_below_min")

    # Ranking order: perfect ≥ partial ≥ weak
    idxes = {r["name"]: i for i, r in enumerate(results)}
    _report("accuracy: Perfect ranked above Partial",
            idxes.get("Perfect", 99) < idxes.get("Partial", 99))
    _report("accuracy: Partial ranked above Weak",
            idxes.get("Partial", 99) < idxes.get("Weak", 99))

    # Weight normalisation continuity
    cfg_alt = ScoringConfig(skill_weight=0.7, semantic_weight=0.3)
    r_alt = score_candidates([c1], required, cfg_alt, "mpnet")
    _report("accuracy: alternative weights produce normalised final_score in [0,1]",
            0.0 <= r_alt[0]["final_score"] <= 1.0)

    # Edge: single candidate — dynamic threshold falls back to model default
    single_results = score_candidates([c1], required, cfg, "mpnet")
    _report("accuracy: single candidate has dynamic_threshold = model default (0.45)",
            abs(single_results[0]["dynamic_threshold"] - 0.45) < 0.001,
            f"got {single_results[0]['dynamic_threshold']}")


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def main():
    suites = [
        test_information_extractor,
        test_scoring_engine,
        test_resume_parser,
        test_semantic_matcher_helpers,
        test_score_accuracy,
    ]

    for suite in suites:
        try:
            suite()
        except Exception as exc:
            print(f"\n  {FAIL}  Suite '{suite.__name__}' raised an unexpected exception:")
            traceback.print_exc()
            _results["failed"] += 1

    total = _results["passed"] + _results["failed"] + _results["skipped"]
    print(f"\n{'═'*55}")
    print(f"  Results: {_results['passed']}/{total} passed  |  "
          f"{_results['failed']} failed  |  {_results['skipped']} skipped")
    print(f"{'═'*55}")
    if _results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
