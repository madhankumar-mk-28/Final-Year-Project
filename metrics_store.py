"""
metrics_store.py
----------------
Dedicated module for recording and analysing ML pipeline performance metrics.

Every screening run can optionally save:
  • Per-resume cosine similarity scores for each embedding model
  • Summary statistics (mean, std-dev, min, max) per model
  • Spearman rank-correlation between model pairs (model agreement / accuracy proxy)
  • Embedding vectors (serialised as lists) for reproducibility — opt-in via
    save_embeddings=True because they can be large

Data is written to ``metrics_log.jsonl`` (one JSON object per line, newest last).
A companion ``embeddings_store.json`` holds raw vectors keyed by run_id when
embeddings are requested.

Usage — run as CLI to inspect stored data:

    python metrics_store.py                  # print summary of every run
    python metrics_store.py --run latest     # detail for the latest run
    python metrics_store.py --compare        # Spearman correlations across runs
    python metrics_store.py --export csv     # dump to metrics_export.csv
    python metrics_store.py --clear          # delete all stored metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("metrics_store")

# ── File paths ────────────────────────────────────────────────────────────────
METRICS_LOG_FILE    = Path("metrics_log.jsonl")
EMBEDDINGS_FILE     = Path("embeddings_store.json")


# ─────────────────────────────────────────────────────────────────────────────
# WRITING / RECORDING
# ─────────────────────────────────────────────────────────────────────────────

def record_run(
    *,
    run_id: str,
    model_key: str,
    job_description: str,
    similarity_results: list[dict],
    per_model_similarities: dict[str, list[dict]] | None = None,
    final_scores: list[dict] | None = None,
    embeddings: dict[str, list[float]] | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Persist a full metrics snapshot for one screening run.

    Parameters
    ----------
    run_id                  Unique run identifier (e.g. UTC timestamp string).
    model_key               Active model key: "mpnet", "mxbai", "arctic", or "ensemble".
    job_description         Job description text (first 200 chars are stored).
    similarity_results      Output of ``rank_resumes_by_similarity`` /
                            ``rank_resumes_ensemble`` — list of
                            {"filename": str, "similarity_score": float}.
    per_model_similarities  When model_key=="ensemble", pass the raw per-model
                            results here as {"mpnet": [...], "mxbai": [...], ...}.
                            Enables pairwise Spearman comparisons.
    final_scores            Serialised screening output — list of candidate dicts
                            containing at least {"filename", "finalScore",
                            "skillScore", "semanticScore"}.
    embeddings              Optional dict of {"filename": vector_list}.  Stored
                            in embeddings_store.json (separate file, can be large).
    extra                   Any additional metadata to include verbatim.

    Returns
    -------
    The complete metrics dict that was written to disk.
    """
    sim_values = [r["similarity_score"] for r in similarity_results if "similarity_score" in r]

    entry: dict[str, Any] = {
        "run_id":          run_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "model_key":       model_key,
        "job_description": (job_description or "")[:200],
        "resume_count":    len(similarity_results),

        # ── Per-resume cosine similarities ────────────────────────────────────
        "cosine_similarities": [
            {"filename": r["filename"], "similarity": r["similarity_score"]}
            for r in similarity_results
        ],

        # ── Descriptive statistics ────────────────────────────────────────────
        "stats": _describe(sim_values),

        # ── Per-model breakdown (ensemble only) ───────────────────────────────
        "per_model": _build_per_model(per_model_similarities) if per_model_similarities else {},

        # ── Spearman rank correlations between model pairs ────────────────────
        "spearman": _spearman_all_pairs(per_model_similarities) if per_model_similarities else {},

        # ── Final scoring metrics (when full pipeline output is available) ────
        "final_score_stats": _describe(
            [c["finalScore"] for c in (final_scores or []) if "finalScore" in c]
        ),
        "skill_score_stats": _describe(
            [c["skillScore"] for c in (final_scores or []) if "skillScore" in c]
        ),
        "semantic_score_stats": _describe(
            [c["semanticScore"] for c in (final_scores or []) if "semanticScore" in c]
        ),

        # ── Candidate ranking list (name, file, final_score) ─────────────────
        "rankings": [
            {
                "rank":       c.get("rank"),
                "name":       c.get("name", "Unknown"),
                "filename":   c.get("filename", ""),
                "skillScore":    round(float(c.get("skillScore", 0)), 4),
                "semanticScore": round(float(c.get("semanticScore", 0)), 4),
                "finalScore":    round(float(c.get("finalScore", 0)), 4),
                "eligible":   c.get("eligible", False),
            }
            for c in (final_scores or [])
        ],

        **(extra or {}),
    }

    # ── Append to JSONL log ───────────────────────────────────────────────────
    try:
        with open(METRICS_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        logger.info("[MetricsStore] Run '%s' recorded to %s", run_id, METRICS_LOG_FILE)
    except OSError as exc:
        logger.warning("[MetricsStore] Could not write metrics log: %s", exc)

    # ── Save embeddings separately (opt-in) ───────────────────────────────────
    if embeddings:
        _append_embeddings(run_id, embeddings)

    return entry


def _describe(values: list[float]) -> dict:
    """Return mean, std (sample), min, max, count for a list of floats."""
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    n    = len(values)
    mean = sum(values) / n
    # Sample standard deviation (ddof=1) — unbiased estimator for n > 1
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {
        "count": n,
        "mean":  round(mean, 4),
        "std":   round(std, 4),
        "min":   round(min(values), 4),
        "max":   round(max(values), 4),
    }


def _build_per_model(per_model: dict[str, list[dict]]) -> dict:
    """Build per-model similarity stats from per_model_similarities."""
    out = {}
    for model_key, results in per_model.items():
        vals = [r["similarity_score"] for r in results if "similarity_score" in r]
        out[model_key] = _describe(vals)
    return out


def _spearman_all_pairs(per_model: dict[str, list[dict]]) -> dict:
    """
    Compute pairwise Spearman rank-correlation coefficients between all model pairs.

    A high rho (e.g. > 0.85) means the two models produce very similar rankings —
    the system is robust.  A lower rho suggests the models disagree on candidate
    ordering, which is worth investigating.
    """
    keys   = list(per_model.keys())
    result = {}

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            pair_key = f"{ka}_vs_{kb}"
            try:
                result[pair_key] = _spearman(per_model[ka], per_model[kb])
            except (ValueError, KeyError, ZeroDivisionError) as exc:
                result[pair_key] = {"rho": None, "error": str(exc)}

    return result


def _spearman(results_a: list[dict], results_b: list[dict]) -> dict:
    """
    Compute the Spearman rank-correlation (rho) between two model result lists.

    Both lists must share filenames; only the intersection is used.
    Returns rho in [-1, 1] where 1 = identical ranking, 0 = no correlation.
    """
    # Build filename → score maps
    map_a = {r["filename"]: r["similarity_score"] for r in results_a}
    map_b = {r["filename"]: r["similarity_score"] for r in results_b}

    common = sorted(set(map_a) & set(map_b))
    n = len(common)

    if n < 2:
        return {"rho": None, "n": n, "note": "Not enough common resumes for correlation"}

    # Rank in descending order (highest score → rank 1)
    def rank_list(scores: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)
        ranks = [0.0] * n
        for rank_pos, idx in enumerate(sorted_idx, start=1):
            ranks[idx] = float(rank_pos)
        return ranks

    scores_a = [map_a[f] for f in common]
    scores_b = [map_b[f] for f in common]

    ranks_a = rank_list(scores_a)
    ranks_b = rank_list(scores_b)

    # Spearman rho = 1 - (6 * sum(d²)) / (n*(n²-1))
    d_sq_sum = sum((ra - rb) ** 2 for ra, rb in zip(ranks_a, ranks_b))
    rho = 1.0 - (6.0 * d_sq_sum) / (n * (n ** 2 - 1))

    return {"rho": round(rho, 4), "n": n}


def _append_embeddings(run_id: str, embeddings: dict[str, list[float]]):
    """Append/update embeddings in embeddings_store.json keyed by run_id."""
    try:
        store: dict = {}
        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, encoding="utf-8") as fh:
                store = json.load(fh)
    except (OSError, json.JSONDecodeError):
        store = {}

    store[run_id] = {
        fname: vec for fname, vec in embeddings.items()
    }

    try:
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2)
        logger.info("[MetricsStore] Embeddings for run '%s' saved to %s", run_id, EMBEDDINGS_FILE)
    except OSError as exc:
        logger.warning("[MetricsStore] Could not write embeddings: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# READING / QUERYING
# ─────────────────────────────────────────────────────────────────────────────

def load_all_runs() -> list[dict]:
    """Load and return all metric entries from metrics_log.jsonl (oldest first)."""
    if not METRICS_LOG_FILE.exists():
        return []
    entries = []
    with open(METRICS_LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def load_latest_run() -> dict | None:
    """Return the most recent metrics entry, or None if no runs exist."""
    runs = load_all_runs()
    return runs[-1] if runs else None


def load_run_by_id(run_id: str) -> dict | None:
    """Return the first entry whose run_id matches, or None."""
    for run in load_all_runs():
        if run.get("run_id") == run_id:
            return run
    return None


def load_embeddings_for_run(run_id: str) -> dict | None:
    """Return the embedding dict for a specific run, or None."""
    if not EMBEDDINGS_FILE.exists():
        return None
    try:
        with open(EMBEDDINGS_FILE, encoding="utf-8") as fh:
            store = json.load(fh)
        return store.get(run_id)
    except (OSError, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI / REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pct(val: float | None) -> str:
    return f"{val * 100:.1f}%" if val is not None else "—"


def _print_run_summary(run: dict):
    """Print a compact single-line summary for one run."""
    stats  = run.get("stats", {})
    count  = run.get("resume_count", "?")
    model  = run.get("model_key", "?")
    ts     = run.get("timestamp", "?")[:19]
    mean   = _fmt_pct(stats.get("mean"))
    mx     = _fmt_pct(stats.get("max"))
    mn     = _fmt_pct(stats.get("min"))
    std    = _fmt_pct(stats.get("std"))
    rid    = run.get("run_id", "?")[:20]
    print(f"  {ts}  model={model:<10}  n={count:<3}  "
          f"cosine — mean={mean}  max={mx}  min={mn}  std={std}  id={rid}")


def _print_run_detail(run: dict):
    """Print full detail for one run."""
    sep = "─" * 70

    print(f"\n{sep}")
    print(f"  Run ID    : {run.get('run_id', '?')}")
    print(f"  Timestamp : {run.get('timestamp', '?')}")
    print(f"  Model     : {run.get('model_key', '?')}")
    print(f"  Resumes   : {run.get('resume_count', '?')}")
    jd = run.get("job_description", "")
    print(f"  JD        : {jd[:80]}{'…' if len(jd) > 80 else ''}")
    print(sep)

    # ── Cosine similarity per resume ──────────────────────────────────────────
    print("\n  Cosine Similarity  (document vs. job description)")
    print(f"  {'Filename':<40}  {'Score':>7}")
    print(f"  {'─'*40}  {'─'*7}")
    for item in run.get("cosine_similarities", []):
        fname = item["filename"][:40]
        sim   = _fmt_pct(item.get("similarity"))
        print(f"  {fname:<40}  {sim:>7}")

    # ── Summary statistics ────────────────────────────────────────────────────
    stats = run.get("stats", {})
    print(f"\n  Cosine Similarity Statistics")
    print(f"    count  : {stats.get('count', '—')}")
    print(f"    mean   : {_fmt_pct(stats.get('mean'))}")
    print(f"    std    : {_fmt_pct(stats.get('std'))}")
    print(f"    min    : {_fmt_pct(stats.get('min'))}")
    print(f"    max    : {_fmt_pct(stats.get('max'))}")

    # ── Per-model (ensemble) breakdown ────────────────────────────────────────
    per_model = run.get("per_model", {})
    if per_model:
        print(f"\n  Per-Model Cosine Similarity Breakdown")
        print(f"  {'Model':<12}  {'Mean':>7}  {'Max':>7}  {'Min':>7}  {'Std':>7}")
        print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
        for mkey, mstats in per_model.items():
            print(
                f"  {mkey:<12}  "
                f"{_fmt_pct(mstats.get('mean')):>7}  "
                f"{_fmt_pct(mstats.get('max')):>7}  "
                f"{_fmt_pct(mstats.get('min')):>7}  "
                f"{_fmt_pct(mstats.get('std')):>7}"
            )

    # ── Spearman correlations ─────────────────────────────────────────────────
    spearman = run.get("spearman", {})
    if spearman:
        print(f"\n  Model Agreement — Spearman Rank Correlation (ρ)")
        print(f"  Higher ρ = models produce more similar candidate rankings.")
        print(f"  {'Pair':<28}  {'ρ':>7}  {'n':>5}")
        print(f"  {'─'*28}  {'─'*7}  {'─'*5}")
        for pair, sp in spearman.items():
            rho = sp.get("rho")
            rho_str = f"{rho:.4f}" if rho is not None else "—"
            print(f"  {pair:<28}  {rho_str:>7}  {sp.get('n', '?'):>5}")

    # ── Final score statistics ────────────────────────────────────────────────
    final_stats = run.get("final_score_stats", {})
    skill_stats = run.get("skill_score_stats", {})
    sem_stats   = run.get("semantic_score_stats", {})
    if final_stats.get("count"):
        print(f"\n  Score Statistics — Shortlisted Candidates")
        for label, s in [("Final", final_stats), ("Skill", skill_stats), ("Semantic", sem_stats)]:
            if s.get("count"):
                print(
                    f"  {label:<10}  mean={_fmt_pct(s['mean'])}  "
                    f"max={_fmt_pct(s['max'])}  min={_fmt_pct(s['min'])}  "
                    f"std={_fmt_pct(s['std'])}  (n={s['count']})"
                )

    # ── Candidate rankings ────────────────────────────────────────────────────
    rankings = run.get("rankings", [])
    if rankings:
        print(f"\n  Candidate Rankings")
        print(f"  {'Rank':<6}  {'Name':<22}  {'Skill':>6}  {'Semantic':>9}  {'Final':>6}  {'Eligible'}")
        print(f"  {'─'*6}  {'─'*22}  {'─'*6}  {'─'*9}  {'─'*6}  {'─'*8}")
        for c in rankings:
            rank = c.get("rank") or "—"
            print(
                f"  {str(rank):<6}  {str(c.get('name', '?'))[:22]:<22}  "
                f"{_fmt_pct(c.get('skillScore')):>6}  "
                f"{_fmt_pct(c.get('semanticScore')):>9}  "
                f"{_fmt_pct(c.get('finalScore')):>6}  "
                f"{'Yes' if c.get('eligible') else 'No'}"
            )

    print(f"\n{sep}")


def export_to_csv(output_path: str = "metrics_export.csv"):
    """Export all runs to a flat CSV file (one row per candidate per run)."""
    runs = load_all_runs()
    if not runs:
        print("No metrics data found. Run the screening pipeline first.")
        return

    fieldnames = [
        "run_id", "timestamp", "model_key", "job_description",
        "resume_count", "cosine_mean", "cosine_std", "cosine_min", "cosine_max",
        "candidate_rank", "candidate_name", "candidate_file",
        "skill_score", "semantic_score", "final_score", "eligible",
    ]

    rows = []
    for run in runs:
        base = {
            "run_id":          run.get("run_id", ""),
            "timestamp":       run.get("timestamp", ""),
            "model_key":       run.get("model_key", ""),
            "job_description": run.get("job_description", "")[:80],
            "resume_count":    run.get("resume_count", ""),
            "cosine_mean":     run.get("stats", {}).get("mean", ""),
            "cosine_std":      run.get("stats", {}).get("std", ""),
            "cosine_min":      run.get("stats", {}).get("min", ""),
            "cosine_max":      run.get("stats", {}).get("max", ""),
        }
        rankings = run.get("rankings", [])
        if rankings:
            for c in rankings:
                rows.append({
                    **base,
                    "candidate_rank":  c.get("rank", ""),
                    "candidate_name":  c.get("name", ""),
                    "candidate_file":  c.get("filename", ""),
                    "skill_score":     c.get("skillScore", ""),
                    "semantic_score":  c.get("semanticScore", ""),
                    "final_score":     c.get("finalScore", ""),
                    "eligible":        c.get("eligible", ""),
                })
        else:
            rows.append({**base, "candidate_rank": "", "candidate_name": "",
                         "candidate_file": "", "skill_score": "", "semantic_score": "",
                         "final_score": "", "eligible": ""})

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} row(s) from {len(runs)} run(s) → {output_path}")


def print_compare():
    """
    Cross-run model comparison — shows average cosine similarity per model
    and average Spearman rho (model agreement) across all ensemble runs.
    """
    runs = load_all_runs()
    if not runs:
        print("No metrics data found.")
        return

    model_means:  dict[str, list[float]] = {}
    spearman_rho: dict[str, list[float]] = {}

    for run in runs:
        mkey = run.get("model_key", "?")
        mean = run.get("stats", {}).get("mean")
        if mean is not None:
            model_means.setdefault(mkey, []).append(mean)

        for pair, sp in run.get("spearman", {}).items():
            rho = sp.get("rho")
            if rho is not None:
                spearman_rho.setdefault(pair, []).append(rho)

    print(f"\n  Cross-Run Model Comparison  ({len(runs)} total run(s))")
    print(f"  {'Model':<12}  {'Runs':>5}  {'Avg Cosine':>11}")
    print(f"  {'─'*12}  {'─'*5}  {'─'*11}")
    for mkey, vals in sorted(model_means.items()):
        avg = sum(vals) / len(vals)
        print(f"  {mkey:<12}  {len(vals):>5}  {_fmt_pct(avg):>11}")

    if spearman_rho:
        print(f"\n  Average Spearman ρ (model agreement) across ensemble runs")
        print(f"  {'Pair':<28}  {'Runs':>5}  {'Avg ρ':>7}")
        print(f"  {'─'*28}  {'─'*5}  {'─'*7}")
        for pair, vals in sorted(spearman_rho.items()):
            avg = sum(vals) / len(vals)
            print(f"  {pair:<28}  {len(vals):>5}  {avg:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN / CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python metrics_store.py",
        description="Inspect and export ML screening metrics (cosine similarity, embeddings, accuracy).",
    )
    p.add_argument(
        "--run", metavar="RUN_ID",
        help="Show detail for a specific run_id, or 'latest' for the most recent run.",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Show cross-run model comparison and Spearman correlations.",
    )
    p.add_argument(
        "--export", metavar="FORMAT",
        choices=["csv"],
        help="Export all metrics to a CSV file (metrics_export.csv).",
    )
    p.add_argument(
        "--clear", action="store_true",
        help="Delete metrics_log.jsonl and embeddings_store.json.",
    )
    p.add_argument(
        "--embeddings", metavar="RUN_ID",
        help="Print the stored embedding vectors for a run (can be large).",
    )
    return p


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ── --clear ───────────────────────────────────────────────────────────────
    if args.clear:
        for f in (METRICS_LOG_FILE, EMBEDDINGS_FILE):
            if f.exists():
                os.remove(f)
                print(f"Deleted {f}")
            else:
                print(f"  {f} not found — nothing to delete.")
        return

    # ── --export csv ──────────────────────────────────────────────────────────
    if args.export == "csv":
        export_to_csv()
        return

    # ── --compare ─────────────────────────────────────────────────────────────
    if args.compare:
        print_compare()
        return

    # ── --embeddings RUN_ID ───────────────────────────────────────────────────
    if args.embeddings:
        rid = args.embeddings
        if rid == "latest":
            run = load_latest_run()
            rid = run["run_id"] if run else None
        if not rid:
            print("No runs found.")
            return
        vecs = load_embeddings_for_run(rid)
        if vecs is None:
            print(f"No embeddings stored for run '{rid}'.")
            print(f"Re-run screening with save_embeddings=True to capture them.")
        else:
            for fname, vec in vecs.items():
                dim  = len(vec)
                norm = math.sqrt(sum(v * v for v in vec))
                preview = ", ".join(f"{v:.4f}" for v in vec[:6])
                print(f"  {fname}: dim={dim}  |v|={norm:.4f}  [{preview}, ...]")
        return

    # ── --run or default (list all) ───────────────────────────────────────────
    if args.run:
        rid = args.run
        run = load_latest_run() if rid == "latest" else load_run_by_id(rid)
        if not run:
            print(f"Run '{rid}' not found in {METRICS_LOG_FILE}")
            sys.exit(1)
        _print_run_detail(run)
    else:
        runs = load_all_runs()
        if not runs:
            print(f"No metrics data found in {METRICS_LOG_FILE}.")
            print("Run the screening pipeline (python app.py + Upload → Run Screening)")
            print("and then re-run this script.")
            return
        print(f"\n  All Screening Runs  ({len(runs)} total)\n")
        for run in runs:
            _print_run_summary(run)
        print()
        print(f"  → python metrics_store.py --run latest      (full detail for latest run)")
        print(f"  → python metrics_store.py --compare         (cross-model accuracy)")
        print(f"  → python metrics_store.py --export csv      (dump to CSV)")


if __name__ == "__main__":
    main()
