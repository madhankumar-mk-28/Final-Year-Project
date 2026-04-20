"""
metrics_store.py — Records and inspects ML pipeline metrics for every screening run.

Provides both a programmatic API (used by app.py after each screening run) and
a CLI interface for post-session analysis.

API Functions:
    record_run(...)     — Append one screening run's metrics to metrics_log.jsonl
    load_all_runs()     — Load every stored run (oldest first)
    load_latest_run()   — Load the most recent run
    load_run_by_id(id)  — Load a specific run by its run_id

CLI Usage:
    python metrics_store.py                    — List all runs (one-line summaries)
    python metrics_store.py --run latest       — Full detail for the latest run
    python metrics_store.py --run <id>         — Full detail for a specific run
    python metrics_store.py --compare          — Cross-run model comparison table
    python metrics_store.py --export csv       — Export all runs to metrics_export.csv
    python metrics_store.py --clear            — Delete all metrics files

Storage:
    metrics_log.jsonl    — One JSON record per screening run (auto-rotated at 500 entries)
    embeddings_store.json — Optional raw embedding vectors (for advanced analysis)

Runs fully offline. No network calls. OS-independent (uses pathlib).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("metrics_store")

# Log file paths (created in the project root directory)
METRICS_LOG_FILE = Path("metrics_log.jsonl")
EMBEDDINGS_FILE  = Path("embeddings_store.json")

# Maximum number of records to keep — oldest entries pruned beyond this limit
MAX_METRICS_ENTRIES = 500

# Protects concurrent JSONL writes from multiple worker threads
_metrics_write_lock = threading.Lock()


def _rotate_metrics_log(max_entries: int = MAX_METRICS_ENTRIES) -> None:
    """Trim the metrics log to keep only the most recent max_entries records.

    Uses an atomic temp-file swap to prevent partial reads during rotation.
    Only triggers when the file exceeds 100KB (avoids unnecessary I/O on small files).
    """
    try:
        if not METRICS_LOG_FILE.exists():
            return

        # Skip rotation for small files (< 100KB) — saves I/O
        if METRICS_LOG_FILE.stat().st_size < 100_000:
            return

        raw = METRICS_LOG_FILE.read_text(encoding="utf-8")
        records = [r.strip() for r in raw.split("\n\n") if r.strip()]
        if len(records) <= max_entries:
            return                                     # Within limit — no action

        keep = records[-max_entries:]

        # Atomic file replacement via temp file to prevent partial reads
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=METRICS_LOG_FILE.parent,
            suffix=".tmp",
            prefix="metrics_",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write("\n\n".join(keep) + "\n\n")
            os.replace(tmp_path, METRICS_LOG_FILE)     # Atomic rename
            logger.info(
                "[MetricsStore] Rotated %s: %d → %d entries.",
                METRICS_LOG_FILE.name, len(records), max_entries,
            )
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    except OSError as exc:
        logger.warning("[MetricsStore] Could not rotate %s: %s", METRICS_LOG_FILE.name, exc)


def _compute_stats(values: list[float]) -> dict:
    """Compute descriptive statistics for a list of numeric values.

    Returns:
        dict with keys: count, mean, std (population), min, max.
        Returns zeros for all fields if the input list is empty.
    """
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    n    = len(values)
    mean = sum(values) / n
    # Population standard deviation (not sample) — appropriate for full-batch metrics
    std  = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n > 1 else 0.0

    return {
        "count": n,
        "mean":  round(mean, 4),
        "std":   round(std,  4),
        "min":   round(min(values), 4),
        "max":   round(max(values), 4),
    }


def record_run(
    run_id: str,
    model_key: str,
    job_description: str,
    similarity_results: list,
    final_scores: list,
    per_model_similarities: dict | None = None,
) -> None:
    """Append one screening run's metrics to metrics_log.jsonl (thread-safe).

    Records per-resume cosine similarities, aggregate statistics for all score
    types, and a compact ranking snapshot for CLI viewers.

    Args:
        run_id:                  Unique identifier for this run (typically ISO timestamp).
        model_key:               Embedding model used ("mpnet", "mxbai", or "arctic").
        job_description:         Job description text (truncated to 200 chars for storage).
        similarity_results:      List of per-resume similarity dicts from semantic_matcher.
        final_scores:            List of serialised candidate dicts from score_candidates.
        per_model_similarities:  Optional cross-model comparison data.
    """
    # Extract score lists for statistical computation
    sims   = [r.get("similarity_score", 0.0) for r in similarity_results]
    finals = [r.get("finalScore",        0.0) for r in final_scores]
    skills = [r.get("skillScore",        0.0) for r in final_scores]
    sems   = [r.get("semanticScore",     0.0) for r in final_scores]

    entry = {
        "run_id":       run_id,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "model_key":    model_key,
        "job_description": job_description[:200],      # Truncate for storage efficiency

        "resume_count": len(sims),

        # Per-resume cosine similarity scores (used by CLI --run viewer)
        "cosine_similarities": [
            {
                "filename":   r.get("filename", ""),
                "similarity": round(r.get("similarity_score", 0.0), 4),
            }
            for r in similarity_results
        ],

        # Aggregate statistics for each score type
        "stats":                _compute_stats(sims),
        "per_model":            per_model_similarities or {},
        "spearman":             {},                     # Reserved for future cross-model correlation
        "final_score_stats":    _compute_stats(finals),
        "skill_score_stats":    _compute_stats(skills),
        "semantic_score_stats": _compute_stats(sems),

        # Compact ranking snapshot for quick CLI viewing
        "rankings": [
            {
                "rank":          r.get("rank"),
                "name":          r.get("name", "Unknown"),
                "filename":      r.get("filename", ""),
                "skillScore":    round(r.get("skillScore",    0.0), 4),
                "semanticScore": round(r.get("semanticScore", 0.0), 4),
                "finalScore":    round(r.get("finalScore",    0.0), 4),
                "eligible":      r.get("eligible", False),
            }
            for r in final_scores
        ],

        # Full candidate data for API stats endpoint
        "final_scores": final_scores,
    }

    try:
        block = json.dumps(entry, indent=2, ensure_ascii=False) + "\n\n"
        with _metrics_write_lock:
            with open(METRICS_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(block)
            _rotate_metrics_log()                       # Trim if file exceeds max entries
        logger.info(
            "[MetricsStore] Run recorded: %s (%d resumes, model=%s).",
            run_id[:19], len(sims), model_key,
        )
    except OSError as exc:
        logger.warning("[MetricsStore] Could not write metrics log: %s", exc)


def load_all_runs() -> list[dict]:
    """Load every stored run from metrics_log.jsonl (oldest first).

    Corrupted/malformed JSON blocks are silently skipped to ensure the
    function always returns a valid list.

    Returns:
        List of run dictionaries, ordered oldest to newest.
    """
    if not METRICS_LOG_FILE.exists():
        return []

    raw = METRICS_LOG_FILE.read_text(encoding="utf-8")
    entries = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        try:
            entries.append(json.loads(block))
        except json.JSONDecodeError:
            pass                                        # Skip malformed blocks

    return entries


def load_latest_run() -> dict | None:
    """Return the most recent run entry, or None if no runs exist."""
    runs = load_all_runs()
    return runs[-1] if runs else None


def load_run_by_id(run_id: str) -> dict | None:
    """Return the first run matching the given run_id, or None if not found."""
    for run in load_all_runs():
        if run.get("run_id") == run_id:
            return run
    return None


def load_embeddings_for_run(run_id: str) -> dict | None:
    """Return stored embedding vectors for a specific run, or None.

    Embeddings are optionally saved to embeddings_store.json for advanced
    analysis (dimensionality reduction, clustering, etc.).
    """
    if not EMBEDDINGS_FILE.exists():
        return None
    try:
        with open(EMBEDDINGS_FILE, encoding="utf-8") as fh:
            store = json.load(fh)
        return store.get(run_id)
    except (OSError, json.JSONDecodeError):
        return None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI DISPLAY FUNCTIONS                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _print_run_summary(run: dict) -> None:
    """Print a compact one-line summary for a single run (used in list view)."""
    timestamp  = run.get("timestamp", "?")[:19]
    model      = run.get("model_key", "?")
    n          = run.get("resume_count", 0)
    rankings   = run.get("rankings", [])
    shortlisted = sum(1 for r in rankings if r.get("eligible"))
    avg_sim    = run.get("stats", {}).get("mean", 0.0)
    run_id     = run.get("run_id", "?")[:19]

    print(
        f"  {timestamp}  model={model:<6}  "
        f"resumes={n:<4}  shortlisted={shortlisted:<4}  "
        f"avg_sim={avg_sim:.4f}  id={run_id}"
    )


def _print_run_detail(run: dict) -> None:
    """Print full statistics and top-10 candidate rankings for a single run."""
    print("\n" + "=" * 64)
    print(f"  Run ID    : {run.get('run_id', '?')}")
    print(f"  Time      : {run.get('timestamp', '?')[:19]}")
    print(f"  Model     : {run.get('model_key', '?')}")
    print(f"  Resumes   : {run.get('resume_count', 0)}")
    print(f"  JD        : {run.get('job_description', '')[:80]}...")
    print("=" * 64)

    # Print statistics for each score type
    stat_sections = [
        ("Cosine similarity", "stats"),
        ("Final score",       "final_score_stats"),
        ("Skill score",       "skill_score_stats"),
        ("Semantic score",    "semantic_score_stats"),
    ]
    for label, key in stat_sections:
        s = run.get(key, {})
        if s.get("count"):
            print(
                f"  {label:<22} "
                f"mean={s.get('mean', 0):.4f}  "
                f"std={s.get('std', 0):.4f}  "
                f"min={s.get('min', 0):.4f}  "
                f"max={s.get('max', 0):.4f}"
            )

    # Print top-10 ranked candidates
    rankings    = run.get("rankings", [])
    shortlisted = [r for r in rankings if r.get("eligible")]
    print(f"\n  Shortlisted: {len(shortlisted)} of {len(rankings)}")
    print(f"\n  {'Rank':<5} {'Name':<28} {'Final':>6} {'Skill':>6} {'Sem':>6}")
    print("  " + "-" * 56)

    for r in rankings[:10]:
        status = "✓" if r.get("eligible") else "✗"
        print(
            f"  {status}{r.get('rank', '?'):<4} "
            f"{str(r.get('name', 'Unknown'))[:27]:<28} "
            f"{r.get('finalScore', 0):.4f}  "
            f"{r.get('skillScore', 0):.4f}  "
            f"{r.get('semanticScore', 0):.4f}"
        )

    if len(rankings) > 10:
        print(f"  ... and {len(rankings) - 10} more candidates")

    print()


def _avg_stat(runs: list[dict], key: str, stat: str) -> float:
    """Compute the mean of a specific statistic across multiple runs."""
    vals = [r.get(key, {}).get(stat, 0.0) for r in runs if r.get(key)]
    return sum(vals) / len(vals) if vals else 0.0


def print_compare() -> None:
    """Print a cross-run comparison table grouped by embedding model.

    Shows average cosine similarity, final score, skill score, and semantic
    score for each model across all stored runs.
    """
    runs = load_all_runs()
    if not runs:
        print("No runs to compare.")
        return

    # Group runs by model
    by_model: dict[str, list[dict]] = {}
    for run in runs:
        model = run.get("model_key", "unknown")
        by_model.setdefault(model, []).append(run)

    print(f"\n  Cross-Run Model Comparison  ({len(runs)} total runs)\n")
    print(
        f"  {'Model':<10} {'Runs':>5}  "
        f"{'Avg sim':>9}  {'Avg final':>9}  "
        f"{'Avg skill':>9}  {'Avg sem':>8}"
    )
    print("  " + "-" * 60)

    for model, model_runs in sorted(by_model.items()):
        print(
            f"  {model:<10} "
            f"{len(model_runs):>5}  "
            f"{_avg_stat(model_runs, 'stats',              'mean'):>9.4f}  "
            f"{_avg_stat(model_runs, 'final_score_stats',  'mean'):>9.4f}  "
            f"{_avg_stat(model_runs, 'skill_score_stats',  'mean'):>9.4f}  "
            f"{_avg_stat(model_runs, 'semantic_score_stats','mean'):>7.4f}"
        )

    print()


def export_to_csv(filepath: str = "metrics_export.csv") -> None:
    """Export all run summaries to a CSV file (one row per screening run).

    Includes: run_id, timestamp, model, resume count, shortlisted/rejected counts,
    cosine similarity stats, final score stats, skill/semantic averages, JD excerpt.
    """
    runs = load_all_runs()
    if not runs:
        print("No runs to export.")
        return

    fieldnames = [
        "run_id", "timestamp", "model_key", "resume_count",
        "shortlisted", "rejected",
        "sim_mean", "sim_std", "sim_min", "sim_max",
        "final_mean", "final_std", "final_min", "final_max",
        "skill_mean", "semantic_mean",
        "job_description",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            rankings    = run.get("rankings", [])
            shortlisted = sum(1 for r in rankings if r.get("eligible"))
            sim_s       = run.get("stats", {})
            fin_s       = run.get("final_score_stats", {})
            sk_s        = run.get("skill_score_stats", {})
            se_s        = run.get("semantic_score_stats", {})

            writer.writerow({
                "run_id":        run.get("run_id", ""),
                "timestamp":     run.get("timestamp", "")[:19],
                "model_key":     run.get("model_key", ""),
                "resume_count":  run.get("resume_count", 0),
                "shortlisted":   shortlisted,
                "rejected":      len(rankings) - shortlisted,
                "sim_mean":      sim_s.get("mean", ""),
                "sim_std":       sim_s.get("std",  ""),
                "sim_min":       sim_s.get("min",  ""),
                "sim_max":       sim_s.get("max",  ""),
                "final_mean":    fin_s.get("mean", ""),
                "final_std":     fin_s.get("std",  ""),
                "final_min":     fin_s.get("min",  ""),
                "final_max":     fin_s.get("max",  ""),
                "skill_mean":    sk_s.get("mean",  ""),
                "semantic_mean": se_s.get("mean",  ""),
                "job_description": run.get("job_description", "")[:120],
            })

    print(f"Exported {len(runs)} run(s) to {filepath}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI ENTRY POINT                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all available commands."""
    p = argparse.ArgumentParser(
        prog="python metrics_store.py",
        description="Inspect and export ML screening metrics.",
    )
    p.add_argument(
        "--run", metavar="RUN_ID",
        help="Show detail for a run_id, or 'latest' for the most recent run.",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Show cross-run model comparison table.",
    )
    p.add_argument(
        "--export", metavar="FORMAT", choices=["csv"],
        help="Export all metrics to metrics_export.csv.",
    )
    p.add_argument(
        "--clear", action="store_true",
        help="Delete metrics_log.jsonl and embeddings_store.json.",
    )
    p.add_argument(
        "--embeddings", metavar="RUN_ID",
        help="Print stored embedding vectors for a run (output can be large).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — parse arguments and dispatch to the appropriate handler."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args   = parser.parse_args(argv)

    # --clear: Delete all metrics files
    if args.clear:
        for f in (METRICS_LOG_FILE, EMBEDDINGS_FILE):
            if f.exists():
                os.remove(f)
                print(f"Deleted {f}")
            else:
                print(f"  {f} not found — nothing to delete.")
        return

    # --export csv: Dump all runs to CSV
    if args.export == "csv":
        export_to_csv()
        return

    # --compare: Cross-run model comparison table
    if args.compare:
        print_compare()
        return

    # --embeddings: Print raw embedding vectors for a specific run
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
            print("Re-run screening with save_embeddings=True to capture them.")
        else:
            for fname, vec in vecs.items():
                dim     = len(vec)
                norm    = math.sqrt(sum(v * v for v in vec))
                preview = ", ".join(f"{v:.4f}" for v in vec[:6])
                print(f"  {fname}: dim={dim}  |v|={norm:.4f}  [{preview}, ...]")
        return

    # --run <id>: Show full detail for a specific run
    if args.run:
        rid = args.run
        run = load_latest_run() if rid == "latest" else load_run_by_id(rid)
        if not run:
            print(f"Run '{rid}' not found in {METRICS_LOG_FILE}")
            sys.exit(1)
        _print_run_detail(run)
        return

    # Default: One-line summary per stored run
    runs = load_all_runs()
    if not runs:
        print(f"No metrics data found in {METRICS_LOG_FILE}.")
        print("Run the screening pipeline and then re-run this script.")
        return

    print(f"\n  All Screening Runs  ({len(runs)} total)\n")
    for run in runs:
        _print_run_summary(run)
    print()
    print("  → python metrics_store.py --run latest       full detail for latest run")
    print("  → python metrics_store.py --compare          cross-model accuracy table")
    print("  → python metrics_store.py --export csv       dump all runs to CSV")


if __name__ == "__main__":
    main()