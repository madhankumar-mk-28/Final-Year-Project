"""
metrics_store.py — Records and inspects ML pipeline metrics for every screening run.
CLI: python metrics_store.py [--run latest|<id>] [--compare] [--export csv] [--clear]
"""

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

METRICS_LOG_FILE = Path("metrics_log.jsonl")
EMBEDDINGS_FILE  = Path("embeddings_store.json")
MAX_METRICS_ENTRIES = 500  # oldest entries pruned when the log exceeds this many lines

_metrics_write_lock = threading.Lock()  # Protects concurrent JSONL writes from worker threads


def _rotate_metrics_log(max_entries: int = MAX_METRICS_ENTRIES) -> None:
    """Trim the metrics log to the most recent max_entries lines using an atomic temp-file swap."""
    try:
        if not METRICS_LOG_FILE.exists():
            return

        if METRICS_LOG_FILE.stat().st_size < 100_000:  # Skip rotation for small files — I/O overhead not worth it
            return

        lines = METRICS_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) <= max_entries:
            return

        keep = lines[-max_entries:]

        # Write to a temp file then atomically replace the original — prevents partial reads
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=METRICS_LOG_FILE.parent,
            suffix=".tmp",
            prefix="metrics_",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(keep) + "\n")
            os.replace(tmp_path, METRICS_LOG_FILE)
            logger.info(
                "[MetricsStore] Rotated %s: %d → %d entries.",
                METRICS_LOG_FILE.name, len(lines), max_entries,
            )
        except Exception:
            try:  # Clean up the orphaned temp file on failure
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    except OSError as exc:
        logger.warning("[MetricsStore] Could not rotate %s: %s", METRICS_LOG_FILE.name, exc)


def _compute_stats(values: list[float]) -> dict:
    """Compute count, mean, std, min, and max for a list of floats."""
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    n    = len(values)
    mean = sum(values) / n
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
    """Append one screening run's metrics to metrics_log.jsonl (thread-safe)."""
    sims   = [r.get("similarity_score", 0.0) for r in similarity_results]
    finals = [r.get("finalScore",        0.0) for r in final_scores]
    skills = [r.get("skillScore",        0.0) for r in final_scores]
    sems   = [r.get("semanticScore",     0.0) for r in final_scores]

    entry = {
        "run_id":       run_id,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "model_key":    model_key,
        "job_description": job_description[:200],
        "resume_count": len(sims),

        "cosine_similarities": [  # per-resume scores used by CLI --run viewer
            {
                "filename":   r.get("filename", ""),
                "similarity": round(r.get("similarity_score", 0.0), 4),
            }
            for r in similarity_results
        ],

        "stats":                _compute_stats(sims),
        "per_model":            per_model_similarities or {},
        "spearman":             {},  # reserved for future cross-model correlation
        "final_score_stats":    _compute_stats(finals),
        "skill_score_stats":    _compute_stats(skills),
        "semantic_score_stats": _compute_stats(sems),

        # Compact ranking snapshot for CLI viewers
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
    }

    try:
        line = json.dumps(entry) + "\n"
        with _metrics_write_lock:
            with open(METRICS_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(line)
            _rotate_metrics_log()
        logger.info(
            "[MetricsStore] Run recorded: %s (%d resumes, model=%s).",
            run_id[:19], len(sims), model_key,
        )
    except OSError as exc:
        logger.warning("[MetricsStore] Could not write metrics log: %s", exc)


def load_all_runs() -> list[dict]:
    """Load every entry from metrics_log.jsonl (oldest first); corrupted lines are silently skipped."""
    if not METRICS_LOG_FILE.exists():
        return []

    entries = []
    with open(METRICS_LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip malformed lines

    return entries


def load_latest_run() -> dict | None:
    """Return the most recent run entry, or None if no runs exist."""
    runs = load_all_runs()
    return runs[-1] if runs else None


def load_run_by_id(run_id: str) -> dict | None:
    """Return the first run whose run_id matches the given string, or None."""
    for run in load_all_runs():
        if run.get("run_id") == run_id:
            return run
    return None


def load_embeddings_for_run(run_id: str) -> dict | None:
    """Return stored embedding vectors for a specific run, or None if not found."""
    if not EMBEDDINGS_FILE.exists():
        return None
    try:
        with open(EMBEDDINGS_FILE, encoding="utf-8") as fh:
            store = json.load(fh)
        return store.get(run_id)
    except (OSError, json.JSONDecodeError):
        return None


def _print_run_summary(run: dict) -> None:
    """Print a compact one-line summary for a single run (used in the list view)."""
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
    """Return the mean of runs[*][key][stat], ignoring runs where key is absent."""
    vals = [r.get(key, {}).get(stat, 0.0) for r in runs if r.get(key)]
    return sum(vals) / len(vals) if vals else 0.0


def print_compare() -> None:
    """Print a cross-run comparison table grouped by model."""
    runs = load_all_runs()
    if not runs:
        print("No runs to compare.")
        return

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
    """Export all run summaries to metrics_export.csv; one row per screening run."""
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


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
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
    """Parse CLI arguments and dispatch to the appropriate reporter."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.clear:  # Delete all metrics files
        for f in (METRICS_LOG_FILE, EMBEDDINGS_FILE):
            if f.exists():
                os.remove(f)
                print(f"Deleted {f}")
            else:
                print(f"  {f} not found — nothing to delete.")
        return

    if args.export == "csv":  # Export every run to CSV
        export_to_csv()
        return

    if args.compare:  # Cross-run model comparison table
        print_compare()
        return

    if args.embeddings:  # Print raw embedding vectors for a specific run
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

    if args.run:  # Show full detail for one specific run
        rid = args.run
        run = load_latest_run() if rid == "latest" else load_run_by_id(rid)
        if not run:
            print(f"Run '{rid}' not found in {METRICS_LOG_FILE}")
            sys.exit(1)
        _print_run_detail(run)
        return

    runs = load_all_runs()  # Default: one-line summary per stored run
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