"""
Forgetting-Curve Parameter Fitter
==================================

Fits the parameters of ExponentialDecayModel and ReviewCountDecayModel
via pooled Maximum Likelihood Estimation (MLE) over a range of training
users.  The resulting params.json is then consumed by evaluate.py.

Why pooled MLE?
  Fitting per-card and averaging ignores cross-card information and can
  be noisy for cards with few reviews.  Pooling all reviews across all
  training users gives a single set of globally optimal parameters that
  best explain the observed recall patterns across the entire population.

Models
------
  ExponentialDecay :  P(recall | t)    = exp(-k · t)
  ReviewCountDecay :  P(recall | t, n) = exp(-k0 · r^n · t)
    where t = elapsed days, n = 1-indexed review number for the card,
    k0 > 0 is the initial decay rate, and r ∈ (0, 2] is the per-review
    scaling factor (r < 1 → memory consolidation over repetitions).

Usage
-----
    # Fit on users 1–2000, save to params.json
    python fit_models.py \\
        --revlog-dir /path/to/revlogs \\
        --user-start 1 --user-end 2000 \\
        --output params.json

    # Verbose (show per-user progress)
    python fit_models.py \
        --revlog-dir /path/to/revlogs \
        --user-start 1 --user-end 2000 \
        --output params.json \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# rating → recalled  (1=Again is forgotten; 2/3/4 are recalled)
RATING_RECALLED: dict[int, int] = {1: 0, 2: 1, 3: 1, 4: 1}

# Chunk of users to load per polars scan (keeps peak memory manageable)
USER_CHUNK_SIZE = 200


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _scan_parquet(revlog_dir: Path):
    """Return a Polars LazyFrame over the revlog parquet files."""
    try:
        return pl.scan_parquet(
            str(revlog_dir / "**/*.parquet"), hive_partitioning=True
        )
    except Exception:
        return pl.scan_parquet(str(revlog_dir), hive_partitioning=True)


def get_available_user_ids(revlog_dir: Path, start: int, end: int) -> list[int]:
    """Return sorted list of user IDs in [start, end] that exist in the data."""
    lf = _scan_parquet(revlog_dir)
    ids = (
        lf.select("user_id")
          .filter((pl.col("user_id") >= start) & (pl.col("user_id") <= end))
          .unique()
          .collect()["user_id"]
          .to_list()
    )
    return sorted(ids)


def load_training_data(
    revlog_dir: Path,
    user_ids:   list[int],
    verbose:    bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load reviews for the given users and extract the three arrays required
    for MLE fitting:

        elapsed_days : gap since the last review of that card (days)
        recalled     : 1 if rating > 1 (Hard/Good/Easy), 0 if Again
        n_counts     : 1-indexed review number for the card at that review

    Only reviews with elapsed_days > 0 are included (first-time card
    introductions and same-day re-reviews are excluded).

    Data is loaded in chunks to cap peak memory usage.

    Returns
    -------
    elapsed_days, recalled, n_counts  — three parallel float32 arrays
    """
    all_elapsed: list[np.ndarray] = []
    all_recalled: list[np.ndarray] = []
    all_n: list[np.ndarray] = []

    total_chunks = max(1, len(user_ids) // USER_CHUNK_SIZE + 1)

    for chunk_idx, chunk_start in enumerate(
        range(0, len(user_ids), USER_CHUNK_SIZE), 1
    ):
        chunk_ids = user_ids[chunk_start: chunk_start + USER_CHUNK_SIZE]

        if verbose:
            print(
                f"  Loading chunk {chunk_idx}/{total_chunks} "
                f"(users {chunk_ids[0]}–{chunk_ids[-1]}) …"
            )

        lf = _scan_parquet(revlog_dir)
        df = (
            lf.filter(pl.col("user_id").is_in(chunk_ids))
              .collect()
              .with_row_index("_row_idx")
              .sort(["user_id", "day_offset", "_row_idx"])
        )

        if df.is_empty():
            continue

        # ── Per-card 1-indexed review counter ──────────────────────────────
        # We need n = "how many times has this card been reviewed, counting
        # the current review itself" (i.e. 1 for the first review, 2 for
        # the second, etc.).
        df = df.with_columns(
            pl.col("rating")
              .cum_count()
              .over(["user_id", "card_id"])
              .alias("card_review_n")              # 1-indexed
        )

        # ── Filter to evaluable reviews ─────────────────────────────────────
        # elapsed_days == -1 → first introduction; == 0 → same-day re-review.
        # Both are excluded from fitting, matching the behaviour of the model's
        # own .fit() method (valid = elapsed_days > 0).
        df = df.filter(pl.col("elapsed_days") > 0)

        if df.is_empty():
            continue

        # ── Build output arrays ─────────────────────────────────────────────
        elapsed_np  = df["elapsed_days"].cast(pl.Float32).to_numpy().copy()
        recalled_np = (df["rating"] > 1).cast(pl.Float32).to_numpy().copy()
        n_np        = df["card_review_n"].cast(pl.Float32).to_numpy().copy()

        all_elapsed.append(elapsed_np)
        all_recalled.append(recalled_np)
        all_n.append(n_np)

        del df  # free polars memory promptly

    if not all_elapsed:
        raise RuntimeError(
            "No evaluable reviews found for the given user range. "
            "Check revlog_dir and user IDs."
        )

    elapsed_days = np.concatenate(all_elapsed,  axis=0).astype(np.float64)
    recalled     = np.concatenate(all_recalled, axis=0).astype(np.float64)
    n_counts     = np.concatenate(all_n,        axis=0).astype(np.float64)

    return elapsed_days, recalled, n_counts


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_exponential(
    elapsed_days: np.ndarray,
    recalled:     np.ndarray,
) -> dict[str, float]:
    """
    Fit  P(recall | t) = exp(-k · t)  via MLE (Bernoulli log-likelihood).

    Returns {"k": <float>}
    """
    eps = 1e-9

    def neg_log_likelihood(k: float) -> float:
        p  = np.exp(-k * elapsed_days)
        ll = recalled * np.log(p + eps) + (1.0 - recalled) * np.log(1.0 - p + eps)
        return -ll.sum()

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(1e-6, 20.0),
        method="bounded",
    )

    if not result.success:
        print(
            f"  [WARNING] ExponentialDecay optimizer did not converge: "
            f"{result.message}"
        )

    k = float(result.x)
    ll = float(-result.fun)
    print(
        f"  [ExponentialDecay] MLE k = {k:.6f}   "
        f"(log-likelihood = {ll:.2f},  n = {len(elapsed_days):,})"
    )
    return {"k": k}


def fit_review_count(
    elapsed_days: np.ndarray,
    recalled:     np.ndarray,
    n_counts:     np.ndarray,
) -> dict[str, float]:
    """
    Fit  P(recall | t, n) = exp(-k0 · r^n · t)  via MLE.

    Parameters
    ----------
    k0 : initial decay rate  (bounded to (1e-6, 20])
    r  : per-review scaling  (bounded to (1e-6, 2.0])
         r < 1 → consolidation; r > 1 → sensitisation (unlikely but allowed)

    Returns {"k0": <float>, "r": <float>}
    """
    eps = 1e-9

    def neg_log_likelihood(params: np.ndarray) -> float:
        k0, r = params
        if k0 <= 0 or r <= 0:
            return 1e12
        p  = np.exp(-k0 * (r ** n_counts) * elapsed_days)
        ll = recalled * np.log(p + eps) + (1.0 - recalled) * np.log(1.0 - p + eps)
        return -ll.sum()

    # Warm start: use MLE k from the exponential model as k0, r=0.9
    warm_k = fit_exponential.__wrapped__(elapsed_days, recalled) if hasattr(
        fit_exponential, "__wrapped__"
    ) else 0.1
    x0 = [warm_k, 0.9]

    result = minimize(
        neg_log_likelihood,
        x0,
        bounds=[(1e-6, 20.0), (1e-6, 2.0)],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )

    if not result.success:
        print(
            f"  [WARNING] ReviewCountDecay optimizer did not fully converge: "
            f"{result.message}"
        )

    k0, r = float(result.x[0]), float(result.x[1])
    ll    = float(-result.fun)
    print(
        f"  [ReviewCountDecay] MLE k0 = {k0:.6f},  r = {r:.6f}   "
        f"(log-likelihood = {ll:.2f},  n = {len(elapsed_days):,})"
    )
    return {"k0": k0, "r": r}


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_data_summary(
    elapsed_days: np.ndarray,
    recalled:     np.ndarray,
    n_counts:     np.ndarray,
) -> None:
    recall_rate = recalled.mean() * 100
    print(f"\n  Total reviews (elapsed > 0) : {len(elapsed_days):>12,}")
    print(f"  Overall recall rate         : {recall_rate:>11.2f}%")
    print(f"  Elapsed days — min/median/max: "
          f"{elapsed_days.min():.1f} / "
          f"{np.median(elapsed_days):.1f} / "
          f"{elapsed_days.max():.1f}")
    print(f"  Review number n — min/median/max: "
          f"{int(n_counts.min())} / "
          f"{int(np.median(n_counts))} / "
          f"{int(n_counts.max())}")

    # Recall rate by elapsed-day bin (useful sanity-check)
    bin_edges = [0, 1, 3, 7, 14, 30, 90, 180, np.inf]
    bin_labels = ["<1d", "1-3d", "3-7d", "7-14d", "14-30d", "30-90d",
                  "90-180d", ">180d"]
    bins = np.digitize(elapsed_days, bin_edges, right=False) - 1
    print("\n  Recall rate by elapsed-day bin:")
    print(f"  {'Bin':>10}  {'N':>10}  {'Recall %':>10}")
    for i, label in enumerate(bin_labels):
        mask = bins == i
        if mask.sum() == 0:
            continue
        rate = recalled[mask].mean() * 100
        print(f"  {label:>10}  {mask.sum():>10,}  {rate:>9.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--revlog-dir", required=True,
        help="Root directory containing partitioned parquet revlog files.",
    )
    p.add_argument(
        "--user-start", type=int, required=True,
        help="First user ID to include in fitting (inclusive).",
    )
    p.add_argument(
        "--user-end", type=int, required=True,
        help="Last user ID to include in fitting (inclusive).",
    )
    p.add_argument(
        "--output", default="params.json",
        help="Output path for the fitted parameters JSON (default: params.json).",
    )
    p.add_argument(
        "--models", nargs="+",
        default=["exponential", "review_count"],
        choices=["exponential", "review_count"],
        help="Which model(s) to fit (default: both).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-chunk loading progress.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    revlog_dir = Path(args.revlog_dir)

    if not revlog_dir.exists():
        print(f"Error: revlog directory not found: {revlog_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Forgetting-Curve Parameter Fitter")
    print("=" * 60)
    print(f"  revlog_dir  : {revlog_dir}")
    print(f"  user range  : {args.user_start} – {args.user_end}")
    print(f"  models      : {', '.join(args.models)}")
    print(f"  output      : {args.output}")

    # ── Discover available users in range ────────────────────────────────
    print(f"\nDiscovering user IDs in [{args.user_start}, {args.user_end}] …")
    t0 = time.time()
    user_ids = get_available_user_ids(revlog_dir, args.user_start, args.user_end)
    print(f"  Found {len(user_ids)} users  ({time.time() - t0:.1f}s)")

    if not user_ids:
        print("No users found. Check --revlog-dir and --user-start/--user-end.",
              file=sys.stderr)
        sys.exit(1)

    # ── Load training data ────────────────────────────────────────────────
    print(f"\nLoading reviews …")
    t0 = time.time()
    elapsed_days, recalled, n_counts = load_training_data(
        revlog_dir, user_ids, verbose=args.verbose
    )
    print(f"  Done  ({time.time() - t0:.1f}s)")
    print_data_summary(elapsed_days, recalled, n_counts)

    # ── MLE fitting ───────────────────────────────────────────────────────
    output_params: dict = {}

    print("\nFitting models …")

    if "exponential" in args.models:
        print("\n  [1/2] ExponentialDecay")
        t0 = time.time()
        output_params["exponential"] = fit_exponential(elapsed_days, recalled)
        print(f"  → Done in {time.time() - t0:.2f}s")

    if "review_count" in args.models:
        print("\n  [2/2] ReviewCountDecay")
        t0 = time.time()
        output_params["review_count"] = fit_review_count(
            elapsed_days, recalled, n_counts
        )
        print(f"  → Done in {time.time() - t0:.2f}s")

    # ── Save results ──────────────────────────────────────────────────────
    output_params["meta"] = {
        "user_start":      args.user_start,
        "user_end":        args.user_end,
        "n_users_fitted":  len(user_ids),
        "n_reviews":       int(len(elapsed_days)),
        "overall_recall":  round(float(recalled.mean()), 6),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_params, f, indent=2)

    print(f"\nParameters written to {output_path}")
    print(json.dumps(
        {k: v for k, v in output_params.items() if k != "meta"},
        indent=2
    ))


if __name__ == "__main__":
    main()