"""
Interval-Normalised Forgetting Curve Models
============================================

THE CORE PROBLEM (and why β fixes it)
--------------------------------------
Raw elapsed_days is confounded by Anki's scheduler: the scheduler gives
LONG intervals to STRONG cards, so in the raw data long elapsed_days
correlates with HIGH recall — the opposite of what P=exp(-k·Δt) assumes.

The original fitting script fixed this by computing:

    t_rel = elapsed_days / prev_interval

and fitting on that ratio alone. This achieved AUC 0.6–0.7. But the
evaluation adapters used raw elapsed_days, undoing the fix.

THE FIX: absorb the normalisation into the model as a parameter β

    P = exp( -k · Δt / Δt_prev^β )

  β = 0  →  raw absolute time         (broken, AUC < 0.5)
  β = 1  →  pure relative delay       (what the filter computed)
  β ∈ (0,1) →  partial normalisation  (let the data decide)

This is still a 2-parameter model (k, β) for ExponentialDecay, and a
3-parameter model (k0, r, β) for ReviewCountDecay.  Both are closed-form
differentiable, so you can write out dL/dk, dL/dβ manually for the IA.

MANUAL GRADIENT DERIVATION (for IB IA)
---------------------------------------
For the exponential model:

    P̂  = exp(−k · u)            where  u = Δt / Δt_prev^β

Log-likelihood for one review:
    ℓ = y·log(P̂) + (1−y)·log(1−P̂)
      = −y·k·u  +  (1−y)·log(1 − exp(−k·u))

Partial derivatives:
    ∂ℓ/∂k = −y·u  +  (1−y)·u·exp(−k·u) / (1 − exp(−k·u))
    ∂ℓ/∂β = k·u·log(Δt_prev) · [−y  +  (1−y)·exp(−k·u)/(1−exp(−k·u))]

These are the exact gradients fed to L-BFGS-B (see neg_log_likelihood_grad
below).  A student can verify them symbolically with the chain rule.

Usage
-----
# Fit:
    python interval_normalised_models.py fit \\
        --revlog-dir /data/revlogs \\
        --user-start 1 --user-end 2000 \\
        --output params.json

# Evaluate:
    python interval_normalised_models.py eval \\
        --revlog-dir /data/revlogs \\
        --user-start 2001 --user-end 3001 \\
        --params-file params.json \\
        --output results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

USER_CHUNK_SIZE    = 200
EPS                = 1e-9
# A meaningful minimum previous interval: 0.5 days.
# Reviews where Anki reset to <0.5 days (just-failed cards) are noisy for
# normalisation; we retain them but clip Δt_prev to this floor so the ratio
# stays finite.
PREV_INTERVAL_FLOOR = 0.5
ELAPSED_BIN_EDGES  = [0, 1, 3, 7, 14, 30, 90, 180, np.inf]
RATING_RECALLED    = {1: 0, 2: 1, 3: 1, 4: 1}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  –  produces (Δt, Δt_prev, recalled, n_counts)
# ─────────────────────────────────────────────────────────────────────────────

def _scan(revlog_dir: Path) -> pl.LazyFrame:
    try:
        return pl.scan_parquet(str(revlog_dir / "**/*.parquet"),
                               hive_partitioning=True)
    except Exception:
        return pl.scan_parquet(str(revlog_dir), hive_partitioning=True)


def get_user_ids(revlog_dir: Path, start: int, end: int) -> list[int]:
    ids = (
        _scan(revlog_dir)
        .select("user_id")
        .filter((pl.col("user_id") >= start) & (pl.col("user_id") <= end))
        .unique()
        .collect()["user_id"]
        .to_list()
    )
    return sorted(ids)


def _load_chunk(revlog_dir: Path, chunk_ids: list[int]) -> pl.DataFrame | None:
    df = (
        _scan(revlog_dir)
        .filter(pl.col("user_id").is_in(chunk_ids))
        .collect()
        .with_row_index("_row_idx")
        .sort(["user_id", "card_id", "day_offset", "_row_idx"])
    )
    return None if df.is_empty() else df


def _build_arrays(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """
    From a sorted DataFrame, compute per-review arrays:
      delta      : elapsed_days (current review)  > 0
      prev_iv    : elapsed_days of the previous review of this card
                   (clipped to PREV_INTERVAL_FLOOR to keep ratio finite)
      recalled   : 1 if rating > 1
      n_counts   : 1-indexed review number for this card (1 = first intro)
    """
    # prev_interval: shift elapsed_days within each card's history
    df = df.with_columns([
        pl.col("elapsed_days")
          .clip(lower_bound=0)                   # –1 → 0 for first intro
          .shift(1)
          .over(["user_id", "card_id"])
          .fill_null(0.0)
          .alias("prev_interval_raw"),

        pl.col("rating")
          .cum_count()
          .over(["user_id", "card_id"])
          .alias("card_review_n"),
    ])

    # Clip prev_interval to floor (avoids division by tiny numbers)
    df = df.with_columns(
        pl.col("prev_interval_raw")
          .clip(lower_bound=PREV_INTERVAL_FLOOR)
          .alias("prev_interval")
    )

    # Keep only evaluable reviews: elapsed_days > 0
    # (first introductions have elapsed_days = -1 or 0)
    df = df.filter(pl.col("elapsed_days") > 0)
    if df.is_empty():
        return {}

    return {
        "delta":    df["elapsed_days"].cast(pl.Float64).to_numpy().copy(),
        "prev_iv":  df["prev_interval"].cast(pl.Float64).to_numpy().copy(),
        "recalled": (df["rating"] > 1).cast(pl.Float64).to_numpy().copy(),
        "n":        df["card_review_n"].cast(pl.Float64).to_numpy().copy(),
    }


def load_arrays(
    revlog_dir: Path,
    user_ids:   list[int],
    verbose:    bool = False,
) -> dict[str, np.ndarray]:
    buckets: dict[str, list] = {k: [] for k in ("delta", "prev_iv", "recalled", "n")}
    n_chunks = max(1, (len(user_ids) - 1) // USER_CHUNK_SIZE + 1)

    for ci, cs in enumerate(range(0, len(user_ids), USER_CHUNK_SIZE), 1):
        chunk = user_ids[cs: cs + USER_CHUNK_SIZE]
        if verbose:
            print(f"  Chunk {ci}/{n_chunks}  (users {chunk[0]}–{chunk[-1]}) …")
        df = _load_chunk(revlog_dir, chunk)
        if df is None:
            continue
        arrs = _build_arrays(df)
        if not arrs:
            continue
        for k in buckets:
            buckets[k].append(arrs[k])
        del df

    if not buckets["delta"]:
        raise RuntimeError("No evaluable reviews found.")

    return {k: np.concatenate(v) for k, v in buckets.items()}


def print_data_summary(arrs: dict[str, np.ndarray]) -> None:
    n        = len(arrs["delta"])
    recall   = arrs["recalled"].mean()
    t_rel    = arrs["delta"] / arrs["prev_iv"]
    print(f"\n  Reviews (elapsed > 0) : {n:>12,}")
    print(f"  Recall rate           : {recall*100:>9.2f}%")
    print(f"  Δt   min/median/max   : "
          f"{arrs['delta'].min():.1f} / {np.median(arrs['delta']):.1f} / "
          f"{arrs['delta'].max():.1f} days")
    print(f"  t_rel min/median/max  : "
          f"{t_rel.min():.3f} / {np.median(t_rel):.3f} / {t_rel.max():.1f}")
    print(f"  (t_rel = Δt / Δt_prev;  1.0 = reviewed exactly on schedule)")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL A  –  ExponentialDecay with β
#
#   P̂ = exp( -k · Δt / Δt_prev^β )
#
#   Parameters: k > 0, β ∈ [0, 2]
# ─────────────────────────────────────────────────────────────────────────────

class ExponentialBeta:
    """
    P̂ = exp( -k · Δt / Δt_prev^β )

    β = 0  : pure absolute time     (original broken model)
    β = 1  : pure relative delay    (what the original filter computed)
    β ∈ (0,1): data-estimated blend

    Manual gradients are written out below for the IB IA.
    """

    def __init__(self, k: float = 0.5, beta: float = 0.9):
        self.k    = float(k)
        self.beta = float(beta)

    def _u(self, delta: np.ndarray, prev_iv: np.ndarray) -> np.ndarray:
        """Normalised time  u = Δt / Δt_prev^β"""
        return delta / (np.clip(prev_iv, PREV_INTERVAL_FLOOR, None) ** self.beta)

    def predict(self, delta: np.ndarray, prev_iv: np.ndarray) -> np.ndarray:
        return np.clip(np.exp(-self.k * self._u(delta, prev_iv)), EPS, 1 - EPS)

    def fit(self, arrs: dict[str, np.ndarray]) -> "ExponentialBeta":
        delta, prev_iv, y = arrs["delta"], arrs["prev_iv"], arrs["recalled"]

        def neg_ll_and_grad(params: np.ndarray):
            k, beta   = float(params[0]), float(params[1])
            prev_clipped = np.clip(prev_iv, PREV_INTERVAL_FLOOR, None)
            u         = delta / (prev_clipped ** beta)
            ku        = k * u
            p         = np.clip(np.exp(-ku), EPS, 1 - EPS)
            q         = 1.0 - p

            # Negative log-likelihood
            nll = -(y * np.log(p) + (1 - y) * np.log(q)).mean()

            # ∂NLL/∂k  =  mean[ u · (p - y) / (p · q) · p · q ]
            #           =  mean[ u · (p - y) ]   ... after simplification
            #   Derive:  ∂log P̂/∂k = -u;  ∂log(1-P̂)/∂k = p·u/(1-p)
            #   → ∂NLL/∂k = mean[ (p - y) · u ]
            dk   = ((p - y) * u).mean()

            # ∂NLL/∂β  chain rule:  ∂u/∂β = -u · log(Δt_prev)
            log_prev = np.log(prev_clipped)
            du_dbeta = -u * log_prev
            # ∂NLL/∂β = mean[ (p - y) · k · (-du_dbeta) ]
            #          = mean[ (p - y) · k · u · log(Δt_prev) ]
            dbeta = (k * (p - y) * u * log_prev).mean()

            return nll, np.array([dk, dbeta])

        print("  [ExponentialBeta] Fitting (k, β) …")
        res = minimize(
            neg_ll_and_grad,
            [self.k, self.beta],
            jac=True,
            method="L-BFGS-B",
            bounds=[(1e-6, 50.0), (0.0, 2.0)],
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-9},
        )
        self.k, self.beta = float(res.x[0]), float(res.x[1])
        print(f"  [ExponentialBeta]  k={self.k:.5f}  β={self.beta:.5f}  "
              f"NLL={res.fun:.4f}  converged={res.success}")
        return self

    def label(self) -> str:
        return f"ExpBeta(k={self.k:.4f}, β={self.beta:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL B  –  ReviewCountDecay with β
#
#   P̂ = exp( -k0 · r^n · Δt / Δt_prev^β )
#
#   Parameters: k0 > 0, r > 0, β ∈ [0, 2]
# ─────────────────────────────────────────────────────────────────────────────

class ReviewCountBeta:
    """
    P̂ = exp( -k0 · r^n · Δt / Δt_prev^β )

    r < 1 : forgetting slows with each repetition   (memory consolidation)
    r > 1 : forgetting accelerates (unusual, but optimizer can find it)

    Three parameters: k0, r, β.  Still manually differentiable.
    """

    def __init__(self, k0: float = 0.5, r: float = 0.85, beta: float = 0.9):
        self.k0   = float(k0)
        self.r    = float(r)
        self.beta = float(beta)

    def _effective_k(self, n: np.ndarray) -> np.ndarray:
        return self.k0 * (self.r ** n)

    def _u(self, delta: np.ndarray, prev_iv: np.ndarray) -> np.ndarray:
        return delta / (np.clip(prev_iv, PREV_INTERVAL_FLOOR, None) ** self.beta)

    def predict(
        self,
        delta:   np.ndarray,
        prev_iv: np.ndarray,
        n:       np.ndarray,
    ) -> np.ndarray:
        u  = self._u(delta, prev_iv)
        eff_k = self._effective_k(n)
        return np.clip(np.exp(-eff_k * u), EPS, 1 - EPS)

    def fit(self, arrs: dict[str, np.ndarray]) -> "ReviewCountBeta":
        delta   = arrs["delta"]
        prev_iv = arrs["prev_iv"]
        y       = arrs["recalled"]
        n       = arrs["n"]
        prev_c  = np.clip(prev_iv, PREV_INTERVAL_FLOOR, None)
        log_prev = np.log(prev_c)

        def neg_ll_and_grad(params: np.ndarray):
            k0, r, beta = float(params[0]), float(params[1]), float(params[2])
            if k0 <= 0 or r <= 0:
                return 1e12, np.zeros(3)

            rn   = r ** n
            u    = delta / (prev_c ** beta)
            ku   = k0 * rn * u
            p    = np.clip(np.exp(-ku), EPS, 1 - EPS)

            nll  = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

            # residual  e = p - y  (appears in all gradients)
            e    = p - y

            # ∂NLL/∂k0  = mean[ e · rn · u ]
            dk0  = (e * rn * u).mean()

            # ∂NLL/∂r   = mean[ e · k0 · n · r^(n-1) · u ]
            #            = mean[ e · k0 · n/r · rn · u ]   (for r > 0)
            dr   = (e * k0 * (n / r) * rn * u).mean()

            # ∂NLL/∂β   = mean[ e · k0 · rn · u · log(Δt_prev) ]
            dbeta = (e * k0 * rn * u * log_prev).mean()

            return nll, np.array([dk0, dr, dbeta])

        print("  [ReviewCountBeta] Fitting (k0, r, β) …")
        res = minimize(
            neg_ll_and_grad,
            [self.k0, self.r, self.beta],
            jac=True,
            method="L-BFGS-B",
            bounds=[(1e-6, 50.0), (0.05, 2.0), (0.0, 2.0)],
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-9},
        )
        self.k0, self.r, self.beta = (
            float(res.x[0]), float(res.x[1]), float(res.x[2])
        )
        print(f"  [ReviewCountBeta]  k0={self.k0:.5f}  r={self.r:.5f}  "
              f"β={self.beta:.5f}  NLL={res.fun:.4f}  converged={res.success}")
        return self

    def label(self) -> str:
        return (f"ReviewCountBeta(k0={self.k0:.4f}, "
                f"r={self.r:.4f}, β={self.beta:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ADAPTERS  –  compute prev_interval per-card at runtime
# ─────────────────────────────────────────────────────────────────────────────

class ExponentialBetaAdapter:
    """Wraps ExponentialBeta for the evaluate harness."""

    def __init__(self, k: float, beta: float):
        self._model = ExponentialBeta(k=k, beta=beta)

    @property
    def name(self) -> str:
        return self._model.label()

    def predict_user(
        self, reviews: list[dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Per-card tracker: last elapsed_days (= the interval Anki just gave)
        card_last_interval: dict = {}

        y_true_list, y_pred_list, elapsed_list = [], [], []

        for rev in reviews:
            cid   = rev["card_id"]
            delta = float(rev["elapsed_days"])
            r     = int(rev["rating"])

            # prev_interval: the elapsed_days recorded on the previous review
            # of this card.  On the first review elapsed_days = -1 (no prev).
            prev_iv = card_last_interval.get(cid, None)

            # Update tracker BEFORE deciding whether to evaluate, so that
            # future reviews of this card will see the current elapsed_days
            if delta > 0:
                card_last_interval[cid] = delta
            elif cid not in card_last_interval:
                card_last_interval[cid] = 1.0  # sentinel for first intro

            # Only evaluate reviews where both delta and prev_iv are available
            if delta <= 0 or prev_iv is None:
                continue

            prev_iv_clipped = max(float(prev_iv), PREV_INTERVAL_FLOOR)
            p = self._model.predict(
                np.array([delta]), np.array([prev_iv_clipped])
            )[0]

            y_true_list.append(RATING_RECALLED[r])
            y_pred_list.append(float(p))
            elapsed_list.append(delta)

        return (
            np.array(y_true_list,  dtype=np.int32),
            np.array(y_pred_list,  dtype=np.float32),
            np.array(elapsed_list, dtype=np.float32),
        )


class ReviewCountBetaAdapter:
    """Wraps ReviewCountBeta for the evaluate harness."""

    def __init__(self, k0: float, r: float, beta: float):
        self._model = ReviewCountBeta(k0=k0, r=r, beta=beta)

    @property
    def name(self) -> str:
        return self._model.label()

    def predict_user(
        self, reviews: list[dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        card_last_interval: dict = {}
        card_review_count:  dict = defaultdict(int)

        y_true_list, y_pred_list, elapsed_list = [], [], []

        for rev in reviews:
            cid   = rev["card_id"]
            delta = float(rev["elapsed_days"])
            r     = int(rev["rating"])

            card_review_count[cid] += 1
            n       = float(card_review_count[cid])
            prev_iv = card_last_interval.get(cid, None)

            if delta > 0:
                card_last_interval[cid] = delta
            elif cid not in card_last_interval:
                card_last_interval[cid] = 1.0

            if delta <= 0 or prev_iv is None:
                continue

            prev_iv_clipped = max(float(prev_iv), PREV_INTERVAL_FLOOR)
            p = self._model.predict(
                np.array([delta]),
                np.array([prev_iv_clipped]),
                np.array([n]),
            )[0]

            y_true_list.append(RATING_RECALLED[r])
            y_pred_list.append(float(p))
            elapsed_list.append(delta)

        return (
            np.array(y_true_list,  dtype=np.int32),
            np.array(y_pred_list,  dtype=np.float32),
            np.array(elapsed_list, dtype=np.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    elapsed_days: np.ndarray,
) -> dict:
    if len(y_true) == 0:
        return {"RMSE": None, "LogLoss": None, "RMSE(bins)": None,
                "AUC": None, "n_reviews": 0}

    y_c      = np.clip(y_pred, EPS, 1 - EPS)
    rmse     = float(root_mean_squared_error(y_true, y_pred))
    logloss  = float(log_loss(y_true, y_c, labels=[0, 1]))
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        auc = None

    bins = np.digitize(elapsed_days, ELAPSED_BIN_EDGES, right=False) - 1
    rows = pd.DataFrame({"bin": bins, "y": y_true, "p": y_c, "w": 1})
    agg  = (rows.groupby("bin", sort=True)
                .agg(y=("y", "mean"), p=("p", "mean"), w=("w", "sum"))
                .reset_index())
    rmse_bins = float(root_mean_squared_error(
        agg["y"], agg["p"], sample_weight=agg["w"]
    ))

    return {
        "RMSE":       round(rmse, 6),
        "LogLoss":    round(logloss, 6),
        "RMSE(bins)": round(rmse_bins, 6),
        "AUC":        round(auc, 6) if auc is not None else None,
        "n_reviews":  int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _load_user_reviews(revlog_dir: Path, user_id: int) -> list[dict]:
    df = (
        _scan(revlog_dir)
        .filter(pl.col("user_id") == user_id)
        .collect()
        .with_row_index("_row_idx")
        .sort(["day_offset", "_row_idx"])
    )
    return df.drop("_row_idx").to_dicts()


def evaluate(
    revlog_dir: Path,
    adapters:   list,
    user_start: int,
    user_end:   int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {a.name: open(output_dir / f"{''.join(c if c.isalnum() else '_' for c in a.name)}.jsonl", "w")
           for a in adapters}
    acc = {a.name: {"y": [], "p": [], "e": []} for a in adapters}

    user_ids = get_user_ids(revlog_dir, user_start, user_end)
    print(f"\n{len(user_ids)} users in [{user_start}, {user_end}]")
    print("=" * 72)

    for i, uid in enumerate(user_ids, 1):
        reviews = _load_user_reviews(revlog_dir, uid)
        print(f"\n[{i}/{len(user_ids)}] User {uid}  ({len(reviews)} reviews)")
        for adapter in adapters:
            y_t, y_p, el = adapter.predict_user(reviews)
            if len(y_t) == 0:
                continue
            m = compute_metrics(y_t, y_p, el)
            out[adapter.name].write(
                json.dumps({"user": uid, "metrics": m}) + "\n"
            )
            acc[adapter.name]["y"].extend(y_t.tolist())
            acc[adapter.name]["p"].extend(y_p.tolist())
            acc[adapter.name]["e"].extend(el.tolist())
            print(f"  {adapter.name:52s}  "
                  f"RMSE={m['RMSE']:.4f}  LL={m['LogLoss']:.4f}  "
                  f"RMSE(b)={m['RMSE(bins)']:.4f}  AUC={m['AUC']}")

    for f in out.values():
        f.close()

    print("\n" + "=" * 72)
    print("GLOBAL METRICS")
    print("=" * 72)
    summary = []
    for adapter in adapters:
        a = acc[adapter.name]
        if not a["y"]:
            continue
        gm = compute_metrics(
            np.array(a["y"], dtype=np.int32),
            np.array(a["p"], dtype=np.float32),
            np.array(a["e"], dtype=np.float32),
        )
        print(f"  {adapter.name:52s}  "
              f"RMSE={gm['RMSE']:.4f}  LL={gm['LogLoss']:.4f}  "
              f"RMSE(b)={gm['RMSE(bins)']:.4f}  AUC={gm['AUC']}  "
              f"n={gm['n_reviews']:,}")
        summary.append({"model": adapter.name, **gm})

    out_path = output_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _fit_cmd(args: argparse.Namespace) -> None:
    revlog_dir = Path(args.revlog_dir)
    print("=" * 60)
    print("Interval-Normalised Model Fitter")
    print("=" * 60)

    user_ids = get_user_ids(revlog_dir, args.user_start, args.user_end)
    print(f"  users found : {len(user_ids)}")

    print("\nLoading arrays …")
    t0   = time.time()
    arrs = load_arrays(revlog_dir, user_ids, verbose=args.verbose)
    print(f"  Done  ({time.time()-t0:.1f}s)")
    print_data_summary(arrs)

    output: dict = {}

    print("\n── ExponentialBeta ──")
    m1 = ExponentialBeta().fit(arrs)
    output["exponential_beta"] = {"k": m1.k, "beta": m1.beta}

    print("\n── ReviewCountBeta ──")
    m2 = ReviewCountBeta().fit(arrs)
    output["review_count_beta"] = {"k0": m2.k0, "r": m2.r, "beta": m2.beta}

    output["meta"] = {
        "user_start":  args.user_start,
        "user_end":    args.user_end,
        "n_users":     len(user_ids),
        "n_reviews":   int(len(arrs["delta"])),
        "recall_rate": round(float(arrs["recalled"].mean()), 6),
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {args.output}")
    print(json.dumps({k: v for k, v in output.items() if k != "meta"}, indent=2))
    print("\nInterpretation guide:")
    print(f"  β  = {m1.beta:.4f}  (1.0 = pure relative delay, 0.0 = raw absolute)")
    print(f"  r  = {m2.r:.4f}  (<1.0 = forgetting slows with repetition ✓)")


def _eval_cmd(args: argparse.Namespace) -> None:
    with open(args.params_file) as f:
        params = json.load(f)

    adapters = []
    if "exp" in args.models:
        p = params["exponential_beta"]
        adapters.append(ExponentialBetaAdapter(k=p["k"], beta=p["beta"]))
    if "rc" in args.models:
        p = params["review_count_beta"]
        adapters.append(ReviewCountBetaAdapter(k0=p["k0"], r=p["r"], beta=p["beta"]))

    evaluate(
        revlog_dir  = Path(args.revlog_dir),
        adapters    = adapters,
        user_start  = args.user_start,
        user_end    = args.user_end,
        output_dir  = Path(args.output),
    )


def build_parser() -> argparse.ArgumentParser:
    p    = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    sub  = p.add_subparsers(dest="cmd", required=True)

    # fit
    fit = sub.add_parser("fit", help="Fit k/β parameters on training users")
    fit.add_argument("--revlog-dir",  required=True)
    fit.add_argument("--user-start",  type=int, required=True)
    fit.add_argument("--user-end",    type=int, required=True)
    fit.add_argument("--output",      default="params.json")
    fit.add_argument("--verbose",     action="store_true")

    # eval
    ev = sub.add_parser("eval", help="Evaluate on test users")
    ev.add_argument("--revlog-dir",   required=True)
    ev.add_argument("--user-start",   type=int, required=True)
    ev.add_argument("--user-end",     type=int, required=True)
    ev.add_argument("--params-file",  required=True)
    ev.add_argument("--models",       nargs="+", default=["exp", "rc"],
                    choices=["exp", "rc"])
    ev.add_argument("--output",       default="results/")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "fit":
        _fit_cmd(args)
    else:
        _eval_cmd(args)

"""
/Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/

# Fit on training users
python fit2.py fit \
    --revlog-dir /Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/ \
    --user-start 1 --user-end 2000 --output params.json

# Evaluate on test users  
python fit2.py eval \
    --revlog-dir /Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/ \
    --user-start 2001 --user-end 2500 \
    --params-file params.json --models exp rc
"""


if __name__ == "__main__":
    main()