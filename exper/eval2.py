"""
Unified Forgetting-Curve Benchmark
====================================
Fits ExponentialBeta and ReviewCountBeta on TRAIN users,
then evaluates all models on TEST users.

Root cause of previous ABNORMAL/overflow failures
---------------------------------------------------
L-BFGS-B returns ABNORMAL at iters=0 when the gradient is NaN or Inf at
the very first evaluation.  Two sources of NaN in this dataset:

  1.  r^n  with r>1 and n up to N_CAP=100:  r=0.85^100 is fine (underflows
      to 0), but scipy's numerical-difference fallback (used when jac=True
      returns NaN) re-evaluates with r=1.2, giving 1.2^100 = 8.2e7 → Inf.
      Fix: compute r^n in log-space so overflow is impossible for any r, n.

  2.  u = Δt / Δt_prev^β can be huge (p99 ~ 200+) when β<1 and a card
      comes back after months.  exp(-k*200) underflows to 0.0 exactly;
      log(0.0) = -Inf; residual * 0 = NaN.
      Fix: hard-clip u to U_CAP before exp; the gradient term (p-y)*u is
      well-behaved at the clip boundary.

  3.  np.errstate(over='ignore', invalid='ignore') wraps ALL numpy ops
      inside the loss so stray overflows never propagate to scipy.

Additional fix: analytic warm-start places k so that
  exp(−k · median_u) ≈ recall_rate,
which puts the optimizer squarely in the loss basin on the very first call.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# ★  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

REVLOG_DIR       = Path("/Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/")
TRAIN_USER_START = 1
TRAIN_USER_END   = 2000
TEST_USER_START  = 2001
TEST_USER_END    = 3001
ML_MODEL_PATH    = "best_model_opt.pth"   # set to None to skip
CHUNK_SIZE       = 200

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EPS             = 1e-9
PREV_IV_FLOOR   = 0.5    # floor for Δt_prev  (not a data filter)
U_CAP           = 50.0   # clip normalised time; exp(−k*50) ≈ 0 for any k>0
N_CAP           = 100    # cap review count for r^n computation
RATING_RECALLED = {1: 0, 2: 1, 3: 1, 4: 1}
BIN_EDGES       = [0, 1, 3, 7, 14, 30, 90, 180, np.inf]

# ─────────────────────────────────────────────────────────────────────────────
# POLARS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _scan() -> pl.LazyFrame:
    try:
        return pl.scan_parquet(str(REVLOG_DIR / "**/*.parquet"),
                               hive_partitioning=True)
    except Exception:
        return pl.scan_parquet(str(REVLOG_DIR), hive_partitioning=True)


def _get_user_ids(start: int, end: int) -> list[int]:
    return sorted(
        _scan().select("user_id")
        .filter((pl.col("user_id") >= start) & (pl.col("user_id") <= end))
        .unique().collect()["user_id"].to_list()
    )


def _load_user(uid: int) -> list[dict]:
    df = (
        _scan().filter(pl.col("user_id") == uid)
        .collect().with_row_index("_i").sort(["day_offset", "_i"])
    )
    return df.drop("_i").to_dicts()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_training_arrays(user_ids: list[int]) -> dict[str, np.ndarray]:
    """
    Returns four parallel float64 arrays (elapsed_days > 0 rows only):
      delta    – elapsed_days of the current review
      prev_iv  – elapsed_days of the previous review of that card,
                 clipped to [PREV_IV_FLOOR, ∞)
      recalled – 1 if rating > 1 else 0
      n        – 1-indexed review count for this card, capped at N_CAP
    """
    bufs: dict[str, list] = {k: [] for k in ("delta", "prev_iv", "recalled", "n")}
    n_chunks = max(1, (len(user_ids) - 1) // CHUNK_SIZE + 1)

    for ci, cs in enumerate(range(0, len(user_ids), CHUNK_SIZE), 1):
        chunk = user_ids[cs: cs + CHUNK_SIZE]
        print(f"  chunk {ci}/{n_chunks}  "
              f"(users {chunk[0]}–{chunk[-1]}) …", end="\r")

        df = (
            _scan().filter(pl.col("user_id").is_in(chunk)).collect()
            .with_row_index("_i")
            .sort(["user_id", "card_id", "day_offset", "_i"])
        )
        if df.is_empty():
            continue

        df = df.with_columns([
            # prev_interval: shift elapsed_days within each card's history.
            # first-intro elapsed_days = –1 is clipped to 0 before shifting.
            pl.col("elapsed_days")
              .clip(lower_bound=0)
              .shift(1)
              .over(["user_id", "card_id"])
              .fill_null(0.0)
              .clip(lower_bound=PREV_IV_FLOOR)
              .alias("prev_interval"),

            pl.col("rating")
              .cum_count()
              .over(["user_id", "card_id"])
              .clip(upper_bound=N_CAP)
              .alias("card_n"),
        ])

        # Sole data filter: keep only evaluable reviews
        df = df.filter(pl.col("elapsed_days") > 0)
        if df.is_empty():
            continue

        bufs["delta"].append(df["elapsed_days"].cast(pl.Float64).to_numpy().copy())
        bufs["prev_iv"].append(df["prev_interval"].cast(pl.Float64).to_numpy().copy())
        bufs["recalled"].append((df["rating"] > 1).cast(pl.Float64).to_numpy().copy())
        bufs["n"].append(df["card_n"].cast(pl.Float64).to_numpy().copy())
        del df

    print()
    if not bufs["delta"]:
        raise RuntimeError("No evaluable reviews in training range.")
    return {k: np.concatenate(v) for k, v in bufs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SAFE LOSS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _safe_nll(y: np.ndarray, p: np.ndarray) -> float:
    """NLL, guaranteed finite (returns large penalty on NaN/Inf)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        v = -(y * np.log(np.clip(p, EPS, 1 - EPS)) +
              (1 - y) * np.log(np.clip(1 - p, EPS, 1 - EPS))).mean()
    return float(v) if np.isfinite(v) else 1e12


def _safe_mean(arr: np.ndarray) -> float:
    """Mean that returns 0.0 on NaN/Inf instead of propagating."""
    with np.errstate(invalid="ignore", over="ignore"):
        v = np.nanmean(arr)
    return float(v) if np.isfinite(v) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL A  –  ExponentialBeta
#
#   u  = clip( Δt / Δt_prev^β,  0,  U_CAP )
#   P̂  = exp( −k · u )
#
#   Gradients (chain rule, for IB IA):
#     ∂NLL/∂k  = mean[ (P̂ − y) · u ]
#     ∂NLL/∂β  = mean[ (P̂ − y) · k · u · log(Δt_prev) ]
# ─────────────────────────────────────────────────────────────────────────────

class ExponentialBeta:
    def __init__(self, k: float = 0.0, beta: float = 0.95):
        self.k    = k
        self.beta = beta

    def _u(self, delta: np.ndarray, prev_iv: np.ndarray,
           beta: float | None = None) -> np.ndarray:
        b = beta if beta is not None else self.beta
        with np.errstate(over="ignore", invalid="ignore"):
            return np.clip(delta / np.maximum(prev_iv, PREV_IV_FLOOR) ** b,
                           0.0, U_CAP)

    def predict(self, delta: np.ndarray, prev_iv: np.ndarray) -> np.ndarray:
        return np.clip(np.exp(-self.k * self._u(delta, prev_iv)), EPS, 1 - EPS)

    def fit(self, arrs: dict[str, np.ndarray]) -> "ExponentialBeta":
        delta, prev_iv, y = arrs["delta"], arrs["prev_iv"], arrs["recalled"]
        prev_c = np.maximum(prev_iv, PREV_IV_FLOOR)
        log_pc = np.log(prev_c)

        # Analytic warm-start: exp(−k · median_u) = recall_rate
        u0     = self._u(delta, prev_iv, beta=self.beta)
        med_u  = max(float(np.median(u0)), 1e-6)
        recall = float(np.clip(y.mean(), EPS, 1 - EPS))
        self.k = -np.log(recall) / med_u
        print(f"  [ExpBeta] analytic warm-start  k={self.k:.5f}  "
              f"β={self.beta:.4f}  (median_u={med_u:.3f})")

        def loss_grad(params: np.ndarray):
            k, b = float(params[0]), float(params[1])
            with np.errstate(over="ignore", invalid="ignore"):
                u  = np.clip(delta / prev_c ** b, 0.0, U_CAP)
                p  = np.clip(np.exp(-k * u), EPS, 1 - EPS)
                e  = p - y
                nll = _safe_nll(y, p)
                dk  = _safe_mean(e * u)
                db  = _safe_mean(e * k * u * log_pc)
            return nll, np.array([dk, db])

        res = minimize(loss_grad, [self.k, self.beta], jac=True,
                       method="L-BFGS-B",
                       bounds=[(1e-6, 50.0), (0.0, 3.0)],
                       options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10})
        self.k, self.beta = float(res.x[0]), float(res.x[1])
        print(f"  [ExpBeta] k={self.k:.6f}  β={self.beta:.6f}  "
              f"NLL={res.fun:.6f}  iters={res.nit}  converged={res.success}")
        if not res.success:
            print(f"  [ExpBeta] WARNING: {res.message}")
        return self

    def label(self) -> str:
        return f"ExpBeta(k={self.k:.5f}, β={self.beta:.5f})"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL B  –  ReviewCountBeta
#
#   u   = clip( Δt / Δt_prev^β,  0,  U_CAP )
#   r^n = exp( n · log r )          ← log-space, no overflow for any n, r
#   P̂   = exp( −k0 · r^n · u )
#
#   Gradients:
#     ∂NLL/∂k0  = mean[ (P̂−y) · r^n · u ]
#     ∂NLL/∂r   = mean[ (P̂−y) · k0 · n · r^(n−1) · u ]
#                = mean[ (P̂−y) · k0 · n · r^n/r · u ]   (r^n reused)
#     ∂NLL/∂β   = mean[ (P̂−y) · k0 · r^n · u · log(Δt_prev) ]
# ─────────────────────────────────────────────────────────────────────────────

class ReviewCountBeta:
    def __init__(self, k0: float = 0.0, r: float = 0.85, beta: float = 0.95):
        self.k0   = k0
        self.r    = r
        self.beta = beta

    @staticmethod
    def _rn(r: float, n: np.ndarray) -> np.ndarray:
        """r^n in log-space — never overflows."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.exp(n * np.log(max(r, 1e-12)))

    def _u(self, delta: np.ndarray, prev_iv: np.ndarray,
           beta: float | None = None) -> np.ndarray:
        b = beta if beta is not None else self.beta
        with np.errstate(over="ignore", invalid="ignore"):
            return np.clip(delta / np.maximum(prev_iv, PREV_IV_FLOOR) ** b,
                           0.0, U_CAP)

    def predict(self, delta: np.ndarray, prev_iv: np.ndarray,
                n: np.ndarray) -> np.ndarray:
        rn = self._rn(self.r, n)
        return np.clip(
            np.exp(-self.k0 * rn * self._u(delta, prev_iv)), EPS, 1 - EPS
        )

    def fit(self, arrs: dict[str, np.ndarray]) -> "ReviewCountBeta":
        delta, prev_iv, y, n = (
            arrs["delta"], arrs["prev_iv"], arrs["recalled"], arrs["n"]
        )
        prev_c = np.maximum(prev_iv, PREV_IV_FLOOR)
        log_pc = np.log(prev_c)

        # Analytic warm-start
        u0     = self._u(delta, prev_iv, beta=self.beta)
        med_u  = max(float(np.median(u0)), 1e-6)
        recall = float(np.clip(y.mean(), EPS, 1 - EPS))
        # At median n, effective k ≈ k0 * r^(N_CAP/2) ≈ k0 * r^50
        r_med_n = float(self._rn(self.r, np.array([float(N_CAP / 2)]))[0])
        r_med_n = max(r_med_n, 1e-6)
        self.k0 = -np.log(recall) / (med_u * r_med_n)
        print(f"  [RCBeta]  analytic warm-start  k0={self.k0:.5f}  "
              f"r={self.r:.4f}  β={self.beta:.4f}")

        def loss_grad(params: np.ndarray):
            k0, r, b = float(params[0]), float(params[1]), float(params[2])
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                rn   = np.exp(n * np.log(max(r, 1e-12)))   # log-space r^n
                u    = np.clip(delta / prev_c ** b, 0.0, U_CAP)
                p    = np.clip(np.exp(-k0 * rn * u), EPS, 1 - EPS)
                e    = p - y
                nll  = _safe_nll(y, p)
                dk0  = _safe_mean(e * rn * u)
                dr   = _safe_mean(e * k0 * n * (rn / max(r, 1e-12)) * u)
                db   = _safe_mean(e * k0 * rn * u * log_pc)
            return nll, np.array([dk0, dr, db])

        res = minimize(loss_grad, [self.k0, self.r, self.beta], jac=True,
                       method="L-BFGS-B",
                       bounds=[(1e-6, 50.0), (0.05, 3.0), (0.0, 3.0)],
                       options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10})
        self.k0, self.r, self.beta = (
            float(res.x[0]), float(res.x[1]), float(res.x[2])
        )
        print(f"  [RCBeta]  k0={self.k0:.6f}  r={self.r:.6f}  "
              f"β={self.beta:.6f}  NLL={res.fun:.6f}  "
              f"iters={res.nit}  converged={res.success}")
        if not res.success:
            print(f"  [RCBeta] WARNING: {res.message}")
        return self

    def label(self) -> str:
        return (f"ReviewCountBeta(k0={self.k0:.5f}, "
                f"r={self.r:.5f}, β={self.beta:.5f})")


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────────────────────────────────────

ML_CONT_COLS = [
    "interval_days", "interval_sec", "interval_days_cum", "interval_sec_cum",
    "duration", "prev_duration", "diff_reviews", "diff_new_cards",
    "cum_reviews_today", "cum_new_cards_today", "day_offset_diff",
    "interval_sec_sin", "interval_sec_cos", "interval_sec_cum_sin",
    "interval_sec_cum_cos",
]

class MLModel:
    MAX_SEQ = 1536

    def __init__(self):
        import torch
        from train import DenseHybridRecallModel
        self._torch = torch
        self._dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = DenseHybridRecallModel(d_model=256, n_heads=8, num_layers=2)
        m.load_state_dict(
            torch.load(ML_MODEL_PATH, map_location=self._dev, weights_only=True)
        )
        m.eval().to(self._dev)
        self._model = m
        print(f"  [MLModel] loaded '{ML_MODEL_PATH}' on {self._dev}")

    def label(self) -> str:
        return "DenseHybridRecallModel"

    @staticmethod
    def engineer(reviews: list[dict]) -> dict | None:
        if len(reviews) < 5:
            return None
        rows = [{
            "card_id":         int(r["card_id"]),
            "day_offset":      int(r["day_offset"]),
            "elapsed_days":    float(r["elapsed_days"]),
            "elapsed_seconds": float(r.get("elapsed_seconds", 0) or 0),
            "rating":          int(r["rating"]),
            "state":           int(r.get("state", 0) or 0),
            "duration":        float(r.get("duration", 0.0) or 0.0),
        } for r in reviews]
        df = pl.DataFrame(rows).with_row_index("_i").sort(["day_offset", "_i"])

        df = df.with_columns([
            (pl.col("elapsed_days") == -1).cast(pl.Int32).alias("is_first"),
            pl.col("elapsed_days").clip(lower_bound=0).alias("interval_days"),
            pl.col("elapsed_seconds").clip(lower_bound=0).alias("interval_sec"),
            (pl.col("rating") > 1).cast(pl.Float32).alias("y"),
        ])
        df = df.with_columns([
            pl.col("interval_days").cum_sum().over("card_id").alias("interval_days_cum"),
            pl.col("interval_sec").cum_sum().over("card_id").alias("interval_sec_cum"),
            (pl.col("rating").cum_count().over("card_id") - 1)
              .clip(lower_bound=0, upper_bound=100).cast(pl.Int32).alias("card_review_count"),
            pl.col("is_first").cum_sum().alias("cum_new_cards"),
            pl.col("rating").cum_count().alias("review_th"),
        ])
        df = df.with_columns([
            pl.col("rating").shift(1).over("card_id").fill_null(0).cast(pl.Int32).alias("last_rating"),
            pl.col("state").shift(1).over("card_id").fill_null(0).cast(pl.Int32).alias("last_state"),
            pl.col("rating").shift(1).fill_null(0).cast(pl.Int32).alias("prev_rating"),
            pl.col("state").shift(1).fill_null(0).cast(pl.Int32).alias("prev_state"),
            pl.col("review_th").shift(1).over("card_id").fill_null(0).alias("lct"),
            pl.col("cum_new_cards").shift(1).over("card_id").fill_null(0).alias("lcnc"),
            pl.col("day_offset").shift(1).fill_null(pl.col("day_offset")).alias("prev_day_offset"),
            pl.col("duration").shift(1).fill_null(0.0).alias("prev_duration"),
        ])
        df = df.with_columns([
            (pl.col("review_th") - pl.col("lct") - 1)
              .clip(lower_bound=0).cast(pl.Float32).alias("diff_reviews"),
            (pl.col("cum_new_cards") - pl.col("lcnc"))
              .clip(lower_bound=0).cast(pl.Float32).alias("diff_new_cards"),
            (pl.col("day_offset") - pl.col("prev_day_offset"))
              .clip(lower_bound=0).cast(pl.Float32).alias("day_offset_diff"),
            pl.col("rating").cum_count().over("day_offset")
              .cast(pl.Float32).alias("cum_reviews_today"),
            pl.col("is_first").cum_sum().over("day_offset")
              .cast(pl.Float32).alias("cum_new_cards_today"),
        ])
        TWO_PI = 2.0 * np.pi / 86400.0
        df = df.with_columns([
            ((pl.col("interval_sec") % 86400) * TWO_PI).sin()
              .cast(pl.Float32).alias("interval_sec_sin"),
            ((pl.col("interval_sec") % 86400) * TWO_PI).cos()
              .cast(pl.Float32).alias("interval_sec_cos"),
            ((pl.col("interval_sec_cum") % 86400) * TWO_PI).sin()
              .cast(pl.Float32).alias("interval_sec_cum_sin"),
            ((pl.col("interval_sec_cum") % 86400) * TWO_PI).cos()
              .cast(pl.Float32).alias("interval_sec_cum_cos"),
        ])
        for c in ["interval_days", "interval_sec", "interval_days_cum",
                  "interval_sec_cum", "duration", "prev_duration",
                  "diff_reviews", "diff_new_cards", "cum_reviews_today",
                  "cum_new_cards_today", "day_offset_diff"]:
            if c in df.columns:
                df = df.with_columns(
                    (pl.col(c).clip(lower_bound=0).log1p() / 5.0)
                    .cast(pl.Float32).alias(c)
                )
        return {
            "maturity":    df["card_review_count"].to_numpy().copy(),
            "prev_rating": df["prev_rating"].to_numpy().copy(),
            "last_rating": df["last_rating"].to_numpy().copy(),
            "prev_state":  df["prev_state"].to_numpy().copy(),
            "last_state":  df["last_state"].to_numpy().copy(),
            "cont":        df.select(ML_CONT_COLS).to_numpy().copy(),
            "y":           df["y"].to_numpy().copy(),
            "elapsed":     df["elapsed_days"].to_numpy().copy(),
        }

    def run_inference(self, feats: dict) -> np.ndarray:
        torch = self._torch
        T     = len(feats["maturity"])
        logits = np.zeros(T, dtype=np.float32)
        lstm   = None
        for start in range(0, T, self.MAX_SEQ):
            end = min(start + self.MAX_SEQ, T)
            sl  = slice(start, end)
            def tl(a): return torch.from_numpy(a[sl]).unsqueeze(0).to(self._dev)
            def tf(a): return torch.from_numpy(a[sl]).unsqueeze(0).to(self._dev, dtype=torch.float32)
            with torch.no_grad():
                out, lstm = self._model(
                    tl(feats["maturity"]), tf(feats["cont"]),
                    tl(feats["prev_rating"]), tl(feats["last_rating"]),
                    tl(feats["prev_state"]), tl(feats["last_state"]),
                    lstm,
                )
                lstm        = tuple(s.detach() for s in lstm)
                logits[sl]  = out.squeeze(0).cpu().numpy()
        return torch.sigmoid(torch.from_numpy(logits)).numpy()

    def predict_user(self, reviews: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = self.engineer(reviews)
        if feats is None:
            return np.array([]), np.array([]), np.array([])
        probs   = self.run_inference(feats)
        mask    = feats["elapsed"] > 0
        return (feats["y"][mask].astype(np.int32),
                probs[mask].astype(np.float32),
                feats["elapsed"][mask].astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ADAPTERS
# ─────────────────────────────────────────────────────────────────────────────

class ExpBetaAdapter:
    def __init__(self, m: ExponentialBeta): self._m = m
    def label(self): return self._m.label()

    def predict_user(self, reviews):
        prev: dict[int, float] = {}
        yt, yp, el = [], [], []
        for rev in reviews:
            cid, d, r = int(rev["card_id"]), float(rev["elapsed_days"]), int(rev["rating"])
            pv = prev.get(cid)
            prev[cid] = max(float(max(d, 0.0)), PREV_IV_FLOOR)
            if d <= 0 or pv is None: continue
            p = float(self._m.predict(np.array([d]), np.array([max(pv, PREV_IV_FLOOR)]))[0])
            yt.append(RATING_RECALLED[r]); yp.append(p); el.append(d)
        return (np.array(yt, dtype=np.int32), np.array(yp, dtype=np.float32),
                np.array(el, dtype=np.float32))


class RCBetaAdapter:
    def __init__(self, m: ReviewCountBeta): self._m = m
    def label(self): return self._m.label()

    def predict_user(self, reviews):
        prev: dict[int, float] = {}
        cnt: dict[int, int]    = defaultdict(int)
        yt, yp, el = [], [], []
        for rev in reviews:
            cid, d, r = int(rev["card_id"]), float(rev["elapsed_days"]), int(rev["rating"])
            cnt[cid] += 1
            n  = float(min(cnt[cid], N_CAP))
            pv = prev.get(cid)
            prev[cid] = max(float(max(d, 0.0)), PREV_IV_FLOOR)
            if d <= 0 or pv is None: continue
            p = float(self._m.predict(np.array([d]),
                                      np.array([max(pv, PREV_IV_FLOOR)]),
                                      np.array([n]))[0])
            yt.append(RATING_RECALLED[r]); yp.append(p); el.append(d)
        return (np.array(yt, dtype=np.int32), np.array(yp, dtype=np.float32),
                np.array(el, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, elapsed) -> dict:
    if len(y_true) == 0:
        return {"RMSE": None, "LogLoss": None, "RMSE(bins)": None,
                "AUC": None, "n_reviews": 0}
    yc    = np.clip(y_pred, EPS, 1 - EPS)
    rmse  = float(root_mean_squared_error(y_true, y_pred))
    ll    = float(log_loss(y_true, yc, labels=[0, 1]))
    try:   auc = float(roc_auc_score(y_true, y_pred))
    except: auc = None
    bins  = np.digitize(elapsed, BIN_EDGES, right=False) - 1
    agg   = (pd.DataFrame({"b": bins, "y": y_true, "p": yc, "w": 1.0})
               .groupby("b").agg(y=("y","mean"), p=("p","mean"), w=("w","sum"))
               .reset_index())
    rmse_b = float(root_mean_squared_error(agg["y"], agg["p"],
                                           sample_weight=agg["w"]))
    return {"RMSE": round(rmse,6), "LogLoss": round(ll,6),
            "RMSE(bins)": round(rmse_b,6),
            "AUC": round(auc,6) if auc is not None else None,
            "n_reviews": int(len(y_true))}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("STEP 1 – FIT")
    print("=" * 72)
    train_ids = _get_user_ids(TRAIN_USER_START, TRAIN_USER_END)
    print(f"Training users: {len(train_ids)}")
    t0   = time.time()
    arrs = load_training_arrays(train_ids)
    print(f"Loaded {len(arrs['delta']):,} evaluable reviews  ({time.time()-t0:.1f}s)")
    print(f"Recall rate : {arrs['recalled'].mean()*100:.2f}%")
    t_rel = arrs["delta"] / arrs["prev_iv"]
    print(f"t_rel – median:{np.median(t_rel):.3f}  p95:{np.percentile(t_rel,95):.1f}  "
          f"max:{t_rel.max():.1f}")

    print()
    m_exp = ExponentialBeta().fit(arrs)
    print()
    m_rc  = ReviewCountBeta().fit(arrs)

    params = {
        "exponential_beta":  {"k": m_exp.k,  "beta": m_exp.beta},
        "review_count_beta": {"k0": m_rc.k0, "r": m_rc.r, "beta": m_rc.beta},
    }
    with open("params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nParams → params.json")
    print(f"  ExpBeta:  k={m_exp.k:.5f}  β={m_exp.beta:.5f}")
    print(f"  RCBeta:   k0={m_rc.k0:.5f}  r={m_rc.r:.5f}  β={m_rc.beta:.5f}")
    print(f"  r {'< 1  ✓  memory consolidation confirmed' if m_rc.r < 1 else '>= 1  ✗  unexpected'}")

    print()
    print("=" * 72)
    print("STEP 2 – EVALUATE")
    print("=" * 72)
    test_ids = _get_user_ids(TEST_USER_START, TEST_USER_END)
    print(f"Test users: {len(test_ids)}")

    adapters: list = [ExpBetaAdapter(m_exp), RCBetaAdapter(m_rc)]
    if ML_MODEL_PATH and Path(ML_MODEL_PATH).exists():
        try:    adapters.append(MLModel())
        except Exception as ex: print(f"  [MLModel] could not load: {ex}")
    elif ML_MODEL_PATH:
        print(f"  [MLModel] '{ML_MODEL_PATH}' not found – skipping")

    Path("results").mkdir(exist_ok=True)
    handles = {
        a.label(): open(
            f"results/{''.join(c if c.isalnum() else '_' for c in a.label())}.jsonl", "w"
        ) for a in adapters
    }
    gy: dict[str,list] = {a.label():[] for a in adapters}
    gp: dict[str,list] = {a.label():[] for a in adapters}
    ge: dict[str,list] = {a.label():[] for a in adapters}

    for idx, uid in enumerate(test_ids, 1):
        reviews = _load_user(uid)
        if not reviews: continue
        print(f"\n[{idx}/{len(test_ids)}] User {uid}  ({len(reviews)} reviews)")
        for adapter in adapters:
            yt, yp, el = adapter.predict_user(reviews)
            if len(yt) == 0: continue
            m = compute_metrics(yt, yp, el)
            handles[adapter.label()].write(json.dumps({"user":uid,"metrics":m})+"\n")
            gy[adapter.label()].extend(yt.tolist())
            gp[adapter.label()].extend(yp.tolist())
            ge[adapter.label()].extend(el.tolist())
            print(f"  {adapter.label():60s}  "
                  f"RMSE={m['RMSE']:.4f}  LL={m['LogLoss']:.4f}  "
                  f"RMSE(b)={m['RMSE(bins)']:.4f}  AUC={m['AUC']}")

    for h in handles.values(): h.close()

    print(); print("=" * 72); print("GLOBAL METRICS"); print("=" * 72)
    summary = []
    for adapter in adapters:
        lb = adapter.label()
        if not gy[lb]: continue
        gm = compute_metrics(np.array(gy[lb], dtype=np.int32),
                             np.array(gp[lb], dtype=np.float32),
                             np.array(ge[lb], dtype=np.float32))
        print(f"  {lb:60s}  RMSE={gm['RMSE']:.4f}  LL={gm['LogLoss']:.4f}  "
              f"RMSE(b)={gm['RMSE(bins)']:.4f}  AUC={gm['AUC']}  n={gm['n_reviews']:,}")
        summary.append({"model": lb, **gm})

    with open("results/summary.json", "w") as f: json.dump(summary, f, indent=2)
    print("\nSummary → results/summary.json")


if __name__ == "__main__":
    main()