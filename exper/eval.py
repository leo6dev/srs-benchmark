"""
Model-Agnostic Evaluation Harness for Spaced Repetition Recall Models
======================================================================

Evaluates one or more models over a specified range of user IDs.
Each user's FULL review history is used for evaluation — there is no
within-user train/test split. The user-level split is:
  - Training users  (e.g. 1    – 2000): used in fit_models.py to find params
  - Evaluation users (e.g. 2001 – 3001): used here; their full history is the
                                          evaluation set

Metrics (matching srs-benchmark's get_stats):
  RMSE       – root mean squared error on raw predictions
  LogLoss    – binary cross-entropy
  RMSE(bins) – calibration RMSE: per-elapsed-day-bin (mean_y vs mean_p),
               weighted by bin size
  AUC        – ROC area under curve

Usage
-----
    # Forgetting-curve models only:
    python eval.py \
        --revlog-dir /Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/ \
        --user-start 2001 --user-end 2500 \
        --params-file params.json \
        --models exponential review_count \
        --output results/

    python eval.py \
        --revlog-dir /Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/ \
        --user-start 2001 --user-end 2500 \
        --params-file params.json \
        --model-path b/Users/leo/PycharmProjects/srs-benchmark/exper/best_model_opt.pth \
        --models ml\
        --output results/

    # All three models including the ML model:
    python evaluate.py \\
        --revlog-dir /path/to/revlogs \\
        --user-start 2001 --user-end 3001 \\
        --params-file params.json \\
        --model-path best_model_opt.pth \\
        --models exponential review_count ml \\
        --output results/

    python eval.py \
        --revlog-dir /Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/ \
        --user-start 2001 --user-end 2500 \
        --params-file params.json \
        --model-path /Users/leo/PycharmProjects/srs-benchmark/exper/best_model_opt.pth \
        --models ml \
        --output results/

        python eval.py \
        --revlog-dir /Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/ \
        --user-start 2001 --user-end 2500 \
        --params-file params.json \
        --models exponential \
        --output results/
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import root_mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# rating → recalled (1=Again is forgotten; 2/3/4 are recalled)
RATING_RECALLED: dict[int, int] = {1: 0, 2: 1, 3: 1, 4: 1}

# Elapsed-day bin edges for RMSE(bins).
# Log-spaced to mirror difficulty tiers in spaced repetition.
# Each review is assigned to the bin matching its elapsed_days value.
ELAPSED_BIN_EDGES: list[float] = [0, 1, 3, 7, 14, 30, 90, 180, np.inf]

# Continuous features required by the ML model (must match training order)
ML_CONT_COLS: list[str] = [
    "interval_days", "interval_sec", "interval_days_cum", "interval_sec_cum",
    "duration", "prev_duration", "diff_reviews", "diff_new_cards",
    "cum_reviews_today", "cum_new_cards_today", "day_offset_diff",
    "interval_sec_sin", "interval_sec_cos", "interval_sec_cum_sin", "interval_sec_cum_cos",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: assign elapsed-day bins
# ─────────────────────────────────────────────────────────────────────────────

def assign_elapsed_bins(elapsed_days: np.ndarray) -> np.ndarray:
    """Map each elapsed_days value to an integer bin index."""
    return np.digitize(elapsed_days, ELAPSED_BIN_EDGES, right=False) - 1


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ADAPTER INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class ModelAdapter(ABC):
    """
    Stateless adapter that wraps a recall model and produces per-review
    (y_true, y_pred, elapsed_days) for all *evaluable* reviews in a single
    user's chronological history.

    Evaluable = reviews where elapsed_days > 0.
      - elapsed_days == -1 → first introduction of the card (no prior history
                             to forget; not a valid recall test).
      - elapsed_days == 0  → same-day re-review; excluded for consistency with
                             how the forgetting-curve models were trained.
      - elapsed_days >  0  → genuine recall test after a gap; INCLUDED.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier used in output files."""
        ...

    @abstractmethod
    def predict_user(
        self, reviews: list[dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            reviews: Full review history for ONE user, sorted chronologically
                     by (day_offset, original row order).  Each dict contains
                     at minimum: card_id, day_offset, elapsed_days,
                     elapsed_seconds, rating, state, duration (may be absent).

        Returns:
            y_true       : (N,) int32  – 1 if recalled, 0 if forgotten
            y_pred       : (N,) float32 – predicted P(recall) ∈ [0, 1]
            elapsed_days : (N,) float32 – raw elapsed days (for RMSE binning)
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER: EXPONENTIAL DECAY  e^{-k·t}
# ─────────────────────────────────────────────────────────────────────────────

class ExponentialDecayAdapter(ModelAdapter):
    """
    Classic Ebbinghaus exponential decay:  P(recall) = exp(-k · elapsed_days)

    Parameters are loaded from the JSON produced by fit_models.py.
    """

    def __init__(self, k: float):
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self._k = float(k)

    @property
    def name(self) -> str:
        return f"ExponentialDecay(k={self._k:.5f})"

    def predict_user(self, reviews):
        y_true_list, y_pred_list, elapsed_list = [], [], []

        for rev in reviews:
            t = float(rev["elapsed_days"])
            if t <= 0:
                continue  # skip first-time introductions and same-day reviews
            p = float(np.exp(-self._k * t))
            y_true_list.append(RATING_RECALLED[rev["rating"]])
            y_pred_list.append(p)
            elapsed_list.append(t)

        return (
            np.array(y_true_list,  dtype=np.int32),
            np.array(y_pred_list,  dtype=np.float32),
            np.array(elapsed_list, dtype=np.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER: REVIEW-COUNT DECAY  e^{-k0 · r^n · t}
# ─────────────────────────────────────────────────────────────────────────────

class ReviewCountDecayAdapter(ModelAdapter):
    """
    Review-count-scaled exponential decay:
        P(recall) = exp(-k0 · r^n · elapsed_days)

    where n is the 1-indexed review number for this card (n=1 for the second
    review of a card, matching the training convention in fit_models.py).
    r < 1 means the decay rate shrinks with each repetition (memory
    consolidation), r > 1 would mean it grows (not physically meaningful but
    the optimizer is free to find it if the data support it).
    """

    def __init__(self, k0: float, r: float):
        if k0 <= 0:
            raise ValueError(f"k0 must be > 0, got {k0}")
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")
        self._k0 = float(k0)
        self._r  = float(r)

    @property
    def name(self) -> str:
        return f"ReviewCountDecay(k0={self._k0:.5f}, r={self._r:.5f})"

    def predict_user(self, reviews):
        # Per-card review counter: incremented BEFORE prediction so that the
        # second review of a card has n=2, matching the training labelling
        # (i=0 → n=1 was the first/new-card event excluded by elapsed>0;
        #  i=1 → n=2 is the first genuine recall test).
        card_review_counts: dict = defaultdict(int)

        y_true_list, y_pred_list, elapsed_list = [], [], []

        for rev in reviews:
            cid = rev["card_id"]
            card_review_counts[cid] += 1
            n = card_review_counts[cid]      # 1-indexed review number

            t = float(rev["elapsed_days"])
            if t <= 0:
                continue

            p = float(np.exp(-self._k0 * (self._r ** n) * t))
            y_true_list.append(RATING_RECALLED[rev["rating"]])
            y_pred_list.append(p)
            elapsed_list.append(t)

        return (
            np.array(y_true_list,  dtype=np.int32),
            np.array(y_pred_list,  dtype=np.float32),
            np.array(elapsed_list, dtype=np.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER: DENSE HYBRID ML MODEL  (Transformer + LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class MLModelAdapter(ModelAdapter):
    """
    Adapter for DenseHybridRecallModel (DeepSeekMLA + LSTM).

    The model is CAUSAL: position t sees reviews 0…t-1 and predicts recall at
    t.  We run the full user sequence through the model in chunks of
    MAX_SEQ_LEN (matching training) and carry the LSTM state across chunks.

    Feature engineering replicates ChunkedUserDataset._process_batch() exactly.
    """

    MAX_SEQ_LEN = 1536  # must match training

    def __init__(self, model_path: str, device: Optional[str] = None):

        import torch
        from train import DenseHybridRecallModel  # train.py must be on sys.path


        self._torch = torch
        self._device = torch.device(
            device if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = DenseHybridRecallModel(d_model=256, n_heads=8, num_layers=2)
        state_dict = torch.load(
            model_path, map_location=self._device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval().to(self._device)
        self._model = model
        print(f"[MLModelAdapter] Loaded '{model_path}' → {self._device}")

    @property
    def name(self) -> str:
        return "DenseHybridRecallModel"

    # ------------------------------------------------------------------
    # Feature engineering (mirrors ChunkedUserDataset._process_batch)
    # ------------------------------------------------------------------
    @staticmethod
    def _engineer_features(reviews: list[dict]) -> Optional[dict]:
        """
        Build a polars DataFrame from the raw review dicts, apply the same
        feature engineering used during training, and return numpy arrays.

        Returns None if the user has fewer than 5 reviews (model unreliable).
        """
        if len(reviews) < 5:
            return None

        # Build DataFrame; fill missing 'duration' with 0.0 if absent
        df = pl.DataFrame([
            {
                "card_id":        r["card_id"],
                "day_offset":     r["day_offset"],
                "elapsed_days":   r["elapsed_days"],
                "elapsed_seconds": r.get("elapsed_seconds", 0),
                "rating":         r["rating"],
                "state":          r.get("state", 0),
                "duration":       float(r.get("duration", 0.0) or 0.0),
            }
            for r in reviews
        ])

        # Preserve chronological order with an explicit row index
        df = df.with_row_index("row_idx")
        df = df.sort(["day_offset", "row_idx"])

        # ── Core flags & clipped intervals ──────────────────────────────────
        df = df.with_columns([
            pl.col("elapsed_days").alias("raw_elapsed_days"),  # keep for masking
            (pl.col("elapsed_days") == -1).cast(pl.Int32).alias("is_first_review"),
            pl.col("elapsed_days").clip(lower_bound=0).alias("interval_days"),
            pl.col("elapsed_seconds").clip(lower_bound=0).alias("interval_sec"),
            (pl.col("rating") > 1).cast(pl.Float32).alias("y"),
        ])

        # ── Card-level cumulative intervals ──────────────────────────────────
        df = df.with_columns([
            pl.col("interval_days").cum_sum().over("card_id")
              .alias("interval_days_cum"),
            pl.col("interval_sec").cum_sum().over("card_id")
              .alias("interval_sec_cum"),
            (pl.col("rating").cum_count().over("card_id") - 1)
              .clip(lower_bound=0, upper_bound=100)
              .cast(pl.Int32).alias("card_review_count"),
        ])

        # ── User-level cumulative counts ─────────────────────────────────────
        df = df.with_columns([
            pl.col("is_first_review").cum_sum().alias("cum_new_cards"),
            pl.col("rating").cum_count().alias("review_th"),
        ])

        # ── Lag features ─────────────────────────────────────────────────────
        df = df.with_columns([
            # Per-card lags
            pl.col("rating").shift(1).over("card_id")
              .fill_null(0).cast(pl.Int32).alias("last_rating"),
            pl.col("state").shift(1).over("card_id")
              .fill_null(0).cast(pl.Int32).alias("last_state"),
            pl.col("review_th").shift(1).over("card_id")
              .fill_null(0).alias("last_card_th"),
            pl.col("cum_new_cards").shift(1).over("card_id")
              .fill_null(0).alias("last_card_new_cards"),
            # User-level lags
            pl.col("rating").shift(1).fill_null(0).cast(pl.Int32).alias("prev_rating"),
            pl.col("state").shift(1).fill_null(0).cast(pl.Int32).alias("prev_state"),
            pl.col("day_offset").shift(1).fill_null(pl.col("day_offset"))
              .alias("prev_day_offset"),
            pl.col("duration").shift(1).fill_null(0.0).alias("prev_duration"),
        ])

        # ── Difference features ──────────────────────────────────────────────
        df = df.with_columns([
            (pl.col("review_th") - pl.col("last_card_th") - 1)
              .clip(lower_bound=0).cast(pl.Float32).alias("diff_reviews"),
            (pl.col("cum_new_cards") - pl.col("last_card_new_cards"))
              .clip(lower_bound=0).cast(pl.Float32).alias("diff_new_cards"),
            (pl.col("day_offset") - pl.col("prev_day_offset"))
              .clip(lower_bound=0).cast(pl.Float32).alias("day_offset_diff"),
            pl.col("rating").cum_count().over("day_offset")
              .cast(pl.Float32).alias("cum_reviews_today"),
            pl.col("is_first_review").cum_sum().over("day_offset")
              .cast(pl.Float32).alias("cum_new_cards_today"),
        ])

        # ── Circadian features ───────────────────────────────────────────────
        TWO_PI_OVER_DAY = 2.0 * np.pi / 86400.0
        df = df.with_columns([
            ((pl.col("interval_sec") % 86400) * TWO_PI_OVER_DAY)
              .sin().cast(pl.Float32).alias("interval_sec_sin"),
            ((pl.col("interval_sec") % 86400) * TWO_PI_OVER_DAY)
              .cos().cast(pl.Float32).alias("interval_sec_cos"),
            ((pl.col("interval_sec_cum") % 86400) * TWO_PI_OVER_DAY)
              .sin().cast(pl.Float32).alias("interval_sec_cum_sin"),
            ((pl.col("interval_sec_cum") % 86400) * TWO_PI_OVER_DAY)
              .cos().cast(pl.Float32).alias("interval_sec_cum_cos"),
        ])

        # ── Log-normalise continuous features (matching training) ────────────
        cols_to_scale = [
            "interval_days", "interval_sec", "interval_days_cum",
            "interval_sec_cum", "duration", "prev_duration",
            "diff_reviews", "diff_new_cards", "cum_reviews_today",
            "cum_new_cards_today", "day_offset_diff",
        ]
        df = df.with_columns([
            (pl.col(c).clip(lower_bound=0).log1p() / 5.0).cast(pl.Float32).alias(c)
            for c in cols_to_scale if c in df.columns
        ])

        return {
            "maturity":         df["card_review_count"].to_numpy().copy(),
            "prev_rating":      df["prev_rating"].to_numpy().copy(),
            "last_rating":      df["last_rating"].to_numpy().copy(),
            "prev_state":       df["prev_state"].to_numpy().copy(),
            "last_state":       df["last_state"].to_numpy().copy(),
            "cont":             df.select(ML_CONT_COLS).to_numpy().copy(),
            "y":                df["y"].to_numpy().copy(),
            "raw_elapsed_days": df["raw_elapsed_days"].to_numpy().copy(),
        }

    def predict_user(self, reviews):
        torch = self._torch
        feats = self._engineer_features(reviews)
        if feats is None:
            return np.array([]), np.array([]), np.array([])

        T = len(feats["maturity"])
        all_logits = np.zeros(T, dtype=np.float32)
        lstm_state = None

        for start in range(0, T, self.MAX_SEQ_LEN):
            end   = min(start + self.MAX_SEQ_LEN, T)
            sl    = slice(start, end)

            def to_long(arr: np.ndarray):
                return torch.from_numpy(arr[sl]).unsqueeze(0).to(self._device)

            def to_float(arr: np.ndarray):
                return (torch.from_numpy(arr[sl]).unsqueeze(0)
                        .to(self._device, dtype=torch.float32))

            with torch.no_grad():
                logits, lstm_state = self._model(
                    to_long(feats["maturity"]),
                    to_float(feats["cont"]),
                    to_long(feats["prev_rating"]),
                    to_long(feats["last_rating"]),
                    to_long(feats["prev_state"]),
                    to_long(feats["last_state"]),
                    lstm_state,
                )
                lstm_state = tuple(s.detach() for s in lstm_state)
                all_logits[sl] = logits.squeeze(0).cpu().numpy()

        # Evaluate only on reviews where elapsed_days > 0
        mask = feats["raw_elapsed_days"] > 0
        y_true = feats["y"][mask].astype(np.int32)
        y_pred = torch.sigmoid(
            torch.from_numpy(all_logits[mask])
        ).numpy().astype(np.float32)
        elapsed = feats["raw_elapsed_days"][mask].astype(np.float32)

        return y_true, y_pred, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_reviews_for_user(revlog_dir: Path, user_id: int) -> list[dict]:
    """
    Load and chronologically sort all reviews for a single user.

    Sorting by (day_offset, row_idx) preserves the true review order within
    a day — the same convention used by ChunkedUserDataset during training.
    """
    try:
        lf = pl.scan_parquet(str(revlog_dir / "**/*.parquet"), hive_partitioning=True)
    except Exception:
        lf = pl.scan_parquet(str(revlog_dir), hive_partitioning=True)

    df = (
        lf.filter(pl.col("user_id") == user_id)
          .collect()
          .with_row_index("_row_idx")
          .sort(["day_offset", "_row_idx"])
    )
    return df.drop("_row_idx").to_dicts()


def iter_user_ids(revlog_dir: Path, start: int, end: int):
    """Yield user IDs in [start, end] that actually have data on disk."""
    try:
        lf = pl.scan_parquet(str(revlog_dir / "**/*.parquet"), hive_partitioning=True)
    except Exception:
        lf = pl.scan_parquet(str(revlog_dir), hive_partitioning=True)

    present = (
        lf.select("user_id")
          .filter((pl.col("user_id") >= start) & (pl.col("user_id") <= end))
          .unique()
          .collect()["user_id"]
          .to_list()
    )
    yield from sorted(present)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS  (mirrors srs-benchmark get_stats)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    elapsed_days: np.ndarray,
) -> dict:
    """
    Compute RMSE, LogLoss, RMSE(bins), and AUC.

    RMSE(bins): each review is assigned to an elapsed-day bin.  Within each
    bin, mean(y_true) and mean(y_pred) are computed; RMSE is then computed
    over these per-bin means weighted by the number of reviews in that bin.
    This measures calibration quality across different retention intervals.

    Returns a dict with keys: RMSE, LogLoss, RMSE(bins), AUC, n_reviews.
    """
    if len(y_true) == 0:
        return {"RMSE": None, "LogLoss": None, "RMSE(bins)": None,
                "AUC": None, "n_reviews": 0}

    # Clip predictions away from 0/1 to avoid log(0) in LogLoss
    eps      = 1e-7
    y_pred_c = np.clip(y_pred, eps, 1 - eps)

    rmse     = float(root_mean_squared_error(y_true, y_pred))
    logloss  = float(log_loss(y_true, y_pred_c, labels=[0, 1]))

    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        auc = None  # only one class present

    # ── RMSE(bins) ────────────────────────────────────────────────────────
    bins = assign_elapsed_bins(elapsed_days)
    rows = pd.DataFrame({"bin": bins, "y": y_true, "p": y_pred_c, "w": 1})
    agg  = (
        rows.groupby("bin", sort=True)
            .agg(y=("y", "mean"), p=("p", "mean"), w=("w", "sum"))
            .reset_index()
    )
    rmse_bins = float(
        root_mean_squared_error(agg["y"], agg["p"], sample_weight=agg["w"])
    )

    return {
        "RMSE":       round(rmse,      6),
        "LogLoss":    round(logloss,   6),
        "RMSE(bins)": round(rmse_bins, 6),
        "AUC":        round(auc, 6) if auc is not None else None,
        "n_reviews":  int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_users(
    revlog_dir: Path,
    adapters:   list[ModelAdapter],
    user_start: int,
    user_end:   int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-model output handles
    out_files = {
        adapter.name: open(output_dir / f"{_safe_name(adapter.name)}.jsonl", "w")
        for adapter in adapters
    }

    # Accumulators for global (across-user) metrics
    global_acc: dict[str, dict] = {
        a.name: {"y_true": [], "y_pred": [], "elapsed": []}
        for a in adapters
    }

    user_ids = list(iter_user_ids(revlog_dir, user_start, user_end))
    print(f"\nFound {len(user_ids)} users in [{user_start}, {user_end}]")
    print("=" * 70)

    for idx, uid in enumerate(user_ids, 1):
        reviews = load_reviews_for_user(revlog_dir, uid)
        if not reviews:
            continue

        print(f"\n[{idx}/{len(user_ids)}] User {uid}  ({len(reviews)} total reviews)")

        for adapter in adapters:
            y_true, y_pred, elapsed = adapter.predict_user(reviews)

            if len(y_true) == 0:
                print(f"  {adapter.name}: no evaluable reviews, skipping")
                continue

            metrics = compute_metrics(y_true, y_pred, elapsed)
            record  = {
                "user":    uid,
                "model":   adapter.name,
                "metrics": metrics,
            }
            out_files[adapter.name].write(json.dumps(record) + "\n")

            # Accumulate for global metrics
            global_acc[adapter.name]["y_true"].extend(y_true.tolist())
            global_acc[adapter.name]["y_pred"].extend(y_pred.tolist())
            global_acc[adapter.name]["elapsed"].extend(elapsed.tolist())

            print(
                f"  {adapter.name:45s}  "
                f"RMSE={metrics['RMSE']:.4f}  "
                f"LL={metrics['LogLoss']:.4f}  "
                f"RMSE(b)={metrics['RMSE(bins)']:.4f}  "
                f"AUC={metrics['AUC'] or 'N/A'}"
            )

    # ── Flush per-user files ──────────────────────────────────────────────
    for f in out_files.values():
        f.close()

    # ── Global (aggregated) metrics ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("GLOBAL METRICS  (aggregated across all evaluated users)")
    print("=" * 70)

    summary_rows = []
    for adapter in adapters:
        acc = global_acc[adapter.name]
        if not acc["y_true"]:
            print(f"  {adapter.name}: no data")
            continue

        gm = compute_metrics(
            np.array(acc["y_true"],   dtype=np.int32),
            np.array(acc["y_pred"],   dtype=np.float32),
            np.array(acc["elapsed"],  dtype=np.float32),
        )
        print(
            f"  {adapter.name:45s}  "
            f"RMSE={gm['RMSE']:.4f}  "
            f"LL={gm['LogLoss']:.4f}  "
            f"RMSE(b)={gm['RMSE(bins)']:.4f}  "
            f"AUC={gm['AUC'] or 'N/A'}  "
            f"n={gm['n_reviews']:,}"
        )
        summary_rows.append({"model": adapter.name, **gm})

    # Write global summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary written to {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    """Convert a model name to a filesystem-safe string."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_adapters(
    model_names: list[str],
    params_file: Optional[str],
    model_path:  Optional[str],
    device:      Optional[str],
) -> list[ModelAdapter]:
    """Instantiate the requested adapters from CLI arguments."""

    adapters: list[ModelAdapter] = []

    needs_params = {"exponential", "review_count"} & set(model_names)
    params: dict = {}
    if needs_params:
        if not params_file:
            raise ValueError(
                "--params-file is required when using 'exponential' or 'review_count'."
            )
        with open(params_file) as f:
            params = json.load(f)

    for name in model_names:
        if name == "exponential":
            k = params.get("exponential", {}).get("k")
            if k is None:
                raise KeyError(
                    "params.json missing 'exponential.k'. "
                    "Run fit_models.py first."
                )
            adapters.append(ExponentialDecayAdapter(k=k))

        elif name == "review_count":
            k0 = params.get("review_count", {}).get("k0")
            r  = params.get("review_count", {}).get("r")
            if k0 is None or r is None:
                raise KeyError(
                    "params.json missing 'review_count.k0' or 'review_count.r'. "
                    "Run fit_models.py first."
                )
            adapters.append(ReviewCountDecayAdapter(k0=k0, r=r))

        elif name == "ml":
            if not model_path:
                raise ValueError("--model-path is required when using the 'ml' model.")
            adapters.append(MLModelAdapter(model_path=model_path, device=device))

        else:
            raise ValueError(
                f"Unknown model '{name}'. "
                "Valid choices: exponential, review_count, ml"
            )

    return adapters


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--revlog-dir", required=True,
        help="Root directory containing partitioned parquet revlog files.",
    )
    p.add_argument(
        "--user-start", type=int, required=True,
        help="First user ID to evaluate (inclusive).",
    )
    p.add_argument(
        "--user-end", type=int, required=True,
        help="Last user ID to evaluate (inclusive).",
    )
    p.add_argument(
        "--models", nargs="+",
        default=["exponential", "review_count"],
        choices=["exponential", "review_count", "ml"],
        help="Which models to benchmark (default: exponential review_count).",
    )
    p.add_argument(
        "--params-file", default=None,
        help="JSON file of fitted forgetting-curve parameters (from fit_models.py).",
    )
    p.add_argument(
        "--model-path", default=None,
        help="Path to pretrained DenseHybridRecallModel weights (.pth).",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device for ML model, e.g. 'cuda' or 'cpu' (auto-detected if omitted).",
    )
    p.add_argument(
        "--output", default="results/",
        help="Directory for per-model JSONL output and summary.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    revlog_dir = Path(args.revlog_dir)
    if not revlog_dir.exists():
        print(f"Error: revlog directory not found: {revlog_dir}", file=sys.stderr)
        sys.exit(1)

    adapters = build_adapters(
        model_names=args.models,
        params_file=args.params_file,
        model_path=args.model_path,
        device=args.device,
    )

    print(f"Evaluating {len(adapters)} model(s) on users {args.user_start}–{args.user_end}")
    for a in adapters:
        print(f"  • {a.name}")

    evaluate_all_users(
        revlog_dir  = revlog_dir,
        adapters    = adapters,
        user_start  = args.user_start,
        user_end    = args.user_end,
        output_dir  = Path(args.output),
    )


if __name__ == "__main__":
    main()