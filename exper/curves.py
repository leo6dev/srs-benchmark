"""
Forgetting Curve Visualizer
============================
Generates three separate plot files:

  1. forgetting_curve_exponential.png
       Simple e^{-kt} baseline — one global k, no stability or interval terms.

  2. forgetting_curve_ml_backsolve.png
       ML model rendered continuously by back-solving an implied decay rate λ
       per segment from the model's point-in-time prediction at each review.

  3. forgetting_curve_ml_sampled.png
       ML model rendered correctly by replaying the cached LSTM hidden state
       and probing at N_PROBES_PER_SEGMENT synthetic "due" times per segment.
       Each probe is fully independent — the LSTM state is NOT advanced between
       probes, so every value answers: "if the card came up right now, what
       would the model predict?"
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ─────────────────────────────────────────────────────────────────────────────
# ★  CONFIGURATION  – edit these
# ─────────────────────────────────────────────────────────────────────────────

REVLOG_DIR    = Path("/Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/")
PARAMS_FILE   = Path("/Users/leo/PycharmProjects/srs-benchmark/exper/params.json")
ML_MODEL_PATH = "best_model_opt.pth"   # set to None to skip ML plots

VIZ_USER_ID          = 2374
N_REVIEWS            = 20
SEED                 = 42
N_PROBES_PER_SEGMENT = 20   # synthetic probe points per inter-review segment

OUTPUT_EXP        = "forgetting_curve_exponential.png"
OUTPUT_ML_BACK    = "forgetting_curve_ml_backsolve.png"
OUTPUT_ML_SAMPLED = "forgetting_curve_ml_sampled.png"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EPS             = 1e-9
RATING_RECALLED = {1: 0, 2: 1, 3: 1, 4: 1}

ML_CONT_COLS = [
    "interval_days", "interval_sec", "interval_days_cum", "interval_sec_cum",
    "duration", "prev_duration", "diff_reviews", "diff_new_cards",
    "cum_reviews_today", "cum_new_cards_today", "day_offset_diff",
    "interval_sec_sin", "interval_sec_cos", "interval_sec_cum_sin",
    "interval_sec_cum_cos",
]
# Fast index lookup into the continuous feature vector
_CI = {c: i for i, c in enumerate(ML_CONT_COLS)}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _scan() -> pl.LazyFrame:
    try:
        return pl.scan_parquet(str(REVLOG_DIR / "**/*.parquet"), hive_partitioning=True)
    except Exception:
        return pl.scan_parquet(str(REVLOG_DIR), hive_partitioning=True)


def load_user_reviews(user_id: int) -> list[dict]:
    df = (
        _scan()
        .filter(pl.col("user_id") == user_id)
        .collect()
        .with_row_index("_i")
        .sort(["day_offset", "_i"])
    )
    return df.drop("_i").to_dicts()


# ─────────────────────────────────────────────────────────────────────────────
# CARD SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_card(
    reviews: list[dict], n_reviews: int, rng: np.random.Generator
) -> list[dict]:
    card_map: dict[int, list] = defaultdict(list)
    for r in reviews:
        card_map[int(r["card_id"])].append(r)

    valid = {c: revs for c, revs in card_map.items() if len(revs) >= n_reviews}
    if not valid:
        raise ValueError(f"No cards with >= {n_reviews} reviews for user {VIZ_USER_ID}.")

    counts      = np.array([len(v) for v in valid.values()])
    max_allowed = max(np.percentile(counts, 75), np.median(counts) + 2)
    culled      = {c: revs for c, revs in valid.items() if len(revs) <= max_allowed}

    cids = list(culled.keys())
    rng.shuffle(cids)

    recall_rates = {
        c: np.mean([RATING_RECALLED[r["rating"]] for r in culled[c][:n_reviews]])
        for c in cids
    }
    med        = np.median(list(recall_rates.values()))
    target_cid = min(cids, key=lambda c: abs(recall_rates[c] - med))
    return culled[target_cid][:n_reviews]


# ─────────────────────────────────────────────────────────────────────────────
# EXPONENTIAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Exponential:
    """Pure e^{-kt} forgetting curve — single global decay rate, no modifiers."""

    def __init__(self, k: float):
        self.k = k

    def predict(self, delta: np.ndarray) -> np.ndarray:
        return np.clip(np.exp(-self.k * delta), EPS, 1 - EPS)

    def label(self) -> str:
        return f"$e^{{-k \\cdot \\Delta t}}$   $(k={self.k:.4f})$"


def build_curve_exponential(
    card_reviews: list[dict], model: Exponential
) -> tuple[list, list]:
    """
    Piecewise exponential curve.  After each review memory resets to P=1,
    then decays as exp(-k * s) where s is seconds elapsed since that review.
    """
    xs, ys = [], []
    n      = len(card_reviews)

    for i, rev in enumerate(card_reviews):
        t_review = float(rev["day_offset"])
        t_next   = float(card_reviews[i + 1]["day_offset"]) if i + 1 < n else t_review

        # Vertical drop to predicted recall just before this review
        if xs:
            seg_len = t_review - float(card_reviews[i - 1]["day_offset"])
            p_drop  = float(model.predict(np.array([seg_len]))[0]) if seg_len > 0 else 1.0
            xs.append(t_review)
            ys.append(p_drop)

        # Memory refresh
        xs.append(t_review)
        ys.append(1.0)

        seg_len = t_next - t_review
        if seg_len > 0:
            t_seg = np.linspace(0, seg_len, 300)
            xs.extend((t_review + t_seg).tolist())
            ys.extend(model.predict(t_seg).tolist())

    return xs, ys


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MLModelPredictor:
    MAX_SEQ = 1536

    def __init__(self):
        import torch
        from train import DenseHybridRecallModel

        self._torch = torch
        self._dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DenseHybridRecallModel(d_model=256, n_heads=8, num_layers=2)
        sd    = torch.load(ML_MODEL_PATH, map_location=self._dev, weights_only=True)
        model.load_state_dict(sd)
        model.eval().to(self._dev)
        self._model = model

    def label(self) -> str:
        return "DenseHybridRecallModel"

    # ── Feature engineering (identical to benchmark.py) ──────────────────────

    def _engineer_full_user(self, all_reviews: list[dict]) -> dict | None:
        if len(all_reviews) < 5:
            return None

        df = pl.DataFrame([{
            "card_id":         int(r["card_id"]),
            "day_offset":      int(r["day_offset"]),
            "elapsed_days":    float(r["elapsed_days"]),
            "elapsed_seconds": float(r.get("elapsed_seconds", 0) or 0),
            "rating":          int(r["rating"]),
            "state":           int(r.get("state", 0) or 0),
            "duration":        float(r.get("duration", 0.0) or 0.0),
        } for r in all_reviews]).with_row_index("_i").sort(["day_offset", "_i"])

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
            (pl.col("review_th") - pl.col("lct") - 1).clip(lower_bound=0).cast(pl.Float32).alias("diff_reviews"),
            (pl.col("cum_new_cards") - pl.col("lcnc")).clip(lower_bound=0).cast(pl.Float32).alias("diff_new_cards"),
            (pl.col("day_offset") - pl.col("prev_day_offset")).clip(lower_bound=0).cast(pl.Float32).alias("day_offset_diff"),
            pl.col("rating").cum_count().over("day_offset").cast(pl.Float32).alias("cum_reviews_today"),
            pl.col("is_first").cum_sum().over("day_offset").cast(pl.Float32).alias("cum_new_cards_today"),
        ])
        TWO_PI = 2.0 * np.pi / 86400.0
        df = df.with_columns([
            ((pl.col("interval_sec") % 86400) * TWO_PI).sin().cast(pl.Float32).alias("interval_sec_sin"),
            ((pl.col("interval_sec") % 86400) * TWO_PI).cos().cast(pl.Float32).alias("interval_sec_cos"),
            ((pl.col("interval_sec_cum") % 86400) * TWO_PI).sin().cast(pl.Float32).alias("interval_sec_cum_sin"),
            ((pl.col("interval_sec_cum") % 86400) * TWO_PI).cos().cast(pl.Float32).alias("interval_sec_cum_cos"),
        ])
        for c in ["interval_days", "interval_sec", "interval_days_cum", "interval_sec_cum",
                  "duration", "prev_duration", "diff_reviews", "diff_new_cards",
                  "cum_reviews_today", "cum_new_cards_today", "day_offset_diff"]:
            if c in df.columns:
                df = df.with_columns(
                    (pl.col(c).clip(lower_bound=0).log1p() / 5.0).cast(pl.Float32).alias(c)
                )
        df = df.with_columns(pl.col("_i").alias("orig_idx"))
        return {
            "maturity":    df["card_review_count"].to_numpy().copy(),
            "prev_rating": df["prev_rating"].to_numpy().copy(),
            "last_rating": df["last_rating"].to_numpy().copy(),
            "prev_state":  df["prev_state"].to_numpy().copy(),
            "last_state":  df["last_state"].to_numpy().copy(),
            "cont":        df.select(ML_CONT_COLS).to_numpy().copy(),
            "card_id":     df["card_id"].to_numpy().copy(),
            "day_offset":  df["day_offset"].to_numpy().copy(),
        }

    def _find_card_indices(
        self, feats: dict, card_reviews: list[dict]
    ) -> list[int] | None:
        T = len(feats["maturity"])
        card_ids  = feats["card_id"]
        day_offs  = feats["day_offset"]
        indices   = []
        for rev in card_reviews:
            cid, doff = int(rev["card_id"]), int(rev["day_offset"])
            found = next(
                (i for i in range(T)
                 if int(card_ids[i]) == cid and int(day_offs[i]) == doff),
                None,
            )
            if found is None:
                return None
            indices.append(found)
        return indices

    # ── Point-in-time predictions (used by the back-solve curve) ─────────────

    def get_probabilities_for_card(
        self, all_reviews: list[dict], card_reviews: list[dict]
    ) -> np.ndarray | None:
        """Return the model's P(recall) at every review event (chunked pass)."""
        torch = self._torch
        feats = self._engineer_full_user(all_reviews)
        if feats is None:
            return None

        T          = len(feats["maturity"])
        all_logits = np.zeros(T, dtype=np.float32)
        lstm_state = None

        for start in range(0, T, self.MAX_SEQ):
            end = min(start + self.MAX_SEQ, T)
            sl  = slice(start, end)
            def tl(a, sl=sl): return torch.from_numpy(a[sl]).unsqueeze(0).to(self._dev)
            def tf(a, sl=sl): return torch.from_numpy(a[sl]).unsqueeze(0).to(self._dev, dtype=torch.float32)
            with torch.no_grad():
                logits, lstm_state = self._model(
                    tl(feats["maturity"]), tf(feats["cont"]),
                    tl(feats["prev_rating"]), tl(feats["last_rating"]),
                    tl(feats["prev_state"]),  tl(feats["last_state"]),
                    lstm_state,
                )
                lstm_state = tuple(s.detach() for s in lstm_state)
                all_logits[sl] = logits.squeeze(0).cpu().numpy()

        all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
        lookup = {
            (int(feats["card_id"][i]), int(feats["day_offset"][i])): all_probs[i]
            for i in range(T)
        }
        return np.array([
            lookup.get((int(r["card_id"]), int(r["day_offset"])), float("nan"))
            for r in card_reviews
        ])

    # ── LSTM state replay for the sampled continuous curve ────────────────────

    def get_sampled_curve(
        self,
        all_reviews:  list[dict],
        card_reviews: list[dict],
        n_probes:     int = N_PROBES_PER_SEGMENT,
    ) -> tuple[list, list]:
        """
        True continuous forgetting curve via LSTM state replay.

        Algorithm per inter-review segment [t_i, t_{i+1}]:
          1. Restore LSTM hidden state to the snapshot saved immediately AFTER
             review i was processed (memory is freshly updated at that point).
          2. For each of n_probes synthetic times t_probe spread across the
             segment, build a feature vector that looks like:
               - Categorical inputs (maturity, last_rating, etc.) copied from
                 what the REAL next review (i+1) sees — these are fixed by card
                 history and do not depend on when the card is presented.
               - Continuous inputs copied from review i+1's vector, then the
                 four elapsed-time features (interval_days/sec + their cumulative
                 versions + trig) are overridden with the probe's actual Δt.
             Run a single LSTM forward step WITHOUT updating the base state, so
             every probe is independently conditioned on exactly the state after
             review i.
          3. The prediction at dt = seg_len should match get_probabilities_for_card
             at review i+1 (same inputs, same base state), providing a built-in
             sanity check that the two curves agree at review boundaries.
        """
        torch = self._torch
        feats = self._engineer_full_user(all_reviews)
        if feats is None:
            return [], []

        T            = len(feats["maturity"])
        card_indices = self._find_card_indices(feats, card_reviews)
        if card_indices is None:
            print("Warning: could not match all card reviews to engineered features.")
            return [], []

        # ── Single-step pass: cache LSTM state after each card review ─────────
        max_needed = max(card_indices)
        target_set = set(card_indices)
        state_cache: dict[int, tuple] = {}
        lstm_state = None

        print(f"  Caching LSTM states (single-step pass, {max_needed + 1} steps) …")
        for idx in range(max_needed + 1):
            sl = slice(idx, idx + 1)
            def tl(a, sl=sl): return torch.from_numpy(a[sl].copy()).unsqueeze(0).to(self._dev)
            def tf(a, sl=sl): return torch.from_numpy(a[sl].copy()).unsqueeze(0).to(self._dev, dtype=torch.float32)
            with torch.no_grad():
                _, lstm_state = self._model(
                    tl(feats["maturity"]), tf(feats["cont"]),
                    tl(feats["prev_rating"]), tl(feats["last_rating"]),
                    tl(feats["prev_state"]),  tl(feats["last_state"]),
                    lstm_state,
                )
                lstm_state = tuple(s.detach() for s in lstm_state)
            if idx in target_set:
                state_cache[idx] = tuple(s.clone() for s in lstm_state)

        # ── Precompute raw cumulative intervals for this card ─────────────────
        # (needed to correctly compute the cumulative elapsed features for probes)
        cum_d, cum_s = 0.0, 0.0
        cum_after: list[tuple[float, float]] = []
        for rev in card_reviews:
            cum_d += max(float(rev["elapsed_days"]), 0.0)
            cum_s += max(float(rev.get("elapsed_seconds", 0) or 0), 0.0)
            cum_after.append((cum_d, cum_s))

        TWO_PI = 2.0 * np.pi / 86400.0
        xs, ys  = [], []
        n_card  = len(card_reviews)

        for i in range(n_card - 1):
            t_i     = float(card_reviews[i]["day_offset"])
            t_next  = float(card_reviews[i + 1]["day_offset"])
            seg_len = t_next - t_i
            if seg_len <= 0:
                continue

            idx_i    = card_indices[i]
            idx_next = card_indices[i + 1]

            # Cached state after review i — cloned for each probe so base is unchanged
            base_state = tuple(s.clone() for s in state_cache[idx_i])

            # Categorical inputs: taken from the REAL next review's position.
            # These reflect the correct card maturity and previous outcomes.
            mat_next  = int(feats["maturity"][idx_next])
            pr_next   = int(feats["prev_rating"][idx_next])
            lr_next   = int(feats["last_rating"][idx_next])
            ps_next   = int(feats["prev_state"][idx_next])
            ls_next   = int(feats["last_state"][idx_next])

            # Continuous feature template: start from next review's values.
            # Context features (session stats, durations) are held fixed.
            # Only the four elapsed-time features will be overridden per probe.
            cont_template = feats["cont"][idx_next].copy()   # shape [15]
            cum_d_i, cum_s_i = cum_after[i]

            # P = 1 immediately after review i (memory freshly encoded)
            xs.append(t_i)
            ys.append(1.0)

            # Probe times: n_probes interior points + the exact endpoint (= t_next).
            # Including the endpoint lets us verify the curve lands on the actual
            # model prediction at the next review.
            probe_dts = np.linspace(seg_len / (n_probes + 1), seg_len, n_probes,
                                    endpoint=False)
            probe_dts = np.append(probe_dts, seg_len)

            for dt in probe_dts:
                interval_d = float(dt)
                interval_s = interval_d * 86400.0
                cum_s_probe = cum_s_i + interval_s

                # Build the probe's continuous feature vector
                cont_probe = cont_template.copy()
                cont_probe[_CI["interval_days"]]     = np.log1p(interval_d) / 5.0
                cont_probe[_CI["interval_sec"]]      = np.log1p(interval_s) / 5.0
                cont_probe[_CI["interval_days_cum"]] = np.log1p(cum_d_i + interval_d) / 5.0
                cont_probe[_CI["interval_sec_cum"]]  = np.log1p(cum_s_i + interval_s) / 5.0
                cont_probe[_CI["interval_sec_sin"]]  = float(np.sin((interval_s  % 86400) * TWO_PI))
                cont_probe[_CI["interval_sec_cos"]]  = float(np.cos((interval_s  % 86400) * TWO_PI))
                cont_probe[_CI["interval_sec_cum_sin"]] = float(np.sin((cum_s_probe % 86400) * TWO_PI))
                cont_probe[_CI["interval_sec_cum_cos"]] = float(np.cos((cum_s_probe % 86400) * TWO_PI))

                def mk_int(v: int):
                    return (torch.from_numpy(np.array([v], dtype=np.int32))
                            .unsqueeze(0).to(self._dev))

                cont_t = (torch.from_numpy(cont_probe.reshape(1, -1))
                          .unsqueeze(0).to(self._dev, dtype=torch.float32))

                with torch.no_grad():
                    logit, _ = self._model(
                        mk_int(mat_next), cont_t,
                        mk_int(pr_next), mk_int(lr_next),
                        mk_int(ps_next), mk_int(ls_next),
                        base_state,   # NOT updated between probes
                    )
                p = float(torch.sigmoid(logit).squeeze())
                xs.append(t_i + dt)
                ys.append(p)

        # Explicitly show the memory refresh at the final review
        last_t = float(card_reviews[-1]["day_offset"])
        xs.append(last_t)
        ys.append(1.0)

        return xs, ys


# ─────────────────────────────────────────────────────────────────────────────
# BACK-SOLVE CONTINUOUS CURVE  (graph 2 helper)
# ─────────────────────────────────────────────────────────────────────────────

def build_curve_ml_backsolve(
    card_reviews: list[dict], ml_probs: np.ndarray
) -> tuple[list, list]:
    """
    Continuous curve implied by the ML model via per-segment back-solving.

    Between reviews i and i+1 the model predicts p_{i+1}.  Assuming memory
    resets to 1.0 after review i, the implied decay rate is:
        λ = −log(p_{i+1}) / (t_{i+1} − t_i)
    and the curve is drawn as  P(s) = exp(−λ·s)  over that segment.

    Note: this is an approximation — it forces an exponential shape onto the
    ML model's behaviour and only anchors at the two review endpoints.
    Use the sampled curve (graph 3) to see the model's actual predictions.
    """
    xs, ys   = [], []
    n        = len(card_reviews)
    last_lam = None

    for i, rev in enumerate(card_reviews):
        t_review = float(rev["day_offset"])

        # Drop to ML-predicted value just before this review
        if xs:
            xs.append(t_review)
            ys.append(float(ml_probs[i]))

        # Memory refresh
        xs.append(t_review)
        ys.append(1.0)

        if i + 1 < n:
            t_next  = float(card_reviews[i + 1]["day_offset"])
            seg_len = t_next - t_review
            p_next  = float(ml_probs[i + 1])

            if seg_len > 0 and p_next > EPS:
                last_lam = -np.log(max(p_next, EPS)) / seg_len
            lam = last_lam if last_lam is not None else 0.0

            if seg_len > 0:
                t_seg = np.linspace(0, seg_len, 300)
                xs.extend((t_review + t_seg).tolist())
                ys.extend(np.clip(np.exp(-lam * t_seg), EPS, 1 - EPS).tolist())
        else:
            # Extend 30 days past the final review using the last decay rate
            if last_lam and last_lam > 0:
                t_seg = np.linspace(0, 30.0, 300)
                xs.extend((t_review + t_seg).tolist())
                ys.extend(np.clip(np.exp(-last_lam * t_seg), EPS, 1 - EPS).tolist())

    return xs, ys


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PLOT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save_single_plot(
    xs: list,
    ys: list,
    card_reviews: list[dict],
    curve_color: str,
    ax_title: str,
    fig_title: str,
    legend_label: str,
    output_file: str,
    linestyle: str = "-",
) -> None:
    first_day = float(card_reviews[0]["day_offset"])
    last_day  = float(card_reviews[-1]["day_offset"])

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(xs, ys, color=curve_color, linestyle=linestyle,
            linewidth=2.0, zorder=3)

    # True recall outcomes as × marks
    for rev in card_reviews:
        recalled = RATING_RECALLED[rev["rating"]]
        ax.plot(
            float(rev["day_offset"]), recalled,
            marker="x", markersize=12, markeredgewidth=2.5,
            color="black", zorder=5,
        )

    curve_h = plt.Line2D([0], [0], color=curve_color, linestyle=linestyle,
                         linewidth=2.0, label=legend_label)
    mark_h  = plt.Line2D([0], [0], marker="x", color="black",
                         linestyle="None", markersize=10,
                         markeredgewidth=2.5, label="True recall outcome")
    ax.legend(handles=[curve_h, mark_h], loc="upper right", fontsize=10)

    ax.set_title(ax_title, fontsize=11, pad=8)
    ax.set_xlabel("Day offset", fontsize=11)
    ax.set_ylabel("P(Recall)", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(first_day, last_day + (last_day - first_day) * 0.02)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(fig_title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load exponential params ──────────────────────────────────────────
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"{PARAMS_FILE} not found. Run benchmark.py first.")
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    m_exp = Exponential(k=params["exponential"]["k"])

    # ── Optionally load ML model ─────────────────────────────────────────
    ml_predictor: MLModelPredictor | None = None
    if ML_MODEL_PATH is not None and Path(ML_MODEL_PATH).exists():
        try:
            ml_predictor = MLModelPredictor()
            print(f"ML model loaded from '{ML_MODEL_PATH}'")
        except Exception as ex:
            print(f"Could not load ML model: {ex}  —  generating exponential plot only.")

    # ── Load reviews & select card ────────────────────────────────────────
    print(f"Loading reviews for user {VIZ_USER_ID} …")
    all_reviews = load_user_reviews(VIZ_USER_ID)
    print(f"  {len(all_reviews)} total reviews")

    rng          = np.random.default_rng(SEED)
    card_reviews = select_card(all_reviews, N_REVIEWS, rng)
    cid          = int(card_reviews[0]["card_id"])
    avg_recall   = np.mean([RATING_RECALLED[r["rating"]] for r in card_reviews])
    print(f"Selected card {cid}  ({len(card_reviews)} reviews, avg recall {avg_recall:.2f})")

    fig_base = (
        f"User {VIZ_USER_ID} · Card {cid} · {N_REVIEWS} reviews · "
        f"avg recall {avg_recall:.2f}"
    )

    # ── Plot 1: Exponential ───────────────────────────────────────────────
    xs_exp, ys_exp = build_curve_exponential(card_reviews, m_exp)
    save_single_plot(
        xs=xs_exp, ys=ys_exp,
        card_reviews=card_reviews,
        curve_color="#c0392b",
        ax_title=(
            f"Simple exponential decay   $e^{{-k \\cdot \\Delta t}}$   "
            f"$k = {m_exp.k:.4f}$\n"
            "Memory resets to 1 after each review; "
            "same decay rate throughout."
        ),
        fig_title=f"Forgetting Curve – Exponential Model\n{fig_base}",
        legend_label=m_exp.label(),
        output_file=OUTPUT_EXP,
    )

    # ── Plots 2 & 3: ML model ─────────────────────────────────────────────
    if ml_predictor is None:
        print("No ML model — skipping ML plots.")
        return

    print("Computing point-in-time ML predictions …")
    ml_probs = ml_predictor.get_probabilities_for_card(all_reviews, card_reviews)
    if ml_probs is None:
        print("Feature engineering failed — skipping ML plots.")
        return

    # Plot 2: back-solved continuous curve
    xs_back, ys_back = build_curve_ml_backsolve(card_reviews, ml_probs)
    save_single_plot(
        xs=xs_back, ys=ys_back,
        card_reviews=card_reviews,
        curve_color="#2980b9",
        ax_title=(
            "DenseHybridRecallModel – implied continuous curve (back-solved λ)\n"
            "Between reviews i→i+1: fits exp(−λ·Δt) so the curve lands exactly "
            "on the model's point prediction at t_{i+1}."
        ),
        fig_title=f"Forgetting Curve – ML Model (Back-Solved)\n{fig_base}",
        legend_label="DenseHybridRecallModel (back-solved λ per segment)",
        output_file=OUTPUT_ML_BACK,
        linestyle="--",
    )

    # Plot 3: LSTM state replay (true sampled curve)
    print(f"Computing sampled ML forgetting curve ({N_PROBES_PER_SEGMENT} probes/segment) …")
    xs_samp, ys_samp = ml_predictor.get_sampled_curve(
        all_reviews, card_reviews, n_probes=N_PROBES_PER_SEGMENT
    )
    save_single_plot(
        xs=xs_samp, ys=ys_samp,
        card_reviews=card_reviews,
        curve_color="#27ae60",
        ax_title=(
            f"DenseHybridRecallModel – true forgetting curve "
            f"(LSTM state replay, {N_PROBES_PER_SEGMENT} probes per segment)\n"
            "LSTM state frozen at post-review snapshot; each probe is an "
            "independent 'what if the card came up at time t?' query."
        ),
        fig_title=f"Forgetting Curve – ML Model\n{fig_base}",
        legend_label=f"ML Model",
        output_file=OUTPUT_ML_SAMPLED,
    )


if __name__ == "__main__":
    main()