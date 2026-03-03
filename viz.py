"""
Forgetting Curve Visualizer — with review-count-scaled decay
===========================================================
R(t, n) = exp( - (k0 * r**n) * t )

- The curve resets to 1.0 immediately after each review
- t on the x-axis = absolute day since the user's first review
- Δt in the formula = time since the last review of that card
- k0 and r are learned by MLE from that card's reviews
- X marks show the model's predicted recall *at* each review moment
- Colour of each X encodes the actual rating given

Rating alignment matches train.py:  rating > 1  →  recalled = True
  1 Again  →  0  (failed)
  2 Hard   →  1
  3 Good   →  1
  4 Easy   →  1
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from abc import ABC, abstractmethod

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  –  edit these to change data source / display
# ─────────────────────────────────────────────────────────────────────────────

USER_ID    = 912  # 912
REVLOG_DIR = Path(f"/Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/user_id={USER_ID}")
N_CARDS    = 20        # number of cards to plot
USE_DEMO   = False      # set True to run without real data

# ─────────────────────────────────────────────────────────────────────────────
# RATING METADATA  (aligned with train.py)
# ─────────────────────────────────────────────────────────────────────────────

RATING_RECALLED = {1: 0, 2: 1, 3: 1, 4: 1}
RATING_LABEL    = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}
RATING_COLOR    = {1: "#e74c3c", 2: "#e67e22", 3: "#2ecc71", 4: "#3498db"}

# ─────────────────────────────────────────────────────────────────────────────
# PLUGGABLE MODEL INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class RecallModel(ABC):
    @abstractmethod
    def predict(self, elapsed_days: np.ndarray, n=None) -> np.ndarray:
        """Return P(recall) for elapsed-days values (delta-t since last review)."""
        ...

    def fit(self, elapsed_days: np.ndarray, recalled: np.ndarray) -> "RecallModel":
        return self

    def label(self) -> str:
        return type(self).__name__


class ExponentialDecayModel(RecallModel):
    """R(delta_t) = e^{-k * delta_t}, with k optimised via scipy."""

    def __init__(self, k: float = 0.1):
        self.k = k

    def fit(self, elapsed_days: np.ndarray, recalled: np.ndarray) -> "ExponentialDecayModel":
        valid = elapsed_days > 0
        if valid.sum() < 2:
            return self
        t = elapsed_days[valid].astype(float)
        y = recalled[valid].astype(float)

        eps = 1e-9  # numerical guard inside log

        def neg_log_likelihood(k):
            p = np.exp(-k * t)
            ll = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            return -ll.sum()

        result = minimize_scalar(neg_log_likelihood, bounds=(1e-4, 20.0), method="bounded")
        self.k = result.x
        print(f"[ExponentialDecayModel] MLE k = {self.k:.5f}  (LL = {-result.fun:.2f})")
        return self

    def predict(self, elapsed_days: np.ndarray, n) -> np.ndarray:
        return np.exp(-self.k * np.clip(elapsed_days, 0, None))

    def label(self) -> str:
        return f"$e^{{-{self.k:.3f}\\,\\Delta t}}$"


class NeuralRecallModel(RecallModel):
    """
    Drop-in wrapper for DenseHybridRecallModel from train.py.
    Pass checkpoint_path to __init__, then use normally.
    Curve is drawn using elapsed_days only; other features are zero-imputed.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def _load(self):
        import torch, sys
        sys.path.insert(0, str(Path(self.checkpoint_path).parent))
        from train import DenseHybridRecallModel
        m = DenseHybridRecallModel(d_model=256, n_heads=8)
        m.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        m.eval()
        self._model = m

    def predict(self, elapsed_days: np.ndarray) -> np.ndarray:
        import torch
        if self._model is None:
            self._load()
        T = len(elapsed_days)
        cont = np.zeros((1, T, 6), dtype=np.float32)
        cont[0, :, 0] = np.log1p(np.clip(elapsed_days, 0, None)) / 5.0
        with torch.no_grad():
            logits, _ = self._model(
                maturity=torch.zeros(1, T, dtype=torch.long),
                cont_feats=torch.from_numpy(cont),
                prev_rating=torch.zeros(1, T, dtype=torch.long),
                last_rating=torch.zeros(1, T, dtype=torch.long),
                prev_state=torch.zeros(1, T, dtype=torch.long),
            )
        return torch.sigmoid(logits[0]).numpy()

    def label(self) -> str:
        return "Neural Recall Model"


# ─────────────────────────────────────────────────────────────────────────────
# New model: Review-count-scaled exponential decay
# R(t, n) = exp( - k0 * r**n * t )
# ─────────────────────────────────────────────────────────────────────────────

class ReviewCountDecayModel(RecallModel):
    """
    R(t, n) = exp( - (k0 * r**n) * t )

    - fit(elapsed_days, recalled, n_counts) : n_counts is 1-based review count per sample
    - predict(elapsed_days, n=...) : n can be scalar or array-like
    """

    def __init__(self, k0: float = 0.1, r: float = 0.9):
        self.k0 = float(k0)
        self.r = float(r)

    def fit(self, elapsed_days: np.ndarray, recalled: np.ndarray, n_counts: np.ndarray = None) -> "ReviewCountDecayModel":
        elapsed_days = np.asarray(elapsed_days, dtype=float)
        recalled     = np.asarray(recalled, dtype=float)

        if n_counts is None:
            n_counts = np.arange(1, len(elapsed_days) + 1, dtype=float)
        else:
            n_counts = np.asarray(n_counts, dtype=float)

        valid = elapsed_days > 0
        if valid.sum() < 2:
            return self

        t = elapsed_days[valid].astype(float)
        y = recalled[valid].astype(float)
        n = n_counts[valid].astype(float)

        eps = 1e-9

        def neg_log_likelihood(params):
            k0, r = params
            # keep params positive; if not, return large penalty
            if k0 <= 0 or r <= 0:
                return 1e12
            p = np.exp(-k0 * (r ** n) * t)
            ll = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            return -ll.sum()

        x0 = np.array([max(1e-6, self.k0), max(1e-6, self.r)])
        bounds = [(1e-6, 20.0), (1e-6, 2.0)]  # adjust r bounds if you want r constrained to <1

        res = minimize(neg_log_likelihood, x0, bounds=bounds, method="L-BFGS-B")
        if res.success:
            self.k0, self.r = float(res.x[0]), float(res.x[1])
            print(f"[ReviewCountDecayModel] MLE k0 = {self.k0:.5f}, r = {self.r:.5f}  (LL = {-res.fun:.2f})")
        else:
            print("[ReviewCountDecayModel] optimization failed, keeping defaults")
        return self

    def predict(self, elapsed_days: np.ndarray, n=None) -> np.ndarray:
        t = np.asarray(elapsed_days, dtype=float)

        # build n array broadcast-compatible with t
        if n is None:
            n_arr = np.ones_like(t, dtype=float)
        elif np.isscalar(n):
            n_arr = np.full_like(t, float(n), dtype=float)
        else:
            n_arr = np.asarray(n, dtype=float)
            if n_arr.shape != t.shape:
                try:
                    n_arr = np.broadcast_to(n_arr, t.shape)
                except Exception:
                    raise ValueError("n must be scalar or broadcastable to elapsed_days shape")

        k0 = float(self.k0)
        r  = float(self.r)
        return np.exp(-k0 * (r ** n_arr) * np.clip(t, 0, None))

    def label(self) -> str:
        return f"$e^{{-k_0 r^n t}}\\; (k_0={self.k0:.3f}, r={self.r:.3f})$"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_user_reviews(revlog_dir: Path, user_id: int) -> list[dict]:
    try:
        import polars as pl
    except ImportError:
        raise ImportError("pip install polars pyarrow")

    try:
        lf = pl.scan_parquet(str(revlog_dir / "**/*.parquet"), hive_partitioning=True)
    except Exception:
        lf = pl.scan_parquet(str(revlog_dir), hive_partitioning=True)

    df = (
        lf.filter(pl.col("user_id") == user_id)
          .collect()
          .sort(["day_offset", "elapsed_seconds"])
    )
    if df.is_empty():
        raise ValueError(f"No reviews found for user_id={user_id}")

    print(f"Loaded {len(df)} reviews for user {user_id}")
    return df.to_dicts()


def make_synthetic_reviews(n_cards: int = 5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    reviews = []
    for card_id in range(n_cards):
        day = 0.0
        interval = 1.0
        for _ in range(rng.integers(5, 12)):
            elapsed = interval
            k_true  = rng.uniform(0.05, 0.3)
            p       = np.exp(-k_true * elapsed)
            recalled = rng.random() < p
            rating = int(rng.choice([3, 4]) if recalled and p > 0.7
                         else 2 if recalled else 1)
            reviews.append({
                "card_id":        card_id,
                "day_offset":     round(day, 3),
                "elapsed_days":   round(elapsed, 3),
                "rating":         rating,
                "state":          2 if elapsed > 1 else 0,
                "duration":       int(rng.integers(1500, 15000)),
                "elapsed_seconds": round(elapsed * 86400),
            })
            interval = interval * rng.uniform(2.0, 3.5) if rating > 1 else 1.0
            day += elapsed
    reviews.sort(key=lambda r: (r["day_offset"], r["card_id"]))
    return reviews


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: ROC / AUC (sklearn if available, else fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_auc_and_roc(y_true: np.ndarray, y_score: np.ndarray):
    """
    Returns (auc, fpr, tpr, thresholds)
    Uses sklearn if available, otherwise a numpy implementation.
    """
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))
        return auc, fpr, tpr, thresholds
    except Exception:
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        n = len(y_true)
        pos = y_true.sum()
        neg = n - pos
        if pos == 0 or neg == 0:
            return float("nan"), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

        thresholds = np.concatenate(([np.inf], np.sort(np.unique(y_score))[::-1], [-np.inf]))
        fpr = []
        tpr = []
        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tpr.append(tp / pos)
            fpr.append(fp / neg)
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        order = np.argsort(fpr)
        auc = np.trapz(tpr[order], fpr[order])
        return float(auc), fpr, tpr, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_forgetting_curve_single_card(reviews: list[dict], model: RecallModel,
                                      title: str = "Forgetting Curve",
                                      n_reviews: int = 5):
    """
    Show the forgetting curve for ONE card over its first n_reviews reviews.

    - Picks the card with the most reviews
    - Curve is continuous: at each review it jumps vertically back to 1,
      then decays as e^{-k * delta_t} until the next review
    - X marks are placed at (day_of_review, true_label):
        true_label = 0 for Again (rating 1), 1 for everything else
    - A single curve colour; no rating colour coding

    Additional: computes ROC AUC for model predictions at review times and
    draws/saves a ROC curve (roc_curve.png). Annotates the main plot with AUC.
    """
    from collections import defaultdict

    # ── Pick one card (most reviews = most interesting trajectory) ───────────
    card_map: dict[int, list[dict]] = defaultdict(list)
    for r in reviews:
        card_map[r["card_id"]].append(r)

    card_reviews = sorted(
        max(card_map.values(), key=len),
        key=lambda r: r["day_offset"]
    )[:n_reviews]

    # ── Fit model on THIS card's reviews only (avoid cross-card bias) ────────
    card_elapsed  = np.array([r["elapsed_days"] for r in card_reviews], dtype=float)
    card_recalled = np.array([RATING_RECALLED[r["rating"]] for r in card_reviews], dtype=float)
    # n_counts: 1-based review number for this card
    n_counts = np.arange(1, len(card_reviews) + 1, dtype=int)

    # If the model supports 3-arg fit, pass n_counts; otherwise pass only (elapsed, recalled)
    try:
        model.fit(card_elapsed, card_recalled, n_counts)
    except TypeError:
        model.fit(card_elapsed, card_recalled)

    # ── Compute predictions at review times (just before each review) ───────
    y_true = card_recalled
    y_score = np.array([
        # prediction *just before* the i-th review uses elapsed_days for that review
        # and the review count n_counts[i]
        model.predict(np.array([float(r["elapsed_days"])]), n=float(n_counts[i]))[0]
        for i, r in enumerate(card_reviews)
    ])

    auc, fpr, tpr, thresholds = _compute_auc_and_roc(y_true, y_score)
    if np.isnan(auc):
        auc_text = "AUC: N/A (need both pos & neg samples)"
    else:
        auc_text = f"AUC = {auc:.3f}"
    print("[ROC] " + auc_text)

    # ── X-axis extent: a bit past the last review ────────────────────────────
    first_day = float(card_reviews[0]["day_offset"])
    last_day  = float(card_reviews[-1]["day_offset"])
    x_end     = last_day  # stop exactly at the last review

    # ── Build continuous (x, y) arrays with vertical jumps ──────────────────
    xs, ys = [], []
    CURVE_COLOR = "#c0392b"
    N_PTS = 300  # points per decay segment

    for i, rev in enumerate(card_reviews):
        t_review = float(rev["day_offset"])
        t_next   = float(card_reviews[i + 1]["day_offset"]) if i + 1 < len(card_reviews) else last_day

        # Vertical jump to 1.0 at review moment (creates the sawtooth reset)
        if xs:  # not the very first point
            xs.append(t_review)
            # value just before the reset:
            xs_val_before = model.predict(np.array([float(rev["elapsed_days"])]),
                                          n=float(n_counts[i]))[0]
            ys.append(xs_val_before)
        xs.append(t_review)
        ys.append(1.0)  # jump to 1 right after review

        # Decay segment from this review to the next
        seg_len = t_next - t_review
        if seg_len > 0:
            t_seg   = np.linspace(t_review, t_next, N_PTS)
            delta_t = t_seg - t_review
            # decay for the *current* review-count (n = i+1)
            p_seg   = model.predict(delta_t, n=float(n_counts[i]))
            xs.extend(t_seg.tolist())
            ys.extend(p_seg.tolist())

    # ── Plot forgetting curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, ys, color=CURVE_COLOR, linestyle="--", linewidth=2.0)

    # ── X marks at TRUE label height ─────────────────────────────────────────
    for rev in card_reviews:
        true_label = float(RATING_RECALLED[rev["rating"]])   # 0 or 1
        ax.plot(
            float(rev["day_offset"]), true_label,
            marker="x", markersize=12, markeredgewidth=2.8,
            color="black", zorder=5,
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    model_lbl  = model.label() if hasattr(model, "label") else type(model).__name__
    curve_line = plt.Line2D([0], [0], color=CURVE_COLOR, linestyle="--",
                            label=f"Model: {model_lbl}")
    mark_handle = plt.Line2D([0], [0], marker="x", color="black", linestyle="None",
                             markersize=10, markeredgewidth=2.5,
                             label="True label  (1 = recalled, 0 = forgotten)")
    ax.legend(handles=[curve_line, mark_handle], loc="upper right", fontsize=9)

    # ── AUC annotation on the main plot ─────────────────────────────────────
    ax.text(
        0.02, 0.95, auc_text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("P(Recall)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(first_day, x_end)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("forgetting_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → forgetting_curve.png")

    # ── Draw and save ROC curve as a separate figure ────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(fpr, tpr, linewidth=2, label=auc_text)
    ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Chance")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("ROC Curve", fontsize=13)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → roc_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if USE_DEMO:
    print("[Demo mode] Using synthetic data.")
    reviews = make_synthetic_reviews(n_cards=N_CARDS)
    title   = "Forgetting Curves – Demo"
else:
    reviews = load_user_reviews(REVLOG_DIR, USER_ID)
    title   = f"Forgetting Curves – User {USER_ID}"

# ── Swap model here to use different model: ExponentialDecayModel, NeuralRecallModel, etc.
model = ReviewCountDecayModel()  # <- the new review-count-scaled model
# model = ExponentialDecayModel()
# model = NeuralRecallModel("best_model_opt.pth")

plot_forgetting_curve_single_card(reviews, model, title=title, n_reviews=N_CARDS)