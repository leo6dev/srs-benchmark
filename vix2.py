"""
Forgetting Curve Visualizer — with review-count-scaled decay
===========================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from abc import ABC, abstractmethod

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

USER_ID    = 1003
REVLOG_DIR = Path(f"/Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/user_id={USER_ID}")
N_CARDS    = 20        # Evaluate strictly on the first N_CARDS reviews of valid cards
USE_DEMO   = False

# ─────────────────────────────────────────────────────────────────────────────
# RATING METADATA
# ─────────────────────────────────────────────────────────────────────────────

RATING_RECALLED = {1: 0, 2: 1, 3: 1, 4: 1}

# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

class RecallModel(ABC):
    @abstractmethod
    def predict(self, elapsed_days: np.ndarray, n=None) -> np.ndarray:
        ...

    def fit(self, elapsed_days: np.ndarray, recalled: np.ndarray) -> "RecallModel":
        return self

    def label(self) -> str:
        return type(self).__name__


class ExponentialDecayModel(RecallModel):
    def __init__(self, k: float = 0.1):
        self.k = k

    def fit(self, elapsed_days: np.ndarray, recalled: np.ndarray) -> "ExponentialDecayModel":
        valid = elapsed_days > 0
        if valid.sum() < 2: return self
        t = elapsed_days[valid].astype(float)
        y = recalled[valid].astype(float)
        eps = 1e-9
        def neg_log_likelihood(k):
            p = np.exp(-k * t)
            ll = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            return -ll.sum()

        result = minimize_scalar(neg_log_likelihood, bounds=(1e-4, 20.0), method="bounded")
        self.k = result.x
        print(f"[ExponentialDecayModel] MLE k = {self.k:.5f}  (LL = {-result.fun:.2f})")
        return self

    def predict(self, elapsed_days: np.ndarray, n=None) -> np.ndarray:
        return np.exp(-self.k * np.clip(elapsed_days, 0, None))

    def label(self) -> str:
        return f"$e^{{-{self.k:.3f}\\,\\Delta t}}$"


class ReviewCountDecayModel(RecallModel):
    def __init__(self, k0: float = 0.1, r: float = 0.9):
        self.k0 = float(k0)
        self.r = float(r)

    def fit(self, elapsed_days: np.ndarray, recalled: np.ndarray, n_counts: np.ndarray = None) -> "ReviewCountDecayModel":
        elapsed_days = np.asarray(elapsed_days, dtype=float)
        recalled     = np.asarray(recalled, dtype=float)
        if n_counts is None: n_counts = np.arange(1, len(elapsed_days) + 1, dtype=float)
        else: n_counts = np.asarray(n_counts, dtype=float)

        valid = elapsed_days > 0
        if valid.sum() < 2: return self

        t, y, n = elapsed_days[valid], recalled[valid], n_counts[valid]
        eps = 1e-9

        def neg_log_likelihood(params):
            k0, r = params
            if k0 <= 0 or r <= 0: return 1e12
            p = np.exp(-k0 * (r ** n) * t)
            ll = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
            return -ll.sum()

        res = minimize(neg_log_likelihood, [max(1e-6, self.k0), max(1e-6, self.r)], bounds=[(1e-6, 20.0), (1e-6, 2.0)], method="L-BFGS-B")
        if res.success:
            self.k0, self.r = float(res.x[0]), float(res.x[1])
            print(f"[ReviewCountDecayModel] MLE k0 = {self.k0:.5f}, r = {self.r:.5f}  (LL = {-res.fun:.2f})")
        return self

    def predict(self, elapsed_days: np.ndarray, n=None) -> np.ndarray:
        t = np.asarray(elapsed_days, dtype=float)
        n_arr = np.ones_like(t) if n is None else (np.full_like(t, float(n)) if np.isscalar(n) else np.asarray(n, dtype=float))
        return np.exp(-self.k0 * (self.r ** n_arr) * np.clip(t, 0, None))

    def label(self) -> str:
        return f"$e^{{-k_0 r^n t}}\\; (k_0={self.k0:.3f}, r={self.r:.3f})$"

# ─────────────────────────────────────────────────────────────────────────────
# DATA & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def load_user_reviews(revlog_dir: Path, user_id: int) -> list[dict]:
    import polars as pl
    try: lf = pl.scan_parquet(str(revlog_dir / "**/*.parquet"), hive_partitioning=True)
    except Exception: lf = pl.scan_parquet(str(revlog_dir), hive_partitioning=True)
    df = lf.filter(pl.col("user_id") == user_id).collect().sort(["day_offset", "elapsed_seconds"])
    return df.to_dicts()


def _compute_auc_and_roc(y_true: np.ndarray, y_score: np.ndarray):
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        return float(roc_auc_score(y_true, y_score)), *roc_curve(y_true, y_score)
    except Exception:
        return float("nan"), None, None, None


def plot_forgetting_curve_single_card(reviews: list[dict], model: RecallModel, title: str, n_reviews: int = 5):
    from collections import defaultdict

    # 1. Group by card
    card_map = defaultdict(list)
    for r in reviews: card_map[r["card_id"]].append(r)
    for cid in card_map: card_map[cid].sort(key=lambda r: r["day_offset"])

    # 2. Base Filters & Culling
    valid_cards = {cid: revs for cid, revs in card_map.items() if len(revs) >= n_reviews}
    if not valid_cards: raise ValueError(f"No cards found with >= {n_reviews} reviews.")

    card_counts = np.array([len(revs) for revs in valid_cards.values()])
    med_revs = np.median(card_counts)
    max_allowed_reviews = max(np.percentile(card_counts, 75), med_revs + 2)

    # Strictly cull super-hard outliers
    culled_cards = {cid: revs for cid, revs in valid_cards.items() if len(revs) <= max_allowed_reviews}

    # 3. Train / Test Split (80/20) at the Card Level
    rng = np.random.default_rng(42)  # Fixed seed for repeatable splits
    cids = list(culled_cards.keys())
    rng.shuffle(cids)

    split_idx = int(len(cids) * 0.8)
    train_cids = set(cids[:split_idx])
    test_cids  = set(cids[split_idx:])

    print(f"\n--- DATA CULLING & SPLIT ---")
    print(f"Total valid cards (>= {n_reviews} revs): {len(valid_cards)}")
    print(f"Aggressively culled {len(valid_cards) - len(culled_cards)} long-tail cards (> {max_allowed_reviews:.1f} revs)")
    print(f"Train set: {len(train_cids)} cards | Test set: {len(test_cids)} cards")

    # 4. Build Training Data
    train_t, train_y, train_n = [], [],[]
    for cid in train_cids:
        for i, rev in enumerate(culled_cards[cid][:n_reviews]):
            train_t.append(rev["elapsed_days"])
            train_y.append(RATING_RECALLED[rev["rating"]])
            train_n.append(i + 1)

    print(f"\nFitting model on {len(train_t)} reviews...")
    try: model.fit(np.array(train_t), np.array(train_y), np.array(train_n))
    except TypeError: model.fit(np.array(train_t), np.array(train_y))

    # 5. Build Global Test Data & Evaluate
    test_t, test_y, test_n = [], [],[]
    for cid in test_cids:
        for i, rev in enumerate(culled_cards[cid][:n_reviews]):
            test_t.append(rev["elapsed_days"])
            test_y.append(RATING_RECALLED[rev["rating"]])
            test_n.append(i + 1)

    test_t = np.array(test_t, dtype=float)
    test_y = np.array(test_y, dtype=float)
    test_n = np.array(test_n, dtype=int)

    # Predict globally across ALL test cards
    try:
        y_score_global = model.predict(test_t, n=test_n)
    except TypeError:
        y_score_global = model.predict(test_t)

    global_auc, fpr, tpr, _ = _compute_auc_and_roc(test_y, y_score_global)
    auc_text = f"Global Test AUC = {global_auc:.3f}" if not np.isnan(global_auc) else "AUC: N/A"
    print(f"[ROC] {auc_text} (Evaluated on {len(test_t)} unseen test reviews)")

    # 6. Select "Typical" Card from TEST SET for Visualization
    test_recalls = {cid: np.mean([RATING_RECALLED[r["rating"]] for r in culled_cards[cid][:n_reviews]]) for cid in test_cids}
    med_recall = np.median(list(test_recalls.values()))

    # Find test card closest to median recall and median length
    std_revs = np.std([len(culled_cards[c]) for c in test_cids]) + 1e-6
    std_recall = np.std(list(test_recalls.values())) + 1e-6

    def card_distance(cid):
        return (abs(len(culled_cards[cid]) - med_revs) / std_revs) + (abs(test_recalls[cid] - med_recall) / std_recall)

    target_card_id = min(test_cids, key=card_distance)
    card_reviews = culled_cards[target_card_id][:n_reviews]
    print(f"Selected Test Card {target_card_id} for visual plot (Recall avg: {test_recalls[target_card_id]:.2f})")

    # 7. Plotting the single card
    first_day = float(card_reviews[0]["day_offset"])
    last_day  = float(card_reviews[-1]["day_offset"])

    xs, ys = [],[]
    CURVE_COLOR = "#c0392b"
    n_counts = np.arange(1, len(card_reviews) + 1, dtype=int)

    for i, rev in enumerate(card_reviews):
        t_review = float(rev["day_offset"])
        t_next   = float(card_reviews[i + 1]["day_offset"]) if i + 1 < len(card_reviews) else last_day

        if xs:
            ys.append(model.predict(np.array([float(rev["elapsed_days"])]), n=float(n_counts[i]))[0])
            xs.append(t_review)

        xs.append(t_review)
        ys.append(1.0)

        seg_len = t_next - t_review
        if seg_len > 0:
            t_seg = np.linspace(t_review, t_next, 300)
            p_seg = model.predict(t_seg - t_review, n=float(n_counts[i]))
            xs.extend(t_seg.tolist())
            ys.extend(p_seg.tolist())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, ys, color=CURVE_COLOR, linestyle="--", linewidth=2.0)

    for rev in card_reviews:
        ax.plot(float(rev["day_offset"]), float(RATING_RECALLED[rev["rating"]]),
                marker="x", markersize=12, markeredgewidth=2.8, color="black", zorder=5)

    curve_line = plt.Line2D([0],[0], color=CURVE_COLOR, linestyle="--", label=f"Model: {model.label()}")
    mark_handle = plt.Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=10, markeredgewidth=2.5, label="True label")
    ax.legend(handles=[curve_line, mark_handle], loc="upper right")

    # Annotate Global AUC onto the plot so we don't confuse it with single-card AUC
    ax.text(0.02, 0.95, auc_text, transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"))

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("P(Recall)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(first_day, last_day)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("forgetting_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    reviews = load_user_reviews(REVLOG_DIR, USER_ID)

    print("\n================ EVALUATING REVIEW COUNT MODEL ================")
    plot_forgetting_curve_single_card(reviews, ReviewCountDecayModel(), f"Forgetting Curves – User {USER_ID}", n_reviews=N_CARDS)

    print("\n================ EVALUATING BASE EXPONENTIAL MODEL ================")
    plot_forgetting_curve_single_card(reviews, ExponentialDecayModel(), f"Forgetting Curves – User {USER_ID}", n_reviews=N_CARDS)