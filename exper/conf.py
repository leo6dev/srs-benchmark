# mini_eval_confusion.py
#  0.00511817511844223

# python conf.py --revlog-dir /Users/leo/PycharmProjects/anki-revlogs-10k/revlogs/ --user-start 2001 --user-end 2101 --k 0.00511817511844223
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import polars as pl

# rating -> recalled (1=recalled, 0=forgotten)
RATING_RECALLED = {1: 0, 2: 1, 3: 1, 4: 1}


class ExponentialDecayAdapter:
    """P(recall) = exp(-k * elapsed_days)"""

    def __init__(self, k: float):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = float(k)

    def predict_user(self, reviews: List[dict]):
        y_true = []
        y_pred = []
        elapsed = []
        for r in reviews:
            t = float(r["elapsed_days"])
            if t <= 0:
                continue
            p = float(np.exp(-self.k * t))
            y_true.append(RATING_RECALLED.get(r["rating"], 0))
            y_pred.append(p)
            elapsed.append(t)
        return np.array(y_true, dtype=np.int32), np.array(y_pred, dtype=np.float32), np.array(elapsed, dtype=np.float32)


def _safe_scan_parquet(root: Path):
    """Try a recursive scan, fallback to direct path if pattern fails."""
    try:
        return pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True)
    except Exception:
        return pl.scan_parquet(str(root), hive_partitioning=True)


def find_present_users(revlog_dir: Path, user_start: int, user_end: int):
    lf = _safe_scan_parquet(revlog_dir)
    present = (
        lf.select("user_id")
          .filter((pl.col("user_id") >= user_start) & (pl.col("user_id") <= user_end))
          .unique()
          .collect()
          .to_series()
          .to_list()
    )
    return sorted(present)


def load_reviews_for_user(revlog_dir: Path, user_id: int):
    lf = _safe_scan_parquet(revlog_dir)
    df = (
        lf.filter(pl.col("user_id") == user_id)
          .collect()
          .with_row_count("_row_idx")
          .sort(["day_offset", "_row_idx"])
    )
    return df.drop("_row_idx").to_dicts()


def compute_confusion(y_true: np.ndarray, y_pred_probs: np.ndarray, thr: float = 0.5):
    if len(y_true) == 0:
        return {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    pred_bin = (y_pred_probs >= thr).astype(np.int32)
    TP = int(((pred_bin == 1) & (y_true == 1)).sum())
    FP = int(((pred_bin == 1) & (y_true == 0)).sum())
    TN = int(((pred_bin == 0) & (y_true == 0)).sum())
    FN = int(((pred_bin == 0) & (y_true == 1)).sum())
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}


def pretty_print_confusion(cm: dict):
    TP, FP, TN, FN = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    total = TP + FP + TN + FN
    print("\nConfusion matrix (threshold=0.5):")
    # conventional layout:
    #           Pred=1   Pred=0
    # True=1     TP       FN
    # True=0     FP       TN
    print(f"            Pred=1    Pred=0")
    print(f" True=1    {TP:8d} {FN:10d}")
    print(f" True=0    {FP:8d} {TN:10d}")
    acc = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\nTotals: n={total:,}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


def main():
    p = argparse.ArgumentParser(description="Mini eval: confusion matrix for exponential model")
    p.add_argument("--revlog-dir", required=True, help="root revlog parquet dir")
    p.add_argument("--user-start", type=int, required=True, help="first user id (inclusive)")
    p.add_argument("--user-end", type=int, required=True, help="last user id (inclusive)")
    p.add_argument("--k", type=float, required=True, help="exponential decay rate k")
    p.add_argument("--max-users", type=int, default=100, help="max users to evaluate (default 100)")
    p.add_argument("--threshold", type=float, default=0.5, help="decision threshold (default 0.5)")
    p.add_argument("--output-json", default=None, help="optional path to write aggregated results as JSON")
    args = p.parse_args()

    revlog_dir = Path(args.revlog_dir)
    if not revlog_dir.exists():
        raise SystemExit(f"revlog dir not found: {revlog_dir}")

    adapter = ExponentialDecayAdapter(k=args.k)

    present = find_present_users(revlog_dir, args.user_start, args.user_end)
    if not present:
        raise SystemExit("No users found in the requested range.")

    user_ids = present[: args.max_users]
    print(f"Evaluating exponential(k={args.k}) on {len(user_ids)} users (IDs {user_ids[0]}..{user_ids[-1]})")

    all_y_true = []
    all_y_pred = []
    n_reviews_per_user = {}

    for i, uid in enumerate(user_ids, start=1):
        reviews = load_reviews_for_user(revlog_dir, uid)
        if not reviews:
            continue
        y_true, y_pred, elapsed = adapter.predict_user(reviews)
        if len(y_true) == 0:
            continue
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        n_reviews_per_user[uid] = int(len(y_true))
        if i % 20 == 0:
            print(f"  processed {i}/{len(user_ids)} users...")

    if not all_y_true:
        raise SystemExit("No evaluable reviews (all elapsed_days <= 0 or no data).")

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    cm = compute_confusion(y_true_all, y_pred_all, thr=args.threshold)
    pretty_print_confusion(cm)

    if args.output_json:
        out = {
            "k": args.k,
            "threshold": args.threshold,
            "confusion": cm,
            "n_users": len(user_ids),
            "n_reviews": int(y_true_all.shape[0]),
            "n_reviews_per_user_sample": dict(list(n_reviews_per_user.items())[:20]),
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote summary to {args.output_json}")


if __name__ == "__main__":
    main()