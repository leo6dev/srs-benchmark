"""
train.py — Polars-native bulk loading with vectorised feature engineering.

Pipeline:
  1. pl.scan_parquet()     — reads entire revlogs dir lazily (Arrow, memory-mapped)
  2. Polars window exprs   — all features computed in one vectorised pass, Rust core
  3. UserChronologicalDataset — tensor assembly with tqdm per user (no pandas)
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    import pandas as pd
    HAS_POLARS = False
    print("[warn] polars not installed — falling back to pandas. pip install polars")

from config import load_config
from features import create_features


# =======================================================================================
# 1. CORE COMPONENTS (unchanged)
# =======================================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def apply_rotary_pos_emb(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos.unsqueeze(0).unsqueeze(0)) + (x_rot * sin.unsqueeze(0).unsqueeze(0))


class DeepSeekMLA(nn.Module):
    def __init__(self, d_model, n_heads, kv_lora_rank, d_rope):
        super().__init__()
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_head      = d_model // n_heads
        self.d_rope      = d_rope
        self.d_content   = self.d_head - d_rope
        self.kv_lora_rank = kv_lora_rank

        self.q_proj         = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.kv_c_proj      = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_c_norm      = RMSNorm(kv_lora_rank)
        self.k_content_proj = nn.Linear(kv_lora_rank, n_heads * self.d_content, bias=False)
        self.v_proj         = nn.Linear(kv_lora_rank, n_heads * self.d_head, bias=False)
        self.k_rope_proj    = nn.Linear(d_model, n_heads * d_rope, bias=False)
        self.o_proj         = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = torch.cat([q[..., :self.d_content],
                       apply_rotary_pos_emb(q[..., self.d_content:], cos, sin)], dim=-1)

        c_kv      = self.kv_c_norm(self.kv_c_proj(x))
        k_content = self.k_content_proj(c_kv).view(B, T, self.n_heads, self.d_content).transpose(1, 2)
        v         = self.v_proj(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k_rope    = apply_rotary_pos_emb(
                        self.k_rope_proj(x).view(B, T, self.n_heads, self.d_rope).transpose(1, 2),
                        cos, sin)
        k = torch.cat([k_content, k_rope], dim=-1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, self.d_model))


class GLMSwiGLUFFN(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# =======================================================================================
# 2. MODEL (unchanged)
# =======================================================================================

class DenseHybridRecallModel(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.d_model = d_model

        self.maturity_emb    = nn.Embedding(101, 32)
        self.rating_emb      = nn.Embedding(6, 16)
        self.last_rating_emb = nn.Embedding(6, 16)
        self.state_emb       = nn.Embedding(5, 16)
        self.cont_proj       = nn.Linear(5, 48)

        self.input_proj = nn.Sequential(nn.Linear(128, d_model), nn.SiLU(), RMSNorm(d_model))

        self.norm1 = RMSNorm(d_model)
        self.attn  = DeepSeekMLA(d_model=d_model, n_heads=n_heads, kv_lora_rank=32, d_rope=16)
        self.norm2 = RMSNorm(d_model)
        self.lstm  = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.norm3 = RMSNorm(d_model)
        self.ffn   = GLMSwiGLUFFN(d_model, d_model * 3)

        self.final_norm  = RMSNorm(d_model)
        self.recall_head = nn.Linear(d_model, 1)
        self.register_buffer("inv_freq",
            1.0 / (10000 ** (torch.arange(0, 16, 2).float() / 16)), persistent=False)

    def forward(self, maturity, cont_feats, prev_rating, last_rating, prev_state, lstm_state=None):
        B, T = maturity.shape
        h = self.input_proj(torch.cat([
            self.maturity_emb(maturity),
            self.rating_emb(prev_rating),
            self.last_rating_emb(last_rating),
            self.state_emb(prev_state),
            self.cont_proj(cont_feats),
        ], dim=-1))

        t     = torch.arange(T, device=h.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(h.dtype), emb.sin().to(h.dtype)

        h = h + self.attn(self.norm1(h), cos, sin)
        lstm_out, new_lstm_state = self.lstm(self.norm2(h), lstm_state)
        h = h + lstm_out
        h = h + self.ffn(self.norm3(h))

        return self.recall_head(self.final_norm(h)).squeeze(-1), new_lstm_state


# =======================================================================================
# 3. BULK DATA LOADING + FEATURE ENGINEERING
#
# Architecture: one scan_parquet call → all features computed in Polars' Rust core.
# No multiprocessing, no pandas, no workers.  Polars is already multithreaded.
# =======================================================================================

CONT_FEATS = ["elapsed_days", "elapsed_seconds", "nth_today", "delta_t", "prev_duration"]


def load_and_build_features(revlog_dir: Path, user_ids: list) -> pl.DataFrame:
    """
    Single-pass pipeline:
      1. scan_parquet  — lazy, reads the whole directory, Arrow memory-mapped
      2. collect()     — materialise into RAM once
      3. Polars window expressions — all derived columns in one vectorised pass

    Raw columns available: card_id, day_offset, duration, elapsed_days,
                           elapsed_seconds, rating, state, user_id

    Derived columns (equivalent to create_features output):
      review_th    — per-user chronological review index (0-based)
      y            — recall label: 1 if rating > 1, else 0
                     ⚠ VERIFY: matches your create_features definition of success
      delta_t      — same as elapsed_days (interval since last review of this card)
                     ⚠ VERIFY: if create_features computes delta_t differently, adjust
      nth_today    — how many reviews the user has done so far today (0-based)
      last_rating  — previous rating given to this specific card (0 = first review)
      card_review_count — how many times this card has been reviewed (0-based, capped 100)
      prev_rating  — rating from the immediately preceding review (any card)
      prev_state   — state from the immediately preceding review
      prev_duration — duration of the immediately preceding review
    """
    print("\n[Stage 1/2]  Bulk parquet scan (lazy, Arrow memory-mapped)...")

    try:
        lf = pl.scan_parquet(str(revlog_dir / "**/*.parquet"), hive_partitioning=True)
    except Exception:
        lf = pl.scan_parquet(str(revlog_dir), hive_partitioning=True)

    if user_ids:
        lf = lf.filter(pl.col("user_id").is_in(user_ids))

    with tqdm(total=1, desc="    Reading parquet", bar_format="{l_bar}{bar}| {elapsed}") as pbar:
        df = lf.collect()
        pbar.update(1)

    print(f"    {len(df):,} rows | {df['user_id'].n_unique():,} users")
    print(f"    Raw columns: {df.columns}")

    # ------------------------------------------------------------------
    # Sort once into chronological order per user.
    # elapsed_days + elapsed_seconds gives a total ordering even without
    # an explicit timestamp column.
    # ------------------------------------------------------------------
    print("\n[Stage 2/2]  Feature engineering (Polars vectorised, all users at once)...")

    df = df.sort(["user_id", "elapsed_days", "elapsed_seconds"])

    feature_steps = [
        "review_th + y + delta_t + nth_today",
        "last_rating",
        "card_review_count",
        "prev_rating / prev_state / prev_duration",
        "log1p scaling",
        "filter short users",
    ]

    with tqdm(total=len(feature_steps), desc="    Building features", unit="step") as pbar:

        # --- Step 1: core create_features equivalents -------------------
        df = df.with_columns([
            # review_th: 0-based chronological index within each user
            (pl.col("rating").cum_count().over("user_id") - 1)
             .cast(pl.Int64).alias("review_th"),

            # y: recall success label — rating > 1 means the user remembered
            # (Again=1 → forgot=0, Hard/Good/Easy=2/3/4 → recalled=1)
            (pl.col("rating") > 1).cast(pl.Int8).alias("y"),

            # delta_t: interval since last review — same as elapsed_days for this dataset
            pl.col("elapsed_days").alias("delta_t"),

            # nth_today: how many reviews done by this user so far today (0-based)
            (pl.col("rating").cum_count().over(["user_id", "day_offset"]) - 1)
             .cast(pl.Int64).alias("nth_today"),
        ])
        pbar.update(1)

        # --- Step 2: last_rating (prev rating for THIS card) ------------
        df = df.with_columns(
            pl.col("rating")
              .shift(1).over(["user_id", "card_id"])
              .fill_null(0).cast(pl.Int64)
              .alias("last_rating")
        )
        pbar.update(1)

        # --- Step 3: card maturity --------------------------------------
        df = df.with_columns(
            (pl.col("rating").cum_count().over(["user_id", "card_id"]) - 1)
              .clip(0, 100).cast(pl.Int64)
              .alias("card_review_count")
        )
        pbar.update(1)

        # --- Step 4: session-level lag features (prev review, any card) -
        df = df.with_columns([
            pl.col("rating")  .shift(1).over("user_id").fill_null(0).cast(pl.Int64).alias("prev_rating"),
            pl.col("state")   .shift(1).over("user_id").fill_null(4).cast(pl.Int64).alias("prev_state"),
            pl.col("duration").shift(1).over("user_id").fill_null(0.0).alias("prev_duration"),
        ])
        pbar.update(1)

        # --- Step 5: log1p-scale all continuous features ----------------
        df = df.with_columns([
            (pl.col(c).clip(lower_bound=0).log1p() / 5.0).alias(c)
            for c in ["elapsed_days", "elapsed_seconds", "nth_today", "delta_t", "prev_duration"]
            if c in df.columns
        ])
        pbar.update(1)

        # --- Step 6: drop users with fewer than 5 reviews ---------------
        counts      = df.group_by("user_id").agg(pl.len().alias("n"))
        valid_users = counts.filter(pl.col("n") >= 5)["user_id"]
        df          = df.filter(pl.col("user_id").is_in(valid_users))
        pbar.update(1)

    print(f"    {df['user_id'].n_unique():,} users | {len(df):,} rows ready for training")
    return df


# =======================================================================================
# 4. DATASET  — works natively on the Polars frame, no .to_pandas() on full data
# =======================================================================================

class UserChronologicalDataset(Dataset):
    """
    Receives the fully-engineered Polars DataFrame.
    Uses partition_by() to split per user — stays in Arrow memory throughout,
    only converting each small per-user slice to numpy for tensor assembly.
    """

    def __init__(self, df: pl.DataFrame, user_ids: list, max_seq_len: int = 2048):
        self.users_data: list = []

        # partition_by returns a list of DataFrames, one per user, in insertion order.
        # maintain_order=True preserves chronological sort from load_and_build_features.
        print(f"\n[Assembling]  Tensor chunks for {len(user_ids)} users...")

        # Build a lookup: user_id → Polars DataFrame (no pandas, no copy)
        user_id_set = set(user_ids)
        partitions  = {
            part["user_id"][0]: part
            for part in df.partition_by("user_id", maintain_order=True)
            if part["user_id"][0] in user_id_set
        }

        skipped = 0
        for user_id in tqdm(user_ids, desc="    Building sequences", unit="user"):
            grp = partitions.get(user_id)
            if grp is None or len(grp) < 5:
                skipped += 1
                continue

            # Extract to numpy — only this user's rows, not the full 144M row frame
            maturity    = grp["card_review_count"].to_numpy().astype(np.int64)
            prev_rating = grp["prev_rating"].to_numpy().astype(np.int64)
            last_rating = grp["last_rating"].to_numpy().astype(np.int64)
            prev_state  = grp["prev_state"].to_numpy().astype(np.int64)
            targets     = grp["y"].to_numpy().astype(np.float32)

            cont_array = np.zeros((len(grp), len(CONT_FEATS)), dtype=np.float32)
            for i, c in enumerate(CONT_FEATS):
                if c in grp.columns:
                    cont_array[:, i] = grp[c].to_numpy().astype(np.float32)

            chunks = []
            for start in range(0, len(grp), max_seq_len):
                end = min(start + max_seq_len, len(grp))
                if (end - start) < 2:
                    continue
                chunks.append({
                    "maturity":    torch.tensor(maturity[start:end]).unsqueeze(0),
                    "cont_feats":  torch.tensor(cont_array[start:end]).unsqueeze(0),
                    "prev_rating": torch.tensor(prev_rating[start:end]).unsqueeze(0),
                    "last_rating": torch.tensor(last_rating[start:end]).unsqueeze(0),
                    "prev_state":  torch.tensor(prev_state[start:end]).unsqueeze(0),
                    "target":      torch.tensor(targets[start:end]).unsqueeze(0),
                })
            if chunks:
                self.users_data.append(chunks)

        print(f"    {len(self.users_data)} users ready | {skipped} skipped")

    def __len__(self):
        return len(self.users_data)

    def __getitem__(self, idx):
        return self.users_data[idx]


# =======================================================================================
# 5. EVALUATION
# =======================================================================================

def evaluate(model, dataset, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for user_chunks in dataset:
            lstm_state = None
            for chunk in user_chunks:
                logits, lstm_state = model(
                    chunk["maturity"].to(device),
                    chunk["cont_feats"].to(device),
                    chunk["prev_rating"].to(device),
                    chunk["last_rating"].to(device),
                    chunk["prev_state"].to(device),
                    lstm_state=lstm_state,
                )
                all_preds.extend(torch.sigmoid(logits).view(-1).cpu().numpy())
                all_targets.extend(chunk["target"].view(-1).cpu().numpy())
    if not all_targets:
        return 0.0
    try:
        return roc_auc_score(all_targets, all_preds)
    except ValueError:
        return 0.5


# =======================================================================================
# 6. TRAINING
# =======================================================================================

def train():
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device  : {DEVICE}")
    print(f"Polars  : {'yes' if HAS_POLARS else 'no (pip install polars)'}")

    config     = load_config()
    revlog_dir = Path(config.data_path) / "revlogs"

    all_users   = list(range(1, 1100))
    train_users = all_users[:1000]
    test_users  = all_users[1000:]

    # ------------------------------------------------------------------
    # Load all data and build all features in one Polars pipeline.
    # No multiprocessing — Polars uses all CPU cores internally.
    # ------------------------------------------------------------------
    df_all = load_and_build_features(revlog_dir, all_users)

    # Build train / test from the same loaded frame
    train_dataset = UserChronologicalDataset(df_all, train_users, max_seq_len=2048)
    test_dataset  = UserChronologicalDataset(df_all, test_users,  max_seq_len=2048)
    del df_all  # tensors are inside the Dataset now; free the raw frame

    # ------------------------------------------------------------------
    # Model + optimiser
    # ------------------------------------------------------------------
    model = DenseHybridRecallModel(d_model=128, n_heads=4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel   : {n_params:.3f}M parameters")

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    decay     = [p for n, p in model.named_parameters()
                 if p.requires_grad and len(p.shape) >= 2 and "norm" not in n.lower()]
    no_decay  = [p for n, p in model.named_parameters()
                 if p.requires_grad and (len(p.shape) < 2 or "norm" in n.lower())]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.02},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=5e-4,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epochs           = 10
    grad_accum_steps = 4
    best_auc         = 0.0
    ckpt_path        = "dense_hybrid_best_ckpt.pth"

    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        optimizer.zero_grad()

        order        = list(range(len(train_dataset)))
        random.shuffle(order)
        total_chunks = sum(len(train_dataset[i]) for i in order)

        pbar = tqdm(total=total_chunks, desc=f"Epoch [{epoch+1:02d}/{epochs}]", unit="chunk")

        for uid in order:
            lstm_state = None
            for chunk in train_dataset[uid]:
                logits, lstm_state = model(
                    chunk["maturity"].to(DEVICE),
                    chunk["cont_feats"].to(DEVICE),
                    chunk["prev_rating"].to(DEVICE),
                    chunk["last_rating"].to(DEVICE),
                    chunk["prev_state"].to(DEVICE),
                    lstm_state=lstm_state,
                )
                lstm_state = tuple(s.detach() for s in lstm_state)

                loss = criterion(logits.view(-1), chunk["target"].to(DEVICE).view(-1))
                (loss / grad_accum_steps).backward()

                total_loss += loss.item()
                steps      += 1

                if steps % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix({"BCE": f"{total_loss / steps:.4f}"})
                pbar.update(1)

        if steps % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        pbar.close()

        test_auc = evaluate(model, test_dataset, DEVICE)

        if test_auc > best_auc:
            best_auc = test_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":  total_loss / max(steps, 1),
                "auc":   test_auc,
            }, ckpt_path)
            print(f"    [+] New best AUC — saved to {ckpt_path}")

        print(f"--> Epoch {epoch+1:02d} | BCE: {total_loss/max(steps,1):.4f} "
              f"| AUC: {test_auc:.4f} | Best: {best_auc:.4f}\n")

    print("Training complete.")
    torch.save(model.state_dict(), "dense_hybrid_srs_model.pth")


if __name__ == "__main__":
    train()