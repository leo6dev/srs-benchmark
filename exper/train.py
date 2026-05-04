import os
import random
from pathlib import Path
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    import polars as pl
except ImportError:
    raise ImportError("pip install polars pyarrow")

# =======================================================================================
# 1. CORE COMPONENTS
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
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_rope = d_rope
        self.d_content = self.d_head - d_rope

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.kv_c_proj = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_c_norm = RMSNorm(kv_lora_rank)
        self.k_content_proj = nn.Linear(kv_lora_rank, n_heads * self.d_content, bias=False)
        self.v_proj = nn.Linear(kv_lora_rank, n_heads * self.d_head, bias=False)
        self.k_rope_proj = nn.Linear(d_model, n_heads * d_rope, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = torch.cat([q[..., :self.d_content],
                       apply_rotary_pos_emb(q[..., self.d_content:], cos, sin)], dim=-1)

        c_kv = self.kv_c_norm(self.kv_c_proj(x))
        k_content = self.k_content_proj(c_kv).view(B, T, self.n_heads, self.d_content).transpose(1, 2)
        v = self.v_proj(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k_rope = apply_rotary_pos_emb(
            self.k_rope_proj(x).view(B, T, self.n_heads, self.d_rope).transpose(1, 2),
            cos, sin)
        k = torch.cat([k_content, k_rope], dim=-1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, self.d_model))


class GLMSwiGLUFFN(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, kv_lora_rank=64, d_rope=16):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = DeepSeekMLA(d_model, n_heads, kv_lora_rank, d_rope)
        self.norm2 = RMSNorm(d_model)
        self.ffn = GLMSwiGLUFFN(d_model, d_model * 3)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


# =======================================================================================
# 2. MODEL ARCHITECTURE (~2.1M Params, Deep Physics Embeddings)
# =======================================================================================

class DenseHybridRecallModel(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.CONT_DIM = 15  # All RWKV features ported

        self.maturity_emb = nn.Embedding(101, 32)
        self.rating_emb = nn.Embedding(6, 16)
        self.last_rating_emb = nn.Embedding(6, 16)
        self.state_emb = nn.Embedding(5, 16)
        self.last_state_emb = nn.Embedding(5, 16)

        # Non-linear interval spacing curve processor
        self.cont_proj = nn.Sequential(
            nn.Linear(self.CONT_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 128)
        )

        self.input_proj = nn.Sequential(
            nn.Linear(128 + 32 + 16 + 16 + 16 + 16, d_model),
            nn.SiLU(),
            RMSNorm(d_model)
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, kv_lora_rank=64, d_rope=16)
            for _ in range(num_layers)
        ])

        self.lstm_norm = RMSNorm(d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)

        self.final_norm = RMSNorm(d_model)
        self.recall_head = nn.Linear(d_model, 1)

        # Prior Bias initialization (Log-odds of ~85% success rate in Anki). Prevents early massive gradients.
        nn.init.constant_(self.recall_head.bias, 1.734)

        # Note: we use d_rope=16, so frequencies are matched to 16
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, 16, 2).float() / 16)), persistent=False)

    def forward(self, maturity, cont_feats, prev_rating, last_rating, prev_state, last_state, lstm_state=None):
        B, T = maturity.shape

        e_mat = self.maturity_emb(maturity)
        e_rate = self.rating_emb(prev_rating)
        e_last = self.last_rating_emb(last_rating)
        e_state = self.state_emb(prev_state)
        e_l_state = self.last_state_emb(last_state)
        e_cont = self.cont_proj(cont_feats)

        x = self.input_proj(torch.cat([e_mat, e_rate, e_last, e_state, e_l_state, e_cont], dim=-1))

        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)

        # 2 Causal Transformer blocks
        for layer in self.layers:
            x = layer(x, cos, sin)

        # 1 Sequential LSTM block
        lstm_out, new_lstm_state = self.lstm(self.lstm_norm(x), lstm_state)
        x = x + lstm_out

        return self.recall_head(self.final_norm(x)).squeeze(-1), new_lstm_state


# =======================================================================================
# 3. CHUNKED ITERABLE DATASET (100% Fixed Sorting & Physics Mappings)
# =======================================================================================

class ChunkedUserDataset(IterableDataset):
    def __init__(self, revlog_dir: Path, user_ids: list, chunk_size=32, max_seq_len=1536):
        self.revlog_dir = revlog_dir
        self.user_ids = user_ids
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len

        self.cont_cols = [
            "interval_days", "interval_sec", "interval_days_cum", "interval_sec_cum",
            "duration", "prev_duration", "diff_reviews", "diff_new_cards",
            "cum_reviews_today", "cum_new_cards_today", "day_offset_diff",
            "interval_sec_sin", "interval_sec_cos", "interval_sec_cum_sin", "interval_sec_cum_cos"
        ]

    def __len__(self):
        return len(self.user_ids)

    def _process_batch(self, batch_user_ids):
        try:
            lf = pl.scan_parquet(str(self.revlog_dir / "**/*.parquet"), hive_partitioning=True)
        except Exception:
            lf = pl.scan_parquet(str(self.revlog_dir), hive_partitioning=True)

        lf = lf.filter(pl.col("user_id").is_in(batch_user_ids))
        df = lf.collect()
        if df.is_empty(): return []

        # THE MOST CRITICAL BUG FIX: Add row index BEFORE sort to maintain true chronolgy!
        # Do NOT sort by interval (elapsed_days).
        df = df.with_row_index("row_idx")
        df = df.sort(["user_id", "day_offset", "row_idx"])

        df = df.with_columns([
            (pl.col("rating").cum_count().over("user_id") - 1).cast(pl.Int32).alias("review_th"),
            (pl.col("rating") > 1).cast(pl.Float32).alias("y"),
            (pl.col("elapsed_days") == -1).cast(pl.Int32).alias("is_first_review"),
            pl.col("elapsed_days").clip(lower_bound=0).alias("interval_days"),
            pl.col("elapsed_seconds").clip(lower_bound=0).alias("interval_sec"),
        ])

        df = df.with_columns([
            pl.col("is_first_review").cum_sum().over("user_id").alias("cum_new_cards"),
            pl.col("interval_days").cum_sum().over(["user_id", "card_id"]).alias("interval_days_cum"),
            pl.col("interval_sec").cum_sum().over(["user_id", "card_id"]).alias("interval_sec_cum"),
        ])

        df = df.with_columns([
            pl.col("rating").shift(1).over(["user_id", "card_id"]).fill_null(0).cast(pl.Int32).alias("last_rating"),
            pl.col("state").shift(1).over(["user_id", "card_id"]).fill_null(0).cast(pl.Int32).alias("last_state"),

            pl.col("rating").shift(1).over("user_id").fill_null(0).cast(pl.Int32).alias("prev_rating"),
            pl.col("state").shift(1).over("user_id").fill_null(0).cast(pl.Int32).alias("prev_state"),

            pl.col("review_th").shift(1).over(["user_id", "card_id"]).fill_null(0).alias("last_card_th"),
            pl.col("cum_new_cards").shift(1).over(["user_id", "card_id"]).fill_null(0).alias("last_card_new_cards"),

            pl.col("day_offset").shift(1).over("user_id").fill_null(pl.col("day_offset")).alias("prev_day_offset"),
            pl.col("duration").shift(1).over("user_id").fill_null(0.0).alias("prev_duration")
        ])

        df = df.with_columns([
            (pl.col("review_th") - pl.col("last_card_th") - 1).clip(lower_bound=0).cast(pl.Float32).alias(
                "diff_reviews"),
            (pl.col("cum_new_cards") - pl.col("last_card_new_cards")).clip(lower_bound=0).cast(pl.Float32).alias(
                "diff_new_cards"),
            (pl.col("day_offset") - pl.col("prev_day_offset")).clip(lower_bound=0).cast(pl.Float32).alias(
                "day_offset_diff"),
            pl.col("rating").cum_count().over(["user_id", "day_offset"]).cast(pl.Float32).alias("cum_reviews_today"),
            pl.col("is_first_review").cum_sum().over(["user_id", "day_offset"]).cast(pl.Float32).alias(
                "cum_new_cards_today"),
            (pl.col("rating").cum_count().over(["user_id", "card_id"]) - 1).clip(lower_bound=0, upper_bound=100).cast(
                pl.Int32).alias("card_review_count")
        ])

        # Circadian rhythms
        df = df.with_columns([
            ((pl.col("interval_sec") % 86400) * 2 * np.pi / 86400).sin().cast(pl.Float32).alias("interval_sec_sin"),
            ((pl.col("interval_sec") % 86400) * 2 * np.pi / 86400).cos().cast(pl.Float32).alias("interval_sec_cos"),
            ((pl.col("interval_sec_cum") % 86400) * 2 * np.pi / 86400).sin().cast(pl.Float32).alias(
                "interval_sec_cum_sin"),
            ((pl.col("interval_sec_cum") % 86400) * 2 * np.pi / 86400).cos().cast(pl.Float32).alias(
                "interval_sec_cum_cos"),
        ])

        cols_to_scale = [
            "interval_days", "interval_sec", "interval_days_cum", "interval_sec_cum",
            "duration", "prev_duration", "diff_reviews", "diff_new_cards",
            "cum_reviews_today", "cum_new_cards_today", "day_offset_diff"
        ]

        df = df.with_columns([
            (pl.col(c).clip(lower_bound=0).log1p() / 5.0).cast(pl.Float32).alias(c)
            for c in cols_to_scale if c in df.columns
        ])

        user_partitions = df.partition_by("user_id", maintain_order=True)
        users_list = []

        for up in user_partitions:
            if len(up) < 5: continue

            data = {
                "maturity": up["card_review_count"].to_numpy().copy(),
                "prev_rating": up["prev_rating"].to_numpy().copy(),
                "last_rating": up["last_rating"].to_numpy().copy(),
                "prev_state": up["prev_state"].to_numpy().copy(),
                "last_state": up["last_state"].to_numpy().copy(),
                "target": up["y"].to_numpy().copy(),
                "cont": up.select(self.cont_cols).to_numpy().copy()
            }

            total_len = len(up)
            user_chunks = []

            for start in range(0, total_len, self.max_seq_len):
                end = min(start + self.max_seq_len, total_len)
                if end - start < 2: continue

                user_chunks.append({
                    "maturity": torch.from_numpy(data["maturity"][start:end]),
                    "prev_rating": torch.from_numpy(data["prev_rating"][start:end]),
                    "last_rating": torch.from_numpy(data["last_rating"][start:end]),
                    "prev_state": torch.from_numpy(data["prev_state"][start:end]),
                    "last_state": torch.from_numpy(data["last_state"][start:end]),
                    "cont_feats": torch.from_numpy(data["cont"][start:end]),
                    "target": torch.from_numpy(data["target"][start:end]),
                })

            if user_chunks:
                users_list.append(user_chunks)

        del df, user_partitions
        gc.collect()
        return users_list

    def __iter__(self):
        random.shuffle(self.user_ids)
        batches = [self.user_ids[i: i + self.chunk_size] for i in range(0, len(self.user_ids), self.chunk_size)]

        for batch_ids in batches:
            user_lists = self._process_batch(batch_ids)
            for user_chunks in user_lists:
                yield user_chunks


def pad_collate_users(batch):
    max_chunks = max(len(u) for u in batch)
    step_batches = []

    for t in range(max_chunks):
        step_items = []
        for user_chunks in batch:
            if t < len(user_chunks):
                step_items.append(user_chunks[t])
            else:
                dummy = {k: torch.zeros_like(v[0:1]) for k, v in user_chunks[0].items()}
                dummy["is_dummy"] = True
                step_items.append(dummy)

        lengths = [b['target'].shape[0] for b in step_items]
        max_len = max(lengths)

        padded_batch = {}
        for key in step_items[0].keys():
            if key == "is_dummy": continue
            padded_items = []
            for b in step_items:
                pad_size = max_len - b[key].shape[0]
                if key == 'cont_feats':
                    padded_items.append(F.pad(b[key], (0, 0, 0, pad_size), value=0))
                else:
                    padded_items.append(F.pad(b[key], (0, pad_size), value=0))
            padded_batch[key] = torch.stack(padded_items)

        mask = torch.zeros((len(step_items), max_len), dtype=torch.bool)
        for i, (l, b) in enumerate(zip(lengths, step_items)):
            if not b.get("is_dummy", False):
                mask[i, :l] = True

        padded_batch['mask'] = mask
        step_batches.append(padded_batch)

    return step_batches
