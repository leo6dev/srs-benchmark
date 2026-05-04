import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random

from config import load_config
from data_loader import UserDataLoader


# =======================================================================================
# 1. CORE COMPONENTS (DeepSeek MLA + GLM FFN)
# =======================================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def apply_rotary_pos_emb(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (x_rot * sin)


class DeepSeekMLA(nn.Module):
    def __init__(self, d_model, n_heads, kv_lora_rank, d_rope):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_rope = d_rope
        self.d_content = self.d_head - self.d_rope
        self.kv_lora_rank = kv_lora_rank

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.kv_c_proj = nn.Linear(self.d_model, self.kv_lora_rank, bias=False)
        self.kv_c_norm = RMSNorm(self.kv_lora_rank)
        self.k_content_proj = nn.Linear(self.kv_lora_rank, self.n_heads * self.d_content, bias=False)
        self.v_proj = nn.Linear(self.kv_lora_rank, self.n_heads * self.d_head, bias=False)
        self.k_rope_proj = nn.Linear(self.d_model, self.n_heads * self.d_rope, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = torch.cat([q[..., :self.d_content], apply_rotary_pos_emb(q[..., self.d_content:], cos, sin)], dim=-1)

        c_kv = self.kv_c_norm(self.kv_c_proj(x))
        k_content = self.k_content_proj(c_kv).view(B, T, self.n_heads, self.d_content).transpose(1, 2)
        v = self.v_proj(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        k_rope = apply_rotary_pos_emb(self.k_rope_proj(x).view(B, T, self.n_heads, self.d_rope).transpose(1, 2), cos,
                                      sin)
        k = torch.cat([k_content, k_rope], dim=-1)

        # PyTorch SDPA is massively optimized and handles the causal mask natively
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model))


class GLMSwiGLUFFN(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# =======================================================================================
# 2. HYBRID RECALL MODEL (Rich Sequential Context)
# =======================================================================================

class DeepHybridRecallModel(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        """
        Total params: ~1.8M.
        Highly capable of learning its own feature engineering using local ID tracking.
        """
        super().__init__()
        self.d_model = d_model

        # --- Embeddings ---
        # Per-user factorized local card ID (Acts as the matching key for attention)
        self.local_card_emb = nn.Embedding(8192, 64)

        # Categorical states
        self.rating_emb = nn.Embedding(6, 16)  # What did the user press *just before* this?
        self.last_rating_emb = nn.Embedding(6, 16)  # What did the user press *last time* on THIS card?
        self.state_emb = nn.Embedding(5, 16)  # Card state

        # Continuous projection (5 rich historical features)
        self.cont_proj = nn.Linear(5, 64)

        # Mix categorical + continuous (64 + 16 + 16 + 16 + 64 = 176)
        self.input_proj = nn.Sequential(
            nn.Linear(176, d_model),
            nn.SiLU(),
            RMSNorm(d_model)
        )

        # --- Local Context Block (DeepSeek MLA) ---
        self.norm1 = RMSNorm(d_model)
        self.attn = DeepSeekMLA(d_model=d_model, n_heads=n_heads, kv_lora_rank=32, d_rope=16)

        # --- Global Context Block (Stateful LSTM) ---
        self.norm2 = RMSNorm(d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)

        # --- FFN ---
        self.norm3 = RMSNorm(d_model)
        self.ffn = GLMSwiGLUFFN(d_model, intermediate_size=d_model * 3)

        self.final_norm = RMSNorm(d_model)
        self.recall_head = nn.Linear(d_model, 1)

        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, 16, 2).float() / 16)), persistent=False)

    def forward(self, local_card, cont_feats, prev_rating, last_rating, prev_state, lstm_state=None):
        B, T = local_card.shape

        # 1. Embeddings
        e_card = self.local_card_emb(local_card)
        e_prev_rat = self.rating_emb(prev_rating)
        e_last_rat = self.last_rating_emb(last_rating)
        e_stat = self.state_emb(prev_state)
        e_cont = self.cont_proj(cont_feats)

        # 2. Combine Features
        x_concat = torch.cat([e_card, e_prev_rat, e_last_rat, e_stat, e_cont], dim=-1)
        h = self.input_proj(x_concat)

        # 3. Setup RoPE (Dynamic for any sequence length)
        t = torch.arange(T, device=h.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(h.dtype), emb.sin().to(h.dtype)

        # 4. Local Context (MLA) - Native Causal SDPA handles masking automatically!
        h = h + self.attn(self.norm1(h), cos, sin)

        # 5. Global Stateful Context (LSTM)
        h_norm = self.norm2(h)
        if lstm_state is not None:
            lstm_out, new_lstm_state = self.lstm(h_norm, lstm_state)
        else:
            lstm_out, new_lstm_state = self.lstm(h_norm)

        h = h + lstm_out

        # 6. FFN Block
        h = h + self.ffn(self.norm3(h))

        h = self.final_norm(h)
        return self.recall_head(h).squeeze(-1), new_lstm_state


# =======================================================================================
# 3. DATASET (Per-User Factorization & True Sequential History)
# =======================================================================================

class UserChronologicalDataset(Dataset):
    def __init__(self, user_ids, data_loader: UserDataLoader, max_seq_len=2048):
        self.users_data = []

        print(f"Extracting rich user chronologies for {len(user_ids)} users...")
        for user_id in tqdm(user_ids):
            try:
                df = data_loader.load_user_data(user_id)
            except Exception:
                continue

            if len(df) < 5:
                continue

            df = df.sort_values("review_th").copy()

            # 1. PER-USER LOCAL FACTORIZATION (The Magic Bullet)
            # This turns global noisy IDs into local working memory keys (0, 1, 2...)
            df['local_card_id'] = pd.factorize(df['card_id'])[0] % 8192

            # 2. Shifted "Immediate Past" History (Session context)
            df['prev_rating'] = df['rating'].shift(1).fillna(0)
            df['prev_state'] = df['state'].shift(1).fillna(4)
            df['prev_duration'] = df['duration'].shift(1).fillna(0.0)

            # 3. The specific card's last rating
            if 'last_rating' not in df.columns:
                df['last_rating'] = 0.0
            df['last_rating'] = df['last_rating'].fillna(0.0)

            # 4. Continuous Log Scaling (Div by 5.0 keeps variance tight for the NN)
            cont_feats = ['elapsed_days', 'elapsed_seconds', 'nth_today', 'delta_t', 'prev_duration']
            for c in cont_feats:
                df[c] = np.log1p(np.maximum(df[c].values, 0.0)) / 5.0

            # Extract arrays
            local_card = df['local_card_id'].values.astype(np.int64)
            cont_array = df[cont_feats].values.astype(np.float32)
            prev_rating = df['prev_rating'].values.astype(np.int64)
            last_rating = df['last_rating'].values.astype(np.int64)
            prev_state = df['prev_state'].values.astype(np.int64)
            targets = df['y'].values.astype(np.float32)

            total_len = len(df)
            user_chunks = []

            # Because we use PyTorch SDPA without padding, we can use exact unpadded chunks!
            for i in range(0, total_len, max_seq_len):
                end = min(i + max_seq_len, total_len)
                if (end - i) < 2:
                    continue

                # Add batch dimension immediately -> Shape: (1, seq_len)
                user_chunks.append({
                    "local_card": torch.tensor(local_card[i:end]).unsqueeze(0),
                    "cont_feats": torch.tensor(cont_array[i:end]).unsqueeze(0),
                    "prev_rating": torch.tensor(prev_rating[i:end]).unsqueeze(0),
                    "last_rating": torch.tensor(last_rating[i:end]).unsqueeze(0),
                    "prev_state": torch.tensor(prev_state[i:end]).unsqueeze(0),
                    "target": torch.tensor(targets[i:end]).unsqueeze(0)
                })

            if len(user_chunks) > 0:
                self.users_data.append(user_chunks)

    def __len__(self):
        return len(self.users_data)

    def __getitem__(self, idx):
        return self.users_data[idx]


# =======================================================================================
# 4. TRAINING & EVALUATION (Accumulated TBPTT Loop)
# =======================================================================================

def evaluate(model, dataset, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for user_chunks in dataset:
            lstm_state = None

            for chunk in user_chunks:
                l_card = chunk["local_card"].to(device)
                cont = chunk["cont_feats"].to(device)
                p_rat = chunk["prev_rating"].to(device)
                l_rat = chunk["last_rating"].to(device)
                p_stat = chunk["prev_state"].to(device)
                targets = chunk["target"].to(device)

                logits, lstm_state = model(l_card, cont, p_rat, l_rat, p_stat, lstm_state=lstm_state)

                probs = torch.sigmoid(logits)
                all_preds.extend(probs.view(-1).cpu().numpy())
                all_targets.extend(targets.view(-1).cpu().numpy())

    if len(all_targets) == 0: return 0.0
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5
    return auc


def train_deep_hybrid():
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    config = load_config()
    data_loader = UserDataLoader(config)

    all_users = list(range(1, 551))
    train_users = all_users[:500]
    test_users = all_users[500:]

    print("\n[1/3] Preparing Datasets...")
    train_dataset = UserChronologicalDataset(train_users, data_loader, max_seq_len=2048)
    test_dataset = UserChronologicalDataset(test_users, data_loader, max_seq_len=2048)

    print("\n[2/3] Initializing Deep Hybrid Model...")
    model = DeepHybridRecallModel(d_model=256, n_heads=8).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model Parameters: {param_count:.3f} M (Optimized for Edge)")

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    decay_params = [p for n, p in model.named_parameters() if
                    p.requires_grad and len(p.shape) >= 2 and "norm" not in n.lower()]
    no_decay_params = [p for n, p in model.named_parameters() if
                       p.requires_grad and (len(p.shape) < 2 or "norm" in n.lower())]

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.02},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=4e-4)

    print("\n[3/3] Starting Training Loop...")
    epochs = 10
    grad_accum_steps = 4  # Smooths out gradients over multiple user chunks

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        optimizer.zero_grad()

        user_indices = list(range(len(train_dataset)))
        random.shuffle(user_indices)

        total_chunks = sum(len(train_dataset[i]) for i in user_indices)
        pbar = tqdm(total=total_chunks, desc=f"Epoch [{epoch + 1:02d}/{epochs}]")

        for uid in user_indices:
            user_chunks = train_dataset[uid]
            lstm_state = None

            for chunk in user_chunks:
                l_card = chunk["local_card"].to(DEVICE)
                cont = chunk["cont_feats"].to(DEVICE)
                p_rat = chunk["prev_rating"].to(DEVICE)
                l_rat = chunk["last_rating"].to(DEVICE)
                p_stat = chunk["prev_state"].to(DEVICE)
                targets = chunk["target"].to(DEVICE)

                # Forward Pass
                logits, lstm_state = model(l_card, cont, p_rat, l_rat, p_stat, lstm_state=lstm_state)

                # Detach State for TBPTT
                lstm_state = tuple(s.detach() for s in lstm_state)

                # Loss Calculation (No padding mask needed since we pass exact tensors!)
                loss = criterion(logits.view(-1), targets.view(-1))
                loss = loss / grad_accum_steps
                loss.backward()

                total_loss += loss.item() * grad_accum_steps
                steps += 1

                if steps % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix({"BCE": f"{total_loss / steps:.4f}"})
                pbar.update(1)

        # Final step if there are leftover gradients
        if steps % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        pbar.close()

        avg_train_loss = total_loss / max(steps, 1)
        test_auc = evaluate(model, test_dataset, DEVICE)

        print(f"--> Epoch {epoch + 1} | Train BCE: {avg_train_loss:.4f} | Test AUC: {test_auc:.4f}\n")

    print("\nTraining Complete.")
    torch.save(model.state_dict(), "deep_hybrid_srs_model.pth")


if __name__ == "__main__":
    train_deep_hybrid()