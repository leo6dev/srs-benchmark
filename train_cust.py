import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# =======================================================================================
# 1. YOUR EXACT DATA LOADER
# =======================================================================================
from config import load_config
from data_loader import UserDataLoader


# =======================================================================================
# 2. DEEPSEEK ATTENTION (DSA) MODEL ARCHITECTURE
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

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = torch.cat([q[..., :self.d_content], apply_rotary_pos_emb(q[..., self.d_content:], cos, sin)], dim=-1)

        c_kv = self.kv_c_norm(self.kv_c_proj(x))
        k_content = self.k_content_proj(c_kv).view(B, T, self.n_heads, self.d_content).transpose(1, 2)
        v = self.v_proj(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        k_rope = apply_rotary_pos_emb(self.k_rope_proj(x).view(B, T, self.n_heads, self.d_rope).transpose(1, 2), cos,
                                      sin)
        k = torch.cat([k_content, k_rope], dim=-1)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model))


class GLMSwiGLUFFN(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DirectRecallModel(nn.Module):
    def __init__(self, num_features, d_model=128, n_layers=6, n_heads=4):
        """
        By default, this creates a model with ~1.6 Million parameters.
        Perfect for running locally on mobile devices.
        """
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model * 2),
            nn.SiLU(),
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model)
        )

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': RMSNorm(d_model),
                'attn': DeepSeekMLA(d_model=d_model, n_heads=n_heads, kv_lora_rank=32, d_rope=16),
                'norm2': RMSNorm(d_model),
                'ffn': GLMSwiGLUFFN(d_model, intermediate_size=d_model * 3)
            }) for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.recall_head = nn.Linear(d_model, 1)

        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, 16, 2).float() / 16)), persistent=False)

    def forward(self, x, pad_mask):
        B, T, _ = x.shape

        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)

        # Causal (prevent looking into future reviews) + Padding Mask
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        pad_mask_expanded = pad_mask.view(B, 1, 1, T).bool()
        combined_mask = causal_mask & pad_mask_expanded

        h = self.input_proj(x)
        for layer in self.layers:
            h = h + layer['attn'](layer['norm1'](h), cos, sin, mask=combined_mask)
            h = h + layer['ffn'](layer['norm2'](h))

        h = self.final_norm(h)
        return self.recall_head(h).squeeze(-1)


# =======================================================================================
# 3. DATASET AND DATALOADER
# =======================================================================================

class CardHistoryDataset(Dataset):
    def __init__(self, user_ids, data_loader: UserDataLoader):
        self.samples = []
        self.feature_columns = None

        print(f"Extracting card histories for {len(user_ids)} users...")
        for user_id in tqdm(user_ids):
            try:
                df = data_loader.load_user_data(user_id)
            except Exception:
                continue

            # Automatically find numeric feature columns, ignoring meta-columns
            if self.feature_columns is None:
                ignore_cols = ['card_id', 'y', 'partition', 'tensor', 'r_history', 't_history']
                self.feature_columns = [
                    col for col in df.columns
                    if df[col].dtype in [np.float32, np.float64, np.int32, np.int64] and col not in ignore_cols
                ]

            # Group by card_id. A sequence is the review history of a SINGLE card.
            # This completely solves the 66,000 sequence length problem!
            for card_id, group in df.groupby("card_id"):
                if len(group) < 2:
                    continue  # Need at least 1 history step to predict a future step

                # Sort chronologically just in case
                group = group.sort_values("review_th")

                # To predict review T+1, we use features from review T.
                # Shift targets so X[t] aligns with Y[t+1]
                features = group[self.feature_columns].values[:-1].astype(np.float32)
                targets = group['y'].values[1:].astype(np.float32)

                self.samples.append({
                    "features": torch.tensor(features),
                    "target": torch.tensor(targets)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    features = [item["features"] for item in batch]
    targets = [item["target"] for item in batch]

    # Pad variable-length card histories
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0.0)

    # 1 for valid tokens, 0 for padded regions
    attention_mask = torch.zeros(features_padded.shape[:2], dtype=torch.bool)
    for i, seq in enumerate(features):
        attention_mask[i, :len(seq)] = True

    return features_padded, targets_padded, attention_mask


# =======================================================================================
# 4. TRAINING & EVALUATION LOOP
# =======================================================================================

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets, attn_mask in dataloader:
            features, targets, attn_mask = [x.to(device) for x in [features, targets, attn_mask]]

            logits = model(features, pad_mask=attn_mask)
            probs = torch.sigmoid(logits)

            # Filter out padded tokens using the attention mask
            valid_idx = attn_mask.view(-1)
            all_preds.extend(probs.view(-1)[valid_idx].cpu().numpy())
            all_targets.extend(targets.view(-1)[valid_idx].cpu().numpy())

    if len(all_targets) == 0:
        return 0.0

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5
    return auc


def train_standalone():
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    config = load_config()
    data_loader = UserDataLoader(config)

    # Target Users (First 100 per your request)
    all_users = list(range(1, 111))

    # 90-10 Split
    train_users = all_users[:100]
    test_users = all_users[100:]

    print("\n[1/3] Preparing Datasets...")
    train_dataset = CardHistoryDataset(train_users, data_loader)
    test_dataset = CardHistoryDataset(test_users, data_loader)

    print(f"Using {len(train_dataset.feature_columns)} features: {train_dataset.feature_columns}")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

    print("\n[2/3] Initializing Model...")
    model = DirectRecallModel(
        num_features=len(train_dataset.feature_columns),
        d_model=128,
        n_layers=6,
        n_heads=4
    ).to(DEVICE)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Don't apply weight decay to LayerNorms and Biases
    decay_params = [p for n, p in model.named_parameters() if
                    p.requires_grad and len(p.shape) >= 2 and "norm" not in n.lower()]
    no_decay_params = [p for n, p in model.named_parameters() if
                       p.requires_grad and (len(p.shape) < 2 or "norm" in n.lower())]

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=5e-4)

    print("\n[3/3] Starting Training Loop...")
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for features, targets, attn_mask in tqdm(train_loader, total=len(train_loader)):
            features, targets, attn_mask = [x.to(device=DEVICE) for x in [features, targets, attn_mask]]

            optimizer.zero_grad()
            logits = model(features, pad_mask=attn_mask)

            loss = criterion(logits, targets)

            # Apply Mask (Zeros out padded sequences)
            masked_loss = (loss * attn_mask.float()).sum()
            valid_tokens = attn_mask.sum()

            if valid_tokens > 0:
                mean_loss = masked_loss / valid_tokens
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += masked_loss.item()
                total_tokens += valid_tokens.item()

        avg_train_loss = total_loss / max(total_tokens, 1)
        test_auc = evaluate(model, test_loader, DEVICE)

        print(f"Epoch [{epoch + 1:02d}/{epochs}] | Train BCE Loss: {avg_train_loss:.4f} | Test AUC: {test_auc:.4f}")

    print("\nTraining Complete.")
    torch.save(model.state_dict(), "dsa_srs_model_final.pth")


if __name__ == "__main__":
    train_standalone()