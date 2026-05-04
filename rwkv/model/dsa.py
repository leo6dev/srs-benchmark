import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================================================================================
# CORE ARCHITECTURE ADAPTATION POINT: JIT Compilation (Removed for modern archs)
# =======================================================================================
ModuleType = torch.nn.Module


def FunctionType(fn): return fn


# =======================================================================================
# CONFIGURATION SHELL (Updated for GLM / DeepSeek)
# =======================================================================================
class ModelConfig(NamedTuple):
    d_model: int
    dropout: float

    # --- New GLM/DeepSeek Hyperparameters ---
    num_hidden_layers: int = 4  # Number of transformer blocks
    num_attention_heads: int = 8  # Number of attention heads

    # DeepSeek MLA specific config:
    # How much to compress the Key/Value representation to save memory
    kv_lora_rank: int = 64
    # How many dimensions per head are dedicated exclusively to Positional (RoPE) info
    qk_rope_head_dim: int = 16
    # The intermediate size for the SwiGLU FFN (typically 8/3 * d_model or 4 * d_model)
    intermediate_size: int = 1024


class SrsIterStatistics(NamedTuple):
    # (Unchanged from original)
    average_loss: torch.Tensor
    loss_tensor: torch.Tensor
    w_loss_avg: torch.Tensor
    ahead_logits_mag_loss_avg: torch.Tensor
    ahead_logits_diff_loss_avg: torch.Tensor
    ahead_avg: torch.Tensor
    ahead_raw_avg: torch.Tensor
    ahead_n: int
    ahead_equalize_avg: torch.Tensor
    ahead_raw_equalize_avg: torch.Tensor
    ahead_equalize_n: int
    imm_avg: torch.Tensor
    imm_n: int
    imm_binary_equalize_avg: torch.Tensor
    imm_binary_equalize_n: int
    p_curve: torch.Tensor
    p_imm: torch.Tensor
    p_imm_all: torch.Tensor
    w: torch.Tensor
    label_rating: torch.Tensor
    label_elapsed_seconds: torch.Tensor
    label_review_th: torch.Tensor
    is_query: torch.Tensor
    has_label: torch.Tensor


SrsRWKVIterStatistics = SrsIterStatistics


@dataclass
class PreparedBatch:
    """
    NEW BATCH FORMAT: Dense 3D Tensors.
    See dataloader instructions at the bottom of the script for how to yield this.
    """
    num_data: int
    features: torch.Tensor  # [Batch, Time, 92]
    attention_mask: torch.Tensor  # [Batch, Time] (1 for valid, 0 for pad)
    labels: torch.Tensor  # [Batch, Time, LabelDim]
    label_review_th: torch.Tensor

    def to(self, device):
        return PreparedBatch(
            num_data=self.num_data,
            features=self.features.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            label_review_th=self.label_review_th.to(device),
        )


# Exempts final SRS projection heads from half-precision downcasting to prevent NaNs
DTYPE_EXCLUDE = ["w_linear", "s_linear", "d_linear", "d_softplus", "k_linear", "p_linear", "ahead_linear"]


def is_excluded(name): return any(query in name for query in DTYPE_EXCLUDE)


# =======================================================================================
# NEW ARCHITECTURE COMPONENTS: GLM + DeepSeek Multi-Head Latent Attention
# =======================================================================================

class RMSNorm(nn.Module):
    """GLM / LLaMA standard RMSNorm for training stability."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def apply_rotary_pos_emb(x, cos, sin):
    """
    Applies Rotary Positional Embeddings.
    Expects x shape: [Batch, Heads, Time, Head_Dim]
    """
    # Split features in half to apply the rotary rotation
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    x_rot = torch.cat([-x2, x1], dim=-1)

    # Reshape cos/sin to broadcast over Batch and Heads: [1, 1, Time, Head_Dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (x_rot * sin)


class DeepSeekMLA(nn.Module):
    """
    DeepSeek Multi-Head Latent Attention (MLA).
    Instead of projecting Q, K, V entirely, we compress the input into a small latent
    vector `c_kv`, then expand it. This drastically reduces KV-cache memory in inference
    and acts as a powerful regularizing bottleneck during training.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.num_attention_heads
        self.d_head = self.d_model // self.n_heads

        # DeepSeek specific: Decouple the RoPE dimension from the content dimension
        self.d_rope = config.qk_rope_head_dim
        self.d_content = self.d_head - self.d_rope

        self.kv_lora_rank = config.kv_lora_rank

        # 1. Query Projection (Standard, though DeepSeek V2/3 compresses this too,
        # standardizing it here keeps training stable for smaller datasets).
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)

        # 2. KV Compression Projection (The "Latent" in MLA)
        # We crush the d_model down into a tiny kv_lora_rank (e.g., 64)
        self.kv_c_proj = nn.Linear(self.d_model, self.kv_lora_rank, bias=False)
        self.kv_c_norm = RMSNorm(self.kv_lora_rank)

        # 3. KV Expansion Projection
        # We expand the latent vector back into Content Keys and Content Values
        self.k_content_proj = nn.Linear(self.kv_lora_rank, self.n_heads * self.d_content, bias=False)
        self.v_proj = nn.Linear(self.kv_lora_rank, self.n_heads * self.d_head, bias=False)

        # 4. Decoupled RoPE Key Projection
        # We project the input independently just to fetch positional information for Keys
        self.k_rope_proj = nn.Linear(self.d_model, self.n_heads * self.d_rope, bias=False)

        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape

        # -- 1. Process Queries --
        # Shape: [B, T, n_heads, d_head] -> [B, n_heads, T, d_head]
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Split query into content and rope sections
        q_content = q[..., :self.d_content]
        q_rope = q[..., self.d_content:]
        # Apply RoPE only to the dedicated portion
        q_rope = apply_rotary_pos_emb(q_rope, cos, sin)
        q = torch.cat([q_content, q_rope], dim=-1)

        # -- 2. Process Keys and Values (The Latent Bottleneck) --
        # Shape: [B, T, kv_lora_rank]
        c_kv = self.kv_c_norm(self.kv_c_proj(x))

        # Expand Latent to Keys Content
        k_content = self.k_content_proj(c_kv).view(B, T, self.n_heads, self.d_content).transpose(1, 2)
        # Expand Latent to Values
        v = self.v_proj(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Generate RoPE Keys directly from input
        k_rope = self.k_rope_proj(x).view(B, T, self.n_heads, self.d_rope).transpose(1, 2)
        k_rope = apply_rotary_pos_emb(k_rope, cos, sin)

        # Concat content and RoPE for final Keys
        k = torch.cat([k_content, k_rope], dim=-1)

        # -- 3. Scaled Dot-Product Attention --
        # F.scaled_dot_product_attention natively supports Flash Attention if available.
        # If mask is provided (for padding + causal), we pass it.
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # Re-merge heads: [B, n_heads, T, d_head] -> [B, T, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.o_proj(attn_out)


class GLMSwiGLUFFN(nn.Module):
    """GLM standard SwiGLU activation network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GLMDeepSeekBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = DeepSeekMLA(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = GLMSwiGLUFFN(config)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GLMDeepSeekBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GLMDeepSeekBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.d_model)

        # Pre-compute RoPE frequencies. 10000 is the standard base.
        # Max length of 8192 is hardcoded here but dynamically sliced during forward pass.
        self.register_buffer("inv_freq", 1.0 / (
                    10000 ** (torch.arange(0, config.qk_rope_head_dim, 2).float() / config.qk_rope_head_dim)),
                             persistent=False)

    def _get_rope_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate for each paired dimension (e.g. [f1, f1, f2, f2...])
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        device = x.device

        # 1. Prepare RoPE mappings
        cos, sin = self._get_rope_cos_sin(T, device, x.dtype)

        # 2. Prepare Attention Mask (Causal + Padding combined)
        # We need a mask that prevents looking forward (Causal) AND ignores `<PAD>` tokens.
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))  # [T, T]

        if attention_mask is not None:
            # attention_mask is [B, T] where 1 is real, 0 is pad
            pad_mask = attention_mask.view(B, 1, 1, T).bool()  # [B, 1, 1, T]
            causal_mask = causal_mask.view(1, 1, T, T)  # [1, 1, T, T]
            # Logical AND: Keep token if it's strictly in the past AND it's a real token
            combined_mask = causal_mask & pad_mask
        else:
            combined_mask = causal_mask.view(1, 1, T, T)

        # 3. Forward Pass through blocks
        for layer in self.layers:
            x = layer(x, cos, sin, mask=combined_mask)

        return self.final_norm(x)


# =======================================================================================
# SRS MAIN MODEL SHELL
# =======================================================================================

class SrsSequenceModelShell(ModuleType):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.card_features_dim = 92
        self.d_model = config.d_model

        # SRS specific head dimensions
        self.features_fc_dim = 4 * self.d_model
        self.ahead_head_dim = 4 * self.d_model
        self.p_head_dim = 4 * self.d_model
        self.w_head_dim = 4 * self.d_model
        self.num_curves = 128

        with torch.no_grad():
            self.features2card = torch.nn.Sequential(
                torch.nn.Linear(self.card_features_dim, self.features_fc_dim),
                torch.nn.SiLU(),
                RMSNorm(self.features_fc_dim),  # Upgraded to RMSNorm
                torch.nn.Linear(self.features_fc_dim, self.d_model),
                torch.nn.SiLU(),
            )

            # --- INJECTED DEEPSEEK/GLM BACKBONE ---
            self.backbone = GLMDeepSeekBackbone(config)

            self.prehead_norm = RMSNorm(self.d_model)  # Upgraded to RMSNorm
            self.prehead_dropout = torch.nn.Dropout(p=config.dropout)

            self.head_ahead_logits = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.ahead_head_dim),
                torch.nn.ReLU(),
            )
            self.head_w = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, 1 * self.d_model),
                torch.nn.ReLU(),
                RMSNorm(1 * self.d_model),  # Upgraded to RMSNorm
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(1 * self.d_model, self.w_head_dim),
            )
            self.head_p = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.p_head_dim),
                torch.nn.ReLU(),
            )

            self.max_e = 21
            self.point_spread = 18.5
            self.num_points = 128
            self.ahead_linear = torch.nn.Linear(self.ahead_head_dim, self.num_points)
            torch.nn.init.zeros_(self.ahead_linear.weight)
            torch.nn.init.zeros_(self.ahead_linear.bias)

            self.w_linear = torch.nn.Linear(self.w_head_dim, self.num_curves)
            torch.nn.init.zeros_(self.w_linear.weight)
            torch.nn.init.zeros_(self.w_linear.bias)

            self.s_point_spread = 18.5
            self.s_max = 22

            self.p_linear = torch.nn.Linear(self.p_head_dim, 4)
            torch.nn.init.zeros_(self.p_linear.weight)
            self.p_linear.bias.copy_(torch.tensor([-0.3512, -0.0802, 0.4297, -0.2041]))

    @FunctionType
    def head_and_out(self, input):
        x = self.prehead_dropout(self.prehead_norm(input))
        out_w_logits = self.w_linear(self.head_w(x).float())
        out_w = torch.nn.functional.softmax(out_w_logits, dim=-1)
        out_w_log_p = torch.nn.functional.log_softmax(out_w_logits, dim=-1)
        out_ahead_logits = self.ahead_linear(self.head_ahead_logits(x).float())
        x_p = self.head_p(x).float()
        return out_ahead_logits, out_w, out_w_log_p, self.p_linear(x_p)

    @FunctionType
    def forgetting_curve(self, w, label_elapsed_seconds):
        s_space_raw = torch.exp(torch.linspace(0, self.s_point_spread, self.num_curves, device=w.device))
        s_space = 0.1 + (s_space_raw - 1) * (np.e ** (self.s_max - self.s_point_spread))
        label_elapsed_seconds = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        return 1e-5 + (1 - 2 * 1e-5) * torch.sum(w * 0.9 ** (label_elapsed_seconds / s_space), dim=-1)

    @FunctionType
    def interp(self, out_ahead_logits, label_elapsed_seconds):
        label_elapsed_seconds = torch.clamp(label_elapsed_seconds.contiguous(), min=1)
        point_space_raw = torch.exp(
            torch.linspace(0, self.point_spread, self.num_points, device=out_ahead_logits.device))
        point_space = 0.5 + (point_space_raw - 1) * (np.e ** (self.max_e - self.point_spread))
        right_idx = torch.searchsorted(point_space, label_elapsed_seconds)
        left_idx = torch.clamp(right_idx - 1, min=0)
        xl, xr = point_space[left_idx], point_space[right_idx]
        yl = torch.gather(out_ahead_logits, dim=-1, index=left_idx)
        yr = torch.gather(out_ahead_logits, dim=-1, index=right_idx)
        res = 1e-5 + (1 - 2 * 1e-5) * (yl + (yr - yl) * (label_elapsed_seconds - xl) / (xr - xl))
        return res.squeeze(-1)

    @FunctionType
    def forward_batch(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.features2card(features)  # [B, T, d_model]
        x = self.backbone(x, attention_mask=attention_mask)
        return self.head_and_out(x)

    @FunctionType
    def nanmin(self, tensor):
        return tensor.nan_to_num(1e9).min()

    @FunctionType
    def nanmax(self, tensor):
        return tensor.nan_to_num(-1e9).max()

    @FunctionType
    def _get_loss(self, features, attention_mask, batch_labels, batch_label_review_th):
        out_ahead_logits, out_w, out_w_log_p, out_p_logits = self.forward_batch(features, attention_mask)

        if torch.isnan(out_ahead_logits).any(): return None

        global_labels = batch_labels.float()
        (label_elapsed_seconds, _, label_y, label_rating, has_label, label_is_equalize,
         is_query) = global_labels.unbind(-1)
        has_label = has_label.int()
        label_is_equalize = label_is_equalize.int()
        is_query = is_query.int()

        label_rating = torch.clamp(label_rating - 1, min=0)
        label_elapsed_seconds = label_elapsed_seconds.unsqueeze(-1)
        curve_probs_raw = self.forgetting_curve(out_w, label_elapsed_seconds)
        curve_logits_raw = torch.log(curve_probs_raw / (1 - curve_probs_raw))
        ahead_logit_residual = self.interp(out_ahead_logits, label_elapsed_seconds)
        curve_logits = curve_logits_raw + ahead_logit_residual
        curve_probs = torch.sigmoid(curve_logits)

        out_p_probs = torch.softmax(out_p_logits, dim=-1)
        out_p_again, out_p_1, out_p_2, out_p_3 = out_p_probs.unbind(dim=-1)
        out_p_binary = torch.clamp(1.0 - out_p_again, min=1e-5, max=1.0 - 1e-5)

        if torch.isnan(curve_probs).any(): raise Exception("nan")

        w_loss = torch.nn.functional.kl_div(input=out_w_log_p, target=torch.ones_like(out_w) / self.num_curves,
                                            reduction="none").mean(dim=-1)
        ahead_mask = (1 - is_query) * has_label
        immediate_mask = is_query * has_label
        ahead_equalize_mask = ahead_mask * label_is_equalize
        immediate_equalize_mask = immediate_mask * label_is_equalize

        curve_loss = torch.nn.functional.binary_cross_entropy_with_logits(curve_logits, label_y, reduction="none")
        curve_raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(curve_logits_raw, label_y,
                                                                              reduction="none")

        NUM_LABELS = 4
        B, T = label_rating.shape
        p_loss = torch.nn.functional.cross_entropy(out_p_logits.view(-1, NUM_LABELS), label_rating.long().view(-1),
                                                   reduction="none").view(B, T)
        p_binary_loss = torch.nn.functional.binary_cross_entropy(out_p_binary, label_y, reduction="none")

        ahead_avg = (curve_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_SCALE = 0.5
        ahead_raw_avg = (curve_raw_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_RAW_SCALE = 0.5
        immediate_avg = (p_loss * immediate_mask).sum() / (1e-8 + immediate_mask.sum())
        w_avg = (w_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        W_LOSS_SCALE = 1e-5

        ahead_logits_mag_loss = torch.sqrt(1e-16 + out_ahead_logits.square().mean(dim=-1))
        ahead_logits_mag_avg = (ahead_logits_mag_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_LOGITS_MAG_LOSS_SCALE = 1e-4

        ahead_logits_diff_loss = torch.sqrt(1e-16 + out_ahead_logits.diff().square().mean(dim=-1))
        ahead_logits_diff_avg = (ahead_logits_diff_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_LOGITS_DIFF_LOSS_SCALE = 1e-3

        loss_avg = (
                AHEAD_SCALE * ahead_avg + immediate_avg + AHEAD_RAW_SCALE * ahead_raw_avg +
                W_LOSS_SCALE * w_avg + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_avg +
                AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_avg
        )
        loss_tensor = (
                AHEAD_SCALE * curve_loss.detach() + p_loss.detach() + AHEAD_RAW_SCALE * curve_raw_loss.detach() +
                W_LOSS_SCALE * w_loss.detach() + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_loss.detach() +
                AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_loss.detach()
        )

        ahead_equalize_avg = (curve_loss * ahead_equalize_mask).sum() / (1e-8 + ahead_equalize_mask.sum())
        ahead_raw_equalize_avg = (curve_raw_loss * ahead_equalize_mask).sum() / (1e-8 + ahead_equalize_mask.sum())
        immediate_binary_equalize_avg = (p_binary_loss * immediate_equalize_mask).sum() / (
                    1e-8 + immediate_equalize_mask.sum())

        return SrsIterStatistics(
            average_loss=loss_avg,
            p_curve=curve_probs.detach(), p_imm=out_p_binary.detach(), p_imm_all=out_p_probs.detach(),
            loss_tensor=loss_tensor.detach(), ahead_avg=ahead_avg.detach(), ahead_raw_avg=ahead_raw_avg.detach(),
            ahead_n=ahead_mask.sum().detach(), ahead_equalize_avg=ahead_equalize_avg.detach(),
            ahead_raw_equalize_avg=ahead_raw_equalize_avg.detach(), ahead_equalize_n=ahead_equalize_mask.sum().detach(),
            imm_avg=immediate_avg.detach(), imm_n=immediate_mask.sum().detach(),
            imm_binary_equalize_avg=immediate_binary_equalize_avg.detach(),
            imm_binary_equalize_n=immediate_equalize_mask.sum().detach(),
            w_loss_avg=w_avg.detach(), ahead_logits_mag_loss_avg=ahead_logits_mag_avg.detach(),
            ahead_logits_diff_loss_avg=ahead_logits_diff_avg.detach(), w=out_w.detach(),
            label_review_th=batch_label_review_th.detach(), label_elapsed_seconds=label_elapsed_seconds.detach(),
            label_rating=label_rating.detach(), is_query=is_query.detach(), has_label=has_label.detach(),
        )

    def get_loss(self, batch: PreparedBatch):
        return self._get_loss(batch.features, batch.attention_mask, batch.labels, batch.label_review_th)

    def selective_cast(self, dtype):
        """Allows backbones to be in half-precision while keeping sensitive math FP32."""
        for name, module in self.named_modules():
            if len(name) == 0: continue
            if not is_excluded(name):
                if dtype == torch.bfloat16:
                    module = module.to(dtype)
                elif dtype == torch.float32:
                    pass
        return self

    def copy_downcast_(self, master_model, dtype):
        master_params = dict(master_model.named_parameters())
        with torch.no_grad():
            for name, param in self.named_parameters():
                target_dtype = torch.float32 if is_excluded(name) else dtype
                assert param.dtype == target_dtype
                param.data.copy_(master_params[name].to(target_dtype))
                assert param.dtype == target_dtype


SrsRWKV = SrsSequenceModelShell


# =======================================================================================
# METRICS EXTRACTOR (Unchanged)
# =======================================================================================
@dataclass
class AnkiDictStatistics:
    ahead_ps: dict[int, float];
    imm_ps: dict[int, float];
    imm_ps_all: dict
    label_ratings: dict[int, float];
    label_elapsed_seconds: dict[int, float];
    w: dict


AnkiRWKVDictStatistics = AnkiDictStatistics


def extract_p(stats: SrsIterStatistics):
    assert stats.label_review_th.size(0) == 1
    ahead_ps_dict, imm_ps_dict, label_ratings_dict, label_elapsed_seconds_dict, imm_ps_all_dict = {}, {}, {}, {}, {}

    label_review_ths = stats.label_review_th.squeeze(0).cpu().numpy()
    label_elapsed_seconds_list = stats.label_elapsed_seconds.squeeze(0).cpu().numpy()
    label_ratings_list = stats.label_rating.squeeze(0).cpu().numpy()
    has_labels = stats.has_label.squeeze(0).cpu().numpy()
    is_querys = stats.is_query.squeeze(0).cpu().numpy()
    p_curves = stats.p_curve.squeeze(0).cpu().numpy()
    p_imms = stats.p_imm.squeeze(0).cpu().numpy()
    p_imm_alls = stats.p_imm_all.squeeze(0).cpu().numpy()
    ws = stats.w.squeeze(0).cpu()

    for i in range(len(label_review_ths)):
        label_review_th = label_review_ths[i]
        label_elapsed_seconds_dict[label_review_th] = label_elapsed_seconds_list[i]
        label_rating = label_ratings_list[i]
        if has_labels[i]:
            label_ratings_dict[label_review_th] = label_rating
            if is_querys[i]:
                imm_ps_dict[label_review_th] = p_imms[i]
                imm_ps_all_dict[label_review_th] = p_imm_alls[i]
            else:
                ahead_ps_dict[label_review_th] = p_curves[i]

    return AnkiDictStatistics(ahead_ps=ahead_ps_dict, imm_ps=imm_ps_dict, imm_ps_all=imm_ps_all_dict,
                              label_ratings=label_ratings_dict, label_elapsed_seconds=label_elapsed_seconds_dict, w=ws)


def greedy_splits(*args, **kwargs): pass  # Deprecated for standard batching


def naive_splits(*args, **kwargs): pass  # Deprecated for standard batching


if __name__ == "__main__":
    config = ModelConfig(d_model=256, dropout=0.1, num_hidden_layers=4, num_attention_heads=8)
    model = SrsRWKV(config)
    print(f"Total Params: {sum(p.numel() for p in model.parameters())}")