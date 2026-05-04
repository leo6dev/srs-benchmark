import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Any

import numpy as np
import torch

# =======================================================================================
# CORE ARCHITECTURE ADAPTATION POINT: JIT Compilation
# =======================================================================================
# Old code used torch.jit for speed because RWKV operations were JIT-friendly.
# Newer architectures like Mamba (which uses custom CUDA scans) or DeepSeek
# (which uses FlashAttention) are typically NOT compatible with TorchScript out of the box.
# Therefore, we default back to standard PyTorch nn.Module instead of ScriptModule.
ModuleType = torch.nn.Module


# A no-op decorator to replace `torch.jit.script_method` safely without breaking syntax.
def FunctionType(fn):
    return fn


# =======================================================================================
# CONFIGURATION SHELL
# =======================================================================================
# We replace AnkiRWKVConfig with a generic config. You should extend this with your
# specific Mamba/DeepSeek parameters (e.g., d_state, d_conv for Mamba, or num_heads for DeepSeek).
class ModelConfig(NamedTuple):
    d_model: int
    dropout: float
    # TODO: Add your new architecture's specific hyperparams here!
    # mamba_d_state: int = 16
    # num_attention_heads: int = 8


# We keep the exact same statistics structure to ensure compatibility with your existing
# training loops and logging systems. Aliased to the old name to prevent import errors.
class SrsIterStatistics(NamedTuple):
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


# =======================================================================================
# CORE ARCHITECTURE ADAPTATION POINT: Dataloader / Batching
# =======================================================================================
@dataclass
class PreparedBatch:
    """
    MASSIVE CHANGE HERE:
    The old RWKV architecture processed sequences via complex lists of gathered chunks
    (`sub_gather`, `time_shift_selects`, `skips`) because it acts as an RNN and wanted
    to avoid padding by grouping similar lengths.

    Standard architectures (Transformers/DeepSeek/Mamba) DO NOT do this. They expect
    standard dense 3D tensors representing padded sequences: [Batch, Time, Features].

    You WILL need to update your DataLoader's `collate_fn` to simply pad the features
    and labels to the maximum sequence length in the batch.
    """
    num_data: int  # Equivalent to Batch Size (B)

    # -> YOUR NEW INPUT FORMAT:
    # shape: [Batch, Time, 92] (where 92 is card_features_dim)
    features: torch.Tensor
    # shape: [Batch, Time] (Booleans or 1/0s, where 1 means real token, 0 means padding)
    attention_mask: torch.Tensor

    # -> KEPT THE SAME (just padded to max Time in batch)
    labels: torch.Tensor  # shape: [Batch, Time, LabelDim]
    label_review_th: torch.Tensor

    def to(self, device):
        return PreparedBatch(
            num_data=self.num_data,
            features=self.features.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            label_review_th=self.label_review_th.to(device),
        )


DTYPE_EXCLUDE = [
    "w_linear",
    "s_linear",
    "d_linear",
    "d_softplus",
    "k_linear",
    "p_linear",
    "ahead_linear",
]


def is_excluded(name):
    """
    Utility for mixed precision (BF16/FP16).
    We keep this intact: Certain final linear heads computing sensitive exponential
    or logit math for the Spaced Repetition System (SRS) MUST remain in FP32 to avoid NaN/Inf errors.
    """
    for query in DTYPE_EXCLUDE:
        if query in name:
            return True
    return False


class SrsSequenceModelShell(ModuleType):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.card_features_dim = 92
        self.d_model = config.d_model

        # SRS specific head dimensions (Do not change unless tuning SRS capacity)
        self.features_fc_dim = 4 * self.d_model
        self.ahead_head_dim = 4 * self.d_model
        self.p_head_dim = 4 * self.d_model
        self.w_head_dim = 4 * self.d_model
        self.num_curves = 128

        # Standard initialization wrapped in no_grad to save memory during init.
        # These WILL still track gradients and train normally.
        with torch.no_grad():
            # 1. Input Projection: Maps raw 92-dim card features into your model's hidden dimension
            self.features2card = torch.nn.Sequential(
                torch.nn.Linear(self.card_features_dim, self.features_fc_dim),
                torch.nn.SiLU(),
                torch.nn.LayerNorm(self.features_fc_dim),
                torch.nn.Linear(self.features_fc_dim, self.d_model),
                torch.nn.SiLU(),
            )

            # =========================================================================
            # TODO: INJECT YOUR NEW ARCHITECTURE HERE
            # =========================================================================
            # OLD CODE:
            # self.rwkv_modules = torch.nn.ModuleList([...RWKV modules...])
            #
            # NEW CODE: Drop in your Mamba or DeepSeek model.
            # E.g., self.backbone = Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2)
            # E.g., self.backbone = DeepSeekTransformer(config)

            self.backbone = None  # <-- INITIALIZE YOUR MODEL HERE

            # =========================================================================

            # 3. Output mapping heads (Unchanged - specific to spaced repetition algorithms)
            self.prehead_norm = torch.nn.LayerNorm(self.d_model)
            self.prehead_dropout = torch.nn.Dropout(p=config.dropout)

            self.head_ahead_logits = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.ahead_head_dim),
                torch.nn.ReLU(),
            )
            self.head_w = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, 1 * self.d_model),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(1 * self.d_model),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(1 * self.d_model, self.w_head_dim),
            )
            self.head_p = torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.p_head_dim),
                torch.nn.ReLU(),
            )

            # SRS Constants
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
        """
        Takes the output of your backbone [Batch, Time, d_model] and branches it out
        into the specific components needed to compute the Spaced Repetition Loss.
        PyTorch natively broadcasts Linear layers over the `Time` dimension, so no changes are needed here.
        """
        x = self.prehead_dropout(self.prehead_norm(input))

        out_w_logits = self.w_linear(self.head_w(x).float())
        out_w = torch.nn.functional.softmax(out_w_logits, dim=-1)
        out_w_log_p = torch.nn.functional.log_softmax(out_w_logits, dim=-1)
        out_ahead_logits = self.ahead_linear(self.head_ahead_logits(x).float())

        x_p = self.head_p(x).float()
        return out_ahead_logits, out_w, out_w_log_p, self.p_linear(x_p)

    @FunctionType
    def forgetting_curve(self, w, label_elapsed_seconds):
        """
        SRS MATH LOGIC:
        Generates probability of recalling a card based on an ensemble of exponential
        decay curves (forgetting curves). Independent of sequence architecture.
        """
        s_space_raw = torch.exp(
            torch.linspace(0, self.s_point_spread, self.num_curves, device=w.device)
        )
        s_space = 0.1 + (s_space_raw - 1) * (np.e ** (self.s_max - self.s_point_spread))
        label_elapsed_seconds = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        return 1e-5 + (1 - 2 * 1e-5) * torch.sum(
            w * 0.9 ** (label_elapsed_seconds / s_space), dim=-1
        )

    @FunctionType
    def interp(self, out_ahead_logits, label_elapsed_seconds):
        """
        SRS MATH LOGIC:
        Calculates a non-linear residual offset for the forgetting curve by interpolating
        between control points on the timeline. Independent of sequence architecture.
        """
        label_elapsed_seconds = torch.clamp(label_elapsed_seconds.contiguous(), min=1)
        point_space_raw = torch.exp(
            torch.linspace(
                0, self.point_spread, self.num_points, device=out_ahead_logits.device
            )
        )
        point_space = 0.5 + (point_space_raw - 1) * (
                np.e ** (self.max_e - self.point_spread)
        )
        right_idx = torch.searchsorted(point_space, label_elapsed_seconds)
        left_idx = torch.clamp(right_idx - 1, min=0)
        xl, xr = point_space[left_idx], point_space[right_idx]
        yl = torch.gather(out_ahead_logits, dim=-1, index=left_idx)
        yr = torch.gather(out_ahead_logits, dim=-1, index=right_idx)
        res = 1e-5 + (1 - 2 * 1e-5) * (
                yl + (yr - yl) * (label_elapsed_seconds - xl) / (xr - xl)
        )
        return res.squeeze(-1)

    @FunctionType
    def forward_batch(
            self,
            features: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        ========================================================================
        CORE ARCHITECTURE ADAPTATION POINT: The Forward Pass
        ========================================================================

        We have completely gutted the old `batch_sub_gather` logic.
        Old RWKV iteratively chunked memory arrays.

        New standard execution flow:
        1. Input `features` shape: [B, T, 92]
        2. Map features to D_model -> shape: [B, T, d_model]
        3. Pass sequentially through Backbone -> shape: [B, T, d_model]
        4. Pass to Heads -> Out
        """
        # 1. Feature embedding
        x = self.features2card(features)  # Shape: [B, T, d_model]

        # 2. Sequence modeling backbone
        if self.backbone is None:
            raise NotImplementedError("You must assign `self.backbone` in __init__!")

        # Example for DeepSeek/Transformers (requires attention mask for padded tokens):
        # x = self.backbone(x, attention_mask=attention_mask)

        # Example for Mamba (usually processes padding normally, mask applied at loss):
        # x = self.backbone(x)

        # TODO: Execute your backbone here!
        x = self.backbone(x)

        # 3. Output heads prediction
        # Shape of x entering here is [B, T, d_model]
        return self.head_and_out(x)

    @FunctionType
    def nanmin(self, tensor):
        output = tensor.nan_to_num(1e9).min()
        return output

    @FunctionType
    def nanmax(self, tensor):
        output = tensor.nan_to_num(-1e9).max()
        return output

    @FunctionType
    def _get_loss(
            self,
            features: torch.Tensor,
            attention_mask: torch.Tensor,
            batch_labels: torch.Tensor,
            batch_label_review_th: torch.Tensor,
    ):
        """
        Computes the custom spaced-repetition loss.

        PADDING WARNING:
        Since we moved to standard [B, T] padding logic, you might worry about padding
        tokens contributing to the loss. You do NOT need to change this function!

        In the global labels, `has_label` acts as a mask.
        Ensure your DataLoader sets `has_label = 0` for all padded tokens in `batch_labels`.
        If `has_label == 0`, the token is zeroed out by `ahead_mask` and `immediate_mask`
        and safely ignored in the `_avg` division calculations below.
        """
        # 1. Perform Forward Pass
        out_ahead_logits, out_w, out_w_log_p, out_p_logits = self.forward_batch(
            features, attention_mask
        )

        if torch.isnan(out_ahead_logits).any():
            return None

        # 2. Unpack Labels
        global_labels = batch_labels.float()
        (
            label_elapsed_seconds,
            _,
            label_y,
            label_rating,
            has_label,
            label_is_equalize,
            is_query,
        ) = global_labels.unbind(-1)

        has_label = has_label.int()
        label_is_equalize = label_is_equalize.int()
        is_query = is_query.int()

        # 3. Calculate Predictions
        label_rating = torch.clamp(label_rating - 1, min=0)
        label_elapsed_seconds = label_elapsed_seconds.unsqueeze(-1)
        curve_probs_raw = self.forgetting_curve(out_w, label_elapsed_seconds)
        curve_logits_raw = torch.log(
            curve_probs_raw / (1 - curve_probs_raw)
        )  # inverse sigmoid
        ahead_logit_residual = self.interp(out_ahead_logits, label_elapsed_seconds)
        curve_logits = curve_logits_raw + ahead_logit_residual
        curve_probs = torch.sigmoid(curve_logits)

        out_p_probs = torch.softmax(out_p_logits, dim=-1)
        out_p_again, out_p_1, out_p_2, out_p_3 = out_p_probs.unbind(dim=-1)
        out_p_binary = torch.clamp(1.0 - out_p_again, min=1e-5, max=1.0 - 1e-5)

        if torch.isnan(curve_probs).any():
            raise Exception("nan")

        # 4. Calculate Raw Losses
        w_loss = torch.nn.functional.kl_div(
            input=out_w_log_p,
            target=torch.ones_like(out_w) / self.num_curves,
            reduction="none",
        ).mean(dim=-1)

        # MASKING: Padded tokens automatically have `has_label=0` here!
        ahead_mask = (1 - is_query) * has_label
        immediate_mask = is_query * has_label
        assert ahead_mask.shape == label_is_equalize.shape
        ahead_equalize_mask = ahead_mask * label_is_equalize

        immediate_equalize_mask = immediate_mask * label_is_equalize
        curve_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve_logits, label_y, reduction="none"
        )
        curve_raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve_logits_raw, label_y, reduction="none"
        )

        NUM_LABELS = 4
        B, T = label_rating.shape
        p_loss = torch.nn.functional.cross_entropy(
            out_p_logits.view(-1, NUM_LABELS),
            label_rating.long().view(-1),
            reduction="none",
        ).view(B, T)

        p_binary_loss = torch.nn.functional.binary_cross_entropy(
            out_p_binary, label_y, reduction="none"
        )

        # 5. Aggregate averages applying the dynamic length masks
        ahead_avg = (curve_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_SCALE = 0.5
        ahead_raw_avg = (curve_raw_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_RAW_SCALE = 0.5
        immediate_avg = (p_loss * immediate_mask).sum() / (1e-8 + immediate_mask.sum())
        w_avg = (w_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        W_LOSS_SCALE = 1e-5
        ahead_logits_mag_loss = torch.sqrt(
            1e-16 + out_ahead_logits.square().mean(dim=-1)
        )
        ahead_logits_mag_avg = (ahead_logits_mag_loss * ahead_mask).sum() / (
                1e-8 + ahead_mask.sum()
        )
        AHEAD_LOGITS_MAG_LOSS_SCALE = 1e-4
        ahead_logits_diff_loss = torch.sqrt(
            1e-16 + out_ahead_logits.diff().square().mean(dim=-1)
        )
        ahead_logits_diff_avg = (ahead_logits_diff_loss * ahead_mask).sum() / (
                1e-8 + ahead_mask.sum()
        )
        AHEAD_LOGITS_DIFF_LOSS_SCALE = 1e-3

        # 6. Final Combined Loss
        loss_avg = (
                AHEAD_SCALE * ahead_avg
                + immediate_avg
                + AHEAD_RAW_SCALE * ahead_raw_avg
                + W_LOSS_SCALE * w_avg
                + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_avg
                + AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_avg
        )
        loss_tensor = (
                AHEAD_SCALE * curve_loss.detach()
                + p_loss.detach()
                + AHEAD_RAW_SCALE * curve_raw_loss.detach()
                + W_LOSS_SCALE * w_loss.detach()
                + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_loss.detach()
                + AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_loss.detach()
        )

        ahead_equalize_avg = (curve_loss * ahead_equalize_mask).sum() / (
                1e-8 + ahead_equalize_mask.sum()
        )
        ahead_raw_equalize_avg = (curve_raw_loss * ahead_equalize_mask).sum() / (
                1e-8 + ahead_equalize_mask.sum()
        )
        immediate_binary_equalize_avg = (
                                                p_binary_loss * immediate_equalize_mask
                                        ).sum() / (1e-8 + immediate_equalize_mask.sum())

        return SrsIterStatistics(
            average_loss=loss_avg,
            p_curve=curve_probs.detach(),
            p_imm=out_p_binary.detach(),
            p_imm_all=out_p_probs.detach(),
            loss_tensor=loss_tensor.detach(),
            ahead_avg=ahead_avg.detach(),
            ahead_raw_avg=ahead_raw_avg.detach(),
            ahead_n=ahead_mask.sum().detach(),
            ahead_equalize_avg=ahead_equalize_avg.detach(),
            ahead_raw_equalize_avg=ahead_raw_equalize_avg.detach(),
            ahead_equalize_n=ahead_equalize_mask.sum().detach(),
            imm_avg=immediate_avg.detach(),
            imm_n=immediate_mask.sum().detach(),
            imm_binary_equalize_avg=immediate_binary_equalize_avg.detach(),
            imm_binary_equalize_n=immediate_equalize_mask.sum().detach(),
            w_loss_avg=w_avg.detach(),
            ahead_logits_mag_loss_avg=ahead_logits_mag_avg.detach(),
            ahead_logits_diff_loss_avg=ahead_logits_diff_avg.detach(),
            w=out_w.detach(),
            label_review_th=batch_label_review_th.detach(),
            label_elapsed_seconds=label_elapsed_seconds.detach(),
            label_rating=label_rating.detach(),
            is_query=is_query.detach(),
            has_label=has_label.detach(),
        )

    def get_loss(self, batch: PreparedBatch):
        # Maps the simplified PrepardBatch layout into the math block.
        return self._get_loss(
            features=batch.features,
            attention_mask=batch.attention_mask,
            batch_labels=batch.labels,
            batch_label_review_th=batch.label_review_th,
        )

    def copy_downcast_(self, master_model, dtype):
        master_params = dict(master_model.named_parameters())
        with torch.no_grad():
            for name, param in self.named_parameters():
                target_dtype = torch.float32 if is_excluded(name) else dtype
                assert param.dtype == target_dtype
                param.data.copy_(master_params[name].to(target_dtype))
                assert param.dtype == target_dtype

    def selective_cast(self, dtype):
        """Allows backbones to be in half-precision while keeping sensitive math FP32."""
        for name, module in self.named_modules():
            if len(name) == 0:
                continue
            if not is_excluded(name):
                if dtype == torch.bfloat16:
                    module = module.to(dtype)
                elif dtype == torch.half:
                    raise ValueError("FP16 not tested. Use BF16.")
                elif dtype == torch.float32:
                    pass
        return self


# Alias for compatibility with external references
SrsRWKV = SrsSequenceModelShell


# =======================================================================================
# METRICS EXTRACTOR
# =======================================================================================
@dataclass
class AnkiDictStatistics:
    ahead_ps: dict[int, float]
    imm_ps: dict[int, float]
    imm_ps_all: dict
    label_ratings: dict[int, float]
    label_elapsed_seconds: dict[int, float]
    w: dict


AnkiRWKVDictStatistics = AnkiDictStatistics


def extract_p(stats: SrsIterStatistics):
    """
    Creates a nicer summary format.
    NOTE: This explicitly asserts batch size == 1 down below (stats.label_review_th.size(0) == 1).
    Keep this in mind if your new architecture tests run with batched inference!
    """
    assert stats.label_review_th.size(0) == 1
    ahead_ps_dict = {}
    imm_ps_dict = {}
    label_ratings_dict = {}
    label_elapsed_seconds_dict = {}
    imm_ps_all_dict = {}

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
        has_label = has_labels[i]
        is_query = is_querys[i]
        imm_p = p_imms[i]
        imm_p_all = p_imm_alls[i]
        ahead_p = p_curves[i]

        if has_label:
            label_ratings_dict[label_review_th] = label_rating
            if is_query:
                imm_ps_dict[label_review_th] = imm_p
                imm_ps_all_dict[label_review_th] = imm_p_all
            else:
                ahead_ps_dict[label_review_th] = ahead_p

    return AnkiDictStatistics(
        ahead_ps=ahead_ps_dict,
        imm_ps=imm_ps_dict,
        imm_ps_all=imm_ps_all_dict,
        label_ratings=label_ratings_dict,
        label_elapsed_seconds=label_elapsed_seconds_dict,
        w=ws,
    )


# =======================================================================================
# DEPRECATED RWKV BATCHING LOGIC
# =======================================================================================
def greedy_splits(*args, **kwargs):
    """
    [DEPRECATED]
    This was built to efficiently chunk and un-chunk memory blocks exclusively for RWKV.
    Mamba, LLaMA, and DeepSeek do NOT need this.
    Use PyTorch's native `torch.nn.utils.rnn.pad_sequence` in your DataLoader.
    Leaving this as a shell in case it's blindly imported somewhere.
    """
    raise DeprecationWarning("greedy_splits is not needed for dense standard batching.")


def naive_splits(*args, **kwargs):
    """[DEPRECATED] See greedy_splits."""
    raise DeprecationWarning("naive_splits is not needed for dense standard batching.")


if __name__ == "__main__":
    # Test shell execution
    dummy_config = ModelConfig(d_model=256, dropout=0.1)
    model = SrsSequenceModelShell(dummy_config)
    t_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", t_param)
    a_param = sum(p.numel() for p in model.parameters())
    print("Number of parameters", a_param)
    print("\n[!] IMPORTANT: Do not forget to assign `self.backbone` in `__init__` before training!")