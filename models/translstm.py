import torch
from torch import nn, Tensor
import torch.nn.functional as F

from config import Config
from models.base import BaseModel


class Feed_Forward_block(nn.Module):
    """
    Feed-forward block as utilized in the custom Transformer encoder.
    """

    def __init__(self, dim_ff: int):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in: Tensor) -> Tensor:
        return self.layer2(F.relu(self.layer1(ffn_in)))


class LastQueryTransformerRNN(BaseModel):
    """
    Adapted from the Kaggle 'Riiid! Answer Correctness Prediction' winning model.
    Uses only the last input as the query in the Transformer Encoder, resulting in O(L)
    complexity, followed by an LSTM and DNN for binary retention prediction.
    """

    # Hyperparameters tuned for an aggressive but stable parameter count (~200k-300k)
    lr: float = 1e-3
    wd: float = 1e-5
    n_epoch: int = 16

    def __init__(
            self, config: Config, state_dict=None, input_mean=None, input_std=None
    ):
        super().__init__(config)

        # Ensure initial parameters are float tensors
        if input_mean is None:
            input_mean = torch.tensor(0.0)
        elif isinstance(input_mean, torch.Tensor):
            input_mean = input_mean.clone().detach().float()
        else:
            input_mean = torch.tensor(input_mean, dtype=torch.float32)

        if input_std is None:
            input_std = torch.tensor(1.0)
        elif isinstance(input_std, torch.Tensor):
            input_std = input_std.clone().detach().float()
        else:
            input_std = torch.tensor(input_std, dtype=torch.float32)

        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)

        self.use_duration_feature = config.lstm_use_duration
        num_main_inputs = 1 + (1 if self.use_duration_feature else 0)
        self.n_input = num_main_inputs + 4  # Rating is expanded to 4 dims via one-hot

        # Dimensions as referenced in the paper (d=128)
        self.d_model = 128
        self.n_heads = 4

        # 1. Input Projection & Embedding
        self.input_proj = nn.Linear(self.n_input, self.d_model)
        self.pos_embed = nn.Embedding(10000, self.d_model)  # Handles sequence lengths up to 10k

        # 2. Transformer Encoder Layer (Custom Last-Query implementation)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.multi_en = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.n_heads, dropout=0.1
        )

        # 3. LSTM over the attention output
        self.use_lstm = True
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.d_model, hidden_size=self.d_model, num_layers=1
            )
            nn.init.orthogonal_(self.lstm.weight_ih_l0)
            nn.init.orthogonal_(self.lstm.weight_hh_l0)
            self.lstm.bias_ih_l0.data.fill_(0)
            self.lstm.bias_hh_l0.data.fill_(0)

        # 4. Feed Forward Network
        self.layer_norm2 = nn.LayerNorm(self.d_model)
        self.ffn_en = Feed_Forward_block(self.d_model)

        # 5. Output DNN
        self.fc = nn.Linear(in_features=self.d_model, out_features=1)

        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            try:
                self.load_state_dict(
                    torch.load(
                        f"./pretrain/{self.config.get_evaluation_file_name()}_pretrain.pth",
                        weights_only=True,
                        map_location=self.config.device,
                    )
                )
            except FileNotFoundError:
                pass

    def set_normalization_params(self, mean_i, std_i):
        # Explicitly move new parameters to the exact device our buffers are actively residing on
        if not isinstance(mean_i, torch.Tensor):
            mean_i = torch.tensor(mean_i, dtype=torch.float32)
        else:
            mean_i = mean_i.to(dtype=torch.float32)

        if not isinstance(std_i, torch.Tensor):
            std_i = torch.tensor(std_i, dtype=torch.float32)
        else:
            std_i = std_i.to(dtype=torch.float32)

        self.register_buffer("input_mean", mean_i.to(self.input_mean.device))
        self.register_buffer("input_std", std_i.to(self.input_std.device))

    def forward(self, x_lni: Tensor, seq_lens: Tensor) -> Tensor:
        # Prevent indexing & masking dimension issues if dataloader ships `seq_lens` as CPU tensors
        seq_lens = seq_lens.to(x_lni.device)

        # Preprocess features identical to the LSTM base
        x_rating = x_lni[..., -1:]
        x_features = x_lni[..., :-1]

        x_delay = torch.log(1e-5 + x_features[..., :1])
        if self.use_duration_feature:
            x_duration = torch.log(
                torch.clamp(x_features[..., 1:2], min=100, max=60000)
            )
            x_main = torch.cat([x_delay, x_duration], dim=-1)
        else:
            x_main = x_delay

        x_main = (x_main - self.input_mean) / self.input_std

        x_rating = torch.maximum(x_rating, torch.ones_like(x_rating))
        x_rating = torch.nn.functional.one_hot(
            x_rating.squeeze(-1).long() - 1, num_classes=4
        ).float()
        x = torch.cat([x_main, x_rating], dim=-1)

        # Project features to d_model
        x = self.input_proj(x)  # Shape: [seq_len, batch_size, d_model]
        seq_len, batch_size, _ = x.shape

        # Positional Encoding
        pos = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(seq_len, batch_size)
        x = x + self.pos_embed(pos)

        x = self.layer_norm1(x)

        # --- LAST QUERY EXTRACTION ---
        # Fetch only the final step (I_L) for each batch as the Query
        batch_idx = torch.arange(batch_size, device=x.device)
        Q = x[seq_lens - 1, batch_idx].unsqueeze(0)  # Shape: [1, batch_size, d_model]

        # --- MULTI-HEAD ATTENTION ---
        # Mask out padding elements in K and V based on sequence length
        mask_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        key_padding_mask = mask_idx >= seq_lens.unsqueeze(1)  # Shape: [batch_size, seq_len]

        # Q is sequence length 1, K & V are the full sequence
        out, _ = self.multi_en(Q, x, x, key_padding_mask=key_padding_mask)

        # Skip connection
        out = out + Q  # Shape: [1, batch_size, d_model]

        # --- LSTM ---
        if self.use_lstm:
            out, _ = self.lstm(out)

        # --- FFN BLOCK ---
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        # --- OUTPUT ---
        # Squeeze out the sequence length dimension since we only computed for L
        out = self.fc(out.squeeze(0))  # Shape: [batch_size, 1]

        return torch.sigmoid(out).squeeze(-1)  # Shape: [batch_size]

    def batch_process(
            self,
            sequences: Tensor,
            delta_ts: Tensor,
            seq_lens: Tensor,
            real_batch_size: int,
    ) -> dict[str, Tensor]:
        """
        By modifying forward to directly compute and return only the predicted retentions,
        we omit the need for arbitrary tensor slicing usually found in the GRU batch_process.
        """
        outputs = self.forward(sequences, seq_lens)
        return {
            "retentions": outputs[:real_batch_size]
        }

    def get_optimizer(self, lr: float, wd: float = 1e-4) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)