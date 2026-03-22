"""
CIPHER — Conformer-based multi-task EEG speech decoder (v2).

Architecture:
  Multi-scale 1D Conv front-end (kernels 3/7/15) → SE channel attention
  → Conformer encoder (6 blocks with stochastic depth)
  → Attention pooling → Deeper multi-task FC heads
  → Optional CTC head for sequence decoding

Changes from v1:
  - Squeeze-and-Excitation (SE) after conv frontend
  - Stochastic depth (drop path) in Conformer blocks
  - 6 Conformer blocks (up from 4)
  - Deeper classification heads with LayerNorm
  - Relative positional encoding option
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sincnet import MultiresolutionSpectralFrontend
from models.graph_attention import EEGSpatialGraphModule
from models.mamba_encoder import BidirectionalMambaBlock, CrossAttentionGraphInjection
from models.sequence_decoder import AttentionSequenceDecoder


# ===================================================================
# Building blocks
# ===================================================================

class DropPath(nn.Module):
    """Stochastic depth (drop entire residual branch with probability p)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, channels)"""
        se = x.mean(dim=1)           # (B, C) — global average pool over time
        se = self.fc(se).unsqueeze(1) # (B, 1, C)
        return x * se


class AttentionPooling(nn.Module):
    """Learned-query attention pooling over the time dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, hidden_dim) → (batch, hidden_dim)"""
        scores = torch.matmul(x, self.query)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class MultiScaleConvFrontEnd(nn.Module):
    """
    EEGNet-style multi-scale 1D convolution front-end.
    Three parallel branches with kernel sizes 3, 7, 15 → concatenated → SE.
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: int = 64,
        dropout: float = 0.1,
        use_multiscale: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_se = use_se
        # Each branch: Conv1d → BatchNorm → GELU → Dropout
        self.branches = nn.ModuleList()
        kernel_list = (3, 7, 15) if use_multiscale else (7,)
        for ks in kernel_list:
            self.branches.append(nn.Sequential(
                nn.Conv1d(input_dim, out_channels, kernel_size=ks,
                          padding=ks // 2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        total_ch = out_channels * len(kernel_list)
        self.proj = nn.Linear(total_ch, total_ch)
        self.se = SqueezeExcitation(total_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, input_dim) → (batch, seq_len, out_channels*3)"""
        # Conv1d expects (batch, channels, seq_len)
        xt = x.transpose(1, 2)
        outs = [branch(xt) for branch in self.branches]
        cat = torch.cat(outs, dim=1)            # (batch, 3*out_ch, seq_len)
        cat = cat.transpose(1, 2)               # (batch, seq_len, 3*out_ch)
        out = self.proj(cat)
        return self.se(out)


class ConformerConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → Swish → pointwise."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Linear(d_model, d_model * 2)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)"""
        residual = x
        x = self.layer_norm(x)
        x = self.pw_conv1(x)
        x = F.glu(x, dim=-1)                       # (B, T, d_model)
        x = x.transpose(1, 2)                       # (B, d_model, T)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = x.transpose(1, 2)                       # (B, T, d_model)
        x = F.silu(x)
        x = self.pw_conv2(x)
        return self.dropout(x) + residual


class ConformerBlock(nn.Module):
    """
    Single Conformer block:
      FFN(½) → MHSA → ConvModule → FFN(½) → LayerNorm
    With stochastic depth (drop path) on each residual.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4,
        ff_expansion: int = 4, conv_kernel: int = 15,
        dropout: float = 0.1, drop_path: float = 0.0,
    ):
        super().__init__()
        # First half-step feed-forward
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        # Convolution module
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        # Second half-step feed-forward
        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)"""
        # Half-step FFN 1 with drop path
        x = x + self.drop_path(0.5 * self.ff1(self.ff1_norm(x)))
        # MHSA with drop path
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(self.attn_dropout(attn_out))
        # Conv module (has its own residual internally)
        x = self.conv(x)
        # Half-step FFN 2 with drop path
        x = x + self.drop_path(0.5 * self.ff2(self.ff2_norm(x)))
        return self.final_norm(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ===================================================================
# Main model: ConformerDecoder (multi-task + optional CTC)
# ===================================================================

MULTI_TASK_HEADS = {
    "phoneme_identity": 11,   # a, b, d, e, i, o, p, s, t, u, z
    "place": 2,               # alveolar, bilabial
    "manner": 2,              # fricative, stop
    "voicing": 2,             # unvoiced, voiced
}


class ConformerDecoder(nn.Module):
    """
    Multi-task EEG decoder v2 with:
      - Multi-scale conv front-end (k=3,7,15) + SE attention
      - Conformer encoder (6 blocks with stochastic depth)
      - Attention-pooled deeper classification heads
      - Optional CTC head for sequence-level decoding

    Parameters
    ----------
    input_dim : int
        Feature dimension per time-step.
    task_n_classes : dict[str, int]
        Mapping from task name to number of classes for each head.
    d_model : int
        Internal model dimension after front-end projection.
    n_conformer_blocks : int
        Number of stacked Conformer blocks.
    n_heads : int
        Attention heads per block.
    conv_channels : int
        Per-branch output channels in multi-scale front-end.
    conv_kernel : int
        Kernel size for Conformer convolution module.
    dropout : float
        Dropout probability.
    drop_path_rate : float
        Maximum stochastic depth rate (linearly increases per block).
    ctc_vocab_size : int or None
        If set, adds a CTC head with this vocabulary size.
    """

    def __init__(
        self,
        input_dim: int,
        task_n_classes: dict | None = None,
        d_model: int = 256,
        n_conformer_blocks: int = 6,
        n_heads: int = 8,
        conv_channels: int = 64,
        conv_kernel: int = 15,
        dropout: float = 0.4,
        drop_path_rate: float = 0.15,
        ctc_vocab_size: int | None = None,
        use_multiscale: bool = True,
        use_se: bool = True,
        use_attention_pool: bool = True,
    ):
        super().__init__()
        self.task_n_classes = task_n_classes or dict(MULTI_TASK_HEADS)
        self.d_model = d_model

        # --- Input normalization ---
        self.input_norm = nn.LayerNorm(input_dim)

        # --- Multi-scale conv front-end with SE ---
        self.conv_frontend = MultiScaleConvFrontEnd(
            input_dim,
            out_channels=conv_channels,
            dropout=dropout,
            use_multiscale=use_multiscale,
            use_se=use_se,
        )
        frontend_dim = conv_channels * (3 if use_multiscale else 1)

        # --- Projection to d_model ---
        self.proj = nn.Sequential(
            nn.Linear(frontend_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # --- Conformer encoder with linearly increasing stochastic depth ---
        dp_rates = [drop_path_rate * i / max(n_conformer_blocks - 1, 1)
                     for i in range(n_conformer_blocks)]
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model, n_heads=n_heads,
                conv_kernel=conv_kernel, dropout=dropout,
                drop_path=dp_rates[i],
            )
            for i in range(n_conformer_blocks)
        ])

        # --- Attention pooling (shared) ---
        self.use_attention_pool = use_attention_pool
        self.attention = AttentionPooling(d_model)

        # --- Multi-task classification heads (deeper, with LayerNorm) ---
        self.task_heads = nn.ModuleDict()
        for task_name, nc in self.task_n_classes.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout + 0.1),  # heavier dropout in heads
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, nc),
            )

        # --- Optional CTC head ---
        self.ctc_head = None
        if ctc_vocab_size is not None:
            self.ctc_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, ctc_vocab_size),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shared encoder: input → multi-scale conv (+SE) → Conformer → sequence output.
        x : (batch, seq_len, input_dim) → (batch, seq_len, d_model)
        """
        x = self.input_norm(x)
        x = self.conv_frontend(x)             # (B, T, 3*conv_ch)
        x = self.proj(x)                      # (B, T, d_model)
        x = self.pos_enc(x)
        for block in self.conformer_blocks:
            x = block(x)                      # (B, T, d_model)
        return x

    def forward(
        self, x: torch.Tensor, tasks: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass returning logits for requested tasks.

        Parameters
        ----------
        x : (batch, seq_len, input_dim)
        tasks : which heads to compute. None = all available heads.

        Returns
        -------
        dict mapping task_name → (batch, n_classes) logits.
        If CTC head exists, also includes "ctc" → (batch, seq_len, vocab).
        """
        encoded = self.encode(x)                    # (B, T, d_model)
        if self.use_attention_pool:
            pooled = self.attention(encoded)         # (B, d_model)
        else:
            pooled = encoded.mean(dim=1)             # (B, d_model)

        tasks = tasks or list(self.task_heads.keys())
        out = {}
        for task_name in tasks:
            if task_name in self.task_heads:
                out[task_name] = self.task_heads[task_name](pooled)

        if self.ctc_head is not None:
            out["ctc"] = self.ctc_head(encoded)      # (B, T, vocab)

        return out


# ===================================================================
# Legacy GRU model (backward compatibility for loading old checkpoints)
# ===================================================================

class GRUDecoder(nn.Module):
    """Bidirectional GRU decoder — kept for loading old checkpoints."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_size * 2
        self.attention = AttentionPooling(gru_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        gru_out, _ = self.gru(x)
        pooled = self.attention(gru_out)
        return self.classifier(pooled)


# ===================================================================
# Ensemble wrapper (ERP + DDA logit averaging)
# ===================================================================

class EnsembleDecoder(nn.Module):
    """
    Averages logits from two ConformerDecoder models (one ERP, one DDA).
    Both models must share the same task_n_classes.
    """

    def __init__(self, model_erp: ConformerDecoder, model_dda: ConformerDecoder):
        super().__init__()
        self.model_erp = model_erp
        self.model_dda = model_dda

    def forward(
        self,
        x_erp: torch.Tensor,
        x_dda: torch.Tensor,
        tasks: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        out_erp = self.model_erp(x_erp, tasks=tasks)
        out_dda = self.model_dda(x_dda, tasks=tasks)
        out = {}
        for key in out_erp:
            out[key] = (out_erp[key] + out_dda[key]) / 2.0
        return out

# ===================================================================
# CIPHER v3 Neuro-Mamba Decoder
# ===================================================================

class NeuroMambaDecoder(nn.Module):
    """
    State-of-the-art EEG decoder integrating:
      - Multiresolution Spectral Frontend (SincNet)
      - Spatial Graph Module (GAT for 10-20 system)
      - Continuous-Time Mamba Encoder with Subject-Adaptive Norm
      - Autoregressive Attention Sequence Decoder
    """
    def __init__(
        self,
        input_dim: int,
        task_n_classes: dict | None = None,
        d_model: int = 256,
        n_mamba_blocks: int = 4,
        n_electrodes: int = 1, # Typically ~64, but using input_dim/coeffs as a proxy if DDA
        dropout: float = 0.3,
        ctc_vocab_size: int | None = None,
    ):
        super().__init__()
        self.task_n_classes = task_n_classes or dict(MULTI_TASK_HEADS)
        self.d_model = d_model
        
        # 1. Spectral Frontend
        self.frontend = MultiresolutionSpectralFrontend(in_channels=input_dim, d_model=d_model)
        
        # 2. Spatial Graph (simplified since input might not be strict (N, Features))
        self.spatial_graph = EEGSpatialGraphModule(n_electrodes=n_electrodes, d_model=d_model, dropout=dropout)
        
        # 3. Mamba Encoder & Graph Injection
        self.mamba_blocks = nn.ModuleList([
            BidirectionalMambaBlock(d_model) for _ in range(n_mamba_blocks)
        ])
        
        # Since we use sequence format, we can inject graph at the end or interleave
        self.graph_injection = CrossAttentionGraphInjection(d_model)
        
        # 4. Pooling & Classification Heads
        self.attention = AttentionPooling(d_model)
        self.task_heads = nn.ModuleDict()
        for task_name, nc in self.task_n_classes.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout + 0.1),
                nn.Linear(256, nc),
            )
            
        # 5. Sequence Decoder (CTC + Attention Seq2Seq)
        self.ctc_head = None
        self.seq_decoder = None
        if ctc_vocab_size is not None:
            self.ctc_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, ctc_vocab_size),
            )
            self.seq_decoder = AttentionSequenceDecoder(d_model=d_model, vocab_size=ctc_vocab_size)

    def encode(self, x: torch.Tensor, subject_emb=None) -> torch.Tensor:
        """
        x : (B, T, input_dim)
        """
        B, T, D = x.size()
        
        # 1. Spectral features
        x = self.frontend(x) # (B, T, d_model)
        
        # 2. Spatial graph (reshape needed if N_electrodes is configured, here N=1 proxy)
        # For full 10-20, x would be (B, T, N, D), assuming mapped correctly. 
        # Using a single pseudo-node for fallback.
        x_graph = x.view(B, T, 1, self.d_model)
        x_graph = self.spatial_graph(x_graph)
        x_graph = x_graph.view(B, T, self.d_model)
        
        # 3. Mamba blocks
        for block in self.mamba_blocks:
            x = block(x, subject_emb)
            
        # Cross-attention injection
        x = self.graph_injection(x, x_graph)
        return x

    def forward(
        self, x: torch.Tensor, tasks: list[str] | None = None, subject_emb=None
    ) -> dict[str, torch.Tensor]:
        
        encoded = self.encode(x, subject_emb)
        pooled = self.attention(encoded)

        tasks = tasks or list(self.task_heads.keys())
        out = {}
        for task_name in tasks:
            if task_name in self.task_heads:
                out[task_name] = self.task_heads[task_name](pooled)

        if self.ctc_head is not None:
            out["ctc"] = self.ctc_head(encoded)

        return out

