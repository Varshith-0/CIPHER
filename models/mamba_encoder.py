import torch
import torch.nn as nn
import torch.nn.functional as F

class SubjectAdaptiveNorm(nn.Module):
    """
    Subject-adaptive normalization (FiLM-like).
    Scales and shifts normalized features based on subject embedding if available.
    For simplicity, falls back to standard LayerNorm if subject info isn't provided.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, subject_emb=None):
        out = self.norm(x)
        if subject_emb is not None:
            # Assume subject_emb provides scale and shift: (Batch, 2 * d_model)
            gamma, beta = subject_emb.chunk(2, dim=-1)
            out = out * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return out


class MinimalMambaBlock(nn.Module):
    """
    Minimal Selective State Space (Mamba) Proxy Model.
    If `mamba_ssm` is available, this could wrap it. For now, it provides
    a parameterized depthwise Conv1d + GLU gating which is an effective 
    local-global proxy for SSM behavior.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner, bias=True,
            kernel_size=d_conv, groups=d_inner, padding=d_conv - 1,
        )
        
        # SSM Parameter Proxies (simplification: linear projection as state mix)
        self.x_proj = nn.Linear(d_inner, d_state + d_state + d_inner, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1) # (B, L, d_inner)
        
        # Conv
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :L]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)
        
        # Gating
        y = x_in * F.silu(z)
        out = self.out_proj(y)
        return out


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional wrapper for Mamba blocks.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = SubjectAdaptiveNorm(d_model)
        self.mamba_fwd = MinimalMambaBlock(d_model)
        self.mamba_bwd = MinimalMambaBlock(d_model)
        
    def forward(self, x, subject_emb=None):
        residual = x
        x_norm = self.norm(x, subject_emb)
        
        out_fwd = self.mamba_fwd(x_norm)
        
        x_rev = torch.flip(x_norm, dims=[1])
        out_bwd = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # Merge directions
        out = out_fwd + out_bwd
        return out + residual


class CrossAttentionGraphInjection(nn.Module):
    """
    Cross-attention to inject graph spatial features into the sequential Mamba stream.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.norm_seq = nn.LayerNorm(d_model)
        self.norm_graph = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, seq_x, graph_x):
        """
        seq_x: (Batch, Seq_len, d_model) -- Sequence stream
        graph_x: (Batch, N_nodes, d_model) -- Graph spatial embeddings
        """
        seq_norm = self.norm_seq(seq_x)
        graph_norm = self.norm_graph(graph_x)
        
        # Query from Sequence, Key/Value from Graph
        attn_out, _ = self.cross_attn(query=seq_norm, key=graph_norm, value=graph_norm)
        return seq_x + attn_out
