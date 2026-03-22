import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer for EEG spatial routing.
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        """
        h: (Batch, N_nodes, in_features)
        adj: (Batch, N_nodes, N_nodes) or (N_nodes, N_nodes)
        
        Returns: (Batch, N_nodes, out_features)
        """
        Wh = self.W(h) # (B, N, F_out)
        
        # Prepare attention inputs
        B, N, _ = Wh.size()
        
        a_input = torch.cat([Wh.repeat(1, 1, N).view(B, N * N, -1), Wh.repeat(1, N, 1)], dim=-1)
        a_input = a_input.view(B, N, N, 2 * self.out_features) # (B, N, N, 2*F_out)
        
        e = self.leakyrelu(self.a(a_input).squeeze(-1)) # (B, N, N)
        
        # Masking attention where adjacency is 0
        zero_vec = -9e15 * torch.ones_like(e)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).repeat(B, 1, 1) # (B, N, N)
            
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        h_prime = torch.bmm(attention, Wh) # (B, N, F_out)
        return F.elu(h_prime)


class EEGSpatialGraphModule(nn.Module):
    """
    Learns dynamic functional connectivity and routes electrode features.
    Builds on top of standard 10-20 layout.
    """
    def __init__(self, n_electrodes=128, d_model=256, dropout=0.2):
        super(EEGSpatialGraphModule, self).__init__()
        
        self.n_electrodes = n_electrodes
        
        # Dynamic adjacency matrix (to be learned) starting from identity or uniform
        self.adj = nn.Parameter(torch.ones(n_electrodes, n_electrodes) / n_electrodes)
        
        self.gat1 = GraphAttentionLayer(in_features=d_model, out_features=d_model, dropout=dropout)
        self.gat2 = GraphAttentionLayer(in_features=d_model, out_features=d_model, dropout=dropout)

    def forward(self, x):
        """
        x: (Batch, seq_len, n_electrodes, d_model) -- features per electrode
        """
        B, T, N, D = x.size()
        assert N == self.n_electrodes, f"Expected {self.n_electrodes} electrodes, got {N}"
        
        x_flat = x.view(B * T, N, D)
        
        # Dynamic thresholding or sharpening of learned adjacency
        adj_learned = F.relu(self.adj)
        adj_learned = adj_learned + torch.eye(N, device=adj_learned.device) # self-loops
        
        h1 = self.gat1(x_flat, adj_learned)
        h2 = self.gat2(h1, adj_learned)
        
        # Residual connection
        out = h2.view(B, T, N, D) + x
        return out
