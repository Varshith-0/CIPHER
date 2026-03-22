import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSequenceDecoder(nn.Module):
    """
    Autoregressive attention-based decoder for sequence-to-sequence decoding
    with hybrid CTC / Attention support.
    """
    def __init__(self, d_model=256, vocab_size=50, num_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None):
        """
        tgt: (B, tgt_seq_len)
        memory: (B, src_seq_len, d_model) from encoder
        """
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_emb = self.pos_encoder(tgt_emb)
        
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        
        logits = self.fc_out(out)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1) # (1, seq_len, d_model) + ...
        return x

def hybrid_ctc_attention_decode(
    encoder_out, ctc_logits, attention_decoder, bos_token, eos_token, 
    beam_size=5, max_len=10, ctc_weight=0.3, lm_scorer=None
):
    """
    A simplified joint CTC/Attention Beam Search implementation.
    In practice, you would integrate Torchaudio CTC beam search + LM 
    with step-wise attention decoder rescoring.
    
    For Phase 6, we provide a placeholder wrapper for sequence generation.
    """
    device = encoder_out.device
    B = encoder_out.size(0)
    
    # Very naive beam search logic - purely attention-based for demo simplicity
    # Integrating full CTC joint search requires extensive DP. 
    # Here, we show the attention decoder execution path.
    all_hypotheses = []
    
    for b in range(B):
        mem = encoder_out[b:b+1] # (1, L, D)
        
        # Start token
        tgt = torch.tensor([[bos_token]], device=device)
        
        for _ in range(max_len):
            logits = attention_decoder(tgt, mem) # (1, t, V)
            next_word = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            tgt = torch.cat([tgt, next_word], dim=1)
            
            if next_word.item() == eos_token:
                break
                
        all_hypotheses.append(tgt[0].cpu().numpy().tolist())
        
    return all_hypotheses
