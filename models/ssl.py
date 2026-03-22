import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedPredictiveLoss(nn.Module):
    """
    Masked Predictive Coding (like wav2vec 2.0 or MAE).
    Masks parts of the sequence and predicts the raw/encoded features.
    """
    def __init__(self, d_target):
        super().__init__()
        self.predictor = nn.Linear(d_target, d_target)
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x_encoded, target_features, mask_indices):
        """
        x_encoded: (B, L, D) - Output from encoder predicting masked frames
        target_features: (B, L, D) - Ground truth features
        mask_indices: (B, L) - Boolean mask where 1 indicates masked tokens
        """
        predictions = self.predictor(x_encoded)
        loss = self.loss_fn(predictions, target_features)
        
        # Only compute loss on masked tokens
        mask_indices = mask_indices.unsqueeze(-1).expand_as(loss)
        loss = (loss * mask_indices).sum() / (mask_indices.sum() + 1e-8)
        return loss

class ContrastiveLoss(nn.Module):
    """
    InfoNCE Contrastive Loss for SSL.
    Selects positive pairs and negative pairs.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, anchors, positives, negatives):
        """
        anchors: (B, L, D) - Current representations
        positives: (B, L, D) - Positive samples (e.g. augmented views)
        negatives: (K, B, L, D) - Negative samples
        """
        pos_sim = self.cosine(anchors, positives) / self.temperature # (B, L)
        
        neg_sims = []
        for i in range(negatives.size(0)):
            neg_sims.append(self.cosine(anchors, negatives[i]) / self.temperature)
        
        neg_sims = torch.stack(neg_sims, dim=-1) # (B, L, K)
        
        # Denominator = exp(pos_sim) + sum(exp(neg_sim))
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sims], dim=-1) # (B, L, 1+K)
        
        # Labels are 0 (the positive entry is at index 0)
        labels = torch.zeros(logits.size(0) * logits.size(1), dtype=torch.long, device=logits.device)
        logits = logits.view(-1, logits.size(-1))
        
        loss = F.cross_entropy(logits, labels)
        return loss
