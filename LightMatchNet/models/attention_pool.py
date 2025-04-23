import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionPool, self).__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, view_embeddings):
        """
        view_embeddings: (N_views, embed_dim)
        returns: fused_embedding: (embed_dim,)
        """
        attn_scores = self.attn_fc(view_embeddings)  # (N_views, 1)
        attn_weights = F.softmax(attn_scores, dim=0)  # (N_views, 1)
        weighted = attn_weights * view_embeddings  # (N_views, embed_dim)
        fused = torch.sum(weighted, dim=0)  # (embed_dim,)
        return F.normalize(fused, p=2, dim=0)  # Normalize for cosine sim
