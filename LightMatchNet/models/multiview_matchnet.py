import torch
import torch.nn as nn
from models.attention_pool import AttentionPool

class MultiViewMatchNet(nn.Module):
    def __init__(self, backbone="efficientnet", embedding_dim=128):
        super(MultiViewMatchNet, self).__init__()
        if backbone == "efficientnet":
            from models.efficientnet_matchnet import EfficientNetMatchNet
            self.encoder = EfficientNetMatchNet(embedding_dim=embedding_dim)
        elif backbone == "mobilenet":
            from models.mobilenetv3_small import MobileNetMatchNet
            self.encoder = MobileNetMatchNet(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.pool = AttentionPool(embed_dim=embedding_dim)

    def forward(self, list_of_images):
        """
        list_of_images: list of tensors (B, 3, H, W)
        Each item is a view of the same object
        Returns: fused embedding vector (1, embedding_dim)
        """
        embeddings = [self.encoder(img.unsqueeze(0)) for img in list_of_images]
        embeddings = torch.cat(embeddings, dim=0)  # (N_views, embed_dim)
        fused_embedding = self.pool(embeddings)  # (embed_dim,)
        return fused_embedding.unsqueeze(0)
