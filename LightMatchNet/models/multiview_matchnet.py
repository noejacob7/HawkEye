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
        list_of_images: list of tensors, each with shape (3, H, W)
        Returns: fused embedding tensor with shape (1, embedding_dim)
        """
        device = next(self.parameters()).device  # Get correct device even in DataParallel
        embeddings = []

        for img in list_of_images:
            if img.ndim == 3:
                img = img.unsqueeze(0)  # Make it (1, 3, H, W)
            img = img.to(device)
            emb = self.encoder(img)  # shape: (1, embed_dim)
            embeddings.append(emb.squeeze(0))  # shape: (embed_dim,)

        stacked = torch.stack(embeddings, dim=0)  # shape: (N_views, embed_dim)
        fused = self.pool(stacked)                # shape: (embed_dim,)
        return fused.unsqueeze(0)                 # shape: (1, embed_dim)

