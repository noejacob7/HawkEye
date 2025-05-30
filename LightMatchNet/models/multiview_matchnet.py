import torch
import torch.nn as nn
from LightMatchNet.models.attention_pool import AttentionPool

class MultiViewMatchNet(nn.Module):
    def __init__(self, backbone="efficientnet", embedding_dim=128):
        super(MultiViewMatchNet, self).__init__()
        if backbone == "efficientnet":
            from LightMatchNet.models.efficientnet_matchnet import EfficientNetMatchNet
            self.encoder = EfficientNetMatchNet(embedding_dim=embedding_dim)
        elif backbone == "mobilenet":
            from LightMatchNet.models.mobilenetv3_small import MobileNetMatchNet
            self.encoder = MobileNetMatchNet(embedding_dim=embedding_dim)
        elif backbone == "swifttracknet":
            from LightMatchNet.models.swifttracknet import SwiftTrackNet
            self.encoder = SwiftTrackNet(embedding_dim=embedding_dim)
        elif backbone == "coarsefilter":
            from LightMatchNet.models.coarse_filter_model import CoarseFilterNet
            self.encoder = CoarseFilterNet(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.pool = AttentionPool(embed_dim=embedding_dim)

    def forward(self, list_of_images):
        """
        list_of_images: list of tensors (3, H, W) — multi-view set
        Returns: fused embedding tensor (1, embedding_dim)
        """
        device = next(self.encoder.parameters()).device
        embeddings = []

        for img in list_of_images:
            if img.ndim == 3:
                img = img.unsqueeze(0)  # (1, 3, H, W)
            img = img.to(device)
            emb = self.encoder(img)     # (1, embed_dim)
            embeddings.append(emb.squeeze(0))  # (embed_dim,)

        stacked = torch.stack(embeddings, dim=0)  # (N_views, embed_dim)
        fused = self.pool(stacked)                # (embed_dim,)
        return fused.unsqueeze(0)                 # (1, embed_dim)
