import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

class EfficientNetMatchNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EfficientNetMatchNet, self).__init__()

        # Load EfficientNet B0 backbone
        backbone = efficientnet_b0(weights='IMAGENET1K_V1')  # PyTorch 1.13+
        self.features = backbone.features  # Extract features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(1280, embedding_dim)  # EfficientNet-B0 outputs 1280 features

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize for cosine sim
