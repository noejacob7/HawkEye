# lightmatchnet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

class LightMatchNet(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(LightMatchNet, self).__init__()

        base = mobilenet_v3_small(pretrained=pretrained)
        self.backbone = base.features  # Output: (B, 576, H/32, W/32)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(576, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
