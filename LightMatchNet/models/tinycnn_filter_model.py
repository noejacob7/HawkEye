import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNNCoarseFilter(nn.Module):
    def __init__(self):
        super(TinyCNNCoarseFilter, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # (8, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (8, 112, 112)

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # (16, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 56, 56)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (32, 56, 56)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (32, 1, 1)
        )

        self.fc = nn.Linear(32 * 2, 1)  # Query + Candidate features

    def forward(self, query_img, candidate_img):
        q_feat = self.encoder(query_img).view(query_img.size(0), -1)  # (B, 32)
        c_feat = self.encoder(candidate_img).view(candidate_img.size(0), -1)
        combined = torch.cat((q_feat, c_feat), dim=1)  # (B, 64)
        out = self.fc(combined)
        return torch.sigmoid(out)  # Score (0–1)
