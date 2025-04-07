import torch
import torch.nn as nn
import torch.nn.functional as F

class CoarseFilterNet(nn.Module):
    def __init__(self):
        super(CoarseFilterNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Output: 16x224x224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x112x112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32x112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x56x56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 64x1x1
        )

        self.fc = nn.Linear(64 * 2, 1)  # Two image features (query + candidate)

    def forward(self, query_img, candidate_img):
        q_feat = self.encoder(query_img).view(query_img.size(0), -1)
        c_feat = self.encoder(candidate_img).view(candidate_img.size(0), -1)
        combined = torch.cat((q_feat, c_feat), dim=1)
        out = self.fc(combined)
        return torch.sigmoid(out)  # Returns probability (0 to 1)
