import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.projection(x)
