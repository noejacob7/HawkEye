import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Core Building Blocks
# -------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, ratio=2):
        super().__init__()
        init_channels = oup // ratio
        new_channels = oup - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class TinySEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

# -------------------------------
# SwiftTrackNet
# -------------------------------

class SwiftTrackNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Input 64x64x3
        self.conv1 = DepthwiseSeparableConv(3, 16, stride=2)  # 32x32x16
        self.ghost1 = GhostModule(16, 32)  # 32x32x32
        self.ghost2 = GhostModule(32, 64)  # 32x32x64

        self.downsample = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)  # 16x16x64
        self.bn_down = nn.BatchNorm2d(64)

        self.attention = TinySEBlock(64)  # 16x16x64

        self.ghost3 = GhostModule(64, 128)  # 16x16x128

        self.gap = nn.AdaptiveAvgPool2d(1)  # 1x1x128
        self.fc = nn.Linear(128, embedding_dim)  # 128 -> 128

    def forward(self, x):
        x = self.conv1(x)
        x = self.ghost1(x)
        x = self.ghost2(x)

        x = self.downsample(x)
        x = self.bn_down(x)
        x = F.relu(x)

        x = self.attention(x)
        x = self.ghost3(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

# -------------------------------
# Usage Example
# -------------------------------

if __name__ == "__main__":
    model = SwiftTrackNet()
    dummy_input = torch.randn(2, 3, 64, 64)  # Batch size 2
    output = model(dummy_input)
    print(output.shape)  # Expected output: (2, 128)
