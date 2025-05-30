import torch
import time
from torchvision import models, transforms
from PIL import Image
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Dummy image tensor for inference
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# Load your model (replace this with your own)
# Example: MobileNetV3 Small pretrained
from models.multiview_matchnet import MultiViewMatchNet
model = MultiViewMatchNet(backbone="swifttracknet").eval().cuda()
model.load_state_dict(torch.load("LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"))


# Warm-up (important for accurate timings)
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# Measure GPU time using CUDA Events
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = []

# Run multiple iterations and average
for _ in range(100):
    starter.record()
    with torch.no_grad():
        _ = model(dummy_input)
    ender.record()
    torch.cuda.synchronize()  # Wait for GPU ops to finish
    curr_time = starter.elapsed_time(ender)  # ms
    timings.append(curr_time)

# Results
avg_time = sum(timings) / len(timings)
fps = 1000 / avg_time
print(f"[✓] Avg Inference Time: {avg_time:.3f} ms")
print(f"[✓] Estimated FPS: {fps:.2f}")
