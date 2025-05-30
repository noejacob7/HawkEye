import os
import time
import torch
import csv
import numpy as np
from PIL import Image
from torchvision import transforms
from detection.yolo_wrapper import YOLOv11Wrapper
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet

# === CONFIGURATION ===
SAMPLE_IMAGE = "data/query/test_001.jpeg"  # <-- Replace with your test image
CHECKPOINT = "LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"
OUTPUT_CSV = "jetson_module_benchmark.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Models ===
yolo = YOLOv11Wrapper(conf_threshold=0.3)
model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# === Image Preparation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open(SAMPLE_IMAGE).convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)

# === Benchmarking Function ===
def benchmark(fn, trials=30):
    times = []
    for _ in range(trials):
        start = time.time()
        fn()
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    return np.mean(times)

# === Inference Latency ===
lat_yolo = benchmark(lambda: yolo.predict(np.array(image)))
lat_embed = benchmark(lambda: model([tensor.squeeze(0)]))
lat_fusion = benchmark(lambda: model.pool(torch.stack([
    model([tensor.squeeze(0)])[0],
    model([tensor.squeeze(0)])[0]
])))

# === Bandwidth Profiling ===
uav_counts = [1, 2, 4, 8, 16]
embedding_size = 128 * 4  # bytes
bandwidth = [(n, round(n * embedding_size / 1024, 2)) for n in uav_counts]  # in KB

# === Save Results ===
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Module", "Latency (ms)"])
    writer.writerow(["YOLOv11-Tiny", f"{lat_yolo:.2f}"])
    writer.writerow(["LightMatchNet Embedding", f"{lat_embed:.2f}"])
    writer.writerow(["Fusion (Attention)", f"{lat_fusion:.2f}"])
    writer.writerow([])
    writer.writerow(["UAV Count", "Bandwidth (KB)"])
    writer.writerows(bandwidth)

print(f"✅ Results written to {OUTPUT_CSV}")
