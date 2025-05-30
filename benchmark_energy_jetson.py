import os
import time
import subprocess
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from detection.yolo_wrapper import YOLOv11Wrapper
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
import imageio

# === CONFIG ===
VIDEO = "data/test/test_001.mp4"
CHECKPOINT = "LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"
TEGRALOG = "tegrastats.log"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Models ===
yolo = YOLOv11Wrapper(conf_threshold=0.3)
model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Tegrastats Logger ===
def start_tegrastats(log_file):
    return subprocess.Popen(
        f"tegrastats --interval 100 --logfile {log_file}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def stop_tegrastats(process):
    process.terminate()
    process.wait()

def parse_power(log_file):
    watts = []
    with open(log_file, "r") as f:
        for line in f:
            if "VDD_CPU_GPU_CV" in line:
                parts = line.split()
                for part in parts:
                    if "VDD_CPU_GPU_CV" in part:
                        mw = int(part.split("@")[0].replace("mW", ""))
                        watts.append(mw / 1000.0)
                        break
    return sum(watts) / len(watts), len(watts) * 0.1  # avg power, total time in s

# === Inference Modules ===
def run_detection(video_path):
    reader = imageio.get_reader(video_path)
    for frame in reader:
        yolo.predict(np.array(frame))
    reader.close()

def run_embedding(video_path):
    reader = imageio.get_reader(video_path)
    for frame in reader:
        img = transform(Image.fromarray(frame).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            model([img.squeeze(0)])
    reader.close()

def run_fusion(video_path):
    reader = imageio.get_reader(video_path)
    images = []
    for frame in reader:
        img = transform(Image.fromarray(frame).convert("RGB")).to(device)
        images.append(img)
        if len(images) >= 5:  # fuse every 5 frames
            with torch.no_grad():
                emb = model(images).cpu()
                model.pool(emb)
            images.clear()
    reader.close()

# === Profile Helper ===
def profile_module(name, function):
    print(f"[INFO] Profiling {name}...")
    if os.path.exists(TEGRALOG):
        os.remove(TEGRALOG)
    tproc = start_tegrastats(TEGRALOG)
    start = time.time()
    function(VIDEO)
    end = time.time()
    stop_tegrastats(tproc)
    avg_watt, duration = parse_power(TEGRALOG)
    energy = avg_watt * duration
    print(f"{name}: {avg_watt:.2f} W, {duration:.2f} s, {energy:.2f} J")
    return [name, f"{avg_watt:.2f}", f"{duration:.2f}", f"{energy:.2f}"]

# === Run Benchmarks ===
results = []
results.append(["Module", "Avg Power (W)", "Duration (s)", "Energy (J)"])
results.append(profile_module("YOLOv11-Tiny Detection", run_detection))
results.append(profile_module("LightMatchNet Embedding", run_embedding))
results.append(profile_module("LightMatchNet Fusion", run_fusion))

# === Save to CSV ===
import csv
with open("jetson_energy_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print("✅ Saved to jetson_energy_metrics.csv")
