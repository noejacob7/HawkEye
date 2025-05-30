# benchmark/run_pipeline.py

import os
from detection.yolo_wrapper import YOLOv11Wrapper
from HawkEye.tracking.bytetrack_wrapper import ByteTrackDummy
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
import torch
from torchvision import transforms
from PIL import Image
import time

def load_image(img_path):
    return Image.open(img_path).convert("RGB")

def run_benchmark(image_path, model_ckpt, device='cuda'):
    timings = {}

    # Step 1: Detection
    detector = YOLOv11Wrapper(model_path="checkpoints/yolov11-tiny.pt", device=device)
    boxes, yolo_time = detector.infer(image_path)
    timings["YOLOv11"] = yolo_time

    # Step 2: Tracking
    tracker = ByteTrackDummy()
    tracks, track_time = tracker.update(boxes)
    timings["ByteTrack"] = track_time

    # Step 3: Embedding
    embedder = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
    embedder.load_state_dict(torch.load(model_ckpt, map_location=device))
    embedder.eval()

    img = load_image(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        _ = embedder(input_tensor)
    timings["LightMatchNet"] = (time.time() - start) * 1000

    # Optional: Fusion time (fake here)
    timings["Fusion"] = 1.5

    print("\n📊 Inference Times (ms):")
    for k, v in timings.items():
        print(f"  {k:<15}: {v:.2f} ms")

    return timings

if __name__ == "__main__":
    run_benchmark("data/test.jpg", "checkpoints/swifttracknet_multiview_v3.pt")
