# generate_ablation_metrics.py
import os
import sys
import csv
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from thop import profile

# Add root path to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.multiview_matchnet import MultiViewMatchNet
from utils.label_parser import parse_veri_labels

# Config
LABEL_PATH = "data/VeRi/test_label.xml"
QUERY_DIR = "data/VeRi/image_query"
GALLERY_DIR = "data/VeRi/image_test"
CHECKPOINTS = {
    "fusion": "checkpoints/swifttracknet_multiview_v2_2.pt",
    "avg_pool": "checkpoints/swifttracknet_avgpool.pt",
    "single": "checkpoints/swifttracknet_single.pt"
}
BACKBONES = {
    "fusion": "swifttracknet",
    "avg_pool": "swifttracknet",
    "single": "swifttracknet"
}
OUTPUT_CSV = "experiments/metrics/ablation_metrics.csv"
TOP_K = [1, 5, 10]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load label info
label_map = parse_veri_labels(LABEL_PATH)
id_to_gallery = defaultdict(list)
for img, (vid, _, _) in label_map.items():
    id_to_gallery[vid].append(img)

# Run evaluation for each variant
results = []
for variant, ckpt in CHECKPOINTS.items():
    print(f"[INFO] Evaluating variant: {variant}")
    backbone = BACKBONES[variant]
    model = MultiViewMatchNet(backbone=backbone, embedding_dim=128)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    # Profile size and FLOPs
    dummy_input = [torch.randn(1, 3, 224, 224).to(device)]
    flops, params = profile(model.encoder, inputs=dummy_input, verbose=False)
    model_size_mb = os.path.getsize(ckpt) / (1024 * 1024)

    correct_at_k = {k: 0 for k in TOP_K}
    total = 0
    total_time = 0.0

    with torch.no_grad():
        for q_img, (q_vid, _, _) in tqdm(label_map.items(), desc=f"{variant} queries"):
            q_path = os.path.join(QUERY_DIR, q_img)
            if not os.path.exists(q_path):
                continue
            q_tensor = transform(Image.open(q_path).convert("RGB")).unsqueeze(0).to(device)
            start = time.time()
            q_emb = model([q_tensor.squeeze(0)]) if variant != "fusion" else model([q_tensor.squeeze(0)])
            total_time += time.time() - start

            sims = []
            for g_vid, g_imgs in id_to_gallery.items():
                for g_img in g_imgs:
                    g_path = os.path.join(GALLERY_DIR, g_img)
                    if not os.path.exists(g_path):
                        continue
                    g_tensor = transform(Image.open(g_path).convert("RGB")).unsqueeze(0).to(device)
                    g_emb = model([g_tensor.squeeze(0)])
                    sim = F.cosine_similarity(q_emb, g_emb).item()
                    sims.append((g_vid, sim))

            sims = sorted(sims, key=lambda x: x[1], reverse=True)
            total += 1
            for k in TOP_K:
                top_k_vids = [vid for vid, _ in sims[:k]]
                if q_vid in top_k_vids:
                    correct_at_k[k] += 1

    avg_time = total_time / total
    row = [variant]
    for k in TOP_K:
        acc = 100 * correct_at_k[k] / total
        row.append(f"{acc:.2f}")
    row += [f"{avg_time:.4f}", f"{model_size_mb:.2f}", f"{params:,}", f"{flops:,}"]
    results.append(row)

# Write results
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variant", "top1", "top5", "top10", "inference_time", "model_size_mb", "params", "flops"])
    writer.writerows(results)

print(f"[✓] Saved ablation metrics to {OUTPUT_CSV}")
