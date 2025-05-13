# generate_ablation_metrics.py (Corrected Evaluation: No query-gallery overlap, single vs fusion analysis)
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
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.multiview_matchnet import MultiViewMatchNet
from utils.label_parser import parse_veri_labels

LABEL_PATH = "data/VeRi/test_label.xml"
QUERY_DIR = "data/VeRi/image_query"
GALLERY_DIR = "data/VeRi/image_query"  # compare only within query set
CHECKPOINTS = {
    "fusion": "checkpoints/swifttracknet_multiview_v2_2.pt",
    "avg_pool": "checkpoints/swifttracknet_multiview_v2_2.pt",
    "single": "checkpoints/swifttracknet_multiview_v2_2.pt"
}
BACKBONES = {
    "fusion": "swifttracknet",
    "avg_pool": "swifttracknet",
    "single": "swifttracknet"
}
OUTPUT_CSV = "experiments/metrics/ablation_metrics.csv"
TOP_K = [1, 5, 10]

# Setup device
device = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

label_map = parse_veri_labels(LABEL_PATH)
vid_to_images = defaultdict(list)
for img, (vid, _, _) in label_map.items():
    vid_to_images[vid].append(img)

results = []
for variant, ckpt in CHECKPOINTS.items():
    print(f"[INFO] Evaluating variant: {variant}")
    backbone = BACKBONES[variant]
    model = MultiViewMatchNet(backbone=backbone, embedding_dim=128)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    dummy_input = [torch.randn(1, 3, 224, 224).to(device)]
    flops, params = profile(model.encoder, inputs=dummy_input, verbose=False)
    model_size_mb = os.path.getsize(ckpt) / (1024 * 1024)

    correct_at_k = {k: 0 for k in TOP_K}
    total_queries = 0
    total_time = 0.0

    with torch.no_grad():
        for vid, img_list in tqdm(vid_to_images.items(), desc=variant):
            if len(img_list) < 2:
                continue

            # --- Query Embedding ---
            if variant == "single":
                query_img = random.choice(img_list)
                q_path = os.path.join(QUERY_DIR, query_img)
                q_tensor = transform(Image.open(q_path).convert("RGB")).unsqueeze(0).to(device)
                start = time.time()
                q_emb = model([q_tensor.squeeze(0)]).cpu()
                total_time += time.time() - start

            else:
                query_tensors = []
                for img in img_list:
                    q_path = os.path.join(QUERY_DIR, img)
                    if os.path.exists(q_path):
                        tensor = transform(Image.open(q_path).convert("RGB")).to(device)
                        query_tensors.append(tensor)
                if len(query_tensors) < 2:
                    continue
                start = time.time()
                q_emb = model(query_tensors).cpu()
                total_time += time.time() - start

            # --- Build Gallery (Exclude current vehicle's images) ---
            gallery_embeddings = []
            gallery_labels = []
            for g_vid, g_imgs in vid_to_images.items():
                if g_vid == vid:
                    continue
                for g_img in g_imgs:
                    g_path = os.path.join(GALLERY_DIR, g_img)
                    if os.path.exists(g_path):
                        tensor = transform(Image.open(g_path).convert("RGB")).unsqueeze(0).to(device)
                        emb = model([tensor.squeeze(0)]).cpu()
                        gallery_embeddings.append(emb)
                        gallery_labels.append(g_vid)

            if len(gallery_embeddings) == 0:
                continue

            gallery_tensor = torch.cat(gallery_embeddings, dim=0)
            sims = F.cosine_similarity(q_emb, gallery_tensor).tolist()
            ranked = sorted(zip(gallery_labels, sims), key=lambda x: x[1], reverse=True)

            total_queries += 1
            for k in TOP_K:
                top_k_ids = [r[0] for r in ranked[:k]]
                if vid in top_k_ids:
                    correct_at_k[k] += 1

    avg_time = total_time / total_queries
    row = [variant]
    for k in TOP_K:
        acc = 100 * correct_at_k[k] / total_queries
        row.append(f"{acc:.2f}")
    row += [f"{avg_time:.4f}", f"{model_size_mb:.2f}", f"{params:,}", f"{flops:,}"]
    results.append(row)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variant", "top1", "top5", "top10", "inference_time", "model_size_mb", "params", "flops"])
    writer.writerows(results)

print(f"[✓] Saved corrected ablation metrics to {OUTPUT_CSV}")
