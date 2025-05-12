# generate_view_fusion_sweep.py
import os
import sys
import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import random

# Add root path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.multiview_matchnet import MultiViewMatchNet
from utils.label_parser import parse_veri_labels

# Config
QUERY_LABEL_PATH = "data/VeRi/test_label.xml"
GALLERY_LABEL_PATH = "data/VeRi/test_label.xml"
QUERY_DIR = "data/VeRi/image_query"
GALLERY_DIR = "data/VeRi/image_test"
CHECKPOINT = "checkpoints/swifttracknet_multiview_v2_2.pt"
BACKBONE = "swifttracknet"
OUTPUT_CSV = "experiments/metrics/view_fusion_sweep.csv"

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
model = MultiViewMatchNet(backbone=BACKBONE, embedding_dim=128)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device)
model.eval()

# Load labels
gallery_labels = parse_veri_labels(GALLERY_LABEL_PATH)
query_labels = parse_veri_labels(QUERY_LABEL_PATH)

id_to_gallery_imgs = defaultdict(list)
for img, (vid, _, _) in gallery_labels.items():
    id_to_gallery_imgs[vid].append(img)

id_to_query_imgs = defaultdict(list)
for img, (vid, _, _) in query_labels.items():
    id_to_query_imgs[vid].append(img)

results = []
random.seed(42)

with torch.no_grad():
    for vid in tqdm(id_to_query_imgs.keys(), desc="Processing vehicles"):
        if vid not in id_to_gallery_imgs:
            continue

        gallery_imgs = id_to_gallery_imgs[vid]
        query_imgs = id_to_query_imgs[vid]

        if len(gallery_imgs) < 2:
            continue

        # Compute query embeddings
        query_embs = []
        for q_img in query_imgs:
            q_path = os.path.join(QUERY_DIR, q_img)
            if not os.path.exists(q_path):
                continue
            q_tensor = transform(Image.open(q_path).convert("RGB")).unsqueeze(0).to(device)
            q_emb = model([q_tensor.squeeze(0)]).squeeze(0)
            query_embs.append(q_emb)

        if not query_embs:
            continue

        avg_query_emb = torch.stack(query_embs).mean(dim=0).unsqueeze(0)

        # Compute all gallery embeddings
        gallery_embs = []
        for g_img in gallery_imgs:
            g_path = os.path.join(GALLERY_DIR, g_img)
            if not os.path.exists(g_path):
                continue
            g_tensor = transform(Image.open(g_path).convert("RGB")).unsqueeze(0).to(device)
            g_emb = model([g_tensor.squeeze(0)]).squeeze(0)
            gallery_embs.append((g_img, g_emb))

        if len(gallery_embs) < 2:
            continue

        # Sort by similarity to query
        sims = [(fname, F.cosine_similarity(avg_query_emb, emb.unsqueeze(0)).item(), emb)
                for fname, emb in gallery_embs]
        sims.sort(key=lambda x: x[1])  # sort by lowest to highest similarity

        fused = []
        sweep = []
        for i, (fname, score, emb) in enumerate(sims):
            fused.append(emb)
            fused_avg = torch.stack(fused).mean(dim=0).unsqueeze(0)
            sim_to_query = F.cosine_similarity(avg_query_emb, fused_avg).item()
            sweep.append([i + 1, vid, len(fused), f"{sim_to_query:.4f}"])

        results.extend(sweep)

# Save CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "vehicle_id", "views_fused", "similarity"])
    writer.writerows(results)

print(f"[✓] Saved ranked fusion sweep to {OUTPUT_CSV}")
