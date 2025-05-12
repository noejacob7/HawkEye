# evaluate_single_reference_fusion.py
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
COARSE_INDEX_PATH = "data/mvp/coarse_index_list.txt"
COARSE_MASK_PATH = "data/mvp/coarse_annotation"
CHECKPOINT = "checkpoints/swifttracknet_multiview_v2_2.pt"
BACKBONE = "swifttracknet"
OUTPUT_CSV = "experiments/metrics/single_reference_query_fusion.csv"

# Define view map
VIEW_MAP = {
    2: "front", 3: "front",
    4: "left", 5: "left",
    6: "right", 7: "right",
    8: "rear", 9: "rear"
}

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

# Load MVP view mapping
view_lookup = defaultdict(list)
with open(COARSE_INDEX_PATH, "r") as f:
    for line in f:
        mid, ds, name = line.strip().split()
        if ds.lower() != "veri":
            continue
        mask_path = os.path.join(COARSE_MASK_PATH, f"{int(mid):05d}.png")
        if not os.path.exists(mask_path):
            continue
        mask = Image.open(mask_path).convert("P")
        parts = set(mask.getdata())
        for pid in parts:
            if pid in VIEW_MAP:
                view_lookup[(name.split('_')[0], VIEW_MAP[pid])].append(name)

results = []
random.seed(42)

with torch.no_grad():
    for vid, query_imgs in tqdm(id_to_query_imgs.items(), desc="Processing vehicles"):
        # Check we have one gallery image per required view
        required_views = ["front", "left", "right", "rear"]
        selected_gallery = []
        for view in required_views:
            key = (vid, view)
            candidates = view_lookup.get(key, [])
            if not candidates:
                break
            g_path = os.path.join(GALLERY_DIR, candidates[0])
            if os.path.exists(g_path):
                g_tensor = transform(Image.open(g_path).convert("RGB")).unsqueeze(0).to(device)
                emb = model([g_tensor.squeeze(0)]).squeeze(0)
                selected_gallery.append((candidates[0], emb))

        if len(selected_gallery) != 4:
            continue

        gallery_emb = torch.stack([e for _, e in selected_gallery]).mean(dim=0).unsqueeze(0)
        gallery_img_list = [f for f, _ in selected_gallery]

        # Evaluate query fusion 1 → N
        q_embs = []
        for q_img in query_imgs:
            q_path = os.path.join(QUERY_DIR, q_img)
            if not os.path.exists(q_path):
                continue
            q_tensor = transform(Image.open(q_path).convert("RGB")).unsqueeze(0).to(device)
            q_emb = model([q_tensor.squeeze(0)]).squeeze(0)
            q_embs.append((q_img, q_emb))

        if len(q_embs) < 2:
            continue

        fused = []
        for i, (q_img, emb) in enumerate(q_embs):
            fused.append(emb)
            fused_avg = torch.stack(fused).mean(dim=0).unsqueeze(0)
            sim = F.cosine_similarity(gallery_emb, fused_avg).item()
            results.append([i+1, vid, len(fused), q_img, "|".join(gallery_img_list), f"{sim:.4f}"])

# Save CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "vehicle_id", "query_views_fused", "query_img", "gallery_imgs", "similarity"])
    writer.writerows(results)

print(f"[✓] Saved query fusion evaluation to {OUTPUT_CSV}")
