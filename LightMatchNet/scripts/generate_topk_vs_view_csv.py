# generate_topk_vs_view.py
import os
import sys
import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# Add root path to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.multiview_matchnet import MultiViewMatchNet
from utils.label_parser import parse_veri_labels

# Config
CHECKPOINT = "checkpoints/swifttracknet_multiview_v2_2.pt"
BACKBONE = "swifttracknet"
LABEL_PATH = "data/VeRi/test_label.xml"
QUERY_DIR = "data/VeRi/image_query"
GALLERY_DIR = "data/VeRi/image_test"
OUTPUT_CSV = "experiments/metrics/topk_accuracy_by_view.csv"
COARSE_INDEX_PATH = "data/mvp/coarse_index_list.txt"
COARSE_MASK_PATH = "data/mvp/coarse_annotation"
TOP_K = [1, 5, 10]

# Define coarse view inference
COARSE_VIEW_MAP = {
    2: "front", 3: "front",
    4: "left", 5: "left",
    6: "right", 7: "right",
    8: "rear", 9: "rear"
}

# Load model
model = MultiViewMatchNet(backbone=BACKBONE, embedding_dim=128)
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Step 1: Parse coarse index list to infer query views
query_views = {}
with open(COARSE_INDEX_PATH, "r") as f:
    for line in f:
        mask_id, dataset, img_name = line.strip().split()
        if dataset.lower() != "veri":
            continue

        mask_path = os.path.join(COARSE_MASK_PATH, f"{int(mask_id):05d}.png")
        if not os.path.exists(mask_path):
            continue

        mask = Image.open(mask_path).convert("P")
        mask_ids = set(mask.getdata())
        view = None
        for pid in mask_ids:
            if pid in COARSE_VIEW_MAP:
                view = COARSE_VIEW_MAP[pid]
                break
        if view:
            query_views[img_name] = view

# Step 2: Load label data
label_map = parse_veri_labels(LABEL_PATH)
id_to_gallery = defaultdict(list)
for img, (vid, _, _) in label_map.items():
    id_to_gallery[vid].append(img)

# Step 3: Compute top-k accuracy per view
view_stats = defaultdict(lambda: {k: 0 for k in TOP_K + ['total']})

with torch.no_grad():
    for q_img, view in tqdm(query_views.items(), desc="Evaluating Top-K accuracy by view"):
        if q_img not in label_map:
            continue

        q_vid, _, _ = label_map[q_img]
        q_path = os.path.join(QUERY_DIR, q_img)
        if not os.path.exists(q_path):
            continue

        q_tensor = transform(Image.open(q_path).convert("RGB")).unsqueeze(0)
        q_emb = model([q_tensor.squeeze(0)])

        sims = []
        for g_vid, g_imgs in id_to_gallery.items():
            for g_img in g_imgs:
                g_path = os.path.join(GALLERY_DIR, g_img)
                if not os.path.exists(g_path):
                    continue
                g_tensor = transform(Image.open(g_path).convert("RGB")).unsqueeze(0)
                g_emb = model([g_tensor.squeeze(0)])
                sim = F.cosine_similarity(q_emb, g_emb).item()
                sims.append((g_vid, sim))

        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        view_stats[view]['total'] += 1
        for k in TOP_K:
            top_k_vids = [vid for vid, _ in sims[:k]]
            if q_vid in top_k_vids:
                view_stats[view][k] += 1

# Step 4: Save CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["view", "top1", "top5", "top10", "total", "top1_acc", "top5_acc", "top10_acc"])
    for view, stats in view_stats.items():
        total = stats['total']
        row = [
            view,
            stats[1],
            stats[5],
            stats[10],
            total,
            f"{100 * stats[1] / total:.2f}",
            f"{100 * stats[5] / total:.2f}",
            f"{100 * stats[10] / total:.2f}"
        ]
        writer.writerow(row)

print(f"[✓] Saved Top-K accuracy by view to {OUTPUT_CSV}")
