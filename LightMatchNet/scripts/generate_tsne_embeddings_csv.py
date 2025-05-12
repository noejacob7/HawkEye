# generate_tsne_embeddings.py
import os
import sys
import csv
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

# Add root path to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.multiview_matchnet import MultiViewMatchNet
from utils.label_parser import parse_veri_labels

# Config
LABEL_PATH = "data/VeRi/test_label.xml"
COARSE_INDEX_PATH = "data/mvp/coarse_index_list.txt"
COARSE_MASK_PATH = "data/mvp/coarse_annotation"
IMAGE_ROOT = "data/VeRi/image_test"
CHECKPOINT = "checkpoints/swifttracknet_multiview_v2_2.pt"
BACKBONE = "swifttracknet"
OUTPUT_CSV = "experiments/metrics/tsne_embeddings.csv"
USE_PCA = False  # Change to True for PCA
SAMPLE_LIMIT = 500

# View map from mask
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

# Load model
model = MultiViewMatchNet(backbone=BACKBONE, embedding_dim=128)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device)
model.eval()

# Parse label file
label_map = parse_veri_labels(LABEL_PATH)

# Infer view from coarse mask
view_lookup = {}
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
        view = next((VIEW_MAP[pid] for pid in parts if pid in VIEW_MAP), None)
        if view:
            view_lookup[name] = view

# Step 1: Collect embeddings
features, image_names, ids, views = [], [], [], []
with torch.no_grad():
    for img, (vid, _, _) in tqdm(label_map.items(), desc="Extracting embeddings"):
        if img not in view_lookup or len(features) >= SAMPLE_LIMIT:
            continue
        path = os.path.join(IMAGE_ROOT, img)
        if not os.path.exists(path):
            continue
        tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        emb = model([tensor.squeeze(0)]).cpu().squeeze().numpy()
        features.append(emb)
        image_names.append(img)
        ids.append(vid)
        views.append(view_lookup[img])

# Step 2: Run t-SNE or PCA
X = np.stack(features)
print("[INFO] Running dimensionality reduction...")
if USE_PCA:
    reduced = PCA(n_components=2).fit_transform(X)
else:
    reduced = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto').fit_transform(X)

# Step 3: Save CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "vehicle_id", "view", "x", "y"])
    for name, vid, view, (x, y) in zip(image_names, ids, views, reduced):
        writer.writerow([name, vid, view, f"{x:.4f}", f"{y:.4f}"])

print(f"[✓] Saved {len(image_names)} embeddings to {OUTPUT_CSV}")
