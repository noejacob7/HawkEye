# generate_intraid_similarity_csv.py
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
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

# Paths (edit these to match your directory structure)
COARSE_INDEX_PATH = "LightMatchNet/data/mvp/coarse_index_list.txt"
COARSE_MASK_PATH = "LightMatchNet/data/mvp/coarse_annotation"
IMAGE_ROOT = "LightMatchNet/data/VeRi/image_test"
OUTPUT_CSV = "LightMatchNet/experiments/metrics/intra_id_similarity_v3.csv"

# Define simple view mapping from coarse mask part IDs
COARSE_VIEW_MAP = {
    2: "front",  # Front-windshield
    3: "front",  # Face
    4: "left",   # Left-window
    5: "left",   # Left-body
    6: "right",  # Right-window
    7: "right",  # Right-body
    8: "rear",   # Rear-windshield
    9: "rear"    # Rear
}

# Load model
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128)
model.load_state_dict(torch.load("LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Step 1: Parse coarse index list
view_index_map = defaultdict(list)
with open(COARSE_INDEX_PATH, "r") as f:
    for line in f:
        mask_id, dataset, img_name = line.strip().split()
        if dataset.lower() != "veri":
            continue

        mask_filename = f"{int(mask_id):05d}.png"  # Ensure 5-digit mask ID with leading zeros
        mask_path = os.path.join(COARSE_MASK_PATH, mask_filename)
        img_path = os.path.join(IMAGE_ROOT, img_name)
        if not os.path.exists(mask_path) or not os.path.exists(img_path):
            continue

        mask = Image.open(mask_path).convert("P")
        mask_ids = set(mask.getdata())

        # Infer coarse view
        view = None
        for pid in mask_ids:
            if pid in COARSE_VIEW_MAP:
                view = COARSE_VIEW_MAP[pid]
                break
        if view is None:
            continue

        vehicle_id = img_name.split('_')[0]
        view_index_map[vehicle_id].append((img_path, view))

# Step 2: Compute pairwise cosine similarities
records = []
with torch.no_grad():
    for vid, entries in tqdm(view_index_map.items(), desc="Computing intra-ID similarities"):
        embeddings = []
        views = []
        for img_path, view in entries:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            emb = model([tensor.squeeze(0)])
            embeddings.append(emb)
            views.append(view)

        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = F.cosine_similarity(embeddings[i], embeddings[j]).item()
                records.append([vid, views[i], views[j], f"{sim:.4f}"])

# Step 3: Save to CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vehicle_id", "view1", "view2", "similarity"])
    writer.writerows(records)

print(f"[✓] Saved intra-ID similarities to {OUTPUT_CSV}")
