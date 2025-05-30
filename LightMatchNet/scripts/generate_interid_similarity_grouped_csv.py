# generate_interid_similarity_grouped_csv.py
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

# Add root path to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
from LightMatchNet.utils.label_parser import parse_veri_labels

# Config
LABEL_PATH = "LightMatchNet/data/VeRi/test_label.xml"  # or train_label.xml
IMAGE_ROOT = "LightMatchNet/data/VeRi/image_test"
OUTPUT_CSV = "LightMatchNet/experiments/metrics/inter_id_similarity_grouped.csv"
CHECKPOINT = "LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"
BACKBONE = "swifttracknet"
PAIR_LIMIT = 300

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

# Step 1: Parse label file
label_map = parse_veri_labels(LABEL_PATH)
id_to_meta = defaultdict(list)
for img, (vid, color, vtype) in label_map.items():
    id_to_meta[vid].append((img, color, vtype))

# Step 2: Prepare pairs by group
records = []
groups = ["different", "same_color_diff_type", "same_type_diff_color", "same_color_type"]
pairs_by_group = defaultdict(list)

ids = list(id_to_meta.keys())
random.shuffle(ids)

for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        if len(records) >= PAIR_LIMIT * len(groups):
            break

        id1, id2 = ids[i], ids[j]
        meta1 = random.choice(id_to_meta[id1])
        meta2 = random.choice(id_to_meta[id2])

        img1, color1, type1 = meta1
        img2, color2, type2 = meta2

        if color1 != color2 and type1 != type2:
            group = "different"
        elif color1 == color2 and type1 != type2:
            group = "same_color_diff_type"
        elif color1 != color2 and type1 == type2:
            group = "same_type_diff_color"
        elif color1 == color2 and type1 == type2:
            group = "same_color_type"
        else:
            continue

        if len(pairs_by_group[group]) >= PAIR_LIMIT:
            continue

        pairs_by_group[group].append((id1, img1, type1, color1, id2, img2, type2, color2))

# Step 3: Compute similarities
with torch.no_grad():
    for group, pairs in tqdm(pairs_by_group.items(), desc="Computing grouped inter-ID similarities"):
        for id1, img1, type1, color1, id2, img2, type2, color2 in pairs:
            p1 = os.path.join(IMAGE_ROOT, img1)
            p2 = os.path.join(IMAGE_ROOT, img2)
            if not os.path.exists(p1) or not os.path.exists(p2):
                continue

            t1 = transform(Image.open(p1).convert("RGB")).unsqueeze(0)
            t2 = transform(Image.open(p2).convert("RGB")).unsqueeze(0)
            e1 = model([t1.squeeze(0)])
            e2 = model([t2.squeeze(0)])

            sim = F.cosine_similarity(e1, e2).item()
            records.append([group, img1, img2, id1, id2, type1, type2, color1, color2, f"{sim:.4f}"])

# Step 4: Save CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["group", "image1", "image2", "vehicle_id1", "vehicle_id2", "type1", "type2", "color1", "color2", "similarity"])
    writer.writerows(records)

print(f"[✓] Saved grouped inter-ID similarities to {OUTPUT_CSV}")
