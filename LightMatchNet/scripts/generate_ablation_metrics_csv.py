# generate_ablation_metrics.py (Optimized for Memory)
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.multiview_matchnet import MultiViewMatchNet
from utils.label_parser import parse_veri_labels

# === Config ===
LABEL_PATH = "data/VeRi/test_label.xml"
QUERY_DIR = "data/VeRi/image_query"
GALLERY_DIR = "data/VeRi/image_test"
OUTPUT_CSV = "experiments/metrics/ablation_metrics_cleaned.csv"

CHECKPOINT_PATH = "checkpoints/swifttracknet_multiview_v2_2.pt"
BACKBONE = "swifttracknet"
TOP_K = [1, 5, 10]
VARIANTS = ["fusion", "avg_pool", "single"]

# === Device ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Model Loader ===
def load_model():
    model = MultiViewMatchNet(backbone=BACKBONE, embedding_dim=128)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model.to(device).eval()

# === Embedding Generator ===
def get_gallery_embeddings(model, label_map, variant):
    vid_to_images = defaultdict(list)
    for fname, (vid, _, _) in label_map.items():
        vid_to_images[vid].append(fname)

    gallery_embs = {}
    with torch.no_grad():
        for vid, img_list in tqdm(vid_to_images.items(), desc=f"Gallery ({variant})"):
            images = []
            for fname in sorted(img_list):
                path = os.path.join(GALLERY_DIR, fname)
                if not os.path.exists(path):
                    continue
                img = transform(Image.open(path).convert("RGB")).to(device)
                images.append(img)

            if variant == "single" and images:
                emb = model([images[0]]).cpu()
            elif variant in ["fusion", "avg_pool"] and len(images) >= 2:
                emb = model(images).cpu()
            else:
                continue

            gallery_embs[vid] = emb
            torch.cuda.empty_cache()

    return gallery_embs

# === Evaluation Loop ===
def run_evaluation():
    label_map = parse_veri_labels(LABEL_PATH)
    model = load_model()

    dummy_input = [torch.randn(1, 3, 224, 224).to(device)]
    flops, params = profile(model.encoder, inputs=dummy_input, verbose=False)
    model_size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)

    results = []

    for variant in VARIANTS:
        gallery_embs = get_gallery_embeddings(model, label_map, variant)

        correct_at_k = {k: 0 for k in TOP_K}
        total_queries = 0
        total_time = 0.0

        with torch.no_grad():
            for fname, (q_vid, _, _) in tqdm(label_map.items(), desc=f"Query ({variant})"):
                path = os.path.join(QUERY_DIR, fname)
                if not os.path.exists(path):
                    continue
                img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                start = time.time()
                q_emb = model([img.squeeze(0)]).cpu()
                total_time += time.time() - start
                torch.cuda.empty_cache()

                sims = []
                for g_vid, g_emb in gallery_embs.items():
                    sim = F.cosine_similarity(q_emb, g_emb).item()
                    sims.append((g_vid, sim))

                sims = sorted(sims, key=lambda x: x[1], reverse=True)
                top_k_vids = [vid for vid, _ in sims]
                total_queries += 1

                for k in TOP_K:
                    if q_vid in top_k_vids[:k]:
                        correct_at_k[k] += 1

        avg_time = total_time / total_queries
        row = [variant]
        for k in TOP_K:
            acc = 100 * correct_at_k[k] / total_queries
            row.append(f"{acc:.2f}")
        row += [f"{avg_time:.4f}", f"{model_size_mb:.2f}", f"{params:,}", f"{flops:,}"]
        results.append(row)

    # === Write to CSV ===
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "top1", "top5", "top10", "inference_time", "model_size_mb", "params", "flops"])
        writer.writerows(results)

    print(f"[✓] Saved cleaned ablation metrics to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_evaluation()
