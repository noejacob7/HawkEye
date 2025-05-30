# generate_ablation_metrics.py — Fused Query and Gallery Evaluation
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
from LightMatchNet.utils.label_parser import parse_veri_labels

LABEL_PATH = "LightMatchNet/data/VeRi/test_label.xml"
QUERY_DIR = "LightMatchNet/data/VeRi/image_query"
GALLERY_DIR = "LightMatchNet/data/VeRi/image_test"
OUTPUT_CSV = "LightMatchNet/experiments/metrics/ablation_metrics_fused_query_and_gallery.csv"

CHECKPOINT_PATH = "LightMatchNet/checkpoints/swifttracknet_multiview_v2_2.pt"
BACKBONE = "swifttracknet"
TOP_K = [1, 5, 10]
VARIANTS = ["fusion", "avg_pool", "single"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = MultiViewMatchNet(backbone=BACKBONE, embedding_dim=128)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model.to(device).eval()

def fuse_embeddings(images, model, variant):
    if variant == "avg_pool":
        embs = [model([img])[0].unsqueeze(0) for img in images]
        emb = torch.mean(torch.cat(embs, dim=0), dim=0, keepdim=True)
    else:  # fusion (attention) or single
        emb = model(images)
    return emb

def get_embeddings(model, label_map, root_dir, variant):
    vid_to_images = defaultdict(list)
    for fname, (vid, _, _) in label_map.items():
        vid_to_images[vid].append(fname)

    embs = {}
    with torch.no_grad():
        for vid, img_list in tqdm(vid_to_images.items(), desc=f"Embeddings ({variant}) from {root_dir}"):
            images = []
            for fname in sorted(img_list):
                path = os.path.join(root_dir, fname)
                if not os.path.exists(path):
                    continue
                img = transform(Image.open(path).convert("RGB")).to(device)
                images.append(img)

            if variant == "single" and images:
                emb = model([images[0]]).cpu()
            elif variant in ["fusion", "avg_pool"] and len(images) >= 2:
                emb = fuse_embeddings(images, model, variant).cpu()
            else:
                continue

            embs[vid] = emb
            torch.cuda.empty_cache()

    return embs

def run_evaluation():
    label_map = parse_veri_labels(LABEL_PATH)
    model = load_model()

    dummy_input = [torch.randn(1, 3, 224, 224).to(device)]
    flops, params = profile(model.encoder, inputs=dummy_input, verbose=False)
    model_size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)

    results = []

    for variant in VARIANTS:
        gallery_embs = get_embeddings(model, label_map, GALLERY_DIR, variant)
        query_embs = get_embeddings(model, label_map, QUERY_DIR, variant)

        correct_at_k = {k: 0 for k in TOP_K}
        total_queries = len(query_embs)
        total_time = 0.0

        with torch.no_grad():
            for q_vid, q_emb in tqdm(query_embs.items(), desc=f"Compare ({variant})"):
                start = time.time()
                sims = []
                for g_vid, g_emb in gallery_embs.items():
                    sim = F.cosine_similarity(q_emb, g_emb).item()
                    sims.append((g_vid, sim))
                total_time += time.time() - start

                sims = sorted(sims, key=lambda x: x[1], reverse=True)
                top_k_vids = [vid for vid, _ in sims]

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

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "top1", "top5", "top10", "inference_time", "model_size_mb", "params", "flops"])
        writer.writerows(results)

    print(f"[✓] Saved ablation metrics to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_evaluation()
