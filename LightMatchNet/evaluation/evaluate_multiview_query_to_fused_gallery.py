# evaluate_multiview_query_to_fused_gallery.py

"""
Evaluates a Re-ID model by comparing multi-view query inputs (fused)
to multi-view fused gallery embeddings. Based on JSON metadata.

Run:
python3 evaluation/evaluate_multiview_query_to_fused_gallery.py \
  --checkpoint checkpoints/your_model.pt \
  --metadata data/VeRi/veri_all_metadata.json \
  --query_dir data/VeRi/image_query \
  --gallery_dir data/VeRi/image_test \
  --output_csv thesis_eval/multiview_query_to_fused_gallery.csv \
  --use_rerank
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import csv
import argparse
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.multiview_matchnet import MultiViewMatchNet
from utils.rerank import re_ranking

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(checkpoint, device, backbone, embedding_dim):
    model = MultiViewMatchNet(backbone=backbone, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval().to(device)
    return model

def get_multiview_query_embeddings(model, query_dir, metadata, device):
    query_groups = defaultdict(list)
    for fname, meta in metadata.items():
        if meta["split"] == "query":
            query_groups[meta["vehicleID"]].append(fname)

    embeddings = {}
    for vid, files in tqdm(query_groups.items(), desc="Fusing queries"):
        tensors = []
        for fname in files:
            path = os.path.join(query_dir, fname)
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img).to(device))
        if tensors:
            with torch.no_grad():
                fused = model(tensors)[0].cpu()
                embeddings[vid] = fused
    return embeddings

def fuse_gallery_embeddings(model, gallery_dir, metadata, device):
    gallery_groups = defaultdict(list)
    for fname, meta in metadata.items():
        if meta["split"] == "test":
            gallery_groups[meta["vehicleID"]].append(fname)

    fused_gallery = {}
    for vid, files in tqdm(gallery_groups.items(), desc="Fusing gallery"):
        tensors = []
        for fname in files:
            path = os.path.join(gallery_dir, fname)
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img).to(device))
        if tensors:
            with torch.no_grad():
                fused = model(tensors)[0].cpu()
                fused_gallery[vid] = fused
    return fused_gallery

def compute_ap(ranked_ids, true_id):
    ap, hits = 0.0, 0
    for i, vid in enumerate(ranked_ids):
        if vid == true_id:
            hits += 1
            ap += hits / (i + 1)
    return ap / hits if hits > 0 else 0.0

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.model, args.embedding_dim)

    with open(args.metadata) as f:
        metadata = json.load(f)

    print(f"[INFO] Building fused multi-view query embeddings...")
    query_embs = get_multiview_query_embeddings(model, args.query_dir, metadata, device)
    print(f"[INFO] Found {len(query_embs)} unique vehicle IDs in query set")

    print(f"[INFO] Building fused multi-view gallery embeddings...")
    fused_gallery = fuse_gallery_embeddings(model, args.gallery_dir, metadata, device)
    gallery_vids = list(fused_gallery.keys())
    gallery_embs = torch.stack([fused_gallery[vid] for vid in gallery_vids])

    results = []
    cmc_ranks = [1, 5, 10, 20, 50]
    cmc_hit_counts = {k: 0 for k in cmc_ranks}
    total_queries = len(query_embs)

    correct_sims, incorrect_sims, ap_list = [], [], []
    top1 = top5 = top10 = 0

    for qvid, q_emb in tqdm(query_embs.items(), desc="Evaluating"):
        if args.use_rerank:
            distmat = re_ranking(q_emb.unsqueeze(0), gallery_embs, k1=20, k2=6, lambda_value=0.3)[0]
            sim_list = [(v, 1.0 - distmat[i]) for i, v in enumerate(gallery_vids)]
        else:
            sims = F.cosine_similarity(q_emb, gallery_embs)
            sim_list = [(v, s.item()) for v, s in zip(gallery_vids, sims)]

        ranked = sorted(sim_list, key=lambda x: x[1], reverse=True)
        ranked_ids = [v for v, _ in ranked]

        for rank_idx, rid in enumerate(ranked_ids):
            if rid == qvid:
                for k in cmc_ranks:
                    if rank_idx < k:
                        cmc_hit_counts[k] += 1
                break

        ap = compute_ap(ranked_ids, qvid)
        ap_list.append(ap)

        topk = ranked_ids[:10]
        top1 += (qvid == topk[0])
        top5 += (qvid in topk[:5])
        top10 += (qvid in topk)

        cos_correct = [s for v, s in ranked if v == qvid]
        correct_sims.append(cos_correct[0] if cos_correct else 0)
        incorrect_sims.extend([s for v, s in ranked if v != qvid])

        results.append({
            "query_vid": qvid,
            "top1_vid": topk[0],
            "top5": qvid in topk[:5],
            "top10": qvid in topk,
            "AP": ap,
            "correct_rank": next((i+1 for i, v in enumerate(ranked_ids) if v == qvid), -1),
            "cosine_top1": ranked[0][1]
        })

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    metrics_path = args.output_csv.replace(".csv", "_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"Top-1,{top1/total_queries:.4f}\n")
        f.write(f"Top-5,{top5/total_queries:.4f}\n")
        f.write(f"Top-10,{top10/total_queries:.4f}\n")
        f.write(f"mAP,{sum(ap_list)/total_queries:.4f}\n")
        f.write(f"Avg_Correct_Sim,{sum(correct_sims)/len(correct_sims):.4f}\n")
        f.write(f"Avg_Incorrect_Sim,{sum(incorrect_sims)/len(incorrect_sims):.4f}\n")
        for k in cmc_ranks:
            f.write(f"CMC@{k},{cmc_hit_counts[k]/total_queries:.4f}\n")

    print("\n[RESULTS]")
    print(f"Top-1: {top1/total_queries:.4f} | Top-5: {top5/total_queries:.4f} | Top-10: {top10/total_queries:.4f}")
    print(f"mAP: {sum(ap_list)/total_queries:.4f}")
    for k in cmc_ranks:
        print(f"CMC@{k}: {cmc_hit_counts[k]/total_queries:.4f}")
    print(f"Avg Cosine Similarity - Correct: {sum(correct_sims)/len(correct_sims):.4f}, Incorrect: {sum(incorrect_sims)/len(incorrect_sims):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--gallery_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="swifttracknet")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--use_rerank", action="store_true", help="Apply k-reciprocal re-ranking")
    args = parser.parse_args()
    evaluate(args)
