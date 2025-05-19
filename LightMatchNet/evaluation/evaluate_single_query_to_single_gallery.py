# evaluate_single_query_to_single_gallery.py

"""
This script evaluates a Re-ID model by comparing single query images
against single gallery images (VeRi-776). It calculates Top-k accuracy,
mAP, cosine similarity statistics, and CMC curve metrics.

Run command:
python3 evaluation/evaluate_single_query_to_single_gallery.py \
  --checkpoint checkpoints/your_model.pt \
  --metadata data/VeRi/veri_all_metadata.json \
  --query_dir data/VeRi/image_query \
  --gallery_dir data/VeRi/image_test \
  --output_csv thesis_eval/single_to_single_results.csv \
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.multiview_matchnet import MultiViewMatchNet
from utils.rerank import re_ranking  # Make sure utils/rerank.py exists

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

def get_embedding(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model([tensor])[0].cpu()

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

    query_items = [(f, m["vehicleID"]) for f, m in metadata.items() if m["split"] == "query"]
    query_lookup = {f: v for f, v in query_items}
    gallery_items = [(f, m["vehicleID"]) for f, m in metadata.items() if m["split"] == "test" and f not in query_lookup]

    print(f"[INFO] {len(query_items)} queries | {len(gallery_items)} gallery images")

    # Compute gallery embeddings
    gallery_embeddings, gallery_labels, gallery_files = [], [], []
    for fname, vid in tqdm(gallery_items, desc="Gallery embeddings"):
        emb = get_embedding(model, os.path.join(args.gallery_dir, fname), device)
        gallery_embeddings.append(emb)
        gallery_labels.append(vid)
        gallery_files.append(fname)
    gallery_matrix = torch.stack(gallery_embeddings)

    # Compute query embeddings
    query_embeddings = []
    for fname, _ in tqdm(query_items, desc="Query embeddings"):
        emb = get_embedding(model, os.path.join(args.query_dir, fname), device)
        query_embeddings.append(emb)
    query_matrix = torch.stack(query_embeddings)

    if args.use_rerank:
        print("[INFO] Applying re-ranking...")
        distmat = re_ranking(query_matrix, gallery_matrix, k1=20, k2=6, lambda_value=0.3)

    results = []
    cmc_ranks = [1, 5, 10, 20, 50]
    cmc_hit_counts = {k: 0 for k in cmc_ranks}
    total_queries = len(query_items)

    correct_sims, incorrect_sims, ap_list = [], [], []
    top1 = top5 = top10 = 0

    for idx, (qname, qvid) in enumerate(tqdm(query_items, desc="Evaluating")):
        if args.use_rerank:
            dists = distmat[idx]
            sim_list = [(f, v, 1.0 - dists[i]) for i, (f, v) in enumerate(zip(gallery_files, gallery_labels))]
        else:
            sims = F.cosine_similarity(query_matrix[idx], gallery_matrix)
            sim_list = [(f, v, s.item()) for f, v, s in zip(gallery_files, gallery_labels, sims)]

        ranked = sorted(sim_list, key=lambda x: x[2], reverse=True)
        ranked_ids = [v for _, v, _ in ranked]

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

        cos_correct = [s for f, v, s in ranked if v == qvid]
        correct_sims.append(cos_correct[0] if cos_correct else 0)
        incorrect_sims.extend([s for f, v, s in ranked if v != qvid])

        results.append({
            "query": qname,
            "gt_vid": qvid,
            "top1_vid": topk[0],
            "top5": qvid in topk[:5],
            "top10": qvid in topk,
            "AP": ap,
            "correct_rank": next((i+1 for i, v in enumerate(ranked_ids) if v == qvid), -1),
            "cosine_top1": ranked[0][2]
        })

    # Save results
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
