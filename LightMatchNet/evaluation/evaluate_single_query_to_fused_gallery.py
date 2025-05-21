# evaluate_single_query_to_fused_gallery.py

"""
Evaluates a Re-ID model by comparing single-view query images to fused gallery
embeddings (multi-view fusion). Uses VeRi XML label files.

Adds:
- --average_by query or vid for metric aggregation
- --query_view_filter front or rear to filter queries by camera view
- --soft_match uses shape + color to compute relaxed identity match

Run:
python3 evaluation/evaluate_single_query_to_fused_gallery.py \
  --checkpoint checkpoints/your_model.pt \
  --query_dir data/VeRi/image_query \
  --query_label data/VeRi/test_label.xml \
  --gallery_dir data/VeRi/image_test \
  --gallery_label data/VeRi/test_label.xml \
  --query_list data/VeRi/name_query.txt \
  --output_csv thesis_eval/single_query_to_fused_gallery.csv \
  --use_rerank \
  --average_by query \
  --query_view_filter front \
  --soft_match
"""

import os, sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.rerank import re_ranking

def parse_label_xml(xml_path):
    with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
        content = f.read()
    root = ET.fromstring(content)
    return {
        item.attrib['imageName']: {
            'vehicleID': item.attrib['vehicleID'],
            'cameraID': item.attrib['cameraID'],
            'colorID': item.attrib['colorID'],
            'typeID': item.attrib['typeID']
        } for item in root.findall(".//Item")
    }

def is_soft_match(query_meta, gallery_meta):
    return (
        query_meta and gallery_meta and
        query_meta['colorID'] == gallery_meta['colorID'] and
        query_meta['typeID'] == gallery_meta['typeID']
    )

def load_query_filenames(query_list_path):
    with open(query_list_path, 'r') as f:
        return set([line.strip() for line in f.readlines()])

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def load_model(model_name, checkpoint_path, device):
    from models.multiview_matchnet import MultiViewMatchNet
    model = MultiViewMatchNet(backbone=model_name, embedding_dim=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()

def get_embedding(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model([tensor])[0].cpu()

def fuse_gallery_embeddings(model, gallery_dir, gallery_labels, query_filenames, device):
    gallery_groups = defaultdict(list)
    for fname, vid in gallery_labels.items():
        if fname not in query_filenames:
            gallery_groups[vid].append(fname)

    fused_gallery = {}
    for vid, files in tqdm(gallery_groups.items(), desc="Fusing gallery"):
        tensors = []
        for fname in files:
            path = os.path.join(gallery_dir, fname)
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img).to(device))
        if tensors:
            with torch.no_grad():
                fused_gallery[vid] = model(tensors)[0].cpu()
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
    model = load_model(args.model, args.checkpoint, device)
    global transform
    transform = get_transform()

    query_filenames = load_query_filenames(args.query_list)
    all_labels = parse_label_xml(args.query_label)
    query_labels = {f: v for f, v in all_labels.items() if f in query_filenames}

    if args.query_view_filter:
        view_map = {'front': ['c001', 'c003'], 'rear': ['c002', 'c004']}
        allowed_cams = view_map.get(args.query_view_filter.lower(), [])
        query_labels = {
            f: v for f, v in query_labels.items() if v['cameraID'] in allowed_cams
        }

    gallery_labels = parse_label_xml(args.gallery_label)
    vid_to_meta = {}
    for fname, meta in gallery_labels.items():
        vid = meta['vehicleID']
        if vid not in vid_to_meta:
            vid_to_meta[vid] = meta
    query_groups = defaultdict(list)
    for fname, meta in query_labels.items():
        query_groups[meta['vehicleID']].append(fname)

    fused_gallery = fuse_gallery_embeddings(
        model, args.gallery_dir,
        {k: v['vehicleID'] for k, v in gallery_labels.items()},
        query_filenames,
        device
    )
    gallery_vids = list(fused_gallery.keys())
    gallery_embs = torch.stack([fused_gallery[vid] for vid in gallery_vids])

    results = []
    cmc_ranks = [1, 5, 10, 20, 50]
    cmc_hit_counts = {k: 0 for k in cmc_ranks}
    correct_sims, incorrect_sims, ap_list = [], [], []
    top1 = top5 = top10 = 0

    if args.average_by == "vid":
        total = len(query_groups)
        for qvid, files in tqdm(query_groups.items(), desc="Evaluating VIDs"):
            tensors = [transform(Image.open(os.path.join(args.query_dir, f)).convert("RGB")).to(device)
                       for f in files if os.path.exists(os.path.join(args.query_dir, f))]
            if not tensors:
                continue
            with torch.no_grad():
                q_emb = model(tensors)[0].cpu()
            if args.use_rerank:
                distmat = re_ranking(q_emb.unsqueeze(0), gallery_embs, k1=20, k2=6, lambda_value=0.3)[0]
                sim_list = [(v, 1.0 - distmat[i]) for i, v in enumerate(gallery_vids)]
            else:
                sims = F.cosine_similarity(q_emb, gallery_embs)
                sim_list = [(v, s.item()) for v, s in zip(gallery_vids, sims)]

            ranked = sorted(sim_list, key=lambda x: x[1], reverse=True)
            ranked_ids = [v for v, _ in ranked]
            query_meta = all_labels[files[0]]
            match_rank = -1
            for rank_idx, gallery_vid in enumerate(ranked_ids):
                gallery_meta = vid_to_meta.get(gallery_vid, {})
                if gallery_vid == qvid or (args.soft_match and is_soft_match(query_meta, gallery_meta)):
                    match_rank = rank_idx
                    for k in cmc_ranks:
                        if rank_idx < k:
                            cmc_hit_counts[k] += 1
                    break
            ap = compute_ap(ranked_ids, qvid)
            ap_list.append(ap)
            topk = ranked_ids[:10]

            def is_match(gid):
                gmeta = vid_to_meta.get(gid, {})
                return (args.soft_match and is_soft_match(query_meta, gmeta)) or (not args.soft_match and gid == qvid)

            top1 += is_match(topk[0])
            top5 += any(is_match(gid) for gid in topk[:5])
            top10 += any(is_match(gid) for gid in topk)

            cos_correct = [s for v, s in ranked if is_match(v)]
            correct_sims.append(cos_correct[0] if cos_correct else 0)
            incorrect_sims.extend([s for v, s in ranked if not is_match(v)])

            results.append({"query_vid": qvid, "AP": ap, "rank": match_rank})

    else:
        flat_items = [(f, v['vehicleID']) for f, v in query_labels.items()]
        total = len(flat_items)
        for fname, qvid in tqdm(flat_items, desc="Evaluating queries"):
            img = Image.open(os.path.join(args.query_dir, fname)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                q_emb = model([tensor])[0].cpu()
            if args.use_rerank:
                distmat = re_ranking(q_emb.unsqueeze(0), gallery_embs, k1=20, k2=6, lambda_value=0.3)[0]
                sim_list = [(v, 1.0 - distmat[i]) for i, v in enumerate(gallery_vids)]
            else:
                sims = F.cosine_similarity(q_emb, gallery_embs)
                sim_list = [(v, s.item()) for v, s in zip(gallery_vids, sims)]

            ranked = sorted(sim_list, key=lambda x: x[1], reverse=True)
            ranked_ids = [v for v, _ in ranked]
            query_meta = all_labels[fname]
            match_rank = -1

            for rank_idx, gallery_vid in enumerate(ranked_ids):
                gallery_meta = vid_to_meta.get(gallery_vid, {})
                matched = (gallery_vid == qvid) or (args.soft_match and is_soft_match(query_meta, gallery_meta))
                if matched:
                    match_rank = rank_idx
                    for k in cmc_ranks:
                        if rank_idx < k:
                            cmc_hit_counts[k] += 1
                    break

            ap = compute_ap(ranked_ids, qvid)
            ap_list.append(ap)
            topk = ranked_ids[:10]

            def is_match(gid):
                gmeta = vid_to_meta.get(gid, {})
                return (args.soft_match and is_soft_match(query_meta, gmeta)) or (not args.soft_match and gid == qvid)

            top1 += is_match(topk[0])
            top5 += any(is_match(gid) for gid in topk[:5])
            top10 += any(is_match(gid) for gid in topk)

            cos_correct = [s for v, s in ranked if is_match(v)]
            correct_sims.append(cos_correct[0] if cos_correct else 0)
            incorrect_sims.extend([s for v, s in ranked if not is_match(v)])

            results.append({"query_vid": qvid, "AP": ap, "rank": match_rank})


    metrics_path = args.output_csv.replace(".csv", "_metrics.csv")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with open(metrics_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"Top-1,{top1/total:.4f}\n")
        f.write(f"Top-5,{top5/total:.4f}\n")
        f.write(f"Top-10,{top10/total:.4f}\n")
        f.write(f"mAP,{sum(ap_list)/total:.4f}\n")
        f.write(f"Avg_Correct_Sim,{sum(correct_sims)/len(correct_sims):.4f}\n")
        f.write(f"Avg_Incorrect_Sim,{sum(incorrect_sims)/len(incorrect_sims):.4f}\n")
        for k in cmc_ranks:
            f.write(f"CMC@{k},{cmc_hit_counts[k]/total:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--query_label", type=str, required=True)
    parser.add_argument("--gallery_dir", type=str, required=True)
    parser.add_argument("--gallery_label", type=str, required=True)
    parser.add_argument("--query_list", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="swifttracknet")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--use_rerank", action="store_true")
    parser.add_argument("--average_by", type=str, choices=["query", "vid"], default="vid")
    parser.add_argument("--query_view_filter", type=str, choices=["front", "rear"])
    parser.add_argument("--soft_match", action="store_true", help="Use relaxed identity match based on shape and color")
    args = parser.parse_args()
    evaluate(args)