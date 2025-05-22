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

from models.multiview_matchnet import MultiViewMatchNet

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

def load_model(backbone, checkpoint_path, device):
    model = MultiViewMatchNet(backbone=backbone, embedding_dim=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_embedding(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model([tensor])[0].cpu()

def compute_ap(ranked_ids, is_match):
    ap, hits = 0.0, 0
    for i, vid in enumerate(ranked_ids):
        if is_match(vid):
            hits += 1
            ap += hits / (i + 1)
    return ap / hits if hits > 0 else 0.0

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.checkpoint, device)
    global transform
    transform = get_transform()

    query_labels = parse_label_xml(args.query_label)
    gallery_labels = parse_label_xml(args.gallery_label)
    query_filenames = load_query_filenames(args.query_list)

    query_items = [(fname, query_labels[fname]) for fname in query_filenames if fname in query_labels]
    gallery_items = [(fname, meta) for fname, meta in gallery_labels.items() if fname not in query_filenames]

    results = []
    cmc_ranks = [1, 5, 10, 20, 50]
    cmc_hit_counts = {k: 0 for k in cmc_ranks}
    correct_sims, incorrect_sims, ap_list = [], [], []
    top1 = top5 = top10 = 0
    total = len(query_items)

    for qfname, qmeta in tqdm(query_items, desc="Evaluating single query to single gallery"):
        qvid = qmeta['vehicleID']
        q_emb = get_embedding(model, os.path.join(args.query_dir, qfname), device)

        gallery_embs, gallery_vids = [], []
        for gfname, gmeta in gallery_items:
            emb = get_embedding(model, os.path.join(args.gallery_dir, gfname), device)
            gallery_embs.append(emb)
            gallery_vids.append(gmeta['vehicleID'])

        gallery_matrix = torch.stack(gallery_embs)

        def is_match(gid):
            gmeta = next((meta for _, meta in gallery_items if meta['vehicleID'] == gid), {})
            return (args.soft_match and is_soft_match(qmeta, gmeta)) or (not args.soft_match and gid == qvid)

        sims = F.cosine_similarity(q_emb, gallery_matrix)
        sim_list = [(gallery_vids[i], sims[i].item()) for i in range(len(gallery_vids))]

        ranked = sorted(sim_list, key=lambda x: x[1], reverse=True)
        ranked_ids = [v for v, _ in ranked]

        match_rank = next((i for i, gid in enumerate(ranked_ids) if is_match(gid)), -1)
        if match_rank >= 0:
            for k in cmc_ranks:
                if match_rank < k:
                    cmc_hit_counts[k] += 1

        ap = compute_ap(ranked_ids, is_match)
        ap_list.append(ap)

        topk = ranked_ids[:10]
        top1 += is_match(topk[0])
        top5 += any(is_match(gid) for gid in topk[:5])
        top10 += any(is_match(gid) for gid in topk)

        cos_correct = [s for v, s in ranked if is_match(v)]
        correct_sims.append(cos_correct[0] if cos_correct else 0)
        incorrect_sims.extend([s for v, s in ranked if not is_match(v)])

        results.append({"query_image": qfname, "gt_vid": qvid, "AP": ap, "rank": match_rank})

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
    parser.add_argument("--soft_match", action="store_true")
    args = parser.parse_args()
    evaluate(args)
