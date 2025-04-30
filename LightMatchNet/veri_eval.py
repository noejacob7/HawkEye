# NEW TESTING SCRIPT FOR VERI DATASET ONLY WITH FULL EVALUATION METRICS
import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
import time

def load_model(model_name, checkpoint_path, device):
    if model_name == "lightmatchnet":
        from models.multiview_matchnet import MultiViewMatchNet
        model = MultiViewMatchNet(backbone="mobilenet", embedding_dim=128)
    elif model_name == "efficientnet":
        from models.multiview_matchnet import MultiViewMatchNet
        model = MultiViewMatchNet(backbone="efficientnet", embedding_dim=128)
    elif model_name == "swifttracknet":
        from models.swifttracknet import SwiftTrackNet
        model = SwiftTrackNet(embedding_dim=128)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def parse_label_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = {}
    for item in root.findall("image"):
        labels[item.attrib['name']] = item.attrib['vehicleID']
    return labels

def extract_embeddings(model, img_dir, label_map, transform, device):
    embeddings_by_id = defaultdict(list)
    total_time = 0.0
    with torch.no_grad():
        for fname in tqdm(os.listdir(img_dir)):
            if not fname.endswith(".jpg"): continue
            img_path = os.path.join(img_dir, fname)
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            start = time.time()
            emb = model(tensor).cpu()
            total_time += time.time() - start
            vehicle_id = label_map.get(fname, None)
            if vehicle_id:
                embeddings_by_id[vehicle_id].append(emb)
    avg_time = total_time / sum(len(v) for v in embeddings_by_id.values())
    return embeddings_by_id, avg_time

def average_embeddings(emb_list):
    return torch.mean(torch.cat(emb_list, dim=0), dim=0, keepdim=True)

def compute_similarity(query_embs, gallery_embs):
    results = []
    top1, top5, top10, correct, total = 0, 0, 0, 0, 0
    for qid, q_emb in query_embs.items():
        q_vec = average_embeddings(q_emb)
        sims = []
        for gid, g_emb in gallery_embs.items():
            g_vec = average_embeddings(g_emb)
            sim = F.cosine_similarity(q_vec, g_vec).item()
            sims.append((gid, sim))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        total += 1
        match_ranks = [i for i, (gid, _) in enumerate(sims) if gid == qid]
        if match_ranks:
            rank = match_ranks[0] + 1
            correct += 1
            if rank == 1: top1 += 1
            if rank <= 5: top5 += 1
            if rank <= 10: top10 += 1
        results.append((qid, sims))
    metrics = {
        "Top-1 Accuracy": top1 / total * 100,
        "Top-5 Accuracy": top5 / total * 100,
        "Top-10 Accuracy": top10 / total * 100,
        "Total Queries": total,
        "Correct Matches": correct,
        "Incorrect Matches": total - correct
    }
    return results, metrics

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    transform = get_transform()
    model = load_model(args.model, args.checkpoint, device)

    print("[INFO] Parsing label files...")
    query_labels = parse_label_xml(args.query_label)
    gallery_labels = parse_label_xml(args.gallery_label)

    print("[INFO] Extracting query embeddings...")
    query_embeddings, query_time = extract_embeddings(model, args.query_dir, query_labels, transform, device)

    print("[INFO] Extracting gallery embeddings...")
    gallery_embeddings, gallery_time = extract_embeddings(model, args.gallery_dir, gallery_labels, transform, device)

    print("[INFO] Computing similarities and metrics...")
    results, metrics = compute_similarity(query_embeddings, gallery_embeddings)

    print("[INFO] Writing results to CSV...")
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query_ID", "Match_Rank", "Gallery_ID", "Similarity"])
        for qid, sims in results:
            for rank, (gid, score) in enumerate(sims[:args.topk]):
                writer.writerow([qid, rank+1, gid, f"{score:.4f}"])

    print("[INFO] Evaluation Summary:")
    for key, val in metrics.items():
        print(f"{key}: {val:.2f}%" if "Accuracy" in key else f"{key}: {val}")
    print(f"Avg Query Inference Time: {query_time:.6f} sec")
    print(f"Avg Gallery Inference Time: {gallery_time:.6f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lightmatchnet", "efficientnet", "swifttracknet"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--query_dir", type=str, required=True, help="Directory of query images")
    parser.add_argument("--query_label", type=str, required=True, help="XML label file for queries")
    parser.add_argument("--gallery_dir", type=str, required=True, help="Directory of gallery images")
    parser.add_argument("--gallery_label", type=str, required=True, help="XML label file for gallery")
    parser.add_argument("--output_csv", type=str, default="veri_eval_results.csv")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    main(args)
