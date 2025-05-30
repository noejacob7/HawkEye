# CLEANED-UP EVALUATION SCRIPT FOR VERI DATASET USING MULTIVIEW_MATCHNET
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

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_model(model_name, checkpoint_path, device):
    from models.multiview_matchnet import MultiViewMatchNet
    model = MultiViewMatchNet(backbone=model_name, embedding_dim=128)
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
    import xml.etree.ElementTree as ET

    # Step 1: Read the file as text using the correct encoding
    with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
        xml_content = f.read()

    # Step 2: Parse from string instead of file
    root = ET.fromstring(xml_content)

    labels = {}
    for item in root.findall(".//Item"):
        labels[item.attrib['imageName']] = item.attrib['vehicleID']
    return labels



def extract_multiview_embeddings(model, img_dir, label_map, transform, device):
    embeddings_by_id = defaultdict(list)
    image_groups = defaultdict(list)
    total_time = 0.0

    for fname in os.listdir(img_dir):
        if not fname.endswith(".jpg"): continue
        vid = label_map.get(fname)
        if vid:
            image_groups[vid].append(fname)

    with torch.no_grad():
        for vid, file_list in tqdm(image_groups.items()):
            tensors = []
            for fname in file_list:
                img_path = os.path.join(img_dir, fname)
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).to(device)
                tensors.append(tensor)
            start = time.time()
            emb = model(tensors).cpu()
            total_time += time.time() - start
            embeddings_by_id[vid].append(emb)

    avg_time = total_time / len(embeddings_by_id)
    return embeddings_by_id, avg_time

def compute_similarity(query_embs, gallery_embs):
    results = []
    top1, top5, top10, correct, total = 0, 0, 0, 0, 0
    for qid, q_emb_list in query_embs.items():
        q_vec = torch.mean(torch.cat(q_emb_list, dim=0), dim=0, keepdim=True)
        sims = []
        for gid, g_emb_list in gallery_embs.items():
            g_vec = torch.mean(torch.cat(g_emb_list, dim=0), dim=0, keepdim=True)
            sim = F.cosine_similarity(q_vec, g_vec).item()
            sims.append((gid, sim))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        total += 1
        match_ranks = [i for i, (gid, _) in enumerate(sims) if gid == qid]
        if match_ranks:
            rank = match_ranks[0] + 1
            if rank <= args.topk:
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

    print("[INFO] Parsing VeRi label files...")
    query_labels = parse_label_xml(args.query_label)
    gallery_labels = parse_label_xml(args.gallery_label)

    print("[INFO] Extracting multiview query embeddings...")
    query_embeddings, query_time = extract_multiview_embeddings(model, args.query_dir, query_labels, transform, device)

    print("[INFO] Extracting multiview gallery embeddings...")
    gallery_embeddings, gallery_time = extract_multiview_embeddings(model, args.gallery_dir, gallery_labels, transform, device)

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
    parser.add_argument("--model", type=str, required=True, choices=["mobilenet", "efficientnet", "swifttracknet"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--query_label", type=str, required=True)
    parser.add_argument("--gallery_dir", type=str, required=True)
    parser.add_argument("--gallery_label", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="veri_eval_results.csv")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    main(args)
