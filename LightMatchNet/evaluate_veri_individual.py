# generate_avg_query_evaluation.py (Modified to average similarity across individual query images)
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
    from models.multiview_matchnet import MultiViewMatchNet
    model = MultiViewMatchNet(backbone=model_name, embedding_dim=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def parse_label_xml(xml_path):
    with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
        xml_content = f.read()
    root = ET.fromstring(xml_content)
    labels = {}
    for item in root.findall(".//Item"):
        labels[item.attrib['imageName']] = item.attrib['vehicleID']
    return labels

def extract_individual_embeddings(model, img_dir, label_map, transform, device):
    embeddings_by_id = defaultdict(list)
    total_time = 0.0

    with torch.no_grad():
        for fname in tqdm(os.listdir(img_dir)):
            if not fname.endswith(".jpg") or fname not in label_map:
                continue
            img_path = os.path.join(img_dir, fname)
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            start = time.time()
            emb = model([tensor.squeeze(0)]).cpu()
            total_time += time.time() - start
            vid = label_map[fname]
            embeddings_by_id[vid].append((fname, emb))

    avg_time = total_time / len(embeddings_by_id)
    return embeddings_by_id, avg_time

def compute_similarity_avg_queries(query_embs, gallery_embs, topk):
    top1, top5, top10, total = 0, 0, 0, 0
    results = []

    for qid, qlist in query_embs.items():
        for qname, q_emb in qlist:
            sims = []
            for gid, g_emb_list in gallery_embs.items():
                g_vec = torch.mean(torch.cat([emb for _, emb in g_emb_list], dim=0), dim=0, keepdim=True)
                sim = F.cosine_similarity(q_emb, g_vec).item()
                sims.append((gid, sim))

            sims = sorted(sims, key=lambda x: x[1], reverse=True)
            total += 1
            match_ranks = [i for i, (gid, _) in enumerate(sims) if gid == qid]
            if match_ranks:
                rank = match_ranks[0] + 1
                if rank == 1: top1 += 1
                if rank <= 5: top5 += 1
                if rank <= 10: top10 += 1
            results.append((qid, qname, sims))

    metrics = {
        "Top-1 Accuracy": top1 / total * 100,
        "Top-5 Accuracy": top5 / total * 100,
        "Top-10 Accuracy": top10 / total * 100,
        "Total Queries": total
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

    print("[INFO] Extracting query embeddings (single images)...")
    query_embeddings, query_time = extract_individual_embeddings(model, args.query_dir, query_labels, transform, device)

    print("[INFO] Extracting multiview gallery embeddings...")
    gallery_embeddings, gallery_time = extract_individual_embeddings(model, args.gallery_dir, gallery_labels, transform, device)

    print("[INFO] Computing similarities and metrics...")
    results, metrics = compute_similarity_avg_queries(query_embeddings, gallery_embeddings, topk=args.topk)

    print("[INFO] Writing results to CSV...")
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query_ID", "Image_Name", "Match_Rank", "Gallery_ID", "Similarity"])
        for qid, qname, sims in results:
            for rank, (gid, score) in enumerate(sims[:args.topk]):
                writer.writerow([qid, qname, rank+1, gid, f"{score:.4f}"])

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
    parser.add_argument("--output_csv", type=str, default="veri_eval_individual_query.csv")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    main(args)
