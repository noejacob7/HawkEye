# FINAL EVALUATION SCRIPT WITH FULL METRICS FOR VERI DATASET
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
from thop import profile


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
    with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
        xml_content = f.read()
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
        if not fname.endswith(".jpg"):
            continue
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


def compute_full_metrics(query_embs, gallery_embs, max_rank=20):
    all_AP = []
    CMC = torch.zeros(max_rank)
    cosine_scores = []
    top1_acc, top5_acc, top10_acc = 0, 0, 0

    total_queries = 0

    for qid, q_emb_list in query_embs.items():
        q_vec = torch.mean(torch.cat(q_emb_list, dim=0), dim=0, keepdim=True)
        sims = []
        for gid, g_emb_list in gallery_embs.items():
            g_vec = torch.mean(torch.cat(g_emb_list, dim=0), dim=0, keepdim=True)
            sim = F.cosine_similarity(q_vec, g_vec).item()
            sims.append((gid, sim))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

        matches = [1 if gid == qid else 0 for gid, _ in sims]
        cosine_scores.append(sims[0][1])

        if sum(matches) == 0:
            continue

        # AP Calculation
        precisions = []
        num_correct = 0
        for i, match in enumerate(matches):
            if match:
                num_correct += 1
                precisions.append(num_correct / (i + 1))
        AP = sum(precisions) / sum(matches)
        all_AP.append(AP)

        # CMC & Top-K Acc
        first_hit = matches.index(1)
        if first_hit < max_rank:
            CMC[first_hit:] += 1
        if first_hit == 0:
            top1_acc += 1
        if first_hit < 5:
            top5_acc += 1
        if first_hit < 10:
            top10_acc += 1

        total_queries += 1

    mAP = sum(all_AP) / total_queries * 100
    CMC = (CMC / total_queries) * 100
    avg_cosine = sum(cosine_scores) / len(cosine_scores)

    return {
        "mAP": mAP,
        "CMC": CMC,
        "Top-1 Accuracy": (top1_acc / total_queries) * 100,
        "Top-5 Accuracy": (top5_acc / total_queries) * 100,
        "Top-10 Accuracy": (top10_acc / total_queries) * 100,
        "Avg Cosine Similarity": avg_cosine,
        "Total Queries": total_queries
    }


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

    print("[INFO] Computing full metrics...")
    metrics = compute_full_metrics(query_embeddings, gallery_embeddings, max_rank=args.max_rank)

    print("[INFO] Measuring FLOPs and Params...")
    dummy_input = [torch.randn(1, 3, 224, 224).to(device)]
    flops, params = profile(model.encoder, inputs=dummy_input, verbose=False)
    model_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)

    print("[INFO] Writing metrics to summary CSV...")
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["mAP", f"{metrics['mAP']:.2f}%"])
        for rank in range(args.max_rank):
            writer.writerow([f"CMC@{rank+1}", f"{metrics['CMC'][rank]:.2f}%"])
        writer.writerow(["Top-1 Accuracy", f"{metrics['Top-1 Accuracy']:.2f}%"])
        writer.writerow(["Top-5 Accuracy", f"{metrics['Top-5 Accuracy']:.2f}%"])
        writer.writerow(["Top-10 Accuracy", f"{metrics['Top-10 Accuracy']:.2f}%"])
        writer.writerow(["Avg Cosine Similarity", f"{metrics['Avg Cosine Similarity']:.4f}"])
        writer.writerow(["Total Queries", metrics['Total Queries']])
        writer.writerow(["Avg Query Inference Time (sec)", f"{query_time:.6f}"])
        writer.writerow(["Avg Gallery Inference Time (sec)", f"{gallery_time:.6f}"])
        writer.writerow(["Model Parameters", f"{params:,}"])
        writer.writerow(["Model FLOPs", f"{flops:,}"])
        writer.writerow(["Model Size (MB)", f"{model_size_mb:.2f}"])

    print("\n[INFO] Final Summary:")
    for key, val in metrics.items():
        if isinstance(val, torch.Tensor):
            for rank in range(args.max_rank):
                print(f"CMC@{rank+1}: {val[rank]:.2f}%")
        else:
            print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")
    print(f"Avg Query Inference Time: {query_time:.6f} sec")
    print(f"Avg Gallery Inference Time: {gallery_time:.6f} sec")
    print(f"Model Parameters: {params:,}")
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mobilenet", "efficientnet", "swifttracknet"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--query_label", type=str, required=True)
    parser.add_argument("--gallery_dir", type=str, required=True)
    parser.add_argument("--gallery_label", type=str, required=True)
    parser.add_argument("--summary_csv", type=str, default="veri_metrics_summary.csv")
    parser.add_argument("--max_rank", type=int, default=20)
    args = parser.parse_args()
    main(args)
