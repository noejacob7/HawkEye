# evaluate_lfw.py — LFW Evaluation Script (Extended for Top-k, mAP, CMC)
import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import csv
from thop import profile
from collections import defaultdict
from models.multiview_matchnet import MultiViewMatchNet

def load_model(model_name, checkpoint_path, device):
    model = MultiViewMatchNet(backbone=model_name, embedding_dim=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def parse_lfw_pairs(pairs_txt_path):
    pairs = []
    identities = set()
    with open(pairs_txt_path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "" or len(line.split()) < 3:
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                name, idx1, idx2 = parts
                file1 = os.path.join(name, f"{name}_{int(idx1):04d}.jpg")
                file2 = os.path.join(name, f"{name}_{int(idx2):04d}.jpg")
                pairs.append((file1, file2, 1))
                identities.add(name)
            elif len(parts) == 4:
                name1, idx1, name2, idx2 = parts
                file1 = os.path.join(name1, f"{name1}_{int(idx1):04d}.jpg")
                file2 = os.path.join(name2, f"{name2}_{int(idx2):04d}.jpg")
                pairs.append((file1, file2, 0))
                identities.update([name1, name2])
    return pairs, list(identities)

def compute_identification_metrics(model, identities, root, transform, device, max_rank):
    id_to_images = defaultdict(list)
    embeddings = {}

    for name in tqdm(identities, desc="Loading faces"):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.endswith(".jpg"):
                continue
            full_path = os.path.join(folder, fname)
            try:
                img = transform(Image.open(full_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model([img.squeeze(0)])
                embeddings[fname] = (emb, name)
                id_to_images[name].append(fname)
            except:
                continue

    cmc = torch.zeros(max_rank)
    ap_list = []
    top1, top5, top10 = 0, 0, 0
    total = 0

    for query_fname, (query_emb, query_id) in embeddings.items():
        print(f"[DEBUG] Starting comparisons for: {query_fname} ({query_id})")
        sims = []
        for gallery_fname, (gallery_emb, gallery_id) in embeddings.items():
            if gallery_fname == query_fname:
                continue
            sim = F.cosine_similarity(query_emb, gallery_emb).item()
            sims.append((gallery_id, sim))

        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        matches = [1 if gid == query_id else 0 for gid, _ in sims]

        if sum(matches) == 0:
            continue

        first_hit = matches.index(1)
        if first_hit < max_rank:
            cmc[first_hit:] += 1
        if first_hit == 0:
            top1 += 1
        if first_hit < 5:
            top5 += 1
        if first_hit < 10:
            top10 += 1

        precisions = []
        num_correct = 0
        for i, m in enumerate(matches):
            if m:
                num_correct += 1
                precisions.append(num_correct / (i + 1))
        ap = sum(precisions) / sum(matches)
        ap_list.append(ap)
        total += 1

    cmc = (cmc / total) * 100
    map_score = sum(ap_list) / total * 100
    return {
        "Top-1 Accuracy": 100.0 * top1 / total,
        "Top-5 Accuracy": 100.0 * top5 / total,
        "Top-10 Accuracy": 100.0 * top10 / total,
        "mAP": map_score,
        "CMC": cmc,
        "Total Queries": total
    }

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = get_transform()
    model = load_model(args.model, args.checkpoint, device)

    print("[INFO] Parsing pairs.txt...")
    _, identities = parse_lfw_pairs(args.pairs_txt)

    print("[INFO] Running face identification evaluation...")
    metrics = compute_identification_metrics(model, identities, args.image_root, transform, device, args.max_rank)

    dummy_input = [torch.randn(1, 3, 224, 224).to(device)]
    flops, params = profile(model.encoder, inputs=dummy_input, verbose=False)
    model_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)

    print("[INFO] Writing metrics to summary CSV...")
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                for r in range(args.max_rank):
                    writer.writerow([f"CMC@{r+1}", f"{v[r]:.2f}%"])
            else:
                writer.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])
        writer.writerow(["Model Parameters", f"{params:,}"])
        writer.writerow(["Model FLOPs", f"{flops:,}"])
        writer.writerow(["Model Size (MB)", f"{model_size_mb:.2f}"])

    print("\n[INFO] Final Summary:")
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            for r in range(args.max_rank):
                print(f"CMC@{r+1}: {v[r]:.2f}%")
        else:
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"Model Parameters: {params:,}")
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pairs_txt", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--summary_csv", type=str, default="results/lfw_identification_metrics.csv")
    parser.add_argument("--max_rank", type=int, default=20)
    args = parser.parse_args()
    main(args)
