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

# Add parent directory to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.multiview_matchnet import MultiViewMatchNet

# Transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(checkpoint_path, device, backbone="swifttracknet", embedding_dim=128):
    model = MultiViewMatchNet(backbone=backbone, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def get_image_embedding(model, img_path, device):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model([tensor])[0]  # returns (1, D)
    return emb.cpu()

def compute_average_precision(ranks, relevant_vid):
    ap, hit_count = 0.0, 0
    for i, vid in enumerate(ranks):
        if vid == relevant_vid:
            hit_count += 1
            ap += hit_count / (i + 1)
    return ap / hit_count if hit_count > 0 else 0.0

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.model, args.embedding_dim)

    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    query_data = [(fname, meta["vehicleID"]) for fname, meta in metadata.items() if meta["split"] == "query"]
    gallery_data = [(fname, meta["vehicleID"]) for fname, meta in metadata.items() if meta["split"] == "test"]

    print(f"[DEBUG] include_query_in_gallery: {args.include_query_in_gallery}")

    if args.include_query_in_gallery:
        gallery_data +=  query_data

    print(f"[INFO] Found {len(query_data)} queries and {len(gallery_data)} gallery images.")

    gallery_embeddings = []
    gallery_labels = []
    for fname, vid in tqdm(gallery_data, desc="Encoding gallery"):
        path = os.path.join(args.gallery_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Gallery image not found: {path}")
        emb = get_image_embedding(model, path, device)
        gallery_embeddings.append(emb)
        gallery_labels.append(vid)

    if len(gallery_embeddings) == 0:
        print("[ERROR] No gallery images were processed. Check the gallery path.")
        return

    gallery_matrix = torch.stack(gallery_embeddings)

    top1, top5, top10 = 0, 0, 0
    all_ap = []
    correct_sim = []
    incorrect_sim = []
    results = []

    for qname, qid in tqdm(query_data, desc="Evaluating queries"):
        qpath = os.path.join(args.query_dir, qname)
        if not os.path.exists(qpath):
            raise FileNotFoundError(f"[ERROR] Query image not found: {qpath}")
        qemb = get_image_embedding(model, qpath, device)

        sims = F.cosine_similarity(qemb, gallery_matrix)
        topk = torch.topk(sims, k=10)
        topk_indices = topk.indices.tolist()
        topk_vids = [gallery_labels[i] for i in topk_indices]

        sorted_vids = [gallery_labels[i] for i in sims.argsort(descending=True).tolist()]
        ap = compute_average_precision(sorted_vids, qid)
        all_ap.append(ap)

        top1_vid = topk_vids[0]
        hit1 = int(qid == top1_vid)
        hit5 = int(qid in topk_vids[:5])
        hit10 = int(qid in topk_vids[:10])

        top1 += hit1
        top5 += hit5
        top10 += hit10

        correct_sim.append(sims[gallery_labels.index(qid)].item()) if qid in gallery_labels else None
        incorrect_sim.extend([sims[i].item() for i in range(len(sims)) if gallery_labels[i] != qid])

        results.append({
            "query": qname,
            "gt_vid": qid,
            "top1_vid": top1_vid,
            "top5": hit5,
            "top10": hit10,
            "correct_rank": next((i+1 for i, vid in enumerate(topk_vids) if vid == qid), -1),
            "cosine_top1": topk.values[0].item(),
            "AP": ap
        })

    if len(results) == 0:
        print("[ERROR] No results to write. Ensure query and gallery directories are correct.")
        return

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    metrics_path = args.output_csv.replace(".csv", "_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"Top-1,{top1/total:.4f}\n")
        f.write(f"Top-5,{top5/total:.4f}\n")
        f.write(f"Top-10,{top10/total:.4f}\n")
        f.write(f"mAP,{sum(all_ap)/len(all_ap):.4f}\n")
        if correct_sim:
            f.write(f"Avg_Correct_Sim,{sum(correct_sim)/len(correct_sim):.4f}\n")
        if incorrect_sim:
            f.write(f"Avg_Incorrect_Sim,{sum(incorrect_sim)/len(incorrect_sim):.4f}\n")

    print("\n[RESULTS]")
    print(f"Top-1 Accuracy: {top1/total:.4f}")
    print(f"Top-5 Accuracy: {top5/total:.4f}")
    print(f"Top-10 Accuracy: {top10/total:.4f}")
    print(f"Mean AP (mAP): {sum(all_ap)/len(all_ap):.4f}")
    print(f"Avg Correct Cosine Similarity: {sum(correct_sim)/len(correct_sim):.4f}" if correct_sim else "No correct matches found.")
    print(f"Avg Incorrect Cosine Similarity: {sum(incorrect_sim)/len(incorrect_sim):.4f}" if incorrect_sim else "No incorrect matches found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--metadata", type=str, required=True, help="Path to veri_all_metadata.json")
    parser.add_argument("--query_dir", type=str, required=True, help="Directory containing query images")
    parser.add_argument("--gallery_dir", type=str, required=True, help="Directory containing gallery images")
    parser.add_argument("--output_csv", type=str, required=True, help="Where to save evaluation results")
    parser.add_argument("--model", type=str, default="swifttracknet", help="Model backbone")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--include_query_in_gallery", action="store_true", help="Whether to include query images in the gallery")
    args = parser.parse_args()

    evaluate(args)
