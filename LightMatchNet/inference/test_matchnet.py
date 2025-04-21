import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Model Loader ===
def load_model(model_name, checkpoint_path, device):
    if model_name == "lightmatchnet":
        from models.lightmatchnet_model import LightMatchNet
        model = LightMatchNet()
    elif model_name == "efficientnet":
        from models.efficientnet_matchnet import EfficientNetMatchNet
        model = EfficientNetMatchNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()
    return model

# === Image Transform ===
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# === Embedding Function ===
def get_embedding(model, img_path, device, transform):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(img).squeeze(0)

# === Gallery Loader ===
def encode_gallery(model, gallery_dir, device, transform):
    gallery_embeddings, gallery_paths = [], []

    for folder in os.listdir(gallery_dir):
        folder_path = os.path.join(gallery_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                emb = get_embedding(model, fpath, device, transform)
                gallery_embeddings.append(emb)
                gallery_paths.append(fpath)
            except:
                continue

    return torch.stack(gallery_embeddings), gallery_paths

# === Main ===
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = load_model(args.model, args.checkpoint, device)
    transform = get_transform()

    print("[INFO] Processing query image...")
    query_emb = get_embedding(model, args.query, device, transform)

    print("[INFO] Encoding gallery...")
    gallery_embs, gallery_paths = encode_gallery(model, args.gallery, device, transform)

    print("[INFO] Matching...")
    sims = F.cosine_similarity(query_emb.unsqueeze(0), gallery_embs)
    topk = torch.topk(sims, k=args.topk)

    query_filename = os.path.basename(args.query)
    print(f"\n[🔍] Query Image: {query_filename}")
    print(f"\nTop {args.topk} Similar Matches:\n" + "-"*50)

    for i, idx in enumerate(topk.indices):
        match_path = gallery_paths[idx]
        match_filename = os.path.basename(match_path)
        similarity = sims[idx].item()
        print(f"{i+1:>2}. {match_filename:30} | Similarity: {similarity:.4f}")


    if args.visualize:
        plt.figure(figsize=(15, 3))
        plt.subplot(1, args.topk + 1, 1)
        plt.imshow(Image.open(args.query))
        plt.title("Query")
        plt.axis("off")

        for i, idx in enumerate(topk.indices):
            plt.subplot(1, args.topk + 1, i + 2)
            plt.imshow(Image.open(gallery_paths[idx]))
            plt.title(f"{sims[idx].item():.2f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

# === CLI Interface ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained matching model.")
    parser.add_argument("--model", type=str, required=True, choices=["lightmatchnet", "efficientnet"],
                        help="Choose model: lightmatchnet or efficientnet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--query", type=str, required=True, help="Path to query image")
    parser.add_argument("--gallery", type=str, required=True, help="Path to gallery folder (subfolders per class)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K results to return")
    parser.add_argument("--visualize", action="store_true", help="Show images of top-k matches")

    args = parser.parse_args()
    main(args)
