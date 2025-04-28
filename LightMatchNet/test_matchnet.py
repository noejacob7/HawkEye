import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_model(model_name, checkpoint_path, device):
    if model_name == "lightmatchnet":
        from models.multiview_matchnet import MultiViewMatchNet
        model = MultiViewMatchNet(backbone="mobilenet", embedding_dim=128)
    elif model_name == "efficientnet":
        from models.multiview_matchnet import MultiViewMatchNet
        model = MultiViewMatchNet(backbone="efficientnet", embedding_dim=128)
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

def load_and_transform(image_path, transform, device):
    img = Image.open(image_path).convert("RGB")
    return transform(img).to(device)

def fuse_folder_views(folder, suffix, transform, device):
    views = []
    for view in ["front", "back", "left", "right"]:
        fname = os.path.join(folder, f"{view}{suffix}.jpg")
        if os.path.exists(fname):
            views.append(load_and_transform(fname, transform, device))
    return views

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    transform = get_transform()
    model = load_model(args.model, args.checkpoint, device)

    # Load query embedding (single image *_02.jpg)
    print("[INFO] Processing query...")
    query_views = fuse_folder_views(args.query, "_02", transform, device)
    if len(query_views) == 0:
        raise ValueError("No *_02.jpg views found in query folder.")
    query_embedding = model(query_views).detach()

    # Encode gallery: each ID folder's *_01.jpg views
    print("[INFO] Processing gallery...")
    gallery_embeddings = []
    gallery_ids = []
    gallery_samples = []

    for folder_name in sorted(os.listdir(args.gallery)):
        folder_path = os.path.join(args.gallery, folder_name)
        if not os.path.isdir(folder_path): continue

        views = fuse_folder_views(folder_path, "_01", transform, device)
        if len(views) < 2: continue

        emb = model(views).detach()
        gallery_embeddings.append(emb)
        gallery_ids.append(folder_name)
        gallery_samples.append([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("_01.jpg")])

    gallery_tensor = torch.cat(gallery_embeddings, dim=0)
    similarities = F.cosine_similarity(query_embedding, gallery_tensor)

    topk = torch.topk(similarities, k=args.topk)
    print(f"\n[📍] Query Folder: {os.path.basename(args.query)}")
    print(f"[📊] Top-{args.topk} Similar Matches:")

    for i, idx in enumerate(topk.indices):
        match_id = gallery_ids[idx]
        match_files = gallery_samples[idx]
        sim_score = similarities[idx].item()
        print(f"{i+1:>2}. ID: {match_id:<15} | Similarity: {sim_score:.4f} | Example File: {os.path.basename(match_files[0])}")

    if args.visualize:
        plt.figure(figsize=(15, 3))
        plt.subplot(1, args.topk + 1, 1)
        img_sample = next(iter(query_views))
        plt.imshow(img_sample.permute(1, 2, 0).cpu())
        plt.title("Query")
        plt.axis("off")

        for i, idx in enumerate(topk.indices):
            example_img = Image.open(gallery_samples[idx][0])
            plt.subplot(1, args.topk + 1, i + 2)
            plt.imshow(example_img)
            plt.title(f"{similarities[idx].item():.2f}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lightmatchnet", "efficientnet"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--query", type=str, required=True, help="Folder containing *_02.jpg views")
    parser.add_argument("--gallery", type=str, required=True, help="Parent folder containing ID folders with *_01.jpg views")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    main(args)
