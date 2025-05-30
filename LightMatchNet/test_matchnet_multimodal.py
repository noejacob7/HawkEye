import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_model(model_name, checkpoint_path, device):
    from models.multiview_matchnet import MultiViewMatchNet
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

def load_and_transform(img_path, transform, device):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def fuse_views(folder, suffix, transform, device):
    tensors = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(suffix + ".jpg"):
            img_path = os.path.join(folder, file)
            tensors.append(load_and_transform(img_path, transform, device))
    if tensors:
        return torch.cat(tensors, dim=0)
    return None

def evaluate_all_ids(model, model_name, data_dir, topk=5):
    transform = get_transform()
    device = next(model.parameters()).device
    all_results = []

    ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    for query_id in tqdm(ids, desc=f"Testing model: {model_name}"):
        query_path = os.path.join(data_dir, query_id)
        query_tensor = fuse_views(query_path, "_02", transform, device)
        if query_tensor is None:
            continue
        query_emb = model(query_tensor).mean(dim=0, keepdim=True)

        gallery_embs = []
        gallery_ids = []

        for gallery_id in ids:
            if gallery_id == query_id:
                continue
            gallery_path = os.path.join(data_dir, gallery_id)
            gallery_tensor = fuse_views(gallery_path, "_01", transform, device)
            if gallery_tensor is None:
                continue
            emb = model(gallery_tensor).mean(dim=0, keepdim=True)
            gallery_embs.append(emb)
            gallery_ids.append(gallery_id)

        if not gallery_embs:
            continue

        gallery_tensor = torch.cat(gallery_embs, dim=0)
        similarities = F.cosine_similarity(query_emb, gallery_tensor)

        actual_k = min(topk, similarities.size(0))
        top_vals, top_idxs = torch.topk(similarities, k=actual_k)
        for rank, (score, idx) in enumerate(zip(top_vals, top_idxs), 1):
            all_results.append({
                "model": model_name,
                "query_id": query_id,
                "match_rank": rank,
                "match_id": gallery_ids[idx],
                "similarity": score.item()
            })

    return pd.DataFrame(all_results)


def main():
    models = {
        "swifttracknet": "checkpoints/swifttracknet_multiview_v1.pt",
        "mobilenet": "checkpoints/mobilenet_multiview_v1.pt",
        "efficientnet": "checkpoints/efficientnet_multiview_v1.pt"
    }

    all_dfs = []
    data_root = "data/hot_wheels"

    for model_name, ckpt in models.items():
        model = load_model(model_name, ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
        df = evaluate_all_ids(model, model_name, data_root, topk=5)
        df.to_csv(f"{model_name}_hotwheels_eval.csv", index=False)
        all_dfs.append(df)

    # Optionally merge
    pd.concat(all_dfs).to_csv("hotwheels_eval_all.csv", index=False)


if __name__ == "__main__":
    main()
