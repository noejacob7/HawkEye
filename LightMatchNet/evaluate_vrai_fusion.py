import os
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.multiview_matchnet import MultiViewMatchNet
from clip_module.utils import compute_cmc_map

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_image_embedding(model, img_path, transform, device):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model([tensor])[0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrai_root", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--dev_pkl", type=str, default="test_dev_annotation.pkl")
    parser.add_argument("--test_pkl", type=str, default="test_annotation.pkl")
    parser.add_argument("--topk", nargs='+', type=int, default=[1, 5, 10])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()

    model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()

    # Load annotations
    dev_anno = load_pkl(os.path.join(args.vrai_root, args.dev_pkl))
    test_anno = load_pkl(os.path.join(args.vrai_root, args.test_pkl))

    dev_im_names = set(dev_anno["dev_im_names"])
    all_im_names = test_anno["test_im_names"]
    all_vids = {name: name.split('_')[0] for name in all_im_names}

    # --- Fuse gallery embeddings from all non-dev images ---
    gallery_groups = defaultdict(list)
    for name in all_im_names:
        if name not in dev_im_names:
            vid = name.split('_')[0]
            gallery_groups[vid].append(name)

    print("Fusing gallery embeddings...")
    gallery_embs, gallery_ids = [], []
    for vid, files in tqdm(gallery_groups.items()):
        embs = []
        for fname in files:
            path = os.path.join(args.vrai_root, "images_dev", fname)
            if os.path.exists(path):
                embs.append(get_image_embedding(model, path, transform, device))
        if embs:
            stacked = torch.stack(embs)
            pooled = model.pool(stacked)
            gallery_embs.append(pooled / pooled.norm())
            gallery_ids.append(vid)

    # --- Get dev query embeddings ---
    print("Encoding dev query images...")
    query_embs, query_ids = [], []
    for fname in tqdm(dev_im_names):
        path = os.path.join(args.vrai_root, "images_dev", fname)
        if os.path.exists(path):
            emb = get_image_embedding(model, path, transform, device)
            query_embs.append(emb)
            query_ids.append(fname.split('_')[0])

    # Evaluate
    query_embs = torch.stack(query_embs).to(device)
    gallery_embs = torch.stack(gallery_embs).to(device)

    print("Evaluating retrieval performance...")
    metrics = compute_cmc_map(
        query_embs=query_embs,
        query_ids=query_ids,
        gallery_embs=gallery_embs,
        gallery_ids=gallery_ids,
        topk=args.topk
    )

    print("\n✅ Evaluation Results:")
    for k, v in metrics["cmc"].items():
        print(f"{k}: {v:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w") as f:
        f.write("metric,value\n")
        for k, v in metrics["cmc"].items():
            f.write(f"{k},{v:.4f}\n")
        f.write(f"mAP,{metrics['mAP']:.4f}\n")

if __name__ == "__main__":
    main()
