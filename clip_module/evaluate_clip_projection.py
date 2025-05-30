import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import clip
import csv
import pandas as pd
from collections import defaultdict

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
from clip_module.projection_head_dcca import DCCAProjectionHead as ProjectionHead
def load_models(light_ckpt, proj_ckpt, device):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    light_model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
    light_model.load_state_dict(torch.load(light_ckpt, map_location=device))
    light_model.eval()
    proj_head = ProjectionHead(input_dim=512, output_dim=128).to(device)
    proj_head.load_state_dict(torch.load(proj_ckpt, map_location=device)["model_state"])
    proj_head.eval()
    return clip_model, clip_preprocess, light_model, proj_head
def load_text_queries(csv_path):
    df = pd.read_csv(csv_path)
    queries = defaultdict(list)
    for _, row in df.iterrows():
        fname = row["image_name"]
        caption = row["caption"]
        vid = "_".join(fname.split("_")[:2])  # Extract "00000000_0002"
        queries[vid].append(caption)
    return queries
def load_gallery_embeddings(gallery_dir, label_map, light_model, transform, device, fuse=True):
    grouped = defaultdict(list)
    for fname, vid in label_map.items():
        grouped[vid].append(fname)
    gallery_embs = {}
    with torch.no_grad():
        for vid, file_list in tqdm(grouped.items(), desc="Fusing gallery images"):
            images = []
            for fname in file_list:
                path = os.path.join(gallery_dir, fname)
                if not os.path.exists(path):
                    continue
                img = transform(Image.open(path).convert("RGB")).to(device)
                images.append(img)
            if not images:
                continue
            if fuse and len(images) > 1:
                embs = [light_model([img])[0].unsqueeze(0) for img in images]
                emb = light_model.pool(torch.cat(embs, dim=0).to(device)).unsqueeze(0)
            else:
                emb = light_model([images[0]])[0].unsqueeze(0)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            gallery_embs[vid] = emb.cpu()
    return gallery_embs
def evaluate(clip_model, proj_head, gallery_embs, queries, device):
    results = []
    top1, top5, top10 = 0, 0, 0
    total = 0
    with torch.no_grad():
        for q_vid, captions in tqdm(queries.items(), desc="Evaluating queries"):
            for caption in captions:
                tokens = clip.tokenize([caption]).to(device)
                txt_emb = clip_model.encode_text(tokens).float()
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                proj = proj_head(txt_emb)
                proj = proj / proj.norm(dim=-1, keepdim=True)
                sims = []
                for g_vid, g_emb in gallery_embs.items():
                    sim = F.cosine_similarity(proj, g_emb.to(device)).item()
                    sims.append((g_vid, sim))
                sims = sorted(sims, key=lambda x: x[1], reverse=True)
                ranked_vids = [vid for vid, _ in sims]
                total += 1
                if q_vid == ranked_vids[0]:
                    top1 += 1
                if q_vid in ranked_vids[:5]:
                    top5 += 1
                if q_vid in ranked_vids[:10]:
                    top10 += 1
                results.append({
                    "query_vid": q_vid,
                    "caption": caption,
                    "top1_match": ranked_vids[0],
                    "correct_in_top5": q_vid in ranked_vids[:5],
                    "correct_in_top10": q_vid in ranked_vids[:10],
                    "cosine_top1": sims[0][1]
                })
    return results, top1, top5, top10, total
def parse_labels_from_filenames(directory):
    label_map = {}
    for fname in os.listdir(directory):
        if fname.endswith(".jpg"):
            vid = fname.split("_")[0]
            label_map[fname] = vid
    return label_map
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    clip_model, clip_preprocess, light_model, proj_head = load_models(
        args.light_ckpt, args.proj_ckpt, device
    )
    queries = load_text_queries(args.query_csv)
    label_map = parse_labels_from_filenames(args.gallery_dir)
    gallery_embs = load_gallery_embeddings(args.gallery_dir, label_map, light_model, clip_preprocess, device, fuse=not args.no_fuse)
    results, top1, top5, top10, total = evaluate(clip_model, proj_head, gallery_embs, queries, device)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print("\\n[INFO] Evaluation complete")
    print(f"Top-1 Accuracy: {100 * top1 / total:.2f}%")
    print(f"Top-5 Accuracy: {100 * top5 / total:.2f}%")
    print(f"Top-10 Accuracy: {100 * top10 / total:.2f}%")
    print(f"Total Queries: {total}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_csv", type=str, required=True, help="CSV with columns: vid, caption")
    parser.add_argument("--gallery_dir", type=str, required=True, help="Directory with gallery images")
    parser.add_argument("--light_ckpt", type=str, required=True, help="Path to LightMatchNet .pt checkpoint")
    parser.add_argument("--proj_ckpt", type=str, required=True, help="Path to projection head checkpoint")
    parser.add_argument("--output_csv", type=str, default="clip_eval_results.csv")
    parser.add_argument("--no_fuse", action="store_true", help="Disable fusion and use single view")
    args = parser.parse_args()
    main(args)