"""
python3 -m clip_module.evaluate_vrai_clip \
  --vrai_pkl_path data/VRAI/test_annotation.pkl \
  --vrai_img_dir data/VRAI/images_test \
  --proj_ckpt clip_module/checkpoints/clip_proj_dcca_vrai/proj_best.pt \
  --light_ckpt LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt

"""

import os
import argparse
import torch
from tqdm import tqdm
import clip

from clip_module.get_clip import load_clip_model
from clip_module.projection_head_dcca import DCCAProjectionHead as ProjectionHead
from clip_module.data.vrai_dataset import VRAITextImageDataset
from clip_module.utils import compute_cmc_map
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet

@torch.no_grad()
def extract_image_embeddings(model, dataset, device):
    model.eval()
    embeddings, ids = [], []
    for img, _, vid in tqdm(dataset, desc="Extracting image embeddings"):
        emb = model([img.unsqueeze(0).to(device)])[0]
        emb = emb / emb.norm()
        embeddings.append(emb.cpu())
        ids.append(vid)
    return torch.stack(embeddings), ids

@torch.no_grad()
def extract_projected_text_embeddings(dataset, clip_model, proj_head, device):
    captions = [dataset[i][1] for i in range(len(dataset))]
    if isinstance(captions[0], str):
        tokenized = clip.tokenize(captions, truncate=True).to(device)
    else:
        tokenized = torch.stack(captions).to(device)
    txt_emb = clip_model.encode_text(tokenized).float()
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    proj_emb = proj_head(txt_emb)
    proj_emb = proj_emb / proj_emb.norm(dim=-1, keepdim=True)
    return proj_emb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--proj_ckpt", type=str, required=True)
    parser.add_argument("--light_ckpt", type=str, required=True)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    clip_model, preprocess, _ = load_clip_model(device=device)
    proj_head = ProjectionHead(input_dim=512, output_dim=128).to(device)
    proj_head.load_state_dict(torch.load(args.proj_ckpt, map_location=device)["model_state"])
    proj_head.eval()

    light_model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
    light_model.load_state_dict(torch.load(args.light_ckpt, map_location=device))
    light_model.eval()

    # Load dataset
    dataset = VRAITextImageDataset(
        annotation_path=args.annotation_path,
        image_dir=args.image_dir,
        split="query",  # query and gallery handled separately
        transform=preprocess
    )

    print("Extracting projected text embeddings...")
    text_embs = extract_projected_text_embeddings(dataset, clip_model, proj_head, device).to(device)

    gallery_dataset = VRAITextImageDataset(
        annotation_path=args.annotation_path,
        image_dir=args.image_dir,
        split="gallery",
        transform=preprocess
    )

    print("Extracting gallery image embeddings...")
    image_embs, image_ids = extract_image_embeddings(light_model, gallery_dataset, device)
    image_embs = image_embs.to(device)

    query_ids = [dataset[i][2] for i in range(len(dataset))]

    print("Evaluating retrieval performance...")
    metrics = compute_cmc_map(
        query_embs=text_embs,
        query_ids=query_ids,
        gallery_embs=image_embs,
        gallery_ids=image_ids,
        topk=args.topk
    )

    print("\n✅ Evaluation Results:")
    for k, v in metrics["cmc"].items():
        print(f"{k}: {v:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")

if __name__ == "__main__":
    main()
