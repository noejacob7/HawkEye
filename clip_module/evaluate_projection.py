import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
import clip

from clip_module.get_clip import load_clip_model
from clip_module.projection_head import ProjectionHead
from clip_module.T2I_VeRi.dataset import T2IVeRiTextImageDataset
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
from clip_module.utils import compute_cmc_map

@torch.no_grad()
def fuse_image_embeddings(model, dataset, device):
    model.eval()
    fused_embeddings, vehicle_ids = [], []
    grouped_images = {}

    for i in range(len(dataset)):
        img, _, vid = dataset[i]
        grouped_images.setdefault(vid, []).append(img)

    for vid, images in tqdm(grouped_images.items(), desc="Fusing gallery images"):
        embeddings = [model([img.unsqueeze(0).to(device)])[0] for img in images]
        stacked = torch.stack(embeddings)
        fused = model.pool(stacked).unsqueeze(0)
        fused = fused / fused.norm(dim=-1, keepdim=True)
        fused_embeddings.append(fused.cpu())
        vehicle_ids.append(vid)

    return torch.cat(fused_embeddings), vehicle_ids

@torch.no_grad()
def extract_projected_text_embeddings(captions, clip_model, proj_head, device):
    tokenized = clip.tokenize(captions, truncate=True).to(device)
    txt_emb = clip_model.encode_text(tokenized).float()
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    proj_emb = proj_head(txt_emb)
    return proj_emb / proj_emb.norm(dim=-1, keepdim=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--proj_ckpt", type=str, required=True)
    parser.add_argument("--light_ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, clip_preprocess, _ = load_clip_model(device=device)
    proj_head = ProjectionHead(input_dim=512, output_dim=128).to(device)
    proj_ckpt = torch.load(args.proj_ckpt, map_location=device)
    proj_head.load_state_dict(proj_ckpt["model_state"])
    proj_head.eval()

    light_model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
    light_model.load_state_dict(torch.load(args.light_ckpt, map_location=device))
    light_model.eval()

    dataset = T2IVeRiTextImageDataset(
        data_json=args.data_json,
        image_root=args.image_root,
        split=args.split,
        transform=clip_preprocess
    )

    print("Extracting fused gallery embeddings...")
    gallery_embs, gallery_vids = fuse_image_embeddings(light_model, dataset, device)
    gallery_embs = gallery_embs.to(device)

    print("Extracting projected text embeddings...")
    captions = [dataset[i][1] for i in range(len(dataset))]
    query_ids = [dataset[i][2] for i in range(len(dataset))]

    text_embs = extract_projected_text_embeddings(captions, clip_model, proj_head, device)

    print("Evaluating retrieval performance...")
    metrics = compute_cmc_map(
        query_embs=text_embs,
        query_ids=query_ids,
        gallery_embs=gallery_embs,
        gallery_ids=gallery_vids,
        topk=args.topk
    )

    print("\n✅ Evaluation Results:")
    for k, v in metrics["cmc"].items():
        print(f"{k}: {v:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")

if __name__ == "__main__":
    main()
