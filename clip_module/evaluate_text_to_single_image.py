import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
import clip

from clip_module.get_clip import load_clip_model
# from clip_module.projection_head import ProjectionHead # old head
from clip_module.projection_head_dcca import DCCAProjectionHead as ProjectionHead
from clip_module.T2I_VeRi.dataset import T2IVeRiTextImageDataset
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
from clip_module.utils import compute_cmc_map

@torch.no_grad()
def extract_image_embeddings(model, dataset, device):
    model.eval()
    embeddings, ids = [], []

    for i in tqdm(range(len(dataset)), desc="Extracting image embeddings"):
        img, _, vid = dataset[i]
        img = img.unsqueeze(0).to(device)
        emb = model([img])[0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())
        ids.append(vid)

    return torch.stack(embeddings), ids

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

    # Load CLIP + projection head
    clip_model, clip_preprocess, _ = load_clip_model(device=device)
    proj_head = ProjectionHead(input_dim=512, output_dim=128).to(device)
    proj_ckpt = torch.load(args.proj_ckpt, map_location=device)
    proj_head.load_state_dict(proj_ckpt["model_state"])
    proj_head.eval()

    # Load LightMatchNet
    light_model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
    light_model.load_state_dict(torch.load(args.light_ckpt, map_location=device))
    light_model.eval()

    # Dataset
    dataset = T2IVeRiTextImageDataset(
        data_json=args.data_json,
        image_root=args.image_root,
        split=args.split,
        transform=clip_preprocess
    )

    print("Extracting gallery image embeddings (no fusion)...")
    gallery_embs, gallery_ids = extract_image_embeddings(light_model, dataset, device)

    print("Extracting projected text embeddings...")
    captions = [dataset[i][1] for i in range(len(dataset))]
    query_ids = [dataset[i][2] for i in range(len(dataset))]
    text_embs = extract_projected_text_embeddings(captions, clip_model, proj_head, device)

    print("Evaluating retrieval performance...")
    metrics = compute_cmc_map(
        query_embs=text_embs.to(device),
        query_ids=query_ids,
        gallery_embs=gallery_embs.to(device),
        gallery_ids=gallery_ids,
        topk=args.topk
    )

    print("\n✅ Evaluation Results:")
    for k, v in metrics["cmc"].items():
        print(f"{k}: {v:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")

if __name__ == "__main__":
    main()
