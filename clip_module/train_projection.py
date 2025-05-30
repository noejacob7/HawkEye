#!/usr/bin/env python3
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import clip
from PIL import Image

from clip_module.utils import set_seed, save_checkpoint
from clip_module.get_clip import load_clip_model
from clip_module.projection_head_dcca import DCCAProjectionHead as ProjectionHead
from clip_module.T2I_VeRi.dataset import T2IVeRiTextImageDataset
from clip_module.data.vrai_dataset import VRAITextImageDataset
from clip_module.dataset_vrai import CSVTextImageDataset


from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet

def train_one_epoch(dataloader, clip_model, light_model, proj_head, optimizer, device):
    clip_model.eval()
    light_model.eval()
    loss_fn = nn.CosineEmbeddingLoss(margin=0.0)

    running_loss = 0.0
    grouped_images = {}
    grouped_texts = {}

    for images, texts, vids in dataloader:
        for img, txt, vid in zip(images, texts, vids):
            grouped_images.setdefault(vid, []).append(img)
            grouped_texts.setdefault(vid, []).append(txt)

    for vid in tqdm(grouped_images.keys(), desc="Training VID fusion"):
        imgs = grouped_images[vid]
        texts = grouped_texts[vid]
        images_tensor = [img.to(device) for img in imgs]

        with torch.no_grad():
            img_embs = torch.stack([
                light_model([img.unsqueeze(0)])[0] for img in images_tensor
            ])
            fused_img = light_model.pool(img_embs).unsqueeze(0)
            fused_img = fused_img / fused_img.norm(dim=-1, keepdim=True)

        # tokenized = clip.tokenize(texts, truncate=True).to(device)
        if isinstance(texts[0], torch.Tensor):  # already tokenized
            tokenized = torch.stack(texts).to(device)
        else:
            tokenized = clip.tokenize(texts, truncate=True).to(device)

        with torch.no_grad():
            txt_emb = clip_model.encode_text(tokenized).float()
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        proj_txt = proj_head(txt_emb)
        proj_txt = proj_txt / proj_txt.norm(dim=-1, keepdim=True)

        if fused_img.ndim == 2 and proj_txt.ndim == 2:
            fused_img = fused_img.expand(proj_txt.size(0), -1)

        targets = torch.ones(proj_txt.size(0), device=device)
        loss = loss_fn(proj_txt, fused_img, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * proj_txt.size(0)

    return running_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--lightmatch_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--proj_hidden_dim", type=int, default=256)
    parser.add_argument("--proj_out_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    clip_model, clip_preprocess, _ = load_clip_model(device=device)
    light_model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
    light_model.load_state_dict(torch.load(args.lightmatch_ckpt, map_location=device))
    light_model.eval()

    proj_head = ProjectionHead(input_dim=512, output_dim=args.proj_out_dim).to(device)
    optimizer = optim.Adam(proj_head.parameters(), lr=args.lr)

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        proj_head.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"✅ Resumed from {args.resume} at epoch {start_epoch - 1} with best_loss={best_loss:.4f}")
    else:
        start_epoch = 1
        best_loss = float("inf")

    # ds = T2IVeRiTextImageDataset(
    #     data_json=args.data_json,
    #     image_root=args.image_root,
    #     split="train",
    #     transform=clip_preprocess,
    #     tokenizer=lambda caps, context_length=77: clip.tokenize(caps, context_length=context_length, truncate=True),
    # )
    # ds = VRAITextImageDataset(
    #     annotation_path="data/VRAI/train_annotation.pkl",
    #     image_dir="data/VRAI/images_train",
    #     transform=clip_preprocess,
    #     tokenizer=lambda caps, context_length=77: clip.tokenize(caps, context_length=context_length, truncate=True),
    #     split="train"
    # )

    ds = CSVTextImageDataset(
    csv_path="clip_module/generated_vrai_text_annotations.csv",
    image_root="data/VRAI/images_train",
    transform=clip_preprocess,
    tokenizer=lambda caps, context_length=77: clip.tokenize(caps, context_length=context_length, truncate=True),
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(dl, clip_model, light_model, proj_head, optimizer, device)
        print(f"[Epoch {epoch}] avg loss: {avg_loss:.4f}")

        save_checkpoint({
            "model_state": proj_head.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        }, os.path.join(args.output_dir, "proj_latest.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint({
                "model_state": proj_head.state_dict(),
                "optim_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            }, os.path.join(args.output_dir, "proj_best.pt"))
            print("  → best checkpoint updated")

    print("✅ Training complete.")

if __name__ == "__main__":
    main()
