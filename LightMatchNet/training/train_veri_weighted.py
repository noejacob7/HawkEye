import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.multiview_matchnet import MultiViewMatchNet
from datasets.triplet_multiview_weighted_dataset import TripletMultiViewWeightedDataset

# Define model getter
def get_model(name, view_mode, embedding_dim=128):
    if view_mode == "multi":
        return MultiViewMatchNet(backbone=name, embedding_dim=embedding_dim)
    else:
        raise NotImplementedError("Only multi-view supported currently.")

# Load similarity matrices
def load_similarity_matrices(veri_dir):
    with open(os.path.join(veri_dir, "type_similarity_matrix.json"), "r") as f:
        type_sim = json.load(f)
    with open(os.path.join(veri_dir, "color_similarity_matrix.json"), "r") as f:
        color_sim = json.load(f)
    return color_sim, type_sim

# Attribute-weighted loss
class AttributeWeightedTripletLoss(nn.Module):
    def __init__(self, margin=0.2, alpha=1.0, beta=1.0, color_sim=None, type_sim=None):
        super().__init__()
        self.triplet = nn.TripletMarginLoss(margin=margin, reduction='none')
        self.alpha = alpha
        self.beta = beta
        self.color_sim = color_sim
        self.type_sim = type_sim

    def forward(self, anchor, positive, negative, anchor_color, neg_color, anchor_type, neg_type):
        base_loss = self.triplet(anchor, positive, negative)
        weights = []
        for ac, nc, at, nt in zip(anchor_color, neg_color, anchor_type, neg_type):
            color_key = f"{ac}|{nc}"
            type_key = f"{at}|{nt}"
            c_sim = self.color_sim.get(color_key, self.color_sim.get(f"{nc}|{ac}", 0.0))
            t_sim = self.type_sim.get(type_key, self.type_sim.get(f"{nt}|{at}", 0.0))
            w = 1 + self.alpha * c_sim + self.beta * t_sim
            weights.append(w)
        weights = torch.tensor(weights).to(base_loss.device)
        return (weights * base_loss).mean()

# Training loop
def train(model, dataloader, loss_fn, optimizer, device, args):
    model.to(device)
    if torch.cuda.device_count() > 1 and not args.no_parallel:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    best_loss = float('inf')
    patience_counter = 0

    log_file = open(args.log, mode="a", newline="")
    log_writer = csv.writer(log_file)
    if os.stat(args.log).st_size == 0:
        log_writer.writerow(["epoch", "batch_loss", "avg_epoch_loss"])

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            optimizer.zero_grad()
            anchor_views, pos_views, neg_views, a_colors, n_colors, a_types, n_types = batch
            anchor = model([img.to(device) for img in anchor_views])
            positive = model([img.to(device) for img in pos_views])
            negative = model([img.to(device) for img in neg_views])

            loss = loss_fn(anchor, positive, negative, a_colors, n_colors, a_types, n_types)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            log_writer.writerow([epoch + 1, loss.item(), ""])

        avg_loss = running_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        log_writer.writerow([epoch + 1, "", avg_loss])

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save)
            print(f"[✓] Model improved and saved to {args.save}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement for {patience_counter} epoch(s)")
            if patience_counter >= args.patience:
                print("[⛔] Early stopping triggered.")
                break

    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="triplet", choices=["triplet"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--label", type=str)
    parser.add_argument("--view_mode", type=str, default="multi")
    parser.add_argument("--anchor_mode", type=str, default="single")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="trained_model.pt")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--log", type=str, default="train_log.csv")
    parser.add_argument("--no_parallel", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--veri_root", type=str, default="data/VeRi")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = get_model(args.model, args.view_mode, embedding_dim=args.embedding_dim)
    color_sim, type_sim = load_similarity_matrices(args.veri_root)
    dataset = TripletMultiViewWeightedDataset(
        args.data, transform=transform, anchor_mode=args.anchor_mode,
        label_file=args.label, dataset_type="veri"
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    loss_fn = AttributeWeightedTripletLoss(margin=0.2, alpha=1.0, beta=1.0, color_sim=color_sim, type_sim=type_sim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, dataloader, loss_fn, optimizer, device, args)