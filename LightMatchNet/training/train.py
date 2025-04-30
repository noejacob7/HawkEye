import argparse
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

def get_model(name, view_mode, embedding_dim=128):
    if view_mode == "multi":
        from models.multiview_matchnet import MultiViewMatchNet
        return MultiViewMatchNet(backbone=name, embedding_dim=embedding_dim)
    else:
        raise NotImplementedError("Only multi-view supported currently.")

from datasets.triplet_multiview_dataset import TripletMultiViewDataset

def get_dataset(method, root_dir, transform, view_mode, anchor_mode, label_path=None, dataset_type="hotwheels"):
    if view_mode == "multi":
        return TripletMultiViewDataset(root_dir, transform=transform, anchor_mode=anchor_mode, label_file=label_path, dataset_type=dataset_type)
    else:
        raise NotImplementedError("Only multi-view supported currently.")


def get_loss_fn(method):
    if method == "triplet":
        return nn.TripletMarginLoss(margin=0.2)
    else:
        raise ValueError(f"Unknown loss function: {method}")

def train(model, dataloader, loss_fn, optimizer, device, epochs, resume_path=None, save_path=None, log_path="train_log.csv", patience=5):
    model.to(device)

    if torch.cuda.device_count() > 1 and not args.no_parallel:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    if resume_path:
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))

    best_loss = float('inf')
    patience_counter = 0

    log_file = open(log_path, mode="a", newline="")
    log_writer = csv.writer(log_file)
    if os.stat(log_path).st_size == 0:
        log_writer.writerow(["epoch", "batch_loss", "avg_epoch_loss"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            if isinstance(batch[0], list):
                anchor_views, pos_views, neg_views = batch
                anchor = model([img.to(device) for img in anchor_views])
                positive = model([img.to(device) for img in pos_views])
                negative = model([img.to(device) for img in neg_views])
            elif isinstance(batch[0], torch.Tensor):
                anchor, pos_views, neg_views = batch
                anchor = model(anchor.unsqueeze(0).to(device))
                positive = model([img.to(device) for img in pos_views])
                negative = model([img.to(device) for img in neg_views])
            else:
                raise ValueError("Unrecognized batch format for anchor")

            loss = loss_fn(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            log_writer.writerow([epoch + 1, loss.item(), ""])

        avg_loss = running_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        log_writer.writerow([epoch + 1, "", avg_loss])

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"[✓] Model improved and saved to {save_path}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement for {patience_counter} epoch(s)")
            if patience_counter >= patience:
                print("[⛔] Early stopping triggered.")
                break

    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Backbone: mobilenet, efficientnet or swifttracknet")
    parser.add_argument("--method", type=str, default="triplet", choices=["triplet"])
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--view_mode", type=str, choices=["multi"], default="multi")
    parser.add_argument("--anchor_mode", type=str, choices=["multi", "single"], default="single", help="Anchor mode: single reference or multi-view")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="trained_model.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log", type=str, default="train_log.csv")
    parser.add_argument("--no_parallel", action="store_true", help="Disable DataParallel")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--label", type=str, help="Path to label XML file (VeRi only)")
    parser.add_argument("--dataset_type", type=str, default="hotwheels", choices=["hotwheels", "veri"], help="Dataset structure type")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    model = get_model(args.model, args.view_mode, embedding_dim=args.embedding_dim)
    dataset = get_dataset(args.method, args.data, transform, args.view_mode, args.anchor_mode, args.label, args.dataset_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    loss_fn = get_loss_fn(args.method)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, dataloader, loss_fn, optimizer, device, args.epochs,
          resume_path=args.resume, save_path=args.save, log_path=args.log, patience=args.patience)
