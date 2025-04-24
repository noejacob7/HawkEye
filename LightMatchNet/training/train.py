import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os, sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Model loader ===
def get_model(name, view_mode, embedding_dim=128):
    if view_mode == "multi":
        from models.multiview_matchnet import MultiViewMatchNet
        return MultiViewMatchNet(backbone=name, embedding_dim=embedding_dim)
    else:
        if name == "mobilenet":
            from models.mobilenetv3_small import MobileNetMatchNet
            return MobileNetMatchNet(embedding_dim=embedding_dim)
        elif name == "efficientnet":
            from models.efficientnet_matchnet import EfficientNetMatchNet
            return EfficientNetMatchNet(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown model: {name}")

# === Dataset loader ===
def get_dataset(method, root_dir, transform, view_mode):
    if view_mode == "multi":
        from datasets.triplet_multiview_dataset import TripletMultiViewDataset
        return TripletMultiViewDataset(root_dir, transform=transform, view_suffix="_01")
    else:
        if method == "triplet":
            from datasets.triplet_dataset import TripletDataset
            return TripletDataset(root_dir, transform=transform)
        else:
            raise ValueError(f"Unsupported method {method} for single-view mode")

# === Loss loader ===
def get_loss_fn(method):
    if method == "triplet":
        return nn.TripletMarginLoss(margin=0.2)
    elif method == "siamese":
        return nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss function: {method}")

# === Training loop ===
def train(model, dataloader, loss_fn, optimizer, device, epochs, resume_path=None, save_path=None, log_path="train_log.csv"):
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    if resume_path:
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))

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

            if isinstance(loss_fn, nn.TripletMarginLoss):
                if isinstance(batch[0], list):  # Multi-view input
                    anchor_views, pos_views, neg_views = batch

                    # Move each view list to device
                    anchor = model([img.to(device) for img in anchor_views])
                    positive = model([img.to(device) for img in pos_views])
                    negative = model([img.to(device) for img in neg_views])

                else:  # Single-view input
                    anchor, positive, negative = [x.to(device) for x in batch]
                    anchor = model(anchor)
                    positive = model(positive)
                    negative = model(negative)

                loss = loss_fn(anchor, positive, negative)

            else:
                raise NotImplementedError("Unsupported loss function")

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            log_writer.writerow([epoch + 1, loss.item(), ""])

        avg_loss = running_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        log_writer.writerow([epoch + 1, "", avg_loss])

    log_file.close()
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"[✓] Model saved to {save_path}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Backbone: mobilenet or efficientnet")
    parser.add_argument("--method", type=str, required=True, choices=["triplet"], help="Training method")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--view_mode", type=str, choices=["single", "multi"], default="single", help="single or multi-view input")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Size of embedding vector")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save", type=str, default="trained_model.pt", help="Path to save model")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume training checkpoint")
    parser.add_argument("--log", type=str, default="train_log.csv", help="Path to CSV log file")

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
    dataset = get_dataset(args.method, args.data, transform, args.view_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    loss_fn = get_loss_fn(args.method)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, dataloader, loss_fn, optimizer, device, args.epochs,
          resume_path=args.resume, save_path=args.save, log_path=args.log)
