import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import csv

# === Dynamic imports ===
def get_model(name):
    if name == "lightmatchnet":
        from models.lightmatchnet_model import LightMatchNet
        return LightMatchNet()
    elif name == "efficientnet":
        from models.efficientnet_matchnet import EfficientNetMatchNet
        return EfficientNetMatchNet()
    # elif name == "ghost":
    #     from models.ghostnet_matchnet import GhostMatchNet
    #     return GhostMatchNet()
    # elif name == "siamese":
    #     from models.siamese_model import SiameseNet
    #     return SiameseNet()
    else:
        raise ValueError(f"Unknown model: {name}")

def get_dataset(method, root_dir, transform):
    if method == "triplet":
        from datasets.triplet_dataset import TripletDataset
        return TripletDataset(root_dir, transform=transform)
    # elif method == "siamese":
    #     from datasets.siamese_dataset import SiameseDataset
    #     return SiameseDataset(root_dir, transform=transform)
    else:
        raise ValueError(f"Unknown method: {method}")

def get_loss_fn(method):
    if method == "triplet":
        return nn.TripletMarginLoss(margin=0.2)
    elif method == "siamese":
        return nn.BCELoss()
    # elif method == "contrastive":
    #     from models.losses import contrastive_loss
    #     return contrastive_loss
    else:
        raise ValueError(f"Unknown loss function: {method}")

# === Training loop ===
def train(model, dataloader, loss_fn, optimizer, device, epochs, resume_path=None, save_path=None):
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    if resume_path:
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))

    log_file = open("train_log.csv", mode="a", newline="")
    log_writer = csv.writer(log_file)
    if os.stat("train_log.csv").st_size == 0:
        log_writer.writerow(["epoch", "batch_loss", "avg_epoch_loss"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            if isinstance(loss_fn, nn.TripletMarginLoss):
                anchor, positive, negative = [x.to(device) for x in batch]
                out_a = model(anchor)
                out_p = model(positive)
                out_n = model(negative)
                loss = loss_fn(out_a, out_p, out_n)

            elif isinstance(loss_fn, nn.BCELoss):
                img1, img2, label = [x.to(device) for x in batch]
                output = model(img1, img2)
                loss = loss_fn(output, label.unsqueeze(1).float())

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

# === CLI interface ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name: lightmatchnet, efficientnet, ghost, siamese")
    parser.add_argument("--method", type=str, required=True, choices=["triplet", "siamese", "contrastive"])
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="trained_model.pt", help="Checkpoint path to save")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from this .pt checkpoint")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    model = get_model(args.model)
    dataset = get_dataset(args.method, args.data, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    loss_fn = get_loss_fn(args.method)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, dataloader, loss_fn, optimizer, device, args.epochs,
          resume_path=args.resume, save_path=args.save)
