import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from lightmatchnet_model import LightMatchNet
from triplet_dataset import TripletDataset
from triplet_miner import batch_hard_triplet_loss  # We'll define this below

if __name__ == "__main__":

    # === Config ===
    DATASET_PATH = "lightmatch_data"
    EPOCHS = 20
    BATCH_SIZE = 32
    LR = 1e-4
    EMBEDDING_DIM = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "lightmatchnet_checkpoint.pt"

    # === Transforms ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # === Dataset and Loader ===
    dataset = TripletDataset(root_dir=DATASET_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # === Model ===
    model = LightMatchNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # === Training Loop ===
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for anchors, positives, negatives in pbar:
            anchors = anchors.to(DEVICE)
            positives = positives.to(DEVICE)
            negatives = negatives.to(DEVICE)

            emb_a = model(anchors)
            emb_p = model(positives)
            emb_n = model(negatives)

            loss = batch_hard_triplet_loss(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(loader):.4f}")

    # === Save Model ===
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")
