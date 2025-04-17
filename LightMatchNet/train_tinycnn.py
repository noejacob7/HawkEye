import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from tinycnn_filter_model import TinyCNNCoarseFilter
from coarse_filter_dataset import CoarseFilterDataset

# === Config ===
DATASET_DIR = "lightmatch_data"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Dataset & DataLoader ===
dataset = CoarseFilterDataset(root_dir=DATASET_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# === Model ===
model = TinyCNNCoarseFilter().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for img1, img2, label in pbar:
        img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.float().to(DEVICE).unsqueeze(1)

        output = model(img1, img2)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(f" Epoch {epoch+1} done. Avg Loss: {running_loss / len(loader):.4f}")

torch.save(model.state_dict(), "coarse_filter_checkpoint.pt")
print("Model saved as coarse_filter_checkpoint.pt")