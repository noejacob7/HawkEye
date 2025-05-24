# LightMatchNet/datasets/vrai_reid_dataset.py

import os
import pickle
from PIL import Image
from torch.utils.data import Dataset

class VRAIVehicleReIDDataset(Dataset):
    def __init__(self, pkl_path, image_root, split="query", transform=None):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        assert split in ["query", "gallery"], "Split must be 'query' or 'gallery'"
        self.image_names = data[f"{split}_order"]
        self.labels = {
            img_name: img_name.split("_")[0]  # vehicle ID is prefix
            for img_name in self.image_names
        }

        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        fname = self.image_names[idx]
        label = self.labels[fname]
        path = os.path.join(self.image_root, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
