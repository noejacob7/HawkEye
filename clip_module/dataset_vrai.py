import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CSVTextImageDataset(Dataset):
    def __init__(self, csv_path, image_root, transform, tokenizer):
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['image_name']
        image_path = os.path.join(self.image_root, row['image_name'])
        image = self.transform(Image.open(image_path).convert("RGB"))
        text = row['caption']
        vid = image_name.split("_")[0]
        return image, text, vid
