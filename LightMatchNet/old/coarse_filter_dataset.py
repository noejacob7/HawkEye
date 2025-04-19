import os
import random
from PIL import Image
from torch.utils.data import Dataset

class CoarseFilterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # Gather image paths grouped by class
        self.class_to_images = {
            d: [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.jpg')]
            for d in self.class_dirs
        }

        self.class_list = list(self.class_to_images.keys())

    def __len__(self):
        return 100000  # Synthetic length for infinite sampling

    def __getitem__(self, idx):
        # Randomly select whether this is a positive or negative pair
        is_positive = random.choice([True, False])

        if is_positive:
            cls = random.choice(self.class_list)
            img1, img2 = random.sample(self.class_to_images[cls], 2)
            label = 1.0
        else:
            cls1, cls2 = random.sample(self.class_list, 2)
            img1 = random.choice(self.class_to_images[cls1])
            img2 = random.choice(self.class_to_images[cls2])
            label = 0.0

        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
