# triplet_dataset.py
# ✅ TripletDataset for LightMatchNet from lightmatch_data structure

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.class_to_imgs = {}
        self.classes = sorted(os.listdir(root_dir))

        for cls in self.classes:
            folder = os.path.join(root_dir, cls)
            imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
            if len(imgs) >= 2:
                self.class_to_imgs[cls] = imgs

        self.class_list = list(self.class_to_imgs.keys())

    def __len__(self):
        return 100000  # infinite-style sampling

    def __getitem__(self, idx):
        anchor_class = random.choice(self.class_list)
        negative_class = random.choice(self.class_list)
        while negative_class == anchor_class:
            negative_class = random.choice(self.class_list)

        anchor_img, positive_img = random.sample(self.class_to_imgs[anchor_class], 2)
        negative_img = random.choice(self.class_to_imgs[negative_class])

        anchor = self.transform(Image.open(anchor_img).convert('RGB'))
        positive = self.transform(Image.open(positive_img).convert('RGB'))
        negative = self.transform(Image.open(negative_img).convert('RGB'))

        return anchor, positive, negative
