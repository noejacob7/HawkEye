import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, anchor_mode="single", view_suffix_train="_01", view_suffix_anchor="_02"):
        """
        root_dir: folder with subfolders (id_001, id_002, ...)
        transform: torchvision transform pipeline
        anchor_mode: 'multi' or 'single' — anchor is fused set or single image
        view_suffix_train: suffix to identify training views (e.g., '_01')
        view_suffix_anchor: suffix to identify anchor views (e.g., '_02')
        """
        self.root_dir = root_dir
        self.transform = transform
        self.anchor_mode = anchor_mode
        self.train_suffix = view_suffix_train
        self.anchor_suffix = view_suffix_anchor
        self.view_types = ["front", "back", "left", "right"]

        self.class_to_train_views = {}
        self.class_to_anchor_images = {}
        self.classes = sorted(os.listdir(root_dir))

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            # Load *_01 training views
            train_views = []
            for view_type in self.view_types:
                path = os.path.join(cls_path, f"{view_type}{self.train_suffix}.jpg")
                if os.path.exists(path):
                    train_views.append(path)
            if len(train_views) >= 2:
                self.class_to_train_views[cls] = train_views

            # Load *_02 anchor reference image
            anchor_imgs = [
                os.path.join(cls_path, f) for f in os.listdir(cls_path)
                if f.endswith(f"{self.anchor_suffix}.jpg")
            ]
            if anchor_imgs:
                self.class_to_anchor_images[cls] = anchor_imgs

        self.class_list = [cls for cls in self.class_to_train_views if cls in self.class_to_anchor_images]

    def __len__(self):
        return 100000

    def load_views(self, paths):
        return [self.transform(Image.open(p).convert("RGB")) for p in paths]

    def __getitem__(self, idx):
        anchor_class = random.choice(self.class_list)
        negative_class = random.choice(self.class_list)
        while negative_class == anchor_class:
            negative_class = random.choice(self.class_list)

        positive_paths = self.class_to_train_views[anchor_class]
        negative_paths = self.class_to_train_views[negative_class]

        positive_views = self.load_views(positive_paths)
        negative_views = self.load_views(negative_paths)

        if self.anchor_mode == "single":
            anchor_path = random.choice(self.class_to_anchor_images[anchor_class])
            anchor_image = self.transform(Image.open(anchor_path).convert("RGB"))
            return anchor_image, positive_views, negative_views
        else:
            anchor_views = self.load_views(positive_paths)
            return anchor_views, positive_views, negative_views
