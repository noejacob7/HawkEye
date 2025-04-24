import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, view_suffix="_01"):
        """
        root_dir: path to directory with object folders (e.g., id_001, id_002)
        transform: torchvision transforms
        view_suffix: used to select view set (_01 for training, _02 for testing)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.view_suffix = view_suffix
        self.view_types = ["front", "back", "left", "right"]

        self.class_to_views = {}
        self.classes = sorted(os.listdir(root_dir))

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            views = []
            for view_type in self.view_types:
                file_name = f"{view_type}{self.view_suffix}.jpg"
                file_path = os.path.join(cls_path, file_name)
                if os.path.exists(file_path):
                    views.append(file_path)

            if len(views) >= 2:
                self.class_to_views[cls] = views

        self.class_list = list(self.class_to_views.keys())

    def __len__(self):
        return 100000  # for infinite sampling

    def load_views(self, paths):
        return [self.transform(Image.open(p).convert('RGB')) for p in paths]

    def __getitem__(self, idx):
        anchor_class = random.choice(self.class_list)
        negative_class = random.choice(self.class_list)
        while negative_class == anchor_class:
            negative_class = random.choice(self.class_list)

        anchor_paths = self.class_to_views[anchor_class]
        positive_paths = self.class_to_views[anchor_class]
        negative_paths = self.class_to_views[negative_class]

        anchor_views = self.load_views(anchor_paths)
        positive_views = self.load_views(positive_paths)
        negative_views = self.load_views(negative_paths)

        return anchor_views, positive_views, negative_views
