import os
from PIL import Image
from torch.utils.data import Dataset

class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, view_suffix="_01"):
        """
        root_dir: Directory with subfolders for each object ID.
        transform: Transform to apply to each image (e.g., resize, normalize).
        view_suffix: View version to load (e.g., '_01' for training, '_02' for testing).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.view_suffix = view_suffix
        self.samples = []

        for obj_id in sorted(os.listdir(root_dir)):
            obj_path = os.path.join(root_dir, obj_id)
            if not os.path.isdir(obj_path):
                continue

            views = []
            for view in ["front", "back", "left", "right"]:
                filename = f"{view}{view_suffix}.jpg"
                fpath = os.path.join(obj_path, filename)
                if os.path.exists(fpath):
                    views.append(fpath)

            if len(views) >= 2:
                self.samples.append(views)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        view_paths = self.samples[idx]
        images = []

        for img_path in view_paths:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        return images  # list of tensors
