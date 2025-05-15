import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletMultiViewWeightedDataset(Dataset):
    def __init__(self, metadata_path, transform=None, anchor_mode="single", label_file=None, dataset_type="veri", image_root="data/VeRi/image_train"):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.transform = transform
        self.anchor_mode = anchor_mode
        self.image_root = image_root

        self.vehicle_to_images = {}
        for fname, meta in self.metadata.items():
            if meta.get("split") != "train":
                continue
            vid = meta["vehicleID"]
            if vid not in self.vehicle_to_images:
                self.vehicle_to_images[vid] = []
            self.vehicle_to_images[vid].append((fname, meta))

        self.vehicle_ids = list(self.vehicle_to_images.keys())

    def __len__(self):
        return 100000

    def load_views(self, entries, num_views=3):
        selected = random.sample(entries, min(num_views, len(entries)))
        images = [self.transform(Image.open(os.path.join(self.image_root, fname)).convert("RGB")) for fname, _ in selected]
        return images

    def __getitem__(self, idx):
        anchor_vid = random.choice(self.vehicle_ids)
        pos_entries = self.vehicle_to_images[anchor_vid]
        neg_vid = random.choice(self.vehicle_ids)
        while neg_vid == anchor_vid:
            neg_vid = random.choice(self.vehicle_ids)
        neg_entries = self.vehicle_to_images[neg_vid]

        pos_views = self.load_views(pos_entries)
        neg_views = self.load_views(neg_entries)

        if self.anchor_mode == "single":
            anchor_img = self.transform(Image.open(os.path.join(self.image_root, pos_entries[0][0])).convert("RGB"))
            anchor_views = [anchor_img]
        else:
            anchor_views = self.load_views(pos_entries)

        anchor_meta = pos_entries[0][1]
        neg_meta = neg_entries[0][1]
        anchor_color = anchor_meta["color_label"]
        anchor_type = anchor_meta["type"]
        neg_color = neg_meta["color_label"]
        neg_type = neg_meta["type"]

        return anchor_views, pos_views, neg_views, [anchor_color], [neg_color], [anchor_type], [neg_type]
