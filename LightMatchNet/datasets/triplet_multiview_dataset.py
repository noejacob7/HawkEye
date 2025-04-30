import os
import random
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from collections import defaultdict

class TripletMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, anchor_mode="single", label_file=None, dataset_type="hotwheels"):
        """
        root_dir: path to training images (flat for veri, folders for hotwheels)
        transform: torchvision transform pipeline
        anchor_mode: 'multi' or 'single'
        label_file: optional XML label file (required for veri)
        dataset_type: 'veri' or 'hotwheels'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.anchor_mode = anchor_mode
        self.dataset_type = dataset_type

        self.class_to_images = defaultdict(list)

        if dataset_type == "veri":
            if label_file is None:
                raise ValueError("label_file must be provided for veri dataset")
            label_map = self.parse_veri_label(label_file)
            for fname in os.listdir(root_dir):
                if fname.endswith(".jpg") and fname in label_map:
                    vid = label_map[fname]
                    self.class_to_images[vid].append(os.path.join(root_dir, fname))
        elif dataset_type == "hotwheels":
            for cls in os.listdir(root_dir):
                cls_path = os.path.join(root_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                for fname in os.listdir(cls_path):
                    if fname.endswith(".jpg"):
                        self.class_to_images[cls].append(os.path.join(cls_path, fname))
        else:
            raise ValueError("Unsupported dataset_type")

        self.class_list = [k for k in self.class_to_images if len(self.class_to_images[k]) >= 2]

    def parse_veri_label(self, label_path):
        with open(label_path, 'r', encoding='gb2312', errors='ignore') as f:
            xml_content = f.read()
        root = ET.fromstring(xml_content)
        label_map = {}
        for item in root.findall(".//Item"):
            label_map[item.attrib['imageName']] = item.attrib['vehicleID']
        return label_map

    def __len__(self):
        return 100000

    def load_views(self, paths):
        return [self.transform(Image.open(p).convert("RGB")) for p in paths]

    def __getitem__(self, idx):
        anchor_class = random.choice(self.class_list)
        negative_class = random.choice(self.class_list)
        while negative_class == anchor_class:
            negative_class = random.choice(self.class_list)

        pos_paths = self.class_to_images[anchor_class]
        neg_paths = self.class_to_images[negative_class]

        # randomly select 2+ positive views
        positive_views = self.load_views(random.sample(pos_paths, min(3, len(pos_paths))))
        negative_views = self.load_views(random.sample(neg_paths, min(3, len(neg_paths))))

        if self.anchor_mode == "single":
            anchor_img = self.transform(Image.open(random.choice(pos_paths)).convert("RGB"))
            return anchor_img, positive_views, negative_views
        else:
            anchor_views = self.load_views(random.sample(pos_paths, min(3, len(pos_paths))))
            return anchor_views, positive_views, negative_views
