# clip_module/T2I_VeRi/dataset.py

import os
import json
from torch.utils.data import Dataset
from PIL import Image

class T2IVeRiTextImageDataset(Dataset):
    """
    A Text–Image dataset for T2I-VeRi.  
    JSON entries look like:
      {
        "split": "train"|"test"|"query",
        "captions": [ "...text..." , … ],
        "file_path": "image/xxxx.jpg",
        "id": <vehicleID>
      }
    """

    def __init__(
        self,
        data_json: str,
        image_root: str,
        split: str = "train",
        transform=None,
        tokenizer=None,
        context_length: int = 77,
    ):
        # load and filter by split
        with open(data_json, "r") as f:
            all_entries = json.load(f)
        self.entries = [e for e in all_entries if e.get("split") == split]

        self.image_root    = image_root
        self.transform     = transform
        self.tokenizer     = tokenizer
        self.context_length = context_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        # 1) load image
        img_path = os.path.join(self.image_root, entry["file_path"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # 2) pick the first caption
        caption = entry["captions"][0]

        # 3) tokenize if we have a tokenizer (e.g. clip.tokenize), else return raw string
        if self.tokenizer:
            # clip.tokenize returns a tensor of shape [1, context_length]
            tokens = self.tokenizer([caption], context_length=self.context_length)[0]
        else:
            tokens = caption

        # 4) the vehicle id
        vid = entry.get("id")

        return img, tokens, vid
