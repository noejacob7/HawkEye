# clip_text_encoder.py

import torch
import clip
from get_clip import load_clip_model

class CLIPTextEncoder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, _ = load_clip_model(device)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a single string into a normalized CLIP embedding.

        Args:
            text (str): The input text description

        Returns:
            torch.Tensor: Normalized text embedding (1 x 512)
        """
        with torch.no_grad():
            tokenized = clip.tokenize([text]).to(self.device)
            embedding = self.model.encode_text(tokenized)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu()
