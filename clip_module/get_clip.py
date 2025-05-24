import torch
import clip

# Load CLIP model (ViT-B/32 is a common lightweight choice)
def load_clip_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def get_clip_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    return load_clip_model(device)

# Encode a list of texts into CLIP embeddings
def encode_texts(texts, model, device):
    with torch.no_grad():
        tokenized = clip.tokenize(texts).to(device)
        text_embeddings = model.encode_text(tokenized)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings.cpu()

if __name__ == "__main__":
    model, _, device = load_clip_model()
    prompts = ["red sedan", "white SUV with roof rack", "black sports car"]
    embeddings = encode_texts(prompts, model, device)
    print("Text embeddings shape:", embeddings.shape)
