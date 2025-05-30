import os
import torch
import imageio
import numpy as np
from PIL import Image
from torchvision import transforms
from detection.yolo_wrapper import YOLOv11Wrapper
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
import torch.nn.functional as F

# === Setup ===
VIDEO_DIR = "data/test"
QUERY_IMAGE = "data/query/test_0011.jpeg"
OUTPUT_DIR = "output/fused"
CHECKPOINT = "LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"
FPS_INTERVAL = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model = model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

yolo = YOLOv11Wrapper(conf_threshold=0.3)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Extract cropped vehicle tensors ===
def extract_crops(video_path, interval_s=0.1):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    step = int(fps * interval_s)
    crops = []

    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        image = Image.fromarray(frame)
        boxes, _, _ = yolo.predict(np.array(image))
        if not boxes:
            continue
        x, y, w, h = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[0]
        crop = image.crop((x, y, x + w, y + h))
        crops.append(transform(crop).to(device))
    reader.close()
    return crops

# === Fuse Embeddings ===
def fuse_embeddings(images):
    with torch.no_grad():
        stacked = torch.stack(images).to(device)
        emb = model(stacked)
        fused = model.pool(emb).unsqueeze(0)
        return fused / fused.norm(dim=-1, keepdim=True)

# === Process Videos ===
gallery_embeddings = {}
for i in range(1, 5):
    video_path = os.path.join(VIDEO_DIR, f"test_00{i}.mp4")
    print(f"[INFO] Processing {video_path}...")
    crops = extract_crops(video_path, interval_s=FPS_INTERVAL)
    if len(crops) >= 2:
        gallery_embeddings[f"video_{i}"] = fuse_embeddings(crops)

# === Embed Query ===
query_tensor = transform(Image.open(QUERY_IMAGE).convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
    query_emb = model(query_tensor)
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

# === Similarity Matching ===
results = []
for vid, gal_emb in gallery_embeddings.items():
    sim = F.cosine_similarity(query_emb, gal_emb.to(device)).item()
    results.append((vid, sim))

results.sort(key=lambda x: x[1], reverse=True)

# === Output Table ===
print("\n[✓] Similarity Results:")
print(f"{'Video':<10} | {'Cosine Similarity':>18}")
print("-" * 32)
for vid, score in results:
    print(f"{vid:<10} | {score:>18.4f}")
