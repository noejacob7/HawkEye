import os
import torch
import imageio
import numpy as np
from PIL import Image
from torchvision import transforms
from detection.yolo_wrapper import YOLOv11Wrapper
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet
import torch.nn.functional as F
from threading import Thread, Barrier, Lock

# === Config ===
VIDEO_DIR = "data/test"
QUERY_IMAGE = "data/query/test_0011.jpeg"
CHECKPOINT = "LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"
FPS_INTERVAL = 0.1
NUM_DRONES = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Shared State ===
gallery_embeddings = {}
lock = Lock()
barrier = Barrier(NUM_DRONES)

# === Model and Transform ===
model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

yolo = YOLOv11Wrapper(conf_threshold=0.3)

# === Functions ===
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

def fuse_embeddings(images):
    with torch.no_grad():
        stacked = torch.stack(images).to(device)
        emb = model(stacked)
        fused = model.pool(emb).unsqueeze(0)
        return fused / fused.norm(dim=-1, keepdim=True)

# === Threaded Drone Function ===
def drone_uav(video_index):
    video_path = os.path.join(VIDEO_DIR, f"test_00{video_index+1}.mp4")
    print(f"[UAV-{video_index+1}] Processing {video_path}...")
    crops = extract_crops(video_path, interval_s=FPS_INTERVAL)

    if len(crops) >= 2:
        fused_emb = fuse_embeddings(crops)
        with lock:
            gallery_embeddings[f"video_{video_index+1}"] = fused_emb

    barrier.wait()  # Sync all drones

# === Launch UAV Threads ===
threads = [Thread(target=drone_uav, args=(i,)) for i in range(NUM_DRONES)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# === Embed Query ===
query_tensor = transform(Image.open(QUERY_IMAGE).convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
    query_emb = model(query_tensor)
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

# === Final Comparison ===
results = []
for vid, gal_emb in gallery_embeddings.items():
    sim = F.cosine_similarity(query_emb, gal_emb.to(device)).item()
    results.append((vid, sim))
results.sort(key=lambda x: x[1], reverse=True)

# === Output Table ===
print("\n[✓] Multi-UAV Similarity Results:")
print(f"{'Video':<10} | {'Cosine Similarity':>18}")
print("-" * 32)
for vid, score in results:
    print(f"{vid:<10} | {score:>18.4f}")
