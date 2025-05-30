import os
import time
import random
import threading
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import imageio
from detection.yolo_wrapper import YOLOv11Wrapper
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet

# === Setup ===
VIDEO_DIR = "data/test"
QUERY_DIR = "data/query"
CHECKPOINT = "LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOP_K = 3

# Load all query images
query_images = sorted([f for f in os.listdir(QUERY_DIR) if f.endswith(".jpeg") or f.endswith(".jpg")])
video_ids = [f"test_00{i}.mp4" for i in range(1, 5)]
NUM_UAVS = min(len(query_images), len(video_ids))

# === Load Models ===
model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

yolo = YOLOv11Wrapper(conf_threshold=0.3)
shared_embeddings = {}
thread_logs = {}
lock = threading.Lock()

def extract_fused_embedding(video_path, log, name):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    step = int(fps * 0.1)
    crops = []

    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        boxes, _, _ = yolo.predict(frame)
        if not boxes:
            continue
        x, y, w, h = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[0]
        img = Image.fromarray(frame).crop((x, y, x + w, y + h))
        crops.append(transform(img).to(DEVICE))  # Ensure crops are on GPU

    reader.close()

    if len(crops) >= 2:
        with torch.no_grad():
            emb = model(crops)  # Keep on DEVICE
            fused = model.pool(emb).unsqueeze(0)  # DEVICE match
            log.append(f"[{name}] Extracted and fused {len(crops)} crops.")
            return fused / fused.norm(dim=-1, keepdim=True)
    log.append(f"[{name}] No valid crops found.")
    return None


# === UAV Thread Function ===
def uav_thread(name, video_file, query_img_path):
    log = []
    log.append(f"[{name}] Assigned to video {video_file} for query {query_img_path}")
    log.append(f"[{name}] Traveling to location...")
    time.sleep(1)

    emb = extract_fused_embedding(os.path.join(VIDEO_DIR, video_file), log, name)
    if emb is not None:
        with lock:
            shared_embeddings[name] = {"embedding": emb, "query": query_img_path}
        log.append(f"[{name}] Shared embedding with others.")
    thread_logs[name] = log

# === Assign Videos and Queries ===
assigned_pairs = list(zip(query_images[:NUM_UAVS], video_ids[:NUM_UAVS]))
random.shuffle(assigned_pairs)
uav_names = [f"UAV_{i+1}" for i in range(NUM_UAVS)]

# === Launch UAV Threads ===
start_time = time.time()
threads = []
for name, (query_img, video_file) in zip(uav_names, assigned_pairs):
    t = threading.Thread(target=uav_thread, args=(name, video_file, os.path.join(QUERY_DIR, query_img)))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
total_time = time.time() - start_time

# === Compare Query to Shared Embeddings ===
results = []
for name in uav_names:
    info = shared_embeddings.get(name)
    if info:
        query_img = transform(Image.open(info["query"]).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            query_emb = model([query_img.squeeze(0)]).cpu()
        sim = F.cosine_similarity(query_emb.to(DEVICE), info["embedding"].to(DEVICE)).item()
        results.append((name, os.path.basename(info["query"]), assigned_pairs[uav_names.index(name)][1], sim))

results.sort(key=lambda x: x[3], reverse=True)

# === Print Results ===
df = pd.DataFrame(results, columns=["UAV", "Query Image", "Video", "Cosine Similarity"])
df["Rank"] = df["Cosine Similarity"].rank(ascending=False).astype(int)
df["Time Taken (s)"] = f"{total_time:.2f}"

for name in uav_names:
    if name in thread_logs:
        df.loc[df["UAV"] == name, "Logs"] = "\n".join(thread_logs[name])

print(df.to_string(index=False))
