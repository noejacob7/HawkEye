import os
import pygame
import threading
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import imageio
from detection.yolo_wrapper import YOLOv11Wrapper
from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet

# === Constants ===
VIDEO_DIR = "data/test"
QUERY_DIR = "data/query"
CHECKPOINT = "LightMatchNet/checkpoints/swifttracknet_multiview_v3.pt"
DRONE_IMAGE = "assets/drone_icon.png"
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FPS = 60

# === Model Setup ===
model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

yolo = YOLOv11Wrapper(conf_threshold=0.3)
video_ids = [f"test_00{i}.mp4" for i in range(1, 5)]
query_ids = [f"test_00{i}.jpeg" for i in [1,3]]
query_ids.extend([f"test_00{i}.jpg" for i in [2,4]])

# === Shared Data ===
shared_results = []
lock = threading.Lock()

# === Extract fused embedding ===
def extract_fused_embedding(video_path):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    step = int(fps * 0.2)
    crops = []

    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        boxes, _, _ = yolo.predict(frame)
        if not boxes:
            continue
        x, y, w, h = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[0]
        img = Image.fromarray(frame).crop((x, y, x + w, y + h))
        crops.append(transform(img).to(DEVICE))
        if len(crops) >= 5:
            break
    reader.close()

    if len(crops) >= 2:
        with torch.no_grad():
            emb = model(crops).to(DEVICE)
            fused = model.pool(emb).unsqueeze(0)
            return fused / fused.norm(dim=-1, keepdim=True), img
    return None, None

# === UAV Thread ===
def uav_worker(uav_name, video_path, query_path):
    time.sleep(2)
    fused_emb, crop_img = extract_fused_embedding(os.path.join(VIDEO_DIR, video_path))
    query_img = transform(Image.open(os.path.join(QUERY_DIR, query_path)).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_emb = model([query_img.squeeze(0)]).to(DEVICE)
        sim = F.cosine_similarity(query_emb, fused_emb).item()

    with lock:
        shared_results.append({"uav": uav_name, "video": video_path, "similarity": sim, "crop": crop_img})

# === Pygame Display ===
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Multi-UAV Collaboration Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Start threads
    threads = []
    for i in range(4):
        t = threading.Thread(target=uav_worker, args=(f"UAV_{i+1}", video_ids[i], query_ids[i]))
        t.start()
        threads.append(t)

    drone_img = pygame.image.load(DRONE_IMAGE)
    drone_img = pygame.transform.scale(drone_img, (80, 80))
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        # Draw Drones and Crops
        for i, result in enumerate(shared_results):
            x, y = 100 + i * 250, 50
            screen.blit(drone_img, (x, y))
            pygame.draw.rect(screen, (0, 0, 0), (x, y + 90, 100, 100), 1)

            # Paste crop image if available
            if result["crop"] is not None:
                crop_resized = result["crop"].resize((100, 100))
                mode = crop_resized.mode
                size = crop_resized.size
                data = crop_resized.tobytes()
                crop_surface = pygame.image.fromstring(data, size, mode)
                screen.blit(crop_surface, (x, y + 90))

        # Draw Results Table
        table_x, table_y = 50, 250
        pygame.draw.line(screen, (0, 0, 0), (table_x, table_y), (table_x + 1000, table_y), 2)
        headers = ["UAV", "Video", "Similarity"]
        for j, header in enumerate(headers):
            text = font.render(header, True, (0, 0, 0))
            screen.blit(text, (table_x + j * 200, table_y + 10))

        for i, result in enumerate(shared_results):
            row_y = table_y + 40 + i * 30
            screen.blit(font.render(result["uav"], True, (0, 0, 0)), (table_x, row_y))
            screen.blit(font.render(result["video"], True, (0, 0, 0)), (table_x + 200, row_y))
            screen.blit(font.render(f"{result['similarity']:.4f}", True, (0, 0, 0)), (table_x + 400, row_y))

        pygame.display.flip()
        clock.tick(FPS)

    for t in threads:
        t.join()
    pygame.quit()

if __name__ == "__main__":
    main()
