# main_simulation.py
import pygame
import time
import random
import os
from simulation.drones import Drone
from simulation.arrows import draw_arrows
from simulation.utils import load_images, create_tables
from simulation.table_display import draw_table

# === Initialize Pygame ===
pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("HawkEye Multi-UAV Embedding Exchange")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20)

# === Load Assets ===
bg_color = (245, 245, 245)
drone_img = pygame.image.load("assets/drone.png")
query_imgs = load_images("assets/query")
crops_imgs = load_images("assets/crops")

# === Simulation Setup ===
uav_positions = [(150, 100), (500, 100), (850, 100), (1150, 100)]
drones = []
logs = {}
top_k_data = {}

for i in range(4):
    name = f"UAV_{i+1}"
    drone = Drone(name, uav_positions[i], drone_img, query_imgs[i], font)
    drones.append(drone)

# === Simulate Event Loop ===
running = True
frame = 0
crop_index = 0
wait_frames = 120

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(bg_color)

    # Draw arrows before drones communicate
    if frame == wait_frames:
        draw_arrows(screen, uav_positions)

    for i, drone in enumerate(drones):
        drone.update(frame)
        drone.draw(screen)

        # Show one crop image every 60 frames (~1 second)
        if frame >= wait_frames and crop_index < len(crops_imgs):
            if frame % 60 == 0:
                drone.show_crop(crops_imgs[crop_index])

        # Show Top-K table for each drone
        if frame == wait_frames + 240:  # ~4 seconds after arrows
            draw_table(screen, font, i, drone.name, drone.query_img, drone.top_k)

    # Advance frame
    if frame % 60 == 0 and crop_index < len(crops_imgs):
        crop_index += 1

    pygame.display.flip()
    frame += 1
    clock.tick(30)

pygame.quit()
