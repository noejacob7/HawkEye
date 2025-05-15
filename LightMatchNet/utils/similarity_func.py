import json
import numpy as np
import os

# Output directory
output_dir = os.path.join("data", "VeRi")
os.makedirs(output_dir, exist_ok=True)

# 1. Define vehicle type list
types = [
    "sedan", "hatchback", "suv", "van", "mpv", "pickup", "bus", "truck", "estate"
]

# 2. Define handcrafted similarity scores based on shape/design proximity
SIMILARITY_MATRIX = {
    ("sedan", "hatchback"): 0.85,
    ("sedan", "estate"): 0.8,
    ("sedan", "suv"): 0.6,
    ("hatchback", "estate"): 0.9,
    ("suv", "van"): 0.7,
    ("suv", "pickup"): 0.75,
    ("van", "mpv"): 0.8,
    ("mpv", "estate"): 0.65,
    ("pickup", "truck"): 0.85,
    ("truck", "bus"): 0.6,
    ("van", "bus"): 0.7,
    ("hatchback", "mpv"): 0.6,
    ("suv", "truck"): 0.5,
    ("pickup", "mpv"): 0.4,
    ("sedan", "truck"): 0.3,
    ("sedan", "bus"): 0.2,
    ("hatchback", "truck"): 0.2,
    ("van", "truck"): 0.5,
    ("estate", "truck"): 0.4
}

# 3. Make the matrix symmetric and fill diagonals with 1.0
type_similarity = {}
for t1 in types:
    for t2 in types:
        if t1 == t2:
            sim = 1.0
        else:
            sim = SIMILARITY_MATRIX.get((t1, t2), SIMILARITY_MATRIX.get((t2, t1), 0.0))
        type_similarity[f"{t1}|{t2}"] = sim

# 4. Define color labels and RGB values
COLOR_RGB = {
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "green": (0, 128, 0),
    "gray": (128, 128, 128),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "golden": (218, 165, 32),
    "brown": (139, 69, 19),
    "black": (0, 0, 0)
}

# 5. Compute color similarity based on normalized inverse Euclidean distance
labels = list(COLOR_RGB.keys())
vectors = np.array([np.array(COLOR_RGB[color]) / 255.0 for color in labels])
color_similarity = {}

for i, c1 in enumerate(labels):
    for j, c2 in enumerate(labels):
        dist = np.linalg.norm(vectors[i] - vectors[j])
        sim = 1 - (dist / np.sqrt(3))  # normalize to [0, 1]
        color_similarity[f"{c1}|{c2}"] = sim

# 6. Save both as JSON files
output_type_path = os.path.join(output_dir, "type_similarity_matrix.json")
output_color_path = os.path.join(output_dir, "color_similarity_matrix.json")

with open(output_type_path, "w") as f:
    json.dump(type_similarity, f, indent=2)

with open(output_color_path, "w") as f:
    json.dump(color_similarity, f, indent=2)

print(f"[✓] Saved type similarity matrix to {output_type_path}")
print(f"[✓] Saved color similarity matrix to {output_color_path}")
