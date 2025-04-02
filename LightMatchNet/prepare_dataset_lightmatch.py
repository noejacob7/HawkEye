# prepare_dataset_lightmatch.py
# ✅ Final working version for KaggleHub TorchVision-compatible Stanford Cars dataset

import os
import shutil
import scipy.io
from tqdm import tqdm

# === CONFIGURATION ===
ROOT_DIR = "stanford_cars"
SOURCE_IMG_DIR = os.path.join(ROOT_DIR, "cars_train")
ANNOTATION_FILE = os.path.join(ROOT_DIR, "devkit", "cars_train_annos.mat")
CLASS_META_FILE = os.path.join(ROOT_DIR, "devkit", "cars_meta.mat")
OUTPUT_DIR = "lightmatch_data"
USE_CLASS_NAMES = False  # True = folder names like '2012_Tesla_Model_S', False = 'id_001'

# === LOAD CLASS NAMES ===
print("[INFO] Loading class names...")
meta = scipy.io.loadmat(CLASS_META_FILE)
class_names_raw = meta['class_names'][0]
class_id_to_name = {
    i + 1: class_names_raw[i][0].replace(" ", "_") for i in range(len(class_names_raw))
}

# === LOAD ANNOTATIONS ===
print("[INFO] Parsing training annotations...")
annos = scipy.io.loadmat(ANNOTATION_FILE)['annotations'][0]  # flat structure
class_id_to_images = {}

for entry in annos:
    img_name = entry[5][0]         # e.g., '00001.jpg'
    class_id = int(entry[4][0])    # class label (1-196)

    if class_id not in class_id_to_images:
        class_id_to_images[class_id] = []
    class_id_to_images[class_id].append(img_name)

# === BUILD DATASET ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Creating folders and copying {sum(len(v) for v in class_id_to_images.values())} images...")

missing_count = 0

for class_id, img_list in tqdm(class_id_to_images.items()):
    folder_name = (
        class_id_to_name[class_id] if USE_CLASS_NAMES else f"id_{str(class_id).zfill(3)}"
    )
    class_folder = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(class_folder, exist_ok=True)

    for img in img_list:
        src = os.path.join(SOURCE_IMG_DIR, img)
        dst = os.path.join(class_folder, img)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"[WARNING] Missing file: {src}")
            missing_count += 1

print("[✅] Dataset ready in:", OUTPUT_DIR)
print(f"[INFO] Total classes: {len(class_id_to_images)}")
print(f"[INFO] Missing images skipped: {missing_count}")
