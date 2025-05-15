import os
import xml.etree.ElementTree as ET
import json

# Paths (adjust if needed)
VERI_ROOT = os.path.join("data", "VeRi")
TRAIN_LABEL = os.path.join(VERI_ROOT, "train_label.xml")
TEST_LABEL = os.path.join(VERI_ROOT, "test_label.xml")
LIST_COLOR = os.path.join(VERI_ROOT, "list_color.txt")
LIST_TYPE = os.path.join(VERI_ROOT, "list_type.txt")
OUTPUT_JSON = os.path.join(VERI_ROOT, "veri_all_metadata.json")

# CameraID → View mapping (can be extended)
CAMERA_TO_VIEW = {
    "c001": ["front"],
    "c002": ["rear"],
    "c003": ["left"],
    "c004": ["right"],
    # You can update based on test_track.txt
}

# Approx RGBs
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


def parse_label_map(filepath):
    label_map = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                label_map[int(parts[0])] = parts[1].lower()
    return label_map


def parse_xml_to_metadata(xml_path, color_map, type_map, split):
    metadata = {}
    with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
        xml_content = f.read()
    root = ET.fromstring(xml_content)
    for item in root.findall(".//Item"):
        fname = item.attrib["imageName"]
        vehicle_id = item.attrib["vehicleID"]
        camera_id = item.attrib["cameraID"]
        color_id = int(item.attrib["colorID"])
        type_id = int(item.attrib["typeID"])

        color_label = color_map.get(color_id, "unknown")
        type_label = type_map.get(type_id, "unknown")
        rgb = [v / 255.0 for v in COLOR_RGB.get(color_label, (0, 0, 0))]
        views = CAMERA_TO_VIEW.get(camera_id, [])

        metadata[fname] = {
            "vehicleID": vehicle_id,
            "cameraID": camera_id,
            "color_label": color_label,
            "color": rgb,
            "type": type_label,
            "views": views,
            "dataset": "veri",
            "split": split
        }

    return metadata



def build_veri_metadata_all():
    print("[INFO] Parsing color and type lists...")
    color_map = parse_label_map(LIST_COLOR)
    type_map = parse_label_map(LIST_TYPE)

    print("[INFO] Parsing train and test XMLs...")
    train_metadata = parse_xml_to_metadata(TRAIN_LABEL, color_map, type_map, split="train")
    test_metadata = parse_xml_to_metadata(TEST_LABEL, color_map, type_map, split="test")

    combined = {**train_metadata, **test_metadata}
    print(f"[✓] Combined metadata for {len(combined)} images.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(combined, f, indent=2)
        print(f"[✓] Written to: {OUTPUT_JSON}")


if __name__ == "__main__":
    build_veri_metadata_all()
