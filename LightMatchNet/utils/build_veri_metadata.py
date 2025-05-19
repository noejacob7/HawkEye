import os
import xml.etree.ElementTree as ET
import json

VERI_ROOT = os.path.join("data", "VeRi")
TRAIN_LABEL = os.path.join(VERI_ROOT, "train_label.xml")
TEST_LABEL = os.path.join(VERI_ROOT, "test_label.xml")
LIST_COLOR = os.path.join(VERI_ROOT, "list_color.txt")
LIST_TYPE = os.path.join(VERI_ROOT, "list_type.txt")
NAME_QUERY = os.path.join(VERI_ROOT, "name_query.txt")
NAME_TEST = os.path.join(VERI_ROOT, "name_test.txt")
OUTPUT_JSON = os.path.join(VERI_ROOT, "veri_all_metadata.json")

CAMERA_TO_VIEW = {
    "c001": ["front"],
    "c002": ["rear"],
    "c003": ["left"],
    "c004": ["right"],
}

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

def parse_xml_to_metadata(xml_path, color_map, type_map, split, valid_filenames=None):
    metadata = {}
    with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
        xml_content = f.read()
    root = ET.fromstring(xml_content)
    for item in root.findall(".//Item"):
        fname = item.attrib["imageName"]
        if valid_filenames and fname not in valid_filenames:
            continue
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

def read_list_file(filepath):
    with open(filepath, "r") as f:
        return set(line.strip() for line in f.readlines())

def build_veri_metadata_all():
    print("[INFO] Parsing color and type lists...")
    color_map = parse_label_map(LIST_COLOR)
    type_map = parse_label_map(LIST_TYPE)

    print("[INFO] Reading query/test lists...")
    query_files = read_list_file(NAME_QUERY)
    test_files = read_list_file(NAME_TEST)

    print("[INFO] Parsing train, query, and test splits...")
    train_metadata = parse_xml_to_metadata(TRAIN_LABEL, color_map, type_map, split="train")
    test_metadata = parse_xml_to_metadata(TEST_LABEL, color_map, type_map, split="test", valid_filenames=test_files)
    query_metadata = parse_xml_to_metadata(TEST_LABEL, color_map, type_map, split="query", valid_filenames=query_files)

    # Combine all: train + test, then overwrite with query if overlapping
    combined = train_metadata.copy()
    combined.update(test_metadata)
    combined.update(query_metadata)

    print(f"[✓] Combined metadata for {len(combined)} images.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(combined, f, indent=2)
        print(f"[✓] Written to: {OUTPUT_JSON}")

if __name__ == "__main__":
    build_veri_metadata_all()
