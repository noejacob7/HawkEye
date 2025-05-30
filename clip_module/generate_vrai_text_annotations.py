import pickle
import os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("print current working directory:", os.getcwd())

# Path to the annotation file
annotation_path = "data/VRAI/train_annotation.pkl"

# Load annotations
with open(annotation_path, "rb") as f:
    annotations = pickle.load(f)

# Label mappings
color_map = {
    1: "white", 2: "black", 3: "gray", 4: "red",
    5: "green", 6: "blue", 7: "yellow", 8: "brown"
}
type_map = {
    1: "sedan", 2: "hatchback", 3: "SUV", 4: "bus",
    5: "lorry", 6: "truck"
}

def describe_attr(val, text):
    return text if val == 1 else None

samples = []



for im_name in annotations["train_im_names"]:
    id_key = int(im_name.split("_")[0])


    color = color_map.get(annotations["color_label"].get(id_key), "vehicle")
    vehicle_type = type_map.get(annotations["type_label"].get(id_key), "vehicle")


    extras = [
        describe_attr(annotations["bumper_label"].get(id_key, 0), "with bumper"),
        describe_attr(annotations["wheel_label"].get(id_key, 0), "visible wheels"),
        describe_attr(annotations["sky_label"].get(id_key, 0), "under open sky"),
        describe_attr(annotations["luggage_label"].get(id_key, 0), "with luggage rack"),
    ]
    extras = [e for e in extras if e]

    caption = f"{color} {vehicle_type}" + (", " + ", ".join(extras) if extras else "")

    samples.append({
        "image_name": im_name,
        "caption": caption
    })

# Save to CSV
output_path = "clip_module/T2I_VeRi/generated_vrai_text_annotations.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pd.DataFrame(samples).to_csv(output_path, index=False)

print(f"✅ Generated {len(samples)} captions.")
print(f"📄 Saved to {output_path}")
