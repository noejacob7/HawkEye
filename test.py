import os
import pickle
from collections import defaultdict

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def check_vrai_structure(vrai_root="data/VRAI"):
    # Load annotations
    test_anno = load_pkl(os.path.join(vrai_root, "test_annotation.pkl"))
    dev_anno  = load_pkl(os.path.join(vrai_root, "test_dev_annotation.pkl"))

    test_names = set(test_anno["test_im_names"])
    dev_names  = set(dev_anno["dev_im_names"])
    all_names  = test_names | dev_names

    print(f"✅ Total test images: {len(test_names)}")
    print(f"✅ Total dev images:  {len(dev_names)}")
    print(f"✅ Overlap:           {len(test_names & dev_names)} (should match dev set)")

    # Check where dev images physically exist
    dev_folder = os.path.join(vrai_root, "images_dev")
    train_folder = os.path.join(vrai_root, "images_train")

    print("\n🔍 Checking where dev images exist:")
    in_dev = []
    in_train = []
    missing = []

    for name in dev_names:
        if os.path.exists(os.path.join(dev_folder, name)):
            in_dev.append(name)
        elif os.path.exists(os.path.join(train_folder, name)):
            in_train.append(name)
        else:
            missing.append(name)

    print(f"🟩 Found in images_dev:    {len(in_dev)}")
    print(f"🟨 Found in images_train:  {len(in_train)}")
    print(f"🟥 Missing:                {len(missing)}")

    # Optionally: Print counts of images per ID
    vid_to_count = defaultdict(int)
    for name in test_names:
        vid = name.split("_")[0]
        vid_to_count[vid] += 1

    multi_view_ids = [vid for vid, count in vid_to_count.items() if count > 1]
    print(f"\n🚘 Vehicle IDs with ≥2 test images: {len(multi_view_ids)}")

if __name__ == "__main__":
    check_vrai_structure("data/VRAI")
