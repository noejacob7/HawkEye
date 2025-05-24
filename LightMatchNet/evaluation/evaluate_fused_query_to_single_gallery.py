#!/usr/bin/env python3
import os
import sys
import argparse
import csv
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.multiview_matchnet import MultiViewMatchNet


def parse_label_xml(xml_path):
    """Parse VeRi XML and return a dict: imageName → metadata dict."""
    import xml.etree.ElementTree as ET
    with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
        root = ET.fromstring(f.read())
    return {
        item.attrib['imageName']: {
            'vehicleID': item.attrib['vehicleID'],
            'cameraID' : item.attrib['cameraID'],
            'colorID'  : item.attrib['colorID'],
            'typeID'   : item.attrib['typeID'],
        }
        for item in root.findall(".//Item")
    }


def load_query_list(path):
    """Load list of query filenames (one per line)."""
    with open(path, 'r') as f:
        return set(line.strip() for line in f)


def get_transform():
    """Standard ImageNet preprocess for 224×224 crops."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_fused_embedding(model, image_paths, transform, device):
    """
    Given a list of image file paths, load + preprocess each,
    pass them into model(list_of_tensors) and return the
    fused embedding (CPU tensor) or None if no valid images.
    """
    tensors = []
    for p in image_paths:
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img).to(device, non_blocking=True))
    if not tensors:
        return None

    with torch.no_grad():
        # model should fuse internally and return a list/tuple
        fused = model(tensors)[0]
    return fused.cpu()


def compute_ap(ranked_files, match_fn):
    """
    Compute Average Precision for one query.
    ranked_files: gallery filenames sorted by decreasing sim.
    match_fn(fname) → bool
    """
    ap = 0.0
    hits = 0
    for i, fname in enumerate(ranked_files, start=1):
        if match_fn(fname):
            hits += 1
            ap += hits / i
    return ap / hits if hits > 0 else 0.0


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load network
    model = MultiViewMatchNet(backbone=args.model,
                              embedding_dim=args.embedding_dim)
    model.load_state_dict(torch.load(args.checkpoint,
                                     map_location=device))
    model.to(device).eval()

    transform      = get_transform()
    query_meta     = parse_label_xml(args.query_label)
    gallery_meta   = parse_label_xml(args.gallery_label)
    query_set      = load_query_list(args.query_list)

    # group query images by vehicleID
    query_groups = defaultdict(list)
    for fname, m in query_meta.items():
        if fname in query_set:
            query_groups[m['vehicleID']].append(fname)

    # gallery excludes query images
    gallery_files = [f for f in gallery_meta if f not in query_set]

    # metrics accumulators
    pairwise_rows = []
    ap_values     = []
    cmc_cutoffs   = [1, 5, 10]
    cmc_hits      = {k: 0 for k in cmc_cutoffs}
    total_queries = len(query_groups)

    for qvid, qfiles in tqdm(query_groups.items(), desc="Queries"):
        # fuse all query views inside the model
        qpaths = [os.path.join(args.query_dir, f) for f in qfiles]
        fused_q = get_fused_embedding(model, qpaths, transform, device)
        if fused_q is None:
            continue  # no valid query images

        # compute similarity to each gallery image
        sims = []
        for gf in gallery_files:
            gpath = os.path.join(args.gallery_dir, gf)
            fused_g = get_fused_embedding(model, [gpath], transform, device)
            if fused_g is None:
                continue
            sim = F.cosine_similarity(
                fused_q.unsqueeze(0),
                fused_g.unsqueeze(0),
                dim=1
            ).item()
            sims.append((gf, gallery_meta[gf]['vehicleID'], sim))

        # rank by similarity desc
        sims.sort(key=lambda x: x[2], reverse=True)
        ranked_files = [gf for gf, _, _ in sims]

        # define match function
        if args.soft_match:
            # color + type must match
            qm = query_meta[qfiles[0]]
            def match_fn(fname):
                gm = gallery_meta[fname]
                return (qm['colorID'] == gm['colorID'] and
                        qm['typeID']  == gm['typeID'])
        else:
            def match_fn(fname):
                return gallery_meta[fname]['vehicleID'] == qvid

        # CMC hits
        for k in cmc_cutoffs:
            if any(match_fn(f) for f in ranked_files[:k]):
                cmc_hits[k] += 1

        # mAP
        ap = compute_ap(ranked_files, match_fn)
        ap_values.append(ap)

        # record each pairwise score
        for gf, gid, sim in sims:
            pairwise_rows.append({
                'query_vid'    : qvid,
                'gallery_image': gf,
                'gallery_vid'  : gid,
                'similarity'   : sim
            })

    # prepare output paths
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pairs_csv   = args.output_csv.replace('.csv', '_pairs.csv')
    metrics_csv = args.output_csv.replace('.csv', '_metrics.csv')

    # write pairwise CSV
    with open(pairs_csv, 'w', newline='') as f:
        w = csv.DictWriter(f,
            fieldnames=['query_vid','gallery_image','gallery_vid','similarity'])
        w.writeheader()
        w.writerows(pairwise_rows)

    # write metrics CSV
    with open(metrics_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','value'])
        w.writerow(['mAP', f"{sum(ap_values)/total_queries:.4f}"])
        for k in cmc_cutoffs:
            w.writerow([f"CMC@{k}", f"{cmc_hits[k]/total_queries:.4f}"])

    print(f"Pairwise results saved to:   {pairs_csv}")
    print(f"Summary metrics saved to:    {metrics_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fused-query re-identification")
    parser.add_argument("--checkpoint",    required=True,
                        help="path to model .pt")
    parser.add_argument("--query_dir",     required=True)
    parser.add_argument("--query_label",   required=True,
                        help="XML file for query set")
    parser.add_argument("--gallery_dir",   required=True)
    parser.add_argument("--gallery_label", required=True,
                        help="XML file for gallery set")
    parser.add_argument("--query_list",    required=True,
                        help="text file listing query image names")
    parser.add_argument("--output_csv",    required=True,
                        help="base path for output CSVs (will append _pairs.csv and _metrics.csv)")
    parser.add_argument("--model",         default="swifttracknet",
                        help="backbone name for MultiViewMatchNet")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--soft_match",    action="store_true",
                        help="use color+type soft match instead of exact vehicleID")
    args = parser.parse_args()
    evaluate(args)