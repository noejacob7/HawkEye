import os, sys
import argparse
import pickle
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from LightMatchNet.models.multiview_matchnet import MultiViewMatchNet


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def get_image_embedding(model, img_path, transform, device):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model([tensor])[0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0)


def compute_cmc_map(query_embs, query_ids, gallery_embs, gallery_ids, topk=(1,5,10), max_rank=20):
    num_q = query_embs.size(0)
    cmc_counts = {r: 0 for r in range(1, max_rank+1)}
    ap_sum = 0.0

    gallery_norm = gallery_embs / gallery_embs.norm(dim=1, keepdim=True)

    for i in range(num_q):
        q_emb = query_embs[i]
        qid = query_ids[i]
        sims = torch.matmul(gallery_norm, q_emb)
        sorted_idxs = torch.argsort(sims, descending=True)
        sorted_gids = [gallery_ids[idx] for idx in sorted_idxs]

        # CMC
        found = False
        for rank, gid in enumerate(sorted_gids, start=1):
            if gid == qid:
                for r in range(rank, max_rank+1):
                    cmc_counts[r] += 1
                found = True
                break
        # AP
        hits, ap = 0, 0.0
        for j, gid in enumerate(sorted_gids, start=1):
            if gid == qid:
                hits += 1
                ap += hits / j
        if hits > 0:
            ap /= hits
        ap_sum += ap

    cmc_full = {r: cmc_counts[r] / num_q for r in cmc_counts}
    cmc_topk = {r: cmc_full[r] for r in topk if r in cmc_full}
    mAP = ap_sum / num_q
    return cmc_topk, cmc_full, mAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrai_root", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--dev_pkl", type=str, default="test_dev_annotation.pkl")
    parser.add_argument("--test_pkl", type=str, default="test_annotation.pkl")
    parser.add_argument("--topk", nargs='+', type=int, default=[1,5,10])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()

    # load model
    model = MultiViewMatchNet(backbone="swifttracknet", embedding_dim=128).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.eval()

    # load annotations
    dev_anno = load_pkl(os.path.join(args.vrai_root, args.dev_pkl))
    test_anno = load_pkl(os.path.join(args.vrai_root, args.test_pkl))
    dev_im_names = set(dev_anno.get("dev_im_names", []))
    all_im_names = test_anno.get("test_im_names", [])

    # fuse gallery embeddings
    gallery_groups = defaultdict(list)
    for name in all_im_names:
        if name not in dev_im_names:
            vid = name.split('_')[0]
            gallery_groups[vid].append(name)

    gallery_embs_list, gallery_ids = [], []
    for vid, files in tqdm(gallery_groups.items(), desc="Fusing gallery embeddings"):
        embs = []
        for fname in files:
            path = os.path.join(args.vrai_root, "images_test", fname)
            if os.path.exists(path):
                embs.append(get_image_embedding(model, path, transform, device))
        if embs:
            stacked = torch.stack(embs)
            pooled = stacked.mean(dim=0)
            normed = pooled / pooled.norm()
            gallery_embs_list.append(normed)
            gallery_ids.append(vid)
    gallery_embs = torch.stack(gallery_embs_list).to(device)

    # encode dev queries
    query_embs_list, query_ids = [], []
    for fname in tqdm(dev_im_names, desc="Encoding dev queries"):
        path = os.path.join(args.vrai_root, "images_dev", fname)
        if os.path.exists(path):
            emb = get_image_embedding(model, path, transform, device)
            query_embs_list.append(emb)
            query_ids.append(fname.split('_')[0])
    query_embs = torch.stack(query_embs_list).to(device)

    # compute metrics
    cmc_topk, cmc_full, mAP = compute_cmc_map(
        query_embs=query_embs,
        query_ids=query_ids,
        gallery_embs=gallery_embs,
        gallery_ids=gallery_ids,
        topk=args.topk
    )

    # print results
    print("\n✅ Evaluation Results:")
    print("Top-k CMC:")
    for k, v in cmc_topk.items():
        print(f"CMC@{k}: {v:.4f}")
    print("\nFull CMC Curve:")
    for k in sorted(cmc_full.keys()):
        print(f"CMC@{k}: {cmc_full[k]:.4f}")
    print(f"mAP: {mAP:.4f}")

    # write output CSV with full curve
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w') as f:
        f.write('metric,value\n')
        for k in sorted(cmc_full.keys()):
            f.write(f"CMC@{k},{cmc_full[k]:.4f}\n")
        f.write(f"mAP,{mAP:.4f}\n")

if __name__ == "__main__":
    main()
