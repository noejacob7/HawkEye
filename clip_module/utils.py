# clip_module/utils.py
import os
import random
import torch
import numpy as np
from collections import defaultdict


def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, filename: str):
    """Save model + optimizer state to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load model weights (and return the checkpoint dict)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def compute_cmc_map(
    query_embs: torch.Tensor,
    query_ids: list,
    gallery_embs: torch.Tensor,
    gallery_ids: list,
    topk: list = (1, 5, 10),
):
    """
    Compute CMC and mAP for a set of query vs gallery embeddings.
    Returns dict with 'cmc' (dict of Top-k accuracies) and 'mAP'.
    """
    # cosine similarity
    sims = query_embs @ gallery_embs.t()  # Q x G
    # for each query
    all_ap = []
    cmc_counts = defaultdict(int)
    num_q = len(query_ids)

    for qi in range(num_q):
        qid = query_ids[qi]
        sim_row = sims[qi].cpu().detach().numpy()
        # sort gallery by descending sim
        idxs = np.argsort(-sim_row)
        ordered_gids = [gallery_ids[i] for i in idxs]

        # hit vector
        matches = np.array([1 if gid == qid else 0 for gid in ordered_gids])
        # CMC
        for k in topk:
            if matches[:k].any():
                cmc_counts[k] += 1

        # mAP
        # precision@i averaging over all correct positions
        cum_hits = np.cumsum(matches)
        precisions = [
            cum_hits[i] / (i + 1)
            for i, m in enumerate(matches)
            if m == 1
        ]
        ap = np.mean(precisions) if precisions else 0.0
        all_ap.append(ap)

    cmc = {f"Top-{k}": cmc_counts[k] / num_q for k in topk}
    mAP = float(np.mean(all_ap))
    return {"cmc": cmc, "mAP": mAP}
