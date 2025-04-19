# triplet_miner.py
import torch
import torch.nn.functional as F

def batch_hard_triplet_loss(anchor, positive, negative, margin=0.2):
    """Standard triplet loss with batch-hard mining"""
    dist_ap = F.pairwise_distance(anchor, positive, p=2)
    dist_an = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(dist_ap - dist_an + margin).mean()
    return loss
