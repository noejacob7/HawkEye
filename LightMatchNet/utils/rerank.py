import numpy as np
import torch

def re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3):
    """
    Re-ranking algorithm using k-reciprocal encoding from CVPR 2017.
    Inputs:
    - qf: Tensor of shape [num_query, feat_dim]
    - gf: Tensor of shape [num_gallery, feat_dim]
    Output:
    - re-ranked distance matrix of shape [num_query, num_gallery]
    """
    qf = qf.numpy()
    gf = gf.numpy()
    all_features = np.vstack((qf, gf))
    all_num = all_features.shape[0]
    query_num = qf.shape[0]

    # Compute original distance
    dist = np.power(all_features, 2).sum(axis=1, keepdims=True) + \
           np.power(all_features, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(all_features, all_features.T)
    dist = np.sqrt(np.maximum(dist, 0))
    original_dist = dist / np.max(dist)
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1)

    for i in range(all_num):
        forward_k = initial_rank[i, :k1 + 1]
        backward_k = initial_rank[forward_k, :k1 + 1]
        reciprocal = forward_k[np.any(backward_k == i, axis=1)]
        reciprocal_exp = reciprocal.copy()
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, :int(k1 / 2) + 1]
            candidate_back = initial_rank[candidate_forward, :int(k1 / 2) + 1]
            if len(set(reciprocal) & set(candidate_forward[np.any(candidate_back == candidate, axis=1)])) > 2/3 * len(candidate_forward):
                reciprocal_exp = np.union1d(reciprocal_exp, candidate_forward)
        weight = np.exp(-original_dist[i, reciprocal_exp])
        V[i, reciprocal_exp] = weight / np.sum(weight)

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = V[initial_rank[i, :k2]].mean(axis=0)
        V = V_qe

    jaccard = np.zeros((query_num, gf.shape[0]), dtype=np.float32)
    for i in range(query_num):
        temp_min = np.minimum(V[i, :], V[query_num:, :]).sum(axis=1)
        jaccard[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard * (1 - lambda_value) + original_dist[:query_num, query_num:] * lambda_value
    return final_dist
