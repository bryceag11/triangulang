import torch


def hungarian_match(pred_masks, gt_masks, num_objects, text_scores=None):
    """Match Q predicted masks to K GT masks using Hungarian algorithm.

    Args:
        pred_masks: [Q, H, W] predicted masks (logits)
        gt_masks: [K, H, W] ground truth masks
        num_objects: K (actual number of objects, may be < gt_masks.shape[0] if padded)
        text_scores: [Q, K] optional per-text scores for cost weighting

    Returns:
        matched_pairs: List[(query_idx, gt_idx)], K pairs
        unmatched_queries: List[int], Q-K unmatched query indices
    """
    from scipy.optimize import linear_sum_assignment

    Q = pred_masks.shape[0]
    K = num_objects
    device = pred_masks.device

    # Compute IoU cost matrix [Q, K]
    pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()  # [Q, H, W]
    cost_matrix = torch.zeros(Q, K, device=device)
    for k in range(K):
        gt_k = (gt_masks[k] > 0.5).float()  # [H, W]
        intersection = (pred_binary * gt_k.unsqueeze(0)).sum(dim=(-2, -1))  # [Q]
        union = pred_binary.sum(dim=(-2, -1)) + gt_k.sum() - intersection
        ious = intersection / union.clamp(min=1.0)  # [Q]
        cost_matrix[:, k] = -ious  # Negative IoU as cost

    # Optional: add text score cost (encourage query-text alignment)
    if text_scores is not None and text_scores.shape[-1] >= K:
        text_cost = -text_scores[:, :K].sigmoid()  # [Q, K]
        cost_matrix = cost_matrix + 0.5 * text_cost

    # Hungarian matching (fast on CPU for typical Q=50, K=10)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    matched_pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
    unmatched = [i for i in range(Q) if i not in set(row_ind.tolist())]
    return matched_pairs, unmatched


def text_greedy_match(text_scores, num_objects):
    """Assign queries to texts using text_scores (greedy, stable matching).

    Unlike Hungarian matching which uses IoU (changes every step),
    text-based matching is more stable because text_scores depend on
    the learned scoring head rather than rapidly-changing mask predictions.

    Args:
        text_scores: [Q, K] text-query alignment scores
        num_objects: K (actual number of objects)

    Returns:
        matched_pairs: List[(query_idx, gt_idx)], K pairs
        unmatched_queries: List[int], Q-K unmatched query indices
    """
    Q = text_scores.shape[0]
    K = num_objects
    scores = text_scores[:, :K].sigmoid().detach()  # [Q, K]

    matched_pairs = []
    used_queries = set()

    # Sort texts by their max score (assign highest-confidence texts first)
    text_max_scores = scores.max(dim=0).values  # [K]
    text_order = text_max_scores.argsort(descending=True)

    for k_idx in text_order.tolist():
        # Find best available query for this text
        text_k_scores = scores[:, k_idx].clone()
        for q in used_queries:
            text_k_scores[q] = -1.0
        best_q = text_k_scores.argmax().item()
        matched_pairs.append((best_q, k_idx))
        used_queries.add(best_q)

    unmatched = [q for q in range(Q) if q not in used_queries]
    return matched_pairs, unmatched
