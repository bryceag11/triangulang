"""Segmentation losses: focal, dice, boundary, Lovasz, point-sampled, align, contrastive."""

import torch
import torch.nn.functional as F
import numpy as np


def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    prob = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
    pt = torch.where(target > 0.5, prob, 1 - prob)
    return (torch.where(target > 0.5, alpha, 1 - alpha) * (1 - pt) ** gamma * ce).mean()


def dice_loss(pred, target, smooth=1.0):
    prob = torch.sigmoid(pred)
    prob_flat = prob.view(prob.shape[0], -1)
    target_flat = target.view(target.shape[0], -1).float()
    intersection = (prob_flat * target_flat).sum(dim=1)
    union = prob_flat.sum(dim=1) + target_flat.sum(dim=1)
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()


def centroid_loss(pred_centroid, gt_centroid):
    """Smooth L1 loss for 3D centroid prediction."""
    return F.smooth_l1_loss(pred_centroid, gt_centroid)


def boundary_loss(pred, target):
    """Distance-transform-based boundary loss (Kervadec et al., MIDL 2019).

    Penalizes predictions that are far from the true boundary.
    Complementary to dice/focal which focus on region overlap.
    """
    from scipy.ndimage import distance_transform_edt
    target_np = target.detach().cpu().numpy()
    dist_maps = []
    for i in range(target_np.shape[0]):
        posmask = target_np[i] > 0.5
        if posmask.sum() == 0 or (~posmask).sum() == 0:
            dist_maps.append(np.zeros_like(target_np[i]))
            continue
        pos_dist = distance_transform_edt(posmask)
        neg_dist = distance_transform_edt(~posmask)
        # Signed distance: negative inside, positive outside
        dist_map = neg_dist * (~posmask).astype(float) - pos_dist * posmask.astype(float)
        # Normalize to [-1, 1] range
        max_val = max(abs(dist_map.min()), abs(dist_map.max()), 1.0)
        dist_map = dist_map / max_val
        dist_maps.append(dist_map)
    dist_map_t = torch.tensor(np.stack(dist_maps), device=pred.device, dtype=pred.dtype)
    return (torch.sigmoid(pred) * dist_map_t).mean()


def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_loss(pred, target):
    """Lovász-Hinge loss for binary segmentation (Berman et al., CVPR 2018).

    Directly optimizes IoU. Known to produce sharper boundaries than dice.
    Operates on logits (not sigmoid).
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1).float()

    signs = 2. * target_flat - 1.
    errors = 1. - pred_flat * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = target_flat[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad) / max(target_flat.sum().item(), 1.0)


def lovasz_loss(pred, target):
    """Batch Lovász-Hinge loss. Averages per-sample Lovász losses."""
    losses = []
    for i in range(pred.shape[0]):
        losses.append(lovasz_hinge_loss(pred[i], target[i]))
    return torch.stack(losses).mean()


def point_sampled_loss(pred, target, focal_fn, dice_fn,
                       focal_weight=2.0, dice_weight=5.0,
                       focal_alpha=0.25, focal_gamma=2.0,
                       num_points=4096, oversample_ratio=3.0,
                       importance_ratio=0.75):
    """Compute focal+dice loss on sampled uncertain points (SAM3-style).

    Samples points near the prediction boundary where sigmoid ~ 0.5,
    focusing gradients on boundary pixels for sharper masks.
    """
    B = pred.shape[0]
    total_loss = 0.0

    for i in range(B):
        p = pred[i:i+1]  # [1, H, W]
        t = target[i:i+1].float()  # [1, H, W]

        # Sample uncertain points
        with torch.no_grad():
            # Oversample random points
            num_sampled = int(num_points * oversample_ratio)
            point_coords = torch.rand(1, num_sampled, 2, device=pred.device)

            # Sample predictions at these points
            point_logits = F.grid_sample(
                p.unsqueeze(1), point_coords.unsqueeze(1) * 2 - 1,
                align_corners=False, mode='bilinear'
            ).squeeze(1).squeeze(1)  # [1, num_sampled]

            # Uncertainty = closeness to 0 (sigmoid ~ 0.5)
            uncertainty = -torch.abs(point_logits)

            # Keep most uncertain + some random
            num_uncertain = int(importance_ratio * num_points)
            num_random = num_points - num_uncertain

            _, idx = uncertainty.topk(num_uncertain, dim=1)
            uncertain_coords = torch.gather(point_coords, 1,
                                           idx.unsqueeze(-1).expand(-1, -1, 2))

            random_coords = torch.rand(1, num_random, 2, device=pred.device)
            selected_coords = torch.cat([uncertain_coords, random_coords], dim=1)

            # Sample GT at selected points
            sampled_gt = F.grid_sample(
                t.unsqueeze(1), selected_coords.unsqueeze(1) * 2 - 1,
                align_corners=False, mode='nearest'
            ).squeeze(1).squeeze(1)  # [1, num_points]

        # Sample pred at selected points (with gradients)
        sampled_pred = F.grid_sample(
            p.unsqueeze(1), selected_coords.unsqueeze(1) * 2 - 1,
            align_corners=False, mode='bilinear'
        ).squeeze(1).squeeze(1)  # [1, num_points]

        # Compute loss on sampled points
        sample_loss = (focal_weight * focal_fn(sampled_pred, sampled_gt,
                                                alpha=focal_alpha, gamma=focal_gamma) +
                       dice_weight * dice_fn(sampled_pred, sampled_gt))
        total_loss = total_loss + sample_loss

    return total_loss / B


def align_loss(logits, actual_ious, alpha=0.5, gamma=2.0, tau=2.0, eps=1e-6):
    """AlignDETR-style IoU-aware focal loss.

    Instead of training a separate IoU head, this modulates the classification
    loss so the logits directly learn to predict mask quality.

    L_align = -t_c * (1-p)^gamma * log(p) - (1-t_c) * p^gamma * log(1-p)
    where t_c = e^(-r/tau) * q, and q = p^alpha * u^(1-alpha)

    Args:
        logits: [B, Q] raw logits (before sigmoid)
        actual_ious: [B, Q] actual IoU of each mask with GT
        alpha: Weight for predicted prob vs actual IoU in quality score (default 0.5)
        gamma: Focal loss gamma - focuses on hard examples (default 2.0)
        tau: Temperature for rank-based weighting (higher = more uniform, default 2.0)
        eps: Small epsilon for numerical stability

    Returns:
        Scalar loss
    """
    # Check for NaN in inputs - return 0 loss CONNECTED TO GRAPH for DDP sync
    if torch.isnan(logits).any() or torch.isnan(actual_ious).any():
        return logits.sum() * 0.0

    # Clamp logits to prevent extreme sigmoid values
    logits_safe = logits.clamp(-15, 15)
    p = torch.sigmoid(logits_safe)
    p = p.clamp(eps, 1 - eps)

    u = actual_ious.detach().clamp(eps, 1 - eps)

    # Quality score: q = p^alpha * u^(1-alpha)
    q = torch.pow(p, alpha) * torch.pow(u, 1 - alpha)
    q = q.clamp(eps, 1 - eps)

    # Rank-based weighting
    ranks = torch.argsort(torch.argsort(-u, dim=1), dim=1).float()
    rank_weight = torch.exp(-ranks / tau)

    # Soft target: t_c = e^(-r/tau) * q
    t_c = (rank_weight * q).clamp(eps, 1 - eps)

    # Actual Align loss formula (autocast safe using logsigmoid):
    # L_align = -t_c * (1-p)^gamma * log(p) - (1-t_c) * p^gamma * log(1-p)
    #
    # Use F.logsigmoid for numerical stability:
    # log(p) = log(sigmoid(x)) = logsigmoid(x)
    # log(1-p) = log(sigmoid(-x)) = logsigmoid(-x)
    log_p = F.logsigmoid(logits_safe)      # log(p), stable
    log_1mp = F.logsigmoid(-logits_safe)   # log(1-p), stable

    # Focal weights inside the loss terms (this is what makes it Align loss)
    # Positive term: focuses on hard positives (low p)
    pos_focal = torch.pow(1 - p, gamma)
    # Negative term: focuses on hard negatives (high p)
    neg_focal = torch.pow(p, gamma)

    # Align loss
    pos_loss = -t_c * pos_focal * log_p
    neg_loss = -(1 - t_c) * neg_focal * log_1mp

    loss = (pos_loss + neg_loss).mean()

    # Final safety check - return 0 connected to graph
    if torch.isnan(loss) or torch.isinf(loss):
        return logits.sum() * 0.0

    return loss


def contrastive_mask_loss(scores, best_idx, margin: float = 0.5):
    """Push best mask's score above others by margin.

    This helps the model learn to rank the correct mask highest,
    enabling score-based selection at inference time.

    Args:
        scores: [B, Q] per-mask scores (IoU predictions or confidence)
        best_idx: [B] index of best mask (by actual IoU with GT)
        margin: minimum gap between best and others
    """
    B, Q = scores.shape
    device = scores.device
    batch_idx = torch.arange(B, device=device)

    best_scores = scores[batch_idx, best_idx]  # [B]

    # Margin ranking: best > other + margin
    # Create mask to exclude best from "others"
    mask = torch.ones(B, Q, device=device, dtype=torch.bool)
    mask[batch_idx, best_idx] = False

    other_scores = scores[mask].view(B, Q - 1)  # [B, Q-1]

    # Hinge loss: max(0, other - best + margin)
    loss = F.relu(other_scores - best_scores.unsqueeze(1) + margin)
    return loss.mean()


def segmentation_loss(pred, target, focal_weight=20.0, dice_weight=1.0, focal_alpha=0.75, focal_gamma=2.0):
    """Combined focal + dice loss."""
    return focal_weight * focal_loss(pred, target, focal_alpha, focal_gamma) + dice_weight * dice_loss(pred, target)
