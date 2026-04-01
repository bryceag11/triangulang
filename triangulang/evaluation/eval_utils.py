"""Metric computation and evaluation utilities."""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from triangulang.utils.spatial_reasoning import get_mask_centroid, get_depth_at_centroid


def create_prompts_from_gt(
    gt_mask: torch.Tensor,
    prompt_type: str,
    num_pos_points: int = 10,
    num_neg_points: int = 2,
    device: str = 'cuda',
) -> Dict:
    """Generate point/box prompts from GT mask for evaluation.

    Args:
        gt_mask: [H, W] binary ground truth mask
        prompt_type: One of 'text_only', 'text_and_points', 'text_and_box',
                     'text_and_points_and_box', 'points_only', 'box_only'
        num_pos_points: Number of positive points to sample
        num_neg_points: Number of negative points to sample
        device: Target device

    Returns:
        Dict with keys: point_prompts, point_labels, box_prompts, box_labels, use_text
    """
    result = {
        'point_prompts': None,
        'point_labels': None,
        'box_prompts': None,
        'box_labels': None,
        'use_text': True,
    }

    gt_binary = (gt_mask > 0.5).float()
    H, W = gt_binary.shape

    # Points
    if 'points' in prompt_type:
        fg_coords = (gt_binary > 0.5).nonzero(as_tuple=False)  # [N, 2] (y, x)
        bg_coords = (gt_binary < 0.5).nonzero(as_tuple=False)

        points = []
        labels = []

        if fg_coords.shape[0] > 0 and num_pos_points > 0:
            idx = torch.randperm(fg_coords.shape[0])[:num_pos_points]
            pos_pts = fg_coords[idx]  # [k, 2] in (y, x)
            # Convert to (x, y) normalized [0, 1]
            pos_xy = torch.stack([pos_pts[:, 1].float() / W, pos_pts[:, 0].float() / H], dim=-1)
            points.append(pos_xy)
            labels.append(torch.ones(pos_xy.shape[0], device=device))

        if bg_coords.shape[0] > 0 and num_neg_points > 0:
            idx = torch.randperm(bg_coords.shape[0])[:num_neg_points]
            neg_pts = bg_coords[idx]
            neg_xy = torch.stack([neg_pts[:, 1].float() / W, neg_pts[:, 0].float() / H], dim=-1)
            points.append(neg_xy)
            labels.append(torch.zeros(neg_xy.shape[0], device=device))

        if points:
            result['point_prompts'] = torch.cat(points, dim=0).unsqueeze(0).to(device)  # [1, P, 2]
            result['point_labels'] = torch.cat(labels, dim=0).unsqueeze(0).to(device)    # [1, P]

    # Box
    if 'box' in prompt_type:
        fg_coords = (gt_binary > 0.5).nonzero(as_tuple=False)  # [N, 2] (y, x)
        if fg_coords.shape[0] > 0:
            y_min, x_min = fg_coords.min(dim=0).values
            y_max, x_max = fg_coords.max(dim=0).values
            # Normalize to [0, 1] in (x1, y1, x2, y2) format
            box = torch.tensor([[
                x_min.float() / W, y_min.float() / H,
                x_max.float() / W, y_max.float() / H
            ]], device=device)
            result['box_prompts'] = box.unsqueeze(0)  # [1, 1, 4]
            result['box_labels'] = torch.ones(1, 1, device=device)

    # Text usage
    if prompt_type in ('points_only', 'box_only', 'points_and_box'):
        result['use_text'] = False

    return result


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute IoU and accuracy metrics.

    Returns dict with: iou, pixel_acc, recall, precision, f1, tp, fp, fn, tn.
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    gt_binary = (gt > 0.5).float()

    tp = (pred_binary * gt_binary).sum()
    fp = (pred_binary * (1 - gt_binary)).sum()
    fn = ((1 - pred_binary) * gt_binary).sum()
    tn = ((1 - pred_binary) * (1 - gt_binary)).sum()
    total_pixels = tp + fp + fn + tn

    union = tp + fp + fn
    iou = (tp / union).item() if union > 0 else 1.0
    pixel_acc = ((tp + tn) / total_pixels).item() if total_pixels > 0 else 1.0
    recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 1.0
    precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'iou': iou, 'pixel_acc': pixel_acc, 'recall': recall,
        'precision': precision, 'f1': f1,
        'tp': tp.item(), 'fp': fp.item(), 'fn': fn.item(), 'tn': tn.item(),
    }


def compute_oracle_iou(all_masks: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute oracle IoU by selecting the mask with best match to GT.

    Args:
        all_masks: [B, Q, H, W] or [Q, H, W] - all mask predictions
        gt: [H, W] or [B, H, W] - ground truth mask

    Returns:
        dict with oracle_iou and best_mask_idx
    """
    if all_masks.dim() == 4:
        all_masks = all_masks[0]
    if gt.dim() == 3:
        gt = gt[0]

    if all_masks.shape[-2:] != gt.shape[-2:]:
        all_masks = F.interpolate(all_masks.unsqueeze(0), size=gt.shape[-2:],
                                  mode='bilinear', align_corners=False).squeeze(0)

    gt_binary = (gt > 0.5).float()
    Q = all_masks.shape[0]

    best_iou = 0.0
    best_idx = 0

    for q in range(Q):
        pred_binary = (torch.sigmoid(all_masks[q]) > threshold).float()
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = (intersection / union).item() if union > 0 else 1.0
        if iou > best_iou:
            best_iou = iou
            best_idx = q

    return {'oracle_iou': best_iou, 'best_mask_idx': best_idx}


def compute_3d_centroid(mask: torch.Tensor, pointmaps: torch.Tensor) -> Optional[torch.Tensor]:
    """Compute 3D centroid from mask and pointmaps.

    Args:
        mask: [H, W] binary mask or logits
        pointmaps: [H, W, 3] world coordinates

    Returns:
        [3] centroid in world coordinates, or None if mask is empty
    """
    if mask.shape != pointmaps.shape[:2]:
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=pointmaps.shape[:2], mode='nearest'
        ).squeeze()

    mask_binary = mask > 0.5
    if mask_binary.sum() < 10:
        return None

    masked_points = pointmaps[mask_binary]
    return masked_points.mean(dim=0)


def compute_centroid_error(pred_centroid: torch.Tensor, gt_centroid: torch.Tensor) -> float:
    """Compute Euclidean distance between predicted and GT centroids in meters."""
    return torch.norm(pred_centroid - gt_centroid).item()


def umeyama_alignment(src_points: np.ndarray, dst_points: np.ndarray,
                      with_scale: bool = True, allow_reflection: bool = True):
    """Umeyama alignment: find optimal rotation, translation, and scale
    that aligns src_points to dst_points (Procrustes).

    Args:
        src_points: [N, 3] source points (e.g. DA3 estimated camera positions)
        dst_points: [N, 3] destination points (e.g. GT camera positions)
        with_scale: whether to estimate scale (7-DoF)
        allow_reflection: if True, allow reflections (det(R) < 0)

    Returns:
        R: [3, 3] rotation/reflection matrix
        t: [3] translation vector
        s: scale factor
    """
    assert src_points.shape == dst_points.shape
    n, dim = src_points.shape

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)

    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    H = src_centered.T @ dst_centered / n
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    d = np.linalg.det(R)
    if not allow_reflection and d < 0:
        Vt[-1, :] *= -1
        S_copy = S.copy()
        S_copy[-1] *= -1
        R = Vt.T @ U.T
        S = S_copy

    if with_scale:
        src_var = (src_centered ** 2).sum() / n
        s = S.sum() / src_var if src_var > 1e-8 else 1.0
    else:
        s = 1.0

    t = dst_mean - s * R @ src_mean
    return R, t, s


def compute_cross_view_consistency(
    pred_masks: List[torch.Tensor],
    pointmaps: torch.Tensor,
    threshold: float = 0.05,
    subsample: int = 1024,
) -> Dict:
    """Compute cross-view consistency: do corresponding 3D points get same prediction?

    Args:
        pred_masks: List of [H, W] predicted masks (logits or probs)
        pointmaps: [N, H, W, 3] world coordinates for all views
        threshold: Distance threshold in meters for "same 3D point"
        subsample: Number of points to sample per view for efficiency

    Returns:
        dict with consistency metrics
    """
    N = len(pred_masks)
    if N < 2:
        return {'consistency': 1.0, 'num_correspondences': 0}

    device = pointmaps.device
    H, W = pred_masks[0].shape[-2:]

    probs = []
    for m in pred_masks:
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear').squeeze()
        probs.append(torch.sigmoid(m) if m.min() < 0 else m)

    if pointmaps.shape[1:3] != (H, W):
        pts = pointmaps.permute(0, 3, 1, 2)
        pts = F.interpolate(pts, size=(H, W), mode='bilinear', align_corners=False)
        pointmaps = pts.permute(0, 2, 3, 1)

    total_agreements = 0
    total_correspondences = 0

    for i in range(N):
        for j in range(i + 1, N):
            pts_i = pointmaps[i].reshape(-1, 3)
            pts_j = pointmaps[j].reshape(-1, 3)
            prob_i = probs[i].reshape(-1)
            prob_j = probs[j].reshape(-1)

            valid_i = pts_i[:, 2] > 0.01
            valid_j = pts_j[:, 2] > 0.01

            pts_i_valid = pts_i[valid_i]
            pts_j_valid = pts_j[valid_j]
            prob_i_valid = prob_i[valid_i]
            prob_j_valid = prob_j[valid_j]

            if pts_i_valid.shape[0] < 10 or pts_j_valid.shape[0] < 10:
                continue

            if pts_i_valid.shape[0] > subsample:
                idx = torch.randperm(pts_i_valid.shape[0], device=device)[:subsample]
                pts_i_valid = pts_i_valid[idx]
                prob_i_valid = prob_i_valid[idx]

            dists = torch.cdist(pts_i_valid, pts_j_valid)
            min_dists, min_indices = dists.min(dim=-1)

            valid_corresp = min_dists < threshold
            if valid_corresp.sum() < 5:
                continue

            pred_i = (prob_i_valid[valid_corresp] > 0.5).float()
            pred_j = (prob_j_valid[min_indices[valid_corresp]] > 0.5).float()

            agreements = (pred_i == pred_j).sum().item()
            correspondences = valid_corresp.sum().item()

            total_agreements += agreements
            total_correspondences += correspondences

    consistency = total_agreements / max(total_correspondences, 1)
    return {
        'consistency': consistency,
        'num_correspondences': total_correspondences,
        'num_agreements': total_agreements,
    }


def compute_spatial_gt(
    masks: List[np.ndarray],
    depth: np.ndarray,
) -> Dict[str, int]:
    """Compute which mask index corresponds to each spatial qualifier.

    Args:
        masks: List of masks for same label [H, W] each
        depth: Depth map [H, W]

    Returns:
        Dict mapping qualifier -> mask index (e.g., {'nearest': 0, 'leftmost': 2})
    """
    if len(masks) <= 1:
        return {}

    centroids = []
    depths_at_centroid = []
    for m in masks:
        cx, cy = get_mask_centroid(m)
        centroids.append((cx, cy))
        depths_at_centroid.append(get_depth_at_centroid(m, depth))

    result = {}

    result['nearest'] = int(np.argmin(depths_at_centroid))
    result['closest'] = result['nearest']
    result['farthest'] = int(np.argmax(depths_at_centroid))

    all_x = [c[0] for c in centroids]
    result['leftmost'] = int(np.argmin(all_x))
    result['left'] = result['leftmost']
    result['rightmost'] = int(np.argmax(all_x))
    result['right'] = result['rightmost']

    all_y = [c[1] for c in centroids]
    result['topmost'] = int(np.argmin(all_y))
    result['top'] = result['topmost']
    result['bottommost'] = int(np.argmax(all_y))
    result['bottom'] = result['bottommost']

    return result
