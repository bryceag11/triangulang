"""Extracted helpers from TrianguLangModel: profiling, prompt utilities, mask selection, depth/pose, and multi-view forward."""
import math
import time
import random

import torch
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np

from sam3.model.geometry_encoders import Prompt
from sam3.model.data_misc import FindStage

def set_profile(self, enabled: bool):
    """Enable/disable profiling mode."""
    self.profile = enabled
    if enabled:
        self._profile_times = {}
        self._profile_counts = {}

def _profile_start(self):
    """Start profiling timer."""
    if self.profile:
        torch.cuda.synchronize()
        return time.time()
    return None

def _profile_end(self, name: str, start_time):
    """End profiling timer and accumulate."""
    if self.profile and start_time is not None:
        torch.cuda.synchronize()
        elapsed = (time.time() - start_time) * 1000  # ms
        if name not in self._profile_times:
            self._profile_times[name] = 0.0
            self._profile_counts[name] = 0
        self._profile_times[name] += elapsed
        self._profile_counts[name] += 1

def get_profile_summary(self) -> str:
    """Get profiling summary as formatted string."""
    if not self._profile_times:
        return "No profiling data"
    total = sum(self._profile_times.values())
    lines = ["Component Timing (avg ms, % of total):"]
    for name, t in sorted(self._profile_times.items(), key=lambda x: -x[1]):
        count = self._profile_counts.get(name, 1)
        avg = t / count if count > 0 else t
        pct = t / total * 100 if total > 0 else 0
        lines.append(f"  {name}: {avg:.1f}ms ({pct:.1f}%)")
    max_count = max(self._profile_counts.values()) if self._profile_counts else 1
    lines.append(f"  TOTAL: {total / max_count:.1f}ms per forward")
    return "\n".join(lines)

def mask_to_box(mask, jitter_ratio: float = 0.05, expand_ratio: float = 0.1):
    """Extract bounding box from mask in cxcywh format (normalized).

    Args:
        mask: [H, W] binary mask
        jitter_ratio: Random jitter for robustness
        expand_ratio: Expand box slightly

    Returns:
        box: [4] normalized box in cxcywh format
    """
    H, W = mask.shape
    device = mask.device

    if mask.sum() == 0:
        return torch.tensor([0.5, 0.5, 0.1, 0.1], device=device)

    # Find bounding box - fully vectorized, no .item() calls
    nonzero = torch.nonzero(mask)
    y_min = nonzero[:, 0].min()
    y_max = nonzero[:, 0].max()
    x_min = nonzero[:, 1].min()
    x_max = nonzero[:, 1].max()

    # Convert to cxcywh (keep as tensors)
    cx = (x_min + x_max).float() / 2 / W
    cy = (y_min + y_max).float() / 2 / H
    w = (x_max - x_min).float() / W
    h = (y_max - y_min).float() / H

    # Add jitter and expand (vectorized random)
    if jitter_ratio > 0:
        jitter = (torch.rand(2, device=device) - 0.5) * 2 * jitter_ratio
        cx = cx + jitter[0] * w
        cy = cy + jitter[1] * h
    if expand_ratio > 0:
        w = w * (1 + expand_ratio)
        h = h * (1 + expand_ratio)

    # Clamp to valid range (vectorized)
    cx = torch.clamp(cx, 0.0, 1.0)
    cy = torch.clamp(cy, 0.0, 1.0)
    w = torch.clamp(w, 0.01, 1.0)
    h = torch.clamp(h, 0.01, 1.0)

    return torch.stack([cx, cy, w, h])

def mask_to_box_batched(masks, jitter_ratio: float = 0.05, expand_ratio: float = 0.1):
    """Extract bounding boxes from batch of masks - fully vectorized.

    Args:
        masks: [B, H, W] binary masks
        jitter_ratio: Random jitter for robustness
        expand_ratio: Expand box slightly

    Returns:
        boxes: [B, 4] normalized boxes in cxcywh format
    """
    B, H, W = masks.shape
    device = masks.device

    # Find bounding boxes for all masks in parallel
    # For each mask, find min/max x and y where mask > 0

    # Create coordinate grids
    y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)

    # Mask out zeros (set to large/small values for min/max)
    mask_bool = masks > 0.5
    has_mask = mask_bool.any(dim=(1, 2))  # [B] - which masks are non-empty

    # For empty masks, we'll handle separately
    y_masked = torch.where(mask_bool, y_coords.float(), torch.tensor(float('inf'), device=device))
    x_masked = torch.where(mask_bool, x_coords.float(), torch.tensor(float('inf'), device=device))
    y_masked_max = torch.where(mask_bool, y_coords.float(), torch.tensor(float('-inf'), device=device))
    x_masked_max = torch.where(mask_bool, x_coords.float(), torch.tensor(float('-inf'), device=device))

    # Find min/max along spatial dimensions
    y_min = y_masked.view(B, -1).min(dim=1).values  # [B]
    y_max = y_masked_max.view(B, -1).max(dim=1).values  # [B]
    x_min = x_masked.view(B, -1).min(dim=1).values  # [B]
    x_max = x_masked_max.view(B, -1).max(dim=1).values  # [B]

    # Convert to cxcywh
    cx = (x_min + x_max) / 2 / W
    cy = (y_min + y_max) / 2 / H
    w = (x_max - x_min) / W
    h = (y_max - y_min) / H

    # Add jitter (batched)
    if jitter_ratio > 0:
        jitter = (torch.rand(B, 2, device=device) - 0.5) * 2 * jitter_ratio
        cx = cx + jitter[:, 0] * w
        cy = cy + jitter[:, 1] * h

    # Expand
    if expand_ratio > 0:
        w = w * (1 + expand_ratio)
        h = h * (1 + expand_ratio)

    # Clamp
    cx = torch.clamp(cx, 0.0, 1.0)
    cy = torch.clamp(cy, 0.0, 1.0)
    w = torch.clamp(w, 0.01, 1.0)
    h = torch.clamp(h, 0.01, 1.0)

    # Handle empty masks - set to default box
    default_box = torch.tensor([0.5, 0.5, 0.1, 0.1], device=device)
    boxes = torch.stack([cx, cy, w, h], dim=1)  # [B, 4]
    boxes = torch.where(has_mask.unsqueeze(1), boxes, default_box.unsqueeze(0).expand(B, 4))

    return boxes

def sample_points_from_mask_batched(masks, num_positive: int = 10, num_negative: int = 2, jitter_ratio: float = 0.02):
    """Sample click points from batch of masks - vectorized.

    Args:
        masks: [B, H, W] binary masks

    Returns:
        points: [B, N_points, 2] normalized points (x, y) in [0, 1]
        labels: [B, N_points] 1=positive, 0=negative
    """
    B, H, W = masks.shape
    device = masks.device
    num_points = num_positive + num_negative

    # Create coordinate grids normalized to [0, 1]
    y_coords = torch.arange(H, device=device).float().view(1, H, 1).expand(B, H, W) / H
    x_coords = torch.arange(W, device=device).float().view(1, 1, W).expand(B, H, W) / W

    all_points = []
    all_labels = []

    for b in range(B):
        mask = masks[b]
        fg_mask = mask > 0.5
        bg_mask = ~fg_mask

        # Sample positive points
        fg_indices = torch.nonzero(fg_mask, as_tuple=False)
        if len(fg_indices) > 0:
            sampled_idx = torch.randint(0, len(fg_indices), (num_positive,), device=device)
            fg_sampled = fg_indices[sampled_idx]
            pos_y = fg_sampled[:, 0].float() / H
            pos_x = fg_sampled[:, 1].float() / W
        else:
            pos_x = torch.full((num_positive,), 0.5, device=device)
            pos_y = torch.full((num_positive,), 0.5, device=device)

        # Sample negative points
        bg_indices = torch.nonzero(bg_mask, as_tuple=False)
        if len(bg_indices) > 0:
            sampled_idx = torch.randint(0, len(bg_indices), (num_negative,), device=device)
            bg_sampled = bg_indices[sampled_idx]
            neg_y = bg_sampled[:, 0].float() / H
            neg_x = bg_sampled[:, 1].float() / W
        else:
            corners = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]], device=device)
            neg_x = corners[torch.arange(num_negative, device=device) % 4, 0]
            neg_y = corners[torch.arange(num_negative, device=device) % 4, 1]

        # Combine and add jitter
        points_x = torch.cat([pos_x, neg_x])
        points_y = torch.cat([pos_y, neg_y])

        if jitter_ratio > 0:
            jitter = (torch.rand(num_points, 2, device=device) - 0.5) * 2 * jitter_ratio
            points_x = torch.clamp(points_x + jitter[:, 0], 0.0, 1.0)
            points_y = torch.clamp(points_y + jitter[:, 1], 0.0, 1.0)

        points = torch.stack([points_x, points_y], dim=1)  # [N_points, 2]
        labels = torch.cat([
            torch.ones(num_positive, device=device, dtype=torch.long),
            torch.zeros(num_negative, device=device, dtype=torch.long)
        ])

        all_points.append(points)
        all_labels.append(labels)

    return torch.stack(all_points), torch.stack(all_labels)

def sample_points_from_mask(mask, num_positive: int = 10, num_negative: int = 2, jitter_ratio: float = 0.02):
    """Sample click points from mask (MV-SAM style prompts).

    Args:
        mask: [H, W] binary mask
        num_positive: Number of positive points to sample from foreground
        num_negative: Number of negative points to sample from background
        jitter_ratio: Random jitter as fraction of image size

    Returns:
        points: [N_points, 2] normalized points (x, y) in [0, 1]
        labels: [N_points] 1=positive, 0=negative
    """
    H, W = mask.shape
    device = mask.device

    # Sample positive points from foreground - fully vectorized
    fg_coords = torch.nonzero(mask > 0.5)  # [N_fg, 2] (y, x)
    if len(fg_coords) > 0:
        indices = torch.randint(0, len(fg_coords), (num_positive,), device=device)
        sampled_fg = fg_coords[indices]  # [num_positive, 2]
        # Vectorized jitter
        jitter = (torch.rand(num_positive, 2, device=device) - 0.5) * 2 * jitter_ratio
        # Convert to normalized (x, y) format - note: nonzero gives (y, x)
        pos_points = torch.zeros(num_positive, 2, device=device)
        pos_points[:, 0] = torch.clamp(sampled_fg[:, 1].float() / W + jitter[:, 0], 0.0, 1.0)  # x
        pos_points[:, 1] = torch.clamp(sampled_fg[:, 0].float() / H + jitter[:, 1], 0.0, 1.0)  # y
    else:
        # No foreground, use center points
        pos_points = torch.full((num_positive, 2), 0.5, device=device)
    pos_labels = torch.ones(num_positive, device=device, dtype=torch.long)

    # Sample negative points from background - fully vectorized
    bg_coords = torch.nonzero(mask <= 0.5)  # [N_bg, 2] (y, x)
    if len(bg_coords) > 0:
        indices = torch.randint(0, len(bg_coords), (num_negative,), device=device)
        sampled_bg = bg_coords[indices]
        jitter = (torch.rand(num_negative, 2, device=device) - 0.5) * 2 * jitter_ratio
        neg_points = torch.zeros(num_negative, 2, device=device)
        neg_points[:, 0] = torch.clamp(sampled_bg[:, 1].float() / W + jitter[:, 0], 0.0, 1.0)  # x
        neg_points[:, 1] = torch.clamp(sampled_bg[:, 0].float() / H + jitter[:, 1], 0.0, 1.0)  # y
    else:
        # No background, use corners
        corners = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]], device=device)
        neg_points = corners[torch.arange(num_negative, device=device) % 4]
    neg_labels = torch.zeros(num_negative, device=device, dtype=torch.long)

    # Concatenate positive and negative points/labels
    points = torch.cat([pos_points, neg_points], dim=0)  # [N_points, 2]
    labels = torch.cat([pos_labels, neg_labels], dim=0)  # [N_points]

    return points, labels

def select_mask_by_confidence(self, mask_preds, logits=None, presence_logit=None):
    """Select mask with highest score, weighted by presence.

    When logits are provided (from DotProductScoring), selection is text-aware.
    Otherwise falls back to text-agnostic mean mask activation.
    When presence_logit is provided, scores are multiplied by sigmoid(presence)
    so low-presence predictions (object not in frame) get suppressed.
    """
    B = mask_preds.shape[0]
    scores = logits if logits is not None else mask_preds.mean(dim=(-2, -1))  # [B, Q]
    # Weight by presence (SAM3-style: score × presence)
    if presence_logit is not None:
        presence_weight = torch.sigmoid(presence_logit)  # [B, 1]
        scores = scores * presence_weight  # [B, Q] * [B, 1] -> broadcast
    best_idx = scores.argmax(dim=1)  # [B]
    batch_idx = torch.arange(B, device=mask_preds.device)
    return mask_preds[batch_idx, best_idx].unsqueeze(1), best_idx

def select_mask_by_iou(self, mask_preds, gt_masks):
    """Select mask with best IoU to GT (SAM3's approach).

    During training, this ensures we're supervising the best matching mask.
    """
    B, Q, H, W = mask_preds.shape
    device = mask_preds.device

    # Handle both [B, H, W] and [B, 1, H, W] input shapes
    if gt_masks.dim() == 4:
        gt_masks = gt_masks.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

    # Resize GT to match prediction resolution
    if gt_masks.shape[-2:] != (H, W):
        gt_resized = F.interpolate(gt_masks.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
    else:
        gt_resized = gt_masks.float()

    # Compute IoU for each mask candidate
    pred_binary = (torch.sigmoid(mask_preds) > 0.5).float()  # [B, Q, H, W]
    gt_binary = (gt_resized > 0.5).float().unsqueeze(1)  # [B, 1, H, W]

    intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))  # [B, Q]
    union = pred_binary.sum(dim=(-2, -1)) + gt_binary.sum(dim=(-2, -1)) - intersection
    ious = intersection / union.clamp(min=1.0)  # [B, Q]

    # In case all IoUs are zero, fall back to confidence
    pred_logits = mask_preds.mean(dim=(-2, -1))
    has_nonzero_ious = (ious > 0).any(dim=1, keepdim=True)  # [B, 1]
    scores = torch.where(has_nonzero_ious.expand_as(ious), ious, pred_logits)

    best_idx = scores.argmax(dim=1)  # [B]
    batch_idx = torch.arange(B, device=device)
    return mask_preds[batch_idx, best_idx].unsqueeze(1), best_idx

def select_mask_by_majority_vote(self, mask_preds, topk: int = 5):
    """Aggregate top-k masks via soft voting.

    Average the top-k masks weighted by their confidence scores.
    """
    B, Q, H, W = mask_preds.shape
    device = mask_preds.device

    # Get confidence scores
    pred_logits = mask_preds.mean(dim=(-2, -1))  # [B, Q]

    # Get top-k indices
    topk = min(topk, Q)
    _, topk_idx = pred_logits.topk(topk, dim=1)  # [B, k]

    # Gather top-k masks
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, topk)
    topk_masks = mask_preds[batch_idx, topk_idx]  # [B, k, H, W]
    topk_scores = pred_logits[batch_idx, topk_idx]  # [B, k]

    # Soft voting: weighted average by softmax of scores
    weights = F.softmax(topk_scores, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, k, 1, 1]
    voted_mask = (topk_masks * weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    return voted_mask, topk_idx[:, 0]  # Return first top idx for logging

def select_mask_by_predicted_iou(self, mask_preds, iou_preds):
    """Select mask with highest predicted IoU (true zero-shot).

    This enables mask selection without GT at inference time.
    Requires use_iou_head=True and trained IoU prediction head.
    """
    B = mask_preds.shape[0]
    device = mask_preds.device
    best_idx = iou_preds.argmax(dim=1)  # [B]
    batch_idx = torch.arange(B, device=device)
    return mask_preds[batch_idx, best_idx].unsqueeze(1), best_idx

def select_mask_by_spatial(self, mask_preds, depth, spatial_qualifier_idx, gt_masks=None, fallback='iou'):
    """Object-aware spatial mask selection.

    Selects mask based on spatial qualifier (nearest, farthest, leftmost, etc.)
    by computing depth/position at each predicted mask's centroid.

    This is OBJECT-AWARE: finds "nearest chair" not "nearest pixel".

    Args:
        mask_preds: [B, Q, H, W] mask predictions (logits)
        depth: [B, 1, H, W] depth map from DA3
        spatial_qualifier_idx: [B] spatial qualifier indices:
            0=none, 1=nearest, 2=farthest, 3=left, 4=right, 5=top, 6=bottom, 7=center
        gt_masks: [B, H, W] optional GT masks for fallback
        fallback: 'iou' (use GT), 'confidence', or 'first' when no spatial qualifier

    Returns:
        selected_masks: [B, 1, H, W]
        best_idx: [B] indices of selected masks
    """
    B, Q, H, W = mask_preds.shape
    device = mask_preds.device

    # Resize depth to match mask resolution
    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
    depth = depth.squeeze(1)  # [B, H, W]

    # Convert logits to probabilities
    mask_probs = torch.sigmoid(mask_preds)  # [B, Q, H, W]

    # Compute centroid and depth for each mask candidate
    # Using soft centroid (probability-weighted mean position)
    y_coords = torch.arange(H, device=device, dtype=mask_probs.dtype).view(1, 1, H, 1).expand(B, Q, H, W)
    x_coords = torch.arange(W, device=device, dtype=mask_probs.dtype).view(1, 1, 1, W).expand(B, Q, H, W)

    # Normalize mask probs for weighted average (add small eps to avoid div by zero)
    mask_sum = mask_probs.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)  # [B, Q, 1, 1]
    mask_normalized = mask_probs / mask_sum

    # Compute centroids [B, Q]
    centroid_y = (mask_normalized * y_coords).sum(dim=(-2, -1))  # [B, Q]
    centroid_x = (mask_normalized * x_coords).sum(dim=(-2, -1))  # [B, Q]

    # Get depth at centroids using bilinear sampling
    # grid_sample expects coords in [-1, 1]
    centroid_y_norm = (centroid_y / (H - 1)) * 2 - 1  # [B, Q]
    centroid_x_norm = (centroid_x / (W - 1)) * 2 - 1  # [B, Q]
    grid = torch.stack([centroid_x_norm, centroid_y_norm], dim=-1).unsqueeze(2)  # [B, Q, 1, 2]
    depth_at_centroid = F.grid_sample(
        depth.unsqueeze(1),  # [B, 1, H, W]
        grid,  # [B, Q, 1, 2]
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(-1).squeeze(1)  # [B, Q]

    # Also compute mask confidence (mean probability) for tie-breaking
    mask_confidence = mask_probs.mean(dim=(-2, -1))  # [B, Q]

    # Select best mask per sample based on spatial qualifier
    best_idx = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        sq = spatial_qualifier_idx[b].item() if spatial_qualifier_idx is not None else 0

        if sq == 0:
            # No spatial qualifier - use fallback
            if fallback == 'iou' and gt_masks is not None:
                # Use IoU with GT
                pred_binary = (mask_probs[b] > 0.5).float()  # [Q, H, W]
                gt_b = gt_masks[b].float() if gt_masks[b].dim() == 2 else gt_masks[b].squeeze(0).float()
                if gt_b.shape != (H, W):
                    gt_b = F.interpolate(gt_b.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze()
                gt_binary = (gt_b > 0.5).float()  # [H, W]
                intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))  # [Q]
                union = pred_binary.sum(dim=(-2, -1)) + gt_binary.sum() - intersection
                ious = intersection / union.clamp(min=1.0)
                best_idx[b] = ious.argmax()
            else:
                # Use confidence
                best_idx[b] = mask_confidence[b].argmax()

        else:
            # Table-driven spatial qualifier dispatch
            # Maps sq -> (values_tensor, find_min, invalid_sentinel)
            dist_to_center = (centroid_y[b] - H / 2) ** 2 + (centroid_x[b] - W / 2) ** 2
            spatial_dispatch = {
                1: (depth_at_centroid[b], True, float('inf')),    # nearest
                2: (depth_at_centroid[b], False, float('-inf')),   # farthest
                3: (centroid_x[b], True, float('inf')),            # leftmost
                4: (centroid_x[b], False, float('-inf')),           # rightmost
                5: (centroid_y[b], True, float('inf')),            # topmost
                6: (centroid_y[b], False, float('-inf')),           # bottommost
                7: (dist_to_center, True, float('inf')),           # center
            }
            if sq in spatial_dispatch:
                values, find_min, sentinel = spatial_dispatch[sq]
                conf_threshold = mask_confidence[b].max() * 0.3
                valid = mask_confidence[b] > conf_threshold
                if valid.any():
                    vals = values.clone()
                    vals[~valid] = sentinel
                    best_idx[b] = vals.argmin() if find_min else vals.argmax()
                else:
                    best_idx[b] = values.argmin() if find_min else values.argmax()

    # Gather selected masks
    batch_idx = torch.arange(B, device=device)
    selected_masks = mask_preds[batch_idx, best_idx].unsqueeze(1)  # [B, 1, H, W]

    return selected_masks, best_idx

@torch.no_grad()
def get_depth_and_pose(self, images):
    """Get depth and camera parameters from DA3.

    NOTE: Multi-view DA3 models (DA3-LARGE, NESTED-GIANT) can estimate poses,
    but ONLY when given multiple views together. Since the training loop
    processes views one at a time, we currently use identity poses.

    For proper multi-view pose estimation, would need to refactor to:
    1. Collect all N views: [B, N, C, H, W]
    2. Run DA3 once on all views together
    3. Get relative poses between views

    For now, all models use identity pose (camera-frame pointmaps).
    Sheaf loss requires external pose information or GT poses to work correctly.
    """
    B, C, H, W = images.shape
    patch_size = 14

    # Use da3_resolution (default 504) instead of full SAM3 resolution (1008) for speed
    # DA3 at 504 is ~4x faster than at 1008, depth quality is still good
    target_res = self.da3_resolution
    da3_H = (target_res // patch_size) * patch_size  # 504 is already patch-aligned
    da3_W = da3_H  # Square

    # Resize to DA3 resolution
    da3_images = F.interpolate(images, size=(da3_H, da3_W), mode='bilinear', align_corners=False)

    da3_output = self.da3.model.forward(
        da3_images.unsqueeze(1), extrinsics=None, intrinsics=None,
        export_feat_layers=[], infer_gs=False
    )
    # NOTE: Could pass GT extrinsics/intrinsics to DA3 here for conditioning,
    # but DA3METRIC-LARGE is per-frame and doesn't use them for multi-view alignment.
    # GT poses are used in forward() for world-space pointmap computation instead.

    depth = da3_output.depth
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)

    # NOTE: For DA3METRIC-LARGE, focal/300 scaling is applied in forward()
    # where we have access to GT intrinsics. Raw depth is returned here.
    # DA3NESTED-GIANT-LARGE outputs meters directly, no scaling needed.

    if depth.shape[-2:] != (H, W):
        depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

    # Use identity pose for all models when processing single views
    # Multi-view pose estimation requires all views together (not supported yet)
    pose = torch.eye(4, device=images.device, dtype=images.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()

    # Try to use estimated intrinsics from DA3 if available
    if hasattr(da3_output, 'intrinsics') and da3_output.intrinsics is not None:
        try:
            intrinsics = da3_output.intrinsics
            # Handle various shapes - must end up as [B, 3, 3]
            if intrinsics.dim() == 2:
                # Check if it's actually 3x3
                if intrinsics.shape == (3, 3):
                    intrinsics = intrinsics.unsqueeze(0)
                else:
                    raise ValueError(f"2D intrinsics must be 3x3, got {intrinsics.shape}")
            elif intrinsics.dim() == 3:
                # Check last two dims are 3x3
                if intrinsics.shape[-2:] != (3, 3):
                    raise ValueError(f"3D intrinsics must be [B, 3, 3], got {intrinsics.shape}")
            else:
                raise ValueError(f"Intrinsics must be 2D or 3D, got dim={intrinsics.dim()}")

            if intrinsics.shape[0] == 1 and B > 1:
                intrinsics = intrinsics.expand(B, -1, -1)
            elif intrinsics.shape[0] != B:
                # Shape mismatch, fall back to estimated
                raise ValueError(f"Intrinsics shape {intrinsics.shape} doesn't match batch size {B}")
            intrinsics = intrinsics.contiguous()
        except Exception:
            # Fall back to estimated intrinsics
            fx = fy = max(H, W) * 1.2
            cx, cy = W / 2, H / 2
            intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                       device=images.device, dtype=images.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    else:
        fx = fy = max(H, W) * 1.2
        cx, cy = W / 2, H / 2
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                   device=images.device, dtype=images.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()

    return depth.unsqueeze(1), pose, intrinsics

@torch.no_grad()
def precompute_sam3_features(self, images, text_prompt):
    """Batch-precompute SAM3 backbone + encoder features for multiple frames.

    Args:
        images: [N, 3, H, W] batch of images (already at SAM3 resolution)
        text_prompt: str, the text prompt (same for all frames)

    Returns:
        list of N dicts, each containing pre-computed SAM3 features for one frame:
            'backbone_fpn': list of FPN feature tensors [1, C, H, W] per level
            'encoder_hidden_states': [1, L, D] encoder memory
            'language_features': [T, 1, D] text features
    """
    N = images.shape[0]
    device = images.device

    # Resize to SAM3 resolution if needed
    if images.shape[-2:] != (self.resolution, self.resolution):
        images = F.interpolate(images, size=(self.resolution, self.resolution),
                               mode='bilinear', align_corners=False)

    with autocast('cuda', dtype=torch.float16):
        # Batched backbone
        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.sam3.backbone.forward_image(images))

        # Text encoding (same prompt for all, but encoder expects N copies)
        text_prompts_expanded = [text_prompt] * N
        text_out = self.sam3.backbone.forward_text(text_prompts_expanded, device=device)
        backbone_out.update(text_out)

        # Batched encoder
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, N, 4, device=device),
            box_mask=torch.zeros(N, 0, device=device, dtype=torch.bool),
        )
        find_input = FindStage(
            img_ids=torch.arange(N, device=device, dtype=torch.long),
            text_ids=torch.arange(N, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )

        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )
        backbone_out, encoder_out, _ = self.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

    # Slice into per-frame results
    encoder_memory = encoder_out["encoder_hidden_states"].transpose(0, 1)  # [N, L, D]
    # Get single-text language features (all copies identical, take first)
    lang_features = backbone_out.get('language_features', None)  # [T, N, D]
    if lang_features is not None:
        lang_features_single = lang_features[:, :1, :]  # [T, 1, D]
    else:
        lang_features_single = None

    per_frame = []
    for i in range(N):
        per_frame.append({
            'backbone_fpn': [f[i:i+1] for f in backbone_out['backbone_fpn']],
            'encoder_hidden_states': encoder_memory[i:i+1],  # [1, L, D]
            'language_features': lang_features_single,  # [T, 1, D] shared
        })

    return per_frame

def forward_multiview(self, images, text_prompts, gt_masks=None,
                      gt_extrinsics=None, gt_intrinsics=None,
                      intrinsics_orig_hw=None, cached_depth=None,
                      point_prompts=None, point_labels=None,
                      box_prompts=None, box_labels=None,
                      da3_extrinsics=None, da3_intrinsics=None,
                      cached_pi3x_pointmaps=None):
    """Multi-view forward with cross-view attention.

    Unlike forward() which processes views independently, this method:
    1. Encodes all N views with SAM3
    2. Concatenates memories from all views: [B, N*L, D]
    3. Concatenates pointmaps from all views: [B, N*L, 3]
    4. GASA attends across ALL views using world-PE for correspondence
    5. Outputs masks for all views at once

    This enables the model to see cross-view correspondences and improve consistency.

    Args:
        images: [B, N, C, H, W] input images (N views per scene)
        text_prompts: List[str] text prompts for each scene in batch (length B)
        gt_masks: [B, N, H, W] optional ground truth masks
        gt_extrinsics: [B, N, 4, 4] camera-to-world transforms (REQUIRED for world-frame PE)
        gt_intrinsics: [B, N, 3, 3] camera intrinsics
        intrinsics_orig_hw: tuple (H, W) original resolution for intrinsics
        cached_depth: [B, N, 1, H, W] optional pre-computed depth
        point_prompts: [B, N, N_pts, 2] optional point prompts per view
        point_labels: [B, N, N_pts] optional point labels per view
        box_prompts: [B, N, N_boxes, 4] optional box prompts per view
        box_labels: [B, N, N_boxes] optional box labels per view

    Returns:
        dict with pred_masks [B, N, H, W], all outputs per-view
    """
    B, N, C, H_img, W_img = images.shape
    device = images.device

    # Reshape for batch processing: [B*N, C, H, W]
    images_flat = images.view(B * N, C, H_img, W_img)

    # Resize to SAM3 resolution
    if images_flat.shape[-2:] != (self.resolution, self.resolution):
        sam3_images = F.interpolate(images_flat, size=(self.resolution, self.resolution),
                                    mode='bilinear', align_corners=False)
    else:
        sam3_images = images_flat

    # 1. Get depth for all views
    da3_live_extrinsics = None  # Will be set if DA3 runs multi-view live
    if self.da3 is None:
        # No DA3 model loaded (no GASA, no PE, no centroid): skip depth entirely
        depth = None
    elif cached_depth is not None:
        # cached_depth: [B, N, 1, H, W] -> [B*N, 1, H, W]
        depth = cached_depth.view(B * N, *cached_depth.shape[2:]).to(device=device, dtype=sam3_images.dtype)
        if depth.shape[-2:] != (self.resolution, self.resolution):
            depth = F.interpolate(depth, size=(self.resolution, self.resolution),
                                  mode='bilinear', align_corners=False)
    else:
        # Run DA3 live: pass all N views together so DA3-NESTED can estimate
        # multi-view consistent depth + poses (instead of per-frame identity poses)
        patch_size = 14
        da3_res = (self.da3_resolution // patch_size) * patch_size
        da3_images = F.interpolate(sam3_images, size=(da3_res, da3_res),
                                   mode='bilinear', align_corners=False)
        # Reshape to [B, N, C, H, W] for multi-view DA3
        da3_images_mv = da3_images.view(B, N, C, da3_res, da3_res)
        with torch.no_grad():
            da3_output = self.da3.model.forward(
                da3_images_mv, extrinsics=None, intrinsics=None,
                export_feat_layers=[], infer_gs=False
            )
        depth = da3_output.depth
        # DA3-NESTED returns depth as [B, N, H, W]: reshape to [B*N, H, W]
        if depth.dim() == 4 and depth.shape[0] == B and depth.shape[1] == N:
            depth = depth.view(B * N, depth.shape[-2], depth.shape[-1])
        elif depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth.squeeze(1)
        elif depth.dim() == 3 and depth.shape[0] == B * N:
            pass  # Already [B*N, H, W]
        elif depth.dim() == 3 and depth.shape[0] == B:
            depth = depth.view(B * N, depth.shape[-2], depth.shape[-1])
        else:
            depth = depth.reshape(B * N, depth.shape[-2], depth.shape[-1])
        # Resize depth to SAM3 resolution if needed
        if depth.shape[-2:] != (self.resolution, self.resolution):
            depth = F.interpolate(depth.unsqueeze(1), size=(self.resolution, self.resolution),
                                  mode='bilinear', align_corners=False).squeeze(1)
        # Extract DA3's estimated extrinsics if available (DA3-NESTED provides these)
        if hasattr(da3_output, 'extrinsics') and da3_output.extrinsics is not None:
            da3_ext = da3_output.extrinsics
            if not isinstance(da3_ext, torch.Tensor):
                da3_ext = torch.from_numpy(da3_ext)
            # Inverse requires float32 (not float16 from autocast)
            da3_ext = da3_ext.to(device=device, dtype=torch.float32)
            # DA3-NESTED returns [B, N, 3, 4]: pad to [B, N, 4, 4]
            if da3_ext.shape[-2] == 3 and da3_ext.shape[-1] == 4:
                pad = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)
                pad = pad.view(1, 1, 1, 4).expand(*da3_ext.shape[:-2], 1, 4)
                da3_ext = torch.cat([da3_ext, pad], dim=-2)  # [B, N, 4, 4]
            da3_ext = da3_ext.view(B * N, 4, 4)
            # DA3-NESTED outputs w2c: invert to c2w for pointmap computation
            da3_live_extrinsics = torch.inverse(da3_ext).to(dtype=depth.dtype)

    # 2. Setup intrinsics and compute pointmaps
    if depth is None:
        # No depth available (no DA3 model loaded): skip pointmap computation
        pointmaps_small = torch.zeros(B * N, self.attn_map_size, self.attn_map_size, 3, device=device)
        pointmaps_full = torch.zeros(B * N, self.resolution, self.resolution, 3, device=device)
        norm_params = None
    else:
        if gt_intrinsics is not None:
            intrinsics = gt_intrinsics.view(B * N, 3, 3).to(device=device, dtype=depth.dtype)
        else:
            focal = self.resolution * 0.8
            cx, cy = self.resolution / 2, self.resolution / 2
            intrinsics = torch.tensor([
                [focal, 0, cx], [0, focal, cy], [0, 0, 1]
            ], device=device, dtype=depth.dtype).unsqueeze(0).expand(B * N, -1, -1).contiguous()

        # Apply DA3METRIC focal/300 scaling
        if 'METRIC' in self.da3_model_name and 'NESTED' not in self.da3_model_name:
            focal = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2
            da3_res = (self.da3_resolution // 14) * 14
            if gt_intrinsics is not None and intrinsics_orig_hw is not None:
                orig_h, orig_w = intrinsics_orig_hw
                focal_at_da3 = focal * (da3_res / orig_h)
            else:
                focal_at_da3 = focal * (da3_res / self.resolution)
            depth_scale = (focal_at_da3 / 300.0).view(B * N, 1, 1, 1)
            depth = depth * depth_scale

        # Scale intrinsics to depth resolution
        depth_h, depth_w = depth.shape[-2:]
        if gt_intrinsics is not None and intrinsics_orig_hw is not None:
            orig_h, orig_w = intrinsics_orig_hw
            scale_x, scale_y = depth_w / orig_w, depth_h / orig_h
            intrinsics_scaled = intrinsics.clone()
            intrinsics_scaled[:, 0, 0] *= scale_x
            intrinsics_scaled[:, 1, 1] *= scale_y
            intrinsics_scaled[:, 0, 2] *= scale_x
            intrinsics_scaled[:, 1, 2] *= scale_y
        else:
            intrinsics_scaled = intrinsics.clone()

        # 3. Compute pointmaps for cross-view attention
        # Priority: pi3x (cached world-frame) > da3_extrinsics > da3_live > gt_extrinsics > identity
        if cached_pi3x_pointmaps is not None:
            # PI3X path: pre-computed world-frame pointmaps bypass PointmapComputer
            _pi3x = cached_pi3x_pointmaps.view(B * N, *cached_pi3x_pointmaps.shape[2:]).to(
                device=device, dtype=sam3_images.dtype)
            if _pi3x.shape[1:3] != (self.resolution, self.resolution):
                pts = _pi3x.permute(0, 3, 1, 2)
                pts = F.interpolate(pts, size=(self.resolution, self.resolution),
                                    mode='bilinear', align_corners=False)
                _pi3x = pts.permute(0, 2, 3, 1)
            if self.pointmap_normalize:
                valid = _pi3x.abs().sum(-1) > 0
                pts_flat = _pi3x.view(B * N, -1, 3)
                valid_flat = valid.view(B * N, -1)
                valid_counts = valid_flat.sum(dim=1, keepdim=True).clamp(min=1)
                center = (pts_flat * valid_flat.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1)
                pts_centered = pts_flat - center
                scale = (pts_centered.abs() * valid_flat.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(-1) / 3.0
                scale = scale.clamp(min=1e-6)
                pointmaps = (pts_centered / scale).view(B * N, *_pi3x.shape[1:3], 3)
                norm_params = {'center': center, 'scale': scale}
            else:
                pointmaps = _pi3x
                norm_params = None
            pointmaps_full = pointmaps
            pts = pointmaps.permute(0, 3, 1, 2)
            pts = F.adaptive_avg_pool2d(pts, (self.attn_map_size, self.attn_map_size))
            pointmaps_small = pts.permute(0, 2, 3, 1)

        else:
            # Standard PointmapComputer path (no PI3X cache)
            if self.use_da3_poses_for_gasa and da3_extrinsics is not None:
                world_pose = da3_extrinsics.view(B * N, 4, 4).to(device=device, dtype=depth.dtype)
            elif self.use_da3_poses_for_gasa and da3_live_extrinsics is not None:
                world_pose = da3_live_extrinsics.to(dtype=depth.dtype)
            elif gt_extrinsics is not None:
                world_pose = gt_extrinsics.view(B * N, 4, 4).to(device=device, dtype=depth.dtype)
            else:
                world_pose = torch.eye(4, device=device, dtype=depth.dtype).unsqueeze(0).expand(B * N, -1, -1).contiguous()
            depth_4d = depth.unsqueeze(1) if depth.dim() == 3 else depth
            pointmaps, norm_params = self.pointmap_computer(depth_4d, world_pose, intrinsics_scaled, normalize=self.pointmap_normalize)
            pointmaps = pointmaps.squeeze(1)  # [B*N, H, W, 3]
            pointmaps_full = pointmaps

            # Downsample pointmaps for decoder
            pts = pointmaps.permute(0, 3, 1, 2)
            pts = F.adaptive_avg_pool2d(pts, (self.attn_map_size, self.attn_map_size))
            pointmaps_small = pts.permute(0, 2, 3, 1)  # [B*N, H', W', 3]

    # 4. Run SAM3 backbone and encoder for all views
    with torch.no_grad():
        backbone_out = {"img_batch_all_stages": sam3_images}
        backbone_out.update(self.sam3.backbone.forward_image(sam3_images))

        # Repeat text prompts for each view
        text_prompts_expanded = []
        for prompt in text_prompts:
            text_prompts_expanded.extend([prompt] * N)
        text_out = self.sam3.backbone.forward_text(text_prompts_expanded, device=device)
        backbone_out.update(text_out)

        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, B * N, 4, device=device),
            box_mask=torch.zeros(B * N, 0, device=device, dtype=torch.bool),
        )
        find_input = FindStage(
            img_ids=torch.arange(B * N, device=device, dtype=torch.long),
            text_ids=torch.arange(B * N, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )

        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )
        backbone_out, encoder_out, _ = self.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

    # Get encoder memories: [B*N, L, D]
    encoder_memory = encoder_out["encoder_hidden_states"].transpose(0, 1)
    L = encoder_memory.shape[1]
    D = encoder_memory.shape[2]

    # 5. RESHAPE FOR CROSS-VIEW ATTENTION
    # Concatenate memories from all views for each scene: [B, N*L, D]
    encoder_memory_mv = encoder_memory.view(B, N, L, D).view(B, N * L, D)

    # Concatenate pointmaps: [B, N*L, 3]
    H_small, W_small = self.attn_map_size, self.attn_map_size
    pointmaps_flat = pointmaps_small.view(B * N, H_small * W_small, 3)  # [B*N, L', 3]
    L_pts = pointmaps_flat.shape[1]

    # Handle size mismatch between encoder memory and pointmaps
    if L != L_pts:
        # Interpolate pointmaps to match encoder memory length
        pts_temp = pointmaps_small.permute(0, 3, 1, 2)  # [B*N, 3, H', W']
        target_size = int(math.sqrt(L))
        pts_temp = F.adaptive_avg_pool2d(pts_temp, (target_size, target_size))
        pointmaps_flat = pts_temp.permute(0, 2, 3, 1).reshape(B * N, -1, 3)
        if pointmaps_flat.shape[1] > L:
            pointmaps_flat = pointmaps_flat[:, :L]
        elif pointmaps_flat.shape[1] < L:
            pad = torch.zeros(B * N, L - pointmaps_flat.shape[1], 3, device=device)
            pointmaps_flat = torch.cat([pointmaps_flat, pad], dim=1)

    pointmaps_mv = pointmaps_flat.view(B, N, L, 3).view(B, N * L, 3)

    # Get text embeddings (use first view's, they're all the same prompt)
    text_embedding = backbone_out.get('language_features', None)
    if text_embedding is not None:
        # [B*N, T, D] -> take first N entries (one per scene)
        text_embedding = text_embedding.transpose(0, 1)
        text_embedding = text_embedding.view(B, N, -1, D)[:, 0]  # [B, T, D]

    # Process prompts for multi-view: concatenate across views for cross-view conditioning
    # point_prompts: [B, N, N_pts, 2] -> [B, N*N_pts, 2]
    # box_prompts: [B, N, N_boxes, 4] -> [B, N*N_boxes, 4]
    mv_point_prompts = None
    mv_point_labels = None
    mv_box_prompts = None
    mv_box_labels = None

    if point_prompts is not None:
        # Concatenate prompts from all views
        mv_point_prompts = point_prompts.view(B, -1, point_prompts.shape[-1])  # [B, N*N_pts, 2]
        if point_labels is not None:
            mv_point_labels = point_labels.view(B, -1)  # [B, N*N_pts]

    if box_prompts is not None:
        mv_box_prompts = box_prompts.view(B, -1, box_prompts.shape[-1])  # [B, N*N_boxes, 4]
        if box_labels is not None:
            mv_box_labels = box_labels.view(B, -1)  # [B, N*N_boxes]

    # 6. Run GASA decoder with cross-view memory
    # Pass per-view poses and intrinsics for cross-view RayRoPE (if pe_type='rayrope')
    poses_per_view = world_pose.view(B, N, 4, 4)       # [B, N, 4, 4] c2w
    intrinsics_pv = intrinsics_scaled.view(B, N, 3, 3)  # [B, N, 3, 3]

    # Returns: queries, presence_logit, centroid_pred, iou_pred, per_query_centroids, text_scores, joint_scores, aux_outputs
    queries, presence_logit, centroid_pred, iou_pred, per_query_centroids, text_scores, joint_scores, aux_outputs = self.gasa_decoder(
        encoder_memory_mv, pointmaps_mv,
        text_embedding=text_embedding,
        point_prompts=mv_point_prompts,
        point_labels=mv_point_labels,
        box_prompts=mv_box_prompts,
        box_labels=mv_box_labels,
        poses_per_view=poses_per_view,
        intrinsics_per_view=intrinsics_pv,
        num_cameras=N,
    )
    # queries: [B, Q, D]
    queries = self.query_proj(queries)
    # Project auxiliary layer queries too (for per-layer align loss)
    aux_queries_proj = None
    if aux_outputs is not None:
        aux_queries_proj = [self.query_proj(aq) for aq in aux_outputs]

    # 7. Generate masks for EACH view separately
    # Queries are shared across views, but upsampled to each view's features
    all_view_masks = []
    for v in range(N):
        # Get this view's backbone features (every Nth sample starting from v)
        # backbone_out has features for B*N images
        view_fpn_features = [f[v::N] for f in backbone_out['backbone_fpn']]

        # Run SAM3's segmentation head
        with torch.no_grad():
            pixel_embed = self.sam3.segmentation_head.pixel_decoder(view_fpn_features)
            instance_embeds = self.sam3.segmentation_head.instance_seg_head(pixel_embed)

        # Get mask predictions using shared queries
        mask_preds = self.sam3.segmentation_head.mask_predictor(queries, instance_embeds)  # [B, Q, H, W]
        all_view_masks.append(mask_preds)

    # Stack: [B, N, Q, H, W]
    all_view_masks = torch.stack(all_view_masks, dim=1)

    # 8. Compute pred_logits
    if self.pred_logits_source == 'text_scoring' and joint_scores is not None:
        pred_logits = joint_scores  # [B, Q] - text-aware
    else:
        pred_logits = all_view_masks.mean(dim=(1, -2, -1))  # [B, Q] - text-agnostic

    # Text scoring for mask selection (same logic as single-view forward)
    use_text_scoring_selection = (
        self.pred_logits_source == 'text_scoring' and joint_scores is not None
    )

    # 9. Select best mask per view
    if use_text_scoring_selection:
        # Text-aware mask selection via DotProductScoring logits
        masks_flat = all_view_masks.view(B * N, *all_view_masks.shape[2:])
        logits_flat = pred_logits.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        pred_masks_flat, best_idx = self.select_mask_by_confidence(masks_flat, logits=logits_flat)
        pred_masks = pred_masks_flat.view(B, N, *pred_masks_flat.shape[1:])
    elif self.mask_selection == 'iou_match' and gt_masks is not None:
        gt_masks_flat = gt_masks.view(B * N, *gt_masks.shape[2:])
        masks_flat = all_view_masks.view(B * N, *all_view_masks.shape[2:])
        pred_masks_flat, best_idx = self.select_mask_by_iou(masks_flat, gt_masks_flat)
        pred_masks = pred_masks_flat.view(B, N, *pred_masks_flat.shape[1:])
    else:
        masks_flat = all_view_masks.view(B * N, *all_view_masks.shape[2:])
        pred_masks_flat, best_idx = self.select_mask_by_confidence(masks_flat)
        pred_masks = pred_masks_flat.view(B, N, *pred_masks_flat.shape[1:])

    # Resize to input resolution
    if pred_masks.shape[-2:] != (H_img, W_img):
        pred_masks = F.interpolate(pred_masks.view(B * N, 1, *pred_masks.shape[-2:]),
                                   size=(H_img, W_img), mode='bilinear', align_corners=False)
        pred_masks = pred_masks.view(B, N, H_img, W_img)

    mv_outputs = {
        'pred_masks': pred_masks,  # [B, N, H, W]
        'pred_logits': pred_logits,  # [B, Q] - text-aware scores for align loss
        'all_masks': all_view_masks,  # [B, N, Q, H, W]
        'depth': depth.view(B, N, *depth.shape[1:]),
        'pointmaps': pointmaps_mv,  # [B, N*L, 3] world-frame
        'pointmaps_full': pointmaps_full.view(B, N, *pointmaps_full.shape[1:]),  # [B, N, H, W, 3] for centroid
        'best_idx': best_idx,  # [B*N] selected query indices
    }
    if presence_logit is not None:
        mv_outputs['presence_logit'] = presence_logit
    if iou_pred is not None:
        mv_outputs['iou_pred'] = iou_pred
    if per_query_centroids is not None:
        mv_outputs['per_query_centroids'] = per_query_centroids
    if text_scores is not None:
        mv_outputs['text_scores'] = text_scores
    if joint_scores is not None:
        mv_outputs['joint_scores'] = joint_scores
    if aux_queries_proj is not None:
        mv_outputs['aux_queries'] = aux_queries_proj  # List of [B, Q, 256] per layer (excl. final)
    return mv_outputs
