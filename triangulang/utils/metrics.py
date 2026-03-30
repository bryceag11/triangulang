"""Metrics utilities for segmentation evaluation."""

import os
import hashlib

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


def compute_iou(pred, target, threshold=0.5, return_tensor=False):
    """Compute IoU between prediction and target.

    Args:
        pred: Predicted logits
        target: Ground truth mask
        threshold: Binarization threshold
        return_tensor: If True, return tensor (no GPU-CPU sync). If False, return float.
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > 0.5).float()
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = intersection / union.clamp(min=1.0)
    if return_tensor:
        return iou
    return iou.item() if union >= 1 else 1.0


def compute_recall(pred, target, threshold=0.5, return_tensor=False):
    """Compute recall = TP / (TP + FN) - what % of GT object did we find.

    This is the standard mAcc metric when averaged per-category.

    Args:
        pred: Predicted logits
        target: Ground truth mask
        threshold: Binarization threshold
        return_tensor: If True, return tensor (no GPU-CPU sync). If False, return float.
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > 0.5).float()

    tp = (pred_binary * target_binary).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    recall = tp / (tp + fn).clamp(min=1.0)
    if return_tensor:
        return recall
    if (tp + fn) > 0:
        return recall.item()
    return 1.0


def compute_per_mask_ious(mask_preds, gt_masks, threshold=0.5):
    """Compute IoU between each predicted mask and GT.

    Args:
        mask_preds: [B, Q, H, W] raw mask logits
        gt_masks: [B, H, W] or [B, 1, H, W] GT masks

    Returns:
        ious: [B, Q] IoU for each mask
    """
    B, Q, H, W = mask_preds.shape
    device = mask_preds.device

    # Handle both [B, H, W] and [B, 1, H, W] input shapes
    if gt_masks.dim() == 4:
        gt_masks = gt_masks.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

    # Resize GT to match prediction resolution
    if gt_masks.shape[-2:] != (H, W):
        gt_resized = F.interpolate(
            gt_masks.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze(1)
    else:
        gt_resized = gt_masks.float()

    # Compute IoU for each mask candidate
    pred_binary = (torch.sigmoid(mask_preds) > threshold).float()  # [B, Q, H, W]
    gt_binary = (gt_resized > 0.5).float().unsqueeze(1)  # [B, 1, H, W]

    intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))  # [B, Q]
    union = pred_binary.sum(dim=(-2, -1)) + gt_binary.sum(dim=(-2, -1)) - intersection
    ious = intersection / union.clamp(min=1.0)  # [B, Q]

    return ious


def compute_mean_accuracy(pred, target, threshold=0.5, return_tensor=False):
    """Compute mean class accuracy (mAcc = average of FG and BG accuracy).

    This balances FG and BG equally, so missing foreground pixels hurts
    even when the mask is small relative to the image.

    Args:
        pred: Predicted logits
        target: Ground truth mask
        threshold: Binarization threshold
        return_tensor: If True, return tensor (no GPU-CPU sync). If False, return float.
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > 0.5).float()

    # Foreground accuracy
    fg_mask = target_binary > 0.5
    if fg_mask.sum() > 0:
        fg_acc = ((pred_binary == target_binary) & fg_mask).sum().float() / fg_mask.sum().float()
    else:
        fg_acc = torch.tensor(1.0, device=pred.device)

    # Background accuracy
    bg_mask = target_binary <= 0.5
    if bg_mask.sum() > 0:
        bg_acc = ((pred_binary == target_binary) & bg_mask).sum().float() / bg_mask.sum().float()
    else:
        bg_acc = torch.tensor(1.0, device=pred.device)

    macc = (fg_acc + bg_acc) / 2
    if return_tensor:
        return macc
    return macc.item()


def compute_gt_centroid(mask, pointmaps):
    """Compute ground truth 3D centroid from mask and pointmaps.

    Args:
        mask: [H, W] binary mask
        pointmaps: [H, W, 3] world coordinates

    Returns:
        centroid: [3] 3D centroid in world coordinates
    """
    # Resize pointmaps to match mask if needed
    if mask.shape != pointmaps.shape[:2]:
        # Resize mask to pointmap resolution
        mask_resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=pointmaps.shape[:2],
            mode='nearest'
        ).squeeze()
    else:
        mask_resized = mask.float()

    # Get masked points
    mask_binary = mask_resized > 0.5
    if mask_binary.sum() == 0:
        # No valid points, return origin
        return torch.zeros(3, device=mask.device)

    # Compute centroid as mean of masked 3D points
    masked_points = pointmaps[mask_binary]  # [N, 3]
    centroid = masked_points.mean(dim=0)  # [3]

    return centroid


class CategoryMetricsTracker:
    """Track per-category IoU for proper mIoU calculation."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Track intersection and union per category for proper averaging
        self.category_intersection = {}  # category -> total intersection
        self.category_union = {}  # category -> total union
        self.category_count = {}  # category -> number of samples

    def update(self, pred, target, category: str, threshold=0.5):
        """Update metrics for a single prediction."""
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = (target > 0.5).float()

        intersection = (pred_binary * target_binary).sum().item()
        union = pred_binary.sum().item() + target_binary.sum().item() - intersection

        if category not in self.category_intersection:
            self.category_intersection[category] = 0.0
            self.category_union[category] = 0.0
            self.category_count[category] = 0

        self.category_intersection[category] += intersection
        self.category_union[category] += union
        self.category_count[category] += 1

    def get_per_category_iou(self):
        """Get IoU for each category."""
        results = {}
        for cat in self.category_intersection:
            if self.category_union[cat] > 0:
                results[cat] = self.category_intersection[cat] / self.category_union[cat]
            else:
                results[cat] = 1.0
        return results

    def get_miou(self):
        """Get mean IoU across all categories (true mIoU)."""
        per_cat = self.get_per_category_iou()
        if not per_cat:
            return 0.0
        return sum(per_cat.values()) / len(per_cat)

    def get_sample_avg_iou(self):
        """Get sample-averaged IoU (what we were computing before)."""
        total_intersection = sum(self.category_intersection.values())
        total_union = sum(self.category_union.values())
        if total_union > 0:
            return total_intersection / total_union
        return 0.0

    @staticmethod
    def _deterministic_hash(s: str) -> int:
        """Compute a deterministic hash for a string.

        IMPORTANT: Python's built-in hash() is randomized per-process for security,
        which breaks DDP sync. Use this instead for consistent hashing across ranks.
        """
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

    def sync_across_ranks(self, ddp):
        """Sync category metrics across DDP ranks by aggregating raw intersection/union.

        This is mathematically correct: we sum intersection and union across ranks,
        then compute IoU from the totals. This is NOT the same as averaging IoUs.

        IMPORTANT: This now gathers ALL categories from ALL ranks, so the final
        per_category_iou includes categories seen by any rank (not just rank 0).

        Only call this in DDP mode. Single-GPU training is unchanged.
        """
        if not ddp.is_distributed:
            return  # No-op for single GPU

        local_cats = sorted(self.category_intersection.keys())

        # Step 1: Gather category NAMES from all ranks using all_gather_object
        # This is slower but necessary to build a global hash->name mapping
        # Category lists are small (typically 50-200), so overhead is minimal
        all_cat_lists = [None] * ddp.world_size
        dist.all_gather_object(all_cat_lists, local_cats)

        # Build global hash->name mapping from all ranks
        # NOTE: Use deterministic hash, NOT Python's hash() which is randomized per-process
        global_hash_to_name = {}
        for cat_list in all_cat_lists:
            for cat in cat_list:
                h = self._deterministic_hash(cat)
                global_hash_to_name[h] = cat

        # Step 2: Use tensor-based gather for metrics (fast)
        local_num_cats = torch.tensor([len(local_cats)], device=ddp.device, dtype=torch.long)
        all_num_cats = [torch.zeros(1, device=ddp.device, dtype=torch.long) for _ in range(ddp.world_size)]
        dist.all_gather(all_num_cats, local_num_cats)

        max_cats = max(n.item() for n in all_num_cats)
        if max_cats == 0:
            return  # All ranks have no categories - nothing to sync

        # Encode category names as integers using deterministic hash
        cat_to_hash = {cat: self._deterministic_hash(cat) for cat in local_cats}

        # Pad to max_cats and gather
        local_hashes = torch.zeros(max_cats, device=ddp.device, dtype=torch.long)
        local_intersection_padded = torch.zeros(max_cats, device=ddp.device, dtype=torch.float32)
        local_union_padded = torch.zeros(max_cats, device=ddp.device, dtype=torch.float32)

        for i, cat in enumerate(local_cats):
            local_hashes[i] = cat_to_hash[cat]
            local_intersection_padded[i] = self.category_intersection[cat]
            local_union_padded[i] = self.category_union[cat]

        # Gather all hashes, intersections, unions
        all_hashes = [torch.zeros_like(local_hashes) for _ in range(ddp.world_size)]
        all_intersections = [torch.zeros_like(local_intersection_padded) for _ in range(ddp.world_size)]
        all_unions = [torch.zeros_like(local_union_padded) for _ in range(ddp.world_size)]

        dist.all_gather(all_hashes, local_hashes)
        dist.all_gather(all_intersections, local_intersection_padded)
        dist.all_gather(all_unions, local_union_padded)

        # Step 3: Aggregate by hash
        hash_to_intersection = {}
        hash_to_union = {}

        for rank_idx in range(ddp.world_size):
            num_cats = all_num_cats[rank_idx].item()
            for i in range(num_cats):
                h = all_hashes[rank_idx][i].item()
                inter = all_intersections[rank_idx][i].item()
                union = all_unions[rank_idx][i].item()
                if h not in hash_to_intersection:
                    hash_to_intersection[h] = 0.0
                    hash_to_union[h] = 0.0
                hash_to_intersection[h] += inter
                hash_to_union[h] += union

        # Step 4: Update local tracker with ALL categories from ALL ranks
        # Use global_hash_to_name to recover category names for categories we didn't see locally
        if os.environ.get('DDP_DEBUG'):
            print(f"[R{ddp.rank}] sync: global_hash_to_name has {len(global_hash_to_name)} categories, "
                  f"hash_to_intersection has {len(hash_to_intersection)} hashes", flush=True)

        for h, name in global_hash_to_name.items():
            if h in hash_to_intersection:
                self.category_intersection[name] = hash_to_intersection[h]
                self.category_union[name] = hash_to_union[h]

        if os.environ.get('DDP_DEBUG'):
            print(f"[R{ddp.rank}] sync: After update, tracker has {len(self.category_intersection)} categories", flush=True)

    def summary(self):
        """Get summary dict for logging."""
        per_cat = self.get_per_category_iou()
        return {
            'mIoU': self.get_miou(),
            'sample_avg_iou': self.get_sample_avg_iou(),
            'num_categories': len(per_cat),
            'per_category_iou': per_cat
        }
