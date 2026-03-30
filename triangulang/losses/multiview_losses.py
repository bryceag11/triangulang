"""
Multi-View Losses for DA3-SAM3 Training

Key innovations:
1. Multi-view consistency loss - objects should triangulate to same 3D point
2. Depth-aware mask loss - use depth for better boundaries
3. Cross-view voting loss - masks should agree across views
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class MultiViewConsistencyLoss(nn.Module):
    """
    Key innovation: Objects should project to same 3D location from all views
    Based on new_idea.md triangulation concept
    """

    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight

    def forward(
        self,
        masks: torch.Tensor,
        depths: torch.Tensor,
        poses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-view consistency loss
        Masks should project to same 3D point across views

        Args:
            masks: [B, 1, H, W] predicted masks
            depths: [B, 1, H, W] depth maps from DA3
            poses: [B, 4, 4] camera poses (world to camera)
        """
        num_views = masks.shape[0]
        if num_views < 2:
            return torch.tensor(0.0, device=masks.device)

        # Get mask centroids in each view
        valid_views = []
        centroids_2d = []
        depths_at_centroid = []

        for i in range(num_views):
            mask = masks[i, 0].sigmoid()  # [H, W], apply sigmoid
            depth = depths[i, 0]  # [H, W]

            # Compute centroid (center of mass)
            mask_sum = mask.sum()

            # Skip if mask is empty or very small
            if mask_sum < 10:  # At least 10 pixels
                continue

            # Compute weighted centroid
            H, W = mask.shape
            y_indices = torch.arange(H, device=mask.device).float().view(-1, 1)
            x_indices = torch.arange(W, device=mask.device).float().view(1, -1)

            centroid_y = (mask * y_indices).sum() / mask_sum
            centroid_x = (mask * x_indices).sum() / mask_sum

            # Check for NaN
            if torch.isnan(centroid_y) or torch.isnan(centroid_x):
                continue

            centroids_2d.append([centroid_x, centroid_y])

            # Get depth at centroid
            cy = int(torch.clamp(centroid_y, 0, H - 1).item())
            cx = int(torch.clamp(centroid_x, 0, W - 1).item())
            depth_val = depth[cy, cx].detach().clone()

            # Check if depth is valid
            if torch.isnan(depth_val) or depth_val <= 0:
                continue

            depths_at_centroid.append(depth_val)
            valid_views.append(i)

        # Need at least 2 valid views for consistency
        if len(valid_views) < 2:
            return torch.tensor(0.0, device=masks.device, requires_grad=True)

        # Compute 3D points from each view (simplified unprojection)
        points_3d = []
        H, W = masks.shape[2], masks.shape[3]

        for i, (cx, cy) in enumerate(centroids_2d):
            depth = depths_at_centroid[i]

            # Simplified unprojection (normalized coordinates)
            x_norm = (cx - W/2) / (W/2)
            y_norm = (cy - H/2) / (H/2)

            # Detach to avoid autograd issues
            depth_val = depth.detach() if hasattr(depth, 'detach') else depth

            point_3d = torch.stack([
                x_norm * depth_val,
                y_norm * depth_val,
                depth_val
            ])

            points_3d.append(point_3d)

        # Compute pairwise distances (should be small if consistent)
        consistency_loss = 0.0
        num_pairs = 0

        for i in range(len(points_3d)):
            for j in range(i + 1, len(points_3d)):
                dist = torch.norm(points_3d[i] - points_3d[j])
                consistency_loss += dist
                num_pairs += 1

        if num_pairs > 0:
            consistency_loss = consistency_loss / num_pairs * self.consistency_weight
        else:
            consistency_loss = torch.tensor(0.0, device=masks.device, requires_grad=True)

        return consistency_loss


class DepthAwareMaskLoss(nn.Module):
    """
    Use depth information to improve mask boundaries
    Penalize masks that cross depth discontinuities
    """

    def __init__(self, depth_threshold: float = 0.1):
        super().__init__()
        self.depth_threshold = depth_threshold

    def forward(
        self,
        masks: torch.Tensor,
        depths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            masks: [B, 1, H, W] predicted masks
            depths: [B, 1, H, W] depth maps
        """
        # Compute depth gradients
        depth_dx = torch.abs(depths[:, :, :, 1:] - depths[:, :, :, :-1])
        depth_dy = torch.abs(depths[:, :, 1:, :] - depths[:, :, :-1, :])

        # Find depth discontinuities
        depth_edges_x = (depth_dx > self.depth_threshold).float()
        depth_edges_y = (depth_dy > self.depth_threshold).float()

        # Compute mask gradients
        mask_sigmoid = masks.sigmoid()
        mask_dx = torch.abs(mask_sigmoid[:, :, :, 1:] - mask_sigmoid[:, :, :, :-1])
        mask_dy = torch.abs(mask_sigmoid[:, :, 1:, :] - mask_sigmoid[:, :, :-1, :])

        # Penalize mask changes at depth discontinuities
        penalty_x = (mask_dx * depth_edges_x).mean()
        penalty_y = (mask_dy * depth_edges_y).mean()

        return penalty_x + penalty_y


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = pred.sigmoid()
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 2.0 * (intersection + 1) / (union + 1)
        return 1 - dice.mean()


class CombinedMultiViewLoss(nn.Module):
    """
    Combined loss for multi-view DA3-SAM3 training

    Components:
    1. Mask supervision loss (BCE + Dice)
    2. Multi-view consistency loss
    3. Depth-aware boundary loss
    """

    def __init__(
        self,
        mask_weight: float = 1.0,
        consistency_weight: float = 0.5,
        depth_boundary_weight: float = 0.2,
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.mask_weight = mask_weight
        self.consistency_weight = consistency_weight
        self.depth_boundary_weight = depth_boundary_weight
        self.dice_weight = dice_weight

        # Loss components
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.consistency_loss = MultiViewConsistencyLoss(consistency_weight)
        self.depth_boundary_loss = DepthAwareMaskLoss()

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        depths: torch.Tensor,
        poses: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_masks: [B, 1, H, W] predicted masks (logits)
            target_masks: [B, 1, H, W] target masks (0-1)
            depths: [B, 1, H, W] depth maps
            poses: [B, 4, 4] camera poses (optional)

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}

        # 1. Mask supervision
        losses['bce'] = self.bce_loss(pred_masks, target_masks)
        losses['dice'] = self.dice_loss(pred_masks, target_masks)
        losses['mask'] = self.mask_weight * (losses['bce'] + self.dice_weight * losses['dice'])

        # 2. Multi-view consistency (if poses provided)
        if poses is not None:
            losses['consistency'] = self.consistency_loss(pred_masks, depths, poses)
        else:
            losses['consistency'] = torch.tensor(0.0, device=pred_masks.device)

        # 3. Depth-aware boundaries
        losses['depth_boundary'] = self.depth_boundary_weight * self.depth_boundary_loss(
            pred_masks, depths
        )

        # Total loss
        losses['total'] = (
            losses['mask'] +
            losses['consistency'] +
            losses['depth_boundary']
        )

        return losses