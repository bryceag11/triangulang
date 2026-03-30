"""
3D Bounding Box utilities.

Computes 3D axis-aligned bounding boxes (AABBs) from predicted masks + depth.
Post-processing only — no model changes, zero inference cost.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List

from triangulang.models.sheaf_embeddings import compute_3d_localization


def compute_3d_bbox(
    pred_masks: torch.Tensor,    # [B, 1, H, W] or [B, H, W]
    depth: torch.Tensor,         # [B, 1, H, W] or [B, H, W]
    intrinsics: torch.Tensor,    # [B, 3, 3]
    threshold: float = 0.5,
    min_points: int = 10,
) -> dict:
    """
    Compute 3D axis-aligned bounding box from mask + depth (camera-relative).

    Pipeline: mask → binary → back-project masked pixels to 3D → AABB from min/max.
    No poses needed — everything is in camera frame.

    Args:
        pred_masks: Predicted segmentation masks
        depth: Metric depth maps (in meters)
        intrinsics: Camera intrinsic matrices
        threshold: Mask threshold for binarization
        min_points: Minimum masked points to compute a valid box

    Returns:
        dict with:
            - bbox_min: [B, 3] min corner (X, Y, Z) in camera frame
            - bbox_max: [B, 3] max corner (X, Y, Z) in camera frame
            - bbox_center: [B, 3] center of AABB
            - bbox_extent: [B, 3] half-extents (width/2, height/2, depth/2)
            - valid: [B] bool — whether each sample has a valid box
            - point_clouds: list of [N_i, 3] point clouds per sample
    """
    loc = compute_3d_localization(pred_masks, depth, intrinsics, threshold)
    point_clouds = loc['point_clouds']

    B = len(point_clouds)
    device = pred_masks.device

    bbox_min = torch.zeros(B, 3, device=device)
    bbox_max = torch.zeros(B, 3, device=device)
    valid = torch.zeros(B, dtype=torch.bool, device=device)

    for b in range(B):
        pc = point_clouds[b]  # [N, 3]
        if pc.shape[0] < min_points:
            continue

        # Filter out depth outliers (>3σ from median)
        depths = pc[:, 2]
        med = depths.median()
        std = depths.std()
        inlier_mask = (depths - med).abs() < 3 * std
        pc_clean = pc[inlier_mask]

        if pc_clean.shape[0] < min_points:
            pc_clean = pc  # Fall back to unfiltered

        bbox_min[b] = pc_clean.min(dim=0).values
        bbox_max[b] = pc_clean.max(dim=0).values
        valid[b] = True

    bbox_center = (bbox_min + bbox_max) / 2
    bbox_extent = (bbox_max - bbox_min) / 2

    return {
        'bbox_min': bbox_min,       # [B, 3]
        'bbox_max': bbox_max,       # [B, 3]
        'bbox_center': bbox_center, # [B, 3]
        'bbox_extent': bbox_extent, # [B, 3] half-sizes
        'valid': valid,             # [B] bool
        'point_clouds': point_clouds,
    }


def compute_3d_bbox_multiview(
    pred_masks: list,        # List of [H, W] masks per view
    depths: list,            # List of [H, W] depth maps per view
    intrinsics: list,        # List of [3, 3] intrinsics per view
    extrinsics: list = None, # List of [4, 4] c2w matrices (optional, for world-frame box)
    threshold: float = 0.5,
    min_points: int = 50,
) -> dict:
    """
    Compute 3D bounding box from multiple views.

    If extrinsics provided: transforms all points to world frame → single world-frame AABB.
    If no extrinsics: returns per-view camera-frame AABBs.

    Args:
        pred_masks: List of per-view masks
        depths: List of per-view metric depth maps
        intrinsics: List of per-view intrinsic matrices
        extrinsics: Optional list of camera-to-world matrices
        threshold: Mask threshold
        min_points: Minimum points for valid box

    Returns:
        dict with world-frame AABB (if extrinsics) or list of camera-frame AABBs
    """
    all_points_world = []

    for i, (mask, depth, K) in enumerate(zip(pred_masks, depths, intrinsics)):
        # Back-project this view
        result = compute_3d_bbox(
            mask.unsqueeze(0), depth.unsqueeze(0), K.unsqueeze(0),
            threshold=threshold, min_points=1,
        )
        pc = result['point_clouds'][0]  # [N, 3] in camera frame

        if pc.shape[0] < 1:
            continue

        if extrinsics is not None and extrinsics[i] is not None:
            # Transform to world frame: p_world = C2W @ p_cam
            c2w = extrinsics[i]  # [4, 4]
            R = c2w[:3, :3]
            t = c2w[:3, 3]
            pc_world = (R @ pc.T).T + t  # [N, 3]
            all_points_world.append(pc_world)
        else:
            all_points_world.append(pc)

    if not all_points_world:
        device = pred_masks[0].device if pred_masks else 'cpu'
        return {
            'bbox_min': torch.zeros(3, device=device),
            'bbox_max': torch.zeros(3, device=device),
            'bbox_center': torch.zeros(3, device=device),
            'bbox_extent': torch.zeros(3, device=device),
            'valid': False,
            'num_points': 0,
        }

    # Merge all points
    merged = torch.cat(all_points_world, dim=0)  # [N_total, 3]

    # Outlier filtering (3σ per axis)
    for axis in range(3):
        med = merged[:, axis].median()
        std = merged[:, axis].std().clamp(min=0.01)
        inlier = (merged[:, axis] - med).abs() < 3 * std
        merged = merged[inlier]

    if merged.shape[0] < min_points:
        device = pred_masks[0].device if pred_masks else 'cpu'
        return {
            'bbox_min': torch.zeros(3, device=device),
            'bbox_max': torch.zeros(3, device=device),
            'bbox_center': torch.zeros(3, device=device),
            'bbox_extent': torch.zeros(3, device=device),
            'valid': False,
            'num_points': merged.shape[0],
        }

    bbox_min = merged.min(dim=0).values
    bbox_max = merged.max(dim=0).values
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_extent = (bbox_max - bbox_min) / 2

    return {
        'bbox_min': bbox_min,        # [3]
        'bbox_max': bbox_max,        # [3]
        'bbox_center': bbox_center,  # [3]
        'bbox_extent': bbox_extent,  # [3] half-sizes (w/2, h/2, d/2)
        'valid': True,
        'num_points': merged.shape[0],
    }
