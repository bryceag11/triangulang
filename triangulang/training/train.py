"""
TrianguLang Training Script - GASA Decoder for Geometry-Aware Segmentation

Architecture:
    SAM3 Backbone + Encoder (FROZEN) -> encoder_memory with semantics
    DA3 (FROZEN) -> depth -> pointmaps (world coordinates)
    GASA Decoder (TRAINABLE) -> object queries with geometric attention bias
    SAM3 SegHead (frozen) -> masks

Usage:
    torchrun --nproc_per_node=8 triangulang/training/train.py \
        --run-name my_run --epochs 30

See CLAUDE.md for full CLI reference and training commands.
"""
import warnings
# Suppress PyTorch scheduler deprecation warning (internal to SequentialLR)
warnings.filterwarnings('ignore', message='.*epoch parameter in.*scheduler.step.*')

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import sys
import os
import gc
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
import random
import math
import time

import tyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'sam3'))
sys.path.insert(0, str(PROJECT_ROOT / 'depth_anything_v3' / 'src'))

# DDP support - auto-detects if running via torchrun (must be after sys.path setup)
from triangulang.utils.ddp_utils import DDPManager

from triangulang.utils.scannetpp_loader import ScanNetPPMultiViewDataset
from triangulang.data.dataset_factory import get_dataset, get_dataset_config

# SAM3 imports
from sam3 import build_sam3_image_model
from sam3.model.geometry_encoders import Prompt
from sam3.model.data_misc import FindStage
from sam3.sam.prompt_encoder import PositionEmbeddingRandom
from sam3.model.model_misc import MLP as SAM3MLP

# DA3 imports
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

# GASA imports
from triangulang.models.gasa import (
    PointmapComputer,
    WorldSpacePositionalEncoding,
    CameraRelativePositionalEncoding,
    PluckerEmbedding,
    RayRoPE3D,
)

# Sheaf consistency losses
from triangulang.losses.sheaf_losses import SheafConsistencyLoss, FeatureSheafLoss, AsymmetricRestrictionSheaf

# Spatial reasoning utilities
from triangulang.utils.spatial_reasoning import (
    parse_spatial_qualifier,
    parse_relational_query,
    get_spatial_qualifier_idx,
    spatial_to_pseudo_point_tensor,
    SpatialAugmentor,
    GTAwareSpatialAugmentor,
    SpatialContext,
    SPATIAL_QUALIFIER_TO_IDX,
)
from triangulang.training.config import TrainConfig

from triangulang import BPE_PATH as _BPE_PATH

# Extracted utilities
from triangulang.utils.lora import LoRALayer, LoRAManager
from triangulang.utils.metrics import (
    compute_iou, compute_recall, compute_per_mask_ious,
    compute_mean_accuracy, compute_gt_centroid, CategoryMetricsTracker,
)
from triangulang.losses.segmentation import (
    focal_loss, dice_loss, centroid_loss, boundary_loss,
    lovasz_grad, lovasz_hinge_loss, lovasz_loss, point_sampled_loss,
    align_loss, contrastive_mask_loss, segmentation_loss,
)


def set_seed(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from triangulang.models.gasa_decoder import (
    GASADecoderLayer,
    MaskRefiner,
    SpatialAttentionBias,
    TextConditionedSpatialBias,
    GASADecoder,
)
from triangulang.models.triangulang_model import TrianguLangModel


# The following classes have been extracted to separate files:
# - GASADecoderLayer, MaskRefiner, SpatialAttentionBias,
#   TextConditionedSpatialBias, GASADecoder -> triangulang.models.gasa_decoder
# - TrianguLangModel -> triangulang.models.triangulang_model


def collate_fn(batch, max_objects=0):
    valid_batch = [b for b in batch if b.get('has_gt_mask', False) and b.get('gt_masks') is not None]
    if not valid_batch:
        return None

    # Check for multi-object mode: gt_masks is [K, N, H, W] when K>1
    is_multi_object = valid_batch[0]['gt_masks'].dim() == 4

    if is_multi_object:
        # Cap K per sample to max_objects (0 = no cap)
        if max_objects > 0:
            for b in valid_batch:
                K_b = b['gt_masks'].shape[0]
                if K_b > max_objects:
                    # Keep only the first max_objects (sorted by mask area in dataset)
                    b['gt_masks'] = b['gt_masks'][:max_objects]
                    if 'multi_object_prompts' in b and b['multi_object_prompts'] is not None:
                        b['multi_object_prompts'] = b['multi_object_prompts'][:max_objects]

        # Multi-object: pad all samples to same K (max K in batch)
        max_K = max(b['gt_masks'].shape[0] for b in valid_batch)
        N = valid_batch[0]['gt_masks'].shape[1]
        H, W = valid_batch[0]['gt_masks'].shape[2:]
        padded_gt_masks = []
        num_objects_list = []
        multi_prompts_list = []
        for b in valid_batch:
            K_b = b['gt_masks'].shape[0]
            num_objects_list.append(K_b)
            multi_prompts_list.append(b.get('multi_object_prompts', [b['prompt']]))
            if K_b < max_K:
                # Pad with zeros
                pad = torch.zeros(max_K - K_b, N, H, W)
                padded_gt_masks.append(torch.cat([b['gt_masks'], pad], dim=0))
                # Pad prompts with empty string
                multi_prompts_list[-1] = multi_prompts_list[-1] + [''] * (max_K - K_b)
            else:
                padded_gt_masks.append(b['gt_masks'])
        result = {
            'images': torch.stack([b['images'] for b in valid_batch]),
            'gt_masks': torch.stack(padded_gt_masks),  # [B, K, N, H, W]
            'prompts': [b['prompt'] for b in valid_batch],  # Primary prompt (backward compat)
            'num_objects': torch.tensor(num_objects_list, dtype=torch.long),  # [B]
            'multi_object_prompts': multi_prompts_list,  # List[List[str]], [B][K]
        }
    else:
        result = {
            'images': torch.stack([b['images'] for b in valid_batch]),
            'gt_masks': torch.stack([b['gt_masks'] for b in valid_batch]),
            'prompts': [b['prompt'] for b in valid_batch],
        }

    # Optional: intrinsics/extrinsics (not all datasets have these)
    if all('intrinsics' in b and b['intrinsics'] is not None for b in valid_batch):
        result['intrinsics'] = torch.stack([b['intrinsics'] for b in valid_batch])
    if all('extrinsics' in b and b['extrinsics'] is not None for b in valid_batch):
        result['extrinsics'] = torch.stack([b['extrinsics'] for b in valid_batch])

    # Optional: point prompts for benchmark datasets (NVOS, SpinNeRF)
    if all('prompt_points' in b for b in valid_batch):
        result['prompt_points'] = torch.stack([b['prompt_points'] for b in valid_batch])
        result['prompt_labels'] = torch.stack([b['prompt_labels'] for b in valid_batch])

    # Optional: cached depth for faster training (bypasses DA3)
    if all('cached_depth' in b and b['cached_depth'] is not None for b in valid_batch):
        result['cached_depth'] = torch.stack([b['cached_depth'] for b in valid_batch])

    # Optional: cached DA3-NESTED poses for world-frame GASA and sheaf loss
    if all('cached_da3_extrinsics' in b and b['cached_da3_extrinsics'] is not None for b in valid_batch):
        result['cached_da3_extrinsics'] = torch.stack([b['cached_da3_extrinsics'] for b in valid_batch])
    if all('cached_da3_intrinsics' in b and b['cached_da3_intrinsics'] is not None for b in valid_batch):
        result['cached_da3_intrinsics'] = torch.stack([b['cached_da3_intrinsics'] for b in valid_batch])

    # Optional: GT 3D centroids for supervision
    if all('centroid_3d' in b and b['centroid_3d'] is not None for b in valid_batch):
        result['centroid_3d'] = torch.stack([b['centroid_3d'] for b in valid_batch])

    # Optional: GT mask coverage at ORIGINAL resolution (for min_mask_coverage filtering)
    if all('gt_mask_coverage' in b and b['gt_mask_coverage'] is not None for b in valid_batch):
        result['gt_mask_coverage'] = torch.stack([b['gt_mask_coverage'] for b in valid_batch])

    # Optional: spatial context for GT-aware spatial augmentation
    # Keep as list of SpatialContext objects (or None for samples without context)
    spatial_contexts = [b.get('spatial_context', None) for b in valid_batch]
    if any(ctx is not None for ctx in spatial_contexts):
        result['spatial_context'] = spatial_contexts

    return result


def spatial_ranking_loss(pred_masks, depth, labels, margin=0.5):
    """Learned spatial ranking loss for multi-instance objects.

    For each group of same-label objects, creates pairwise ranking targets
    based on depth (nearest/farthest) and position (left/right/top/bottom).
    Uses margin ranking loss so the model learns to produce masks whose
    centroids match the correct spatial ordering.

    Args:
        pred_masks: [K, H, W] predicted mask logits for K objects
        depth: [1, H, W] or [H, W] depth map
        labels: list of K label strings
        margin: ranking margin (default 0.5)

    Returns:
        scalar loss (0 if no multi-instance groups)
    """
    K, H, W = pred_masks.shape
    device = pred_masks.device

    if depth.dim() == 3:
        depth = depth.squeeze(0)  # [H, W]

    # Group objects by label
    from collections import defaultdict
    label_groups = defaultdict(list)
    for k, label in enumerate(labels):
        # Strip spatial qualifiers to get base label
        base = label
        for prefix in ['nearest ', 'farthest ', 'leftmost ', 'rightmost ',
                        'topmost ', 'bottommost ', 'closest ', 'center ']:
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        label_groups[base].append(k)

    # Only process groups with 2+ instances
    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    # Compute soft centroids for all objects
    mask_probs = torch.sigmoid(pred_masks)  # [K, H, W]
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)

    mask_sum = mask_probs.sum(dim=(-2, -1)).clamp(min=1e-6)  # [K]
    # Weighted centroid coordinates
    centroid_y = (mask_probs * y_coords.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum  # [K]
    centroid_x = (mask_probs * x_coords.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum  # [K]

    # Depth at centroid via bilinear sampling
    cy_norm = (centroid_y / (H - 1)) * 2 - 1  # [-1, 1]
    cx_norm = (centroid_x / (W - 1)) * 2 - 1
    grid = torch.stack([cx_norm, cy_norm], dim=-1).view(1, K, 1, 2)  # [1, K, 1, 2]
    depth_at_cent = F.grid_sample(
        depth.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        grid, mode='bilinear', padding_mode='border', align_corners=True
    ).view(K)  # [K]

    ranking_loss_fn = torch.nn.MarginRankingLoss(margin=margin, reduction='mean')

    for base_label, indices in label_groups.items():
        if len(indices) < 2:
            continue

        # Create pairwise ranking targets for depth (smaller = nearer)
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                ki, kj = indices[i_idx], indices[j_idx]

                # Depth ranking: if depth[ki] < depth[kj], ki is nearer
                # MarginRankingLoss: loss(x1, x2, y) where y=1 means x1 should be ranked higher
                di = depth_at_cent[ki].unsqueeze(0)
                dj = depth_at_cent[kj].unsqueeze(0)
                # Target: -1 if di < dj (ki nearer), +1 if dj < di (kj nearer)
                with torch.no_grad():
                    depth_target = torch.sign(dj - di).clamp(-1, 1)
                    if depth_target.item() == 0:
                        depth_target = torch.tensor([1.0], device=device)
                total_loss = total_loss + ranking_loss_fn(di, dj, depth_target)
                n_pairs += 1

                # X-position ranking (left-right)
                xi = centroid_x[ki].unsqueeze(0)
                xj = centroid_x[kj].unsqueeze(0)
                with torch.no_grad():
                    x_target = torch.sign(xj - xi).clamp(-1, 1)
                    if x_target.item() == 0:
                        x_target = torch.tensor([1.0], device=device)
                total_loss = total_loss + ranking_loss_fn(xi, xj, x_target)
                n_pairs += 1

    if n_pairs > 0:
        return total_loss / n_pairs
    return total_loss


def spatial_selection_loss(pred_masks, gt_masks, depth, labels, spatial_indices):
    """Spatial selection loss: trains spatial tokens to select the correct instance.

    When a spatial qualifier is active (e.g., "nearest chair" with spatial_idx=1),
    this loss checks whether the predicted mask for that object actually corresponds
    to the spatially correct GT instance (the nearest one by depth).

    Uses cross-entropy: for each spatially-qualified object, creates a target
    distribution over GT instances based on the qualifier, and penalizes the
    model's mask if it matches the wrong instance.

    Args:
        pred_masks: [K, H, W] predicted mask logits (one per object)
        gt_masks: [K, H, W] GT masks (one per object)
        depth: [1, H, W] or [H, W] depth map
        labels: list of K label strings (may include spatial prefixes)
        spatial_indices: [K] tensor of spatial qualifier indices (0=none, 1=nearest, etc.)

    Returns:
        scalar loss (0 if no spatial qualifiers active)
    """
    K, H, W = pred_masks.shape
    device = pred_masks.device

    if depth.dim() == 3:
        depth = depth.squeeze(0)

    # Strip spatial prefixes to get base labels
    base_labels = []
    for label in labels:
        base = label
        for prefix in ['nearest ', 'farthest ', 'leftmost ', 'rightmost ',
                        'topmost ', 'bottommost ', 'closest ', 'center ']:
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        base_labels.append(base)

    # Group objects by base label
    from collections import defaultdict
    label_groups = defaultdict(list)
    for k, base in enumerate(base_labels):
        label_groups[base].append(k)

    total_loss = torch.tensor(0.0, device=device)
    n_spatial = 0

    # Compute GT mask centroids and depths (detached — these are targets)
    with torch.no_grad():
        gt_binary = (gt_masks > 0.5).float()
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
        x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)

        gt_sum = gt_binary.sum(dim=(-2, -1)).clamp(min=1e-6)
        gt_cy = (gt_binary * y_coords.unsqueeze(0)).sum(dim=(-2, -1)) / gt_sum
        gt_cx = (gt_binary * x_coords.unsqueeze(0)).sum(dim=(-2, -1)) / gt_sum

        # Depth at GT centroids
        cy_norm = (gt_cy / (H - 1)) * 2 - 1
        cx_norm = (gt_cx / (W - 1)) * 2 - 1
        grid = torch.stack([cx_norm, cy_norm], dim=-1).view(1, K, 1, 2)
        gt_depth = F.grid_sample(
            depth.unsqueeze(0).unsqueeze(0), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        ).view(K)

    # For each spatially-qualified object, compute IoU of its prediction against
    # ALL same-label GT masks, then penalize if the best-matching GT isn't the
    # spatially correct one
    for base, indices in label_groups.items():
        if len(indices) < 2:
            continue

        # Find which indices have spatial qualifiers
        for k in indices:
            sq = spatial_indices[k].item() if isinstance(spatial_indices, torch.Tensor) else spatial_indices[k]
            if sq == 0:
                continue  # No spatial qualifier

            # Determine which GT instance is the "correct" one for this qualifier
            group_depths = gt_depth[indices]
            group_cx = gt_cx[indices]
            group_cy = gt_cy[indices]
            group_valid = gt_sum[indices] > 1  # Has visible GT

            if group_valid.sum() < 2:
                continue

            # Find target index within group
            # Table-driven spatial target selection
            spatial_dispatch = {
                1: (group_depths, True, 1e6),     # nearest
                2: (group_depths, False, -1e6),    # farthest
                3: (group_cx, True, 1e6),          # leftmost
                4: (group_cx, False, -1e6),         # rightmost
                5: (group_cy, True, 1e6),          # topmost
                6: (group_cy, False, -1e6),         # bottommost
            }
            if sq not in spatial_dispatch:
                continue
            values, find_min, sentinel = spatial_dispatch[sq]
            masked = values.clone()
            masked[~group_valid] = sentinel
            target_in_group = (masked.argmin() if find_min else masked.argmax()).item()

            target_k = indices[target_in_group]

            # Compute IoU of this prediction against each same-label GT
            pred_prob = torch.sigmoid(pred_masks[k])  # [H, W]
            ious = []
            for j in indices:
                if not group_valid[indices.index(j)]:
                    ious.append(torch.tensor(0.0, device=device))
                    continue
                gt_j = gt_binary[j]
                inter = (pred_prob * gt_j).sum()
                union = pred_prob.sum() + gt_j.sum() - inter
                ious.append(inter / union.clamp(min=1e-6))
            ious = torch.stack(ious)  # [num_group]

            # Cross-entropy: prediction should best match the spatially-correct GT
            # Treat IoUs as logits, target is the correct instance index
            target_idx = torch.tensor(target_in_group, device=device, dtype=torch.long)
            # Scale ious to make softmax sharper
            selection_loss = F.cross_entropy(ious.unsqueeze(0) * 10.0, target_idx.unsqueeze(0))
            total_loss = total_loss + selection_loss
            n_spatial += 1

    if n_spatial > 0:
        return total_loss / n_spatial
    return total_loss


def hungarian_match(pred_masks, gt_masks, num_objects, text_scores=None):
    """Match Q predicted masks to K GT masks using Hungarian algorithm.

    Args:
        pred_masks: [Q, H, W] predicted masks (logits)
        gt_masks: [K, H, W] ground truth masks
        num_objects: K (actual number of objects, may be < gt_masks.shape[0] if padded)
        text_scores: [Q, K] optional per-text scores for cost weighting

    Returns:
        matched_pairs: List[(query_idx, gt_idx)] — K pairs
        unmatched_queries: List[int] — Q-K unmatched query indices
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
        matched_pairs: List[(query_idx, gt_idx)] — K pairs
        unmatched_queries: List[int] — Q-K unmatched query indices
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


def triangulate_centroid(masks, extrinsics, intrinsics, mask_threshold=0.5):
    """Multi-view triangulation for 3D centroid estimation.

    Casts rays from each camera through mask centroids and finds the 3D point
    that minimizes squared distances to all rays (least-squares triangulation).

    Args:
        masks: [N, H, W] predicted mask logits for N views
        extrinsics: [N, 4, 4] camera-to-world transformation matrices
        intrinsics: [N, 3, 3] camera intrinsic matrices

    Returns:
        centroid: [3] triangulated 3D centroid in world frame
        valid: bool, whether triangulation succeeded
    """
    device = masks.device
    N = masks.shape[0]

    # Collect valid rays
    ray_origins = []
    ray_dirs = []

    for i in range(N):
        mask = masks[i]
        mask_binary = (torch.sigmoid(mask) > mask_threshold).float()

        # Skip if no valid mask
        if mask_binary.sum() < 10:
            continue

        # Compute 2D mask centroid (weighted by mask confidence)
        H, W = mask.shape
        y_coords = torch.arange(H, device=device).float().view(-1, 1).expand(H, W)
        x_coords = torch.arange(W, device=device).float().view(1, -1).expand(H, W)

        mask_sum = mask_binary.sum()
        u = (x_coords * mask_binary).sum() / mask_sum  # x centroid
        v = (y_coords * mask_binary).sum() / mask_sum  # y centroid

        # Compute ray in camera frame
        K_inv = torch.inverse(intrinsics[i])  # [3, 3]
        pixel_homo = torch.tensor([u, v, 1.0], device=device)  # [3]
        ray_cam = K_inv @ pixel_homo  # [3] direction in camera frame
        ray_cam = ray_cam / ray_cam.norm()  # normalize

        # Transform to world frame
        # extrinsics is camera-to-world: T_cw
        R = extrinsics[i, :3, :3]  # [3, 3] rotation
        t = extrinsics[i, :3, 3]   # [3] translation (camera position in world)

        ray_world = R @ ray_cam  # direction in world frame
        ray_world = ray_world / ray_world.norm()  # normalize
        origin_world = t  # camera origin in world frame

        ray_origins.append(origin_world)
        ray_dirs.append(ray_world)

    # Need at least 2 rays for triangulation
    if len(ray_origins) < 2:
        return torch.zeros(3, device=device), False

    # Stack rays
    origins = torch.stack(ray_origins)  # [M, 3]
    dirs = torch.stack(ray_dirs)        # [M, 3]
    M = origins.shape[0]

    # Least-squares triangulation:
    # Find point c that minimizes sum of squared distances to rays
    # For ray r_i(t) = o_i + t * d_i, distance to point c is:
    # ||(c - o_i) - ((c - o_i) . d_i) * d_i||
    #
    # Closed form: c = (sum_i (I - d_i d_i^T))^{-1} (sum_i (I - d_i d_i^T) o_i)

    I = torch.eye(3, device=device)
    A = torch.zeros(3, 3, device=device)
    b = torch.zeros(3, device=device)

    for i in range(M):
        d = dirs[i]
        o = origins[i]
        P = I - torch.outer(d, d)  # projection matrix orthogonal to ray
        A = A + P
        b = b + P @ o

    # Solve Ac = b (use lstsq to avoid SIGABRT on singular matrices)
    try:
        result = torch.linalg.lstsq(A.unsqueeze(0), b.unsqueeze(0).unsqueeze(-1))
        centroid = result.solution.squeeze()
    except RuntimeError:
        # Singular matrix (rays are parallel)
        # Fall back to midpoint of closest approach between first two rays
        o1, d1 = origins[0], dirs[0]
        o2, d2 = origins[1], dirs[1]
        # Solve for t1, t2 that minimize ||(o1 + t1*d1) - (o2 + t2*d2)||
        # This is a 2x2 linear system
        w0 = o1 - o2
        a = d1.dot(d1)
        b_val = d1.dot(d2)
        c = d2.dot(d2)
        d_val = d1.dot(w0)
        e = d2.dot(w0)
        denom = a * c - b_val * b_val
        if abs(denom) < 1e-8:
            # Rays are parallel, use midpoint of origins
            centroid = (o1 + o2) / 2
        else:
            t1 = (b_val * e - c * d_val) / denom
            t2 = (a * e - b_val * d_val) / denom
            p1 = o1 + t1 * d1
            p2 = o2 + t2 * d2
            centroid = (p1 + p2) / 2

    return centroid, True



def run_validation(model, val_dataloader, device, ddp, args, scaler=None):
    """
    Run validation loop and return metrics.

    Args:
        model: The model to evaluate
        val_dataloader: Validation DataLoader
        device: Device to run on
        ddp: DDP utilities
        args: Training arguments
        scaler: GradScaler for AMP (optional, used for consistent forward pass)

    Returns:
        dict with validation metrics: val_loss, val_iou, val_miou, val_mAcc, val_recall, num_categories, per_category_iou
    """
    model.eval()

    # Get base model (unwrap DDP if needed)
    base_model = model.module if hasattr(model, 'module') else model

    # Category IoU tracker for validation
    val_cat_metrics = CategoryMetricsTracker()

    total_loss = 0.0
    total_iou = 0.0
    total_macc = 0.0
    total_recall = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            images = batch['images'].to(device, non_blocking=True)
            gt_masks = batch['gt_masks'].to(device, non_blocking=True)
            prompts = batch['prompts']
            categories = batch.get('categories', prompts)

            B, N = images.shape[:2]

            # Get optional inputs
            intrinsics = batch.get('intrinsics')
            extrinsics = batch.get('extrinsics')
            cached_depth = batch.get('cached_depth')
            cached_da3_extrinsics = batch.get('cached_da3_extrinsics')
            cached_da3_intrinsics = batch.get('cached_da3_intrinsics')

            if intrinsics is not None:
                intrinsics = intrinsics.to(device, non_blocking=True)
            if extrinsics is not None:
                extrinsics = extrinsics.to(device, non_blocking=True)
            if cached_depth is not None:
                cached_depth = cached_depth.to(device, non_blocking=True)
            if cached_da3_extrinsics is not None:
                cached_da3_extrinsics = cached_da3_extrinsics.to(device, non_blocking=True)
            if cached_da3_intrinsics is not None:
                cached_da3_intrinsics = cached_da3_intrinsics.to(device, non_blocking=True)

            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                # Flatten batch for model: [B, N, C, H, W] -> [B*N, C, H, W]
                flat_images = images.view(B * N, *images.shape[2:])
                flat_gt = gt_masks.view(B * N, *gt_masks.shape[2:])

                # Expand prompts for each view
                flat_prompts = []
                flat_categories = []
                for b_idx in range(B):
                    for v_idx in range(N):
                        flat_prompts.append(prompts[b_idx])
                        flat_categories.append(categories[b_idx] if isinstance(categories, list) else categories)

                # Flatten cached depth
                flat_cached_depth = None
                if cached_depth is not None:
                    flat_cached_depth = cached_depth.view(B * N, *cached_depth.shape[2:])

                # Forward pass
                outputs = base_model(
                    flat_images,
                    text_prompts=flat_prompts,
                    gt_intrinsics=intrinsics.view(B * N, 3, 3) if intrinsics is not None else None,
                    gt_extrinsics=extrinsics.view(B * N, 4, 4) if extrinsics is not None else None,
                    cached_depth=flat_cached_depth,
                    da3_extrinsics=cached_da3_extrinsics.view(B * N, 4, 4) if cached_da3_extrinsics is not None else None,
                    da3_intrinsics=cached_da3_intrinsics.view(B * N, 3, 3) if cached_da3_intrinsics is not None else None,
                )

                pred_masks = outputs.get('pred_masks')
                if pred_masks is None:
                    continue

                # Compute loss per sample
                for i in range(B * N):
                    view_pred = pred_masks[i:i+1]
                    view_gt = flat_gt[i:i+1]

                    # Skip empty GT masks
                    if view_gt.sum() == 0:
                        continue

                    # Resize predictions to match GT mask size if needed
                    if view_pred.shape[-2:] != view_gt.shape[-2:]:
                        view_pred = F.interpolate(
                            view_pred.unsqueeze(1) if view_pred.dim() == 3 else view_pred,
                            size=view_gt.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)

                    # Compute loss (same as training: focal + dice)
                    view_loss = (args.focal_weight * focal_loss(view_pred, view_gt, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                args.dice_weight * dice_loss(view_pred.unsqueeze(1), view_gt.unsqueeze(1)))

                    # Compute metrics
                    pred_binary = (torch.sigmoid(view_pred) > 0.5).float()
                    gt_binary = (view_gt > 0.5).float()

                    # IoU
                    intersection = (pred_binary * gt_binary).sum()
                    union = pred_binary.sum() + gt_binary.sum() - intersection
                    iou = (intersection / union).item() if union > 0 else 1.0

                    # mAcc (mean of FG and BG accuracy)
                    fg_correct = ((pred_binary == 1) & (gt_binary == 1)).sum()
                    fg_total = (gt_binary == 1).sum()
                    bg_correct = ((pred_binary == 0) & (gt_binary == 0)).sum()
                    bg_total = (gt_binary == 0).sum()
                    fg_acc = (fg_correct / fg_total).item() if fg_total > 0 else 1.0
                    bg_acc = (bg_correct / bg_total).item() if bg_total > 0 else 1.0
                    macc = (fg_acc + bg_acc) / 2

                    # Recall = TP / (TP + FN)
                    recall = fg_acc  # Same as FG accuracy

                    total_loss += view_loss.item()
                    total_iou += iou
                    total_macc += macc
                    total_recall += recall
                    num_samples += 1

                    # Update category metrics (pass raw logits - update() applies sigmoid internally)
                    cat = flat_categories[i]
                    val_cat_metrics.update(view_pred.squeeze(), view_gt.squeeze(), cat)

    # Sync metrics across DDP ranks
    if ddp.is_distributed:
        metrics_tensor = torch.tensor([total_loss, total_iou, total_macc, total_recall, num_samples],
                                      device=device, dtype=torch.float32)
        torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss, total_iou, total_macc, total_recall, num_samples = metrics_tensor.tolist()

        # Sync category metrics
        val_cat_metrics.sync_across_ranks(ddp)

    # Compute averages
    if num_samples > 0:
        avg_loss = total_loss / num_samples
        avg_iou = total_iou / num_samples
        avg_macc = total_macc / num_samples
        avg_recall = total_recall / num_samples
    else:
        avg_loss = avg_iou = avg_macc = avg_recall = 0.0

    miou = val_cat_metrics.get_miou()
    cat_summary = val_cat_metrics.summary()

    model.train()

    return {
        'val_loss': avg_loss,
        'val_iou': avg_iou,
        'val_miou': miou,
        'val_mAcc': avg_macc,
        'val_recall': avg_recall,
        'val_num_samples': int(num_samples),
        'val_num_categories': cat_summary['num_categories'],
        'val_per_category_iou': cat_summary['per_category_iou'],
    }


def visualize_predictions(run_dir, epoch, images, gt_masks, outputs, prompts, max_samples=4):
    """Visualize predictions with SAM3-style mask overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    vis_dir = run_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    pred_masks = outputs.get('pred_masks')
    depth = outputs.get('depth')
    B = min(images.shape[0], max_samples)

    # 5 columns: Input+Prompt+Box, Depth, Pred Overlay, GT Overlay, Comparison
    fig, axes = plt.subplots(B, 5, figsize=(20, 4 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for i in range(B):
        # Prepare image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        H, W = img.shape[:2]

        # Get masks and resize to image resolution
        gt_mask = gt_masks[i].cpu().numpy()
        if gt_mask.shape != (H, W):
            from PIL import Image as PILImage
            gt_mask = np.array(PILImage.fromarray((gt_mask * 255).astype(np.uint8)).resize((W, H), PILImage.NEAREST)) / 255.0

        if pred_masks is not None:
            pred_mask = torch.sigmoid(pred_masks[i, 0]).cpu().numpy()
            if pred_mask.shape != (H, W):
                from PIL import Image as PILImage
                pred_mask = np.array(PILImage.fromarray((pred_mask * 255).astype(np.uint8)).resize((W, H), PILImage.BILINEAR)) / 255.0
            pred_binary = (pred_mask > 0.5).astype(float)
        else:
            pred_mask = np.zeros((H, W))
            pred_binary = pred_mask

        # Compute IoU
        intersection = (pred_binary * gt_mask).sum()
        union = pred_binary.sum() + gt_mask.sum() - intersection
        iou = intersection / union if union > 0 else 0

        # Extract box from GT mask for visualization
        box_rect = None
        if gt_mask.sum() > 0:
            nonzero = np.nonzero(gt_mask > 0.5)
            if len(nonzero[0]) > 0:
                y_min, y_max = nonzero[0].min(), nonzero[0].max()
                x_min, x_max = nonzero[1].min(), nonzero[1].max()
                box_rect = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Column 0: Input image with prompt label and box
        axes[i, 0].imshow(img)
        if box_rect is not None:
            rect = patches.Rectangle((box_rect[0], box_rect[1]), box_rect[2], box_rect[3],
                                      linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
            axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f'Prompt: "{prompts[i]}"', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Column 1: Depth
        if depth is not None:
            axes[i, 1].imshow(visualize_depth(depth[i, 0].cpu().numpy(), cmap="Spectral"))
            axes[i, 1].set_title('Depth (DA3)', fontsize=10)
        axes[i, 1].axis('off')

        # Column 2: Prediction overlay (SAM3 style - blue)
        pred_overlay = img.copy()
        pred_color = np.array([0.0, 0.5, 1.0])  # Blue
        for c in range(3):
            pred_overlay[..., c] = np.where(pred_binary > 0.5,
                                            pred_overlay[..., c] * 0.5 + pred_color[c] * 0.5,
                                            pred_overlay[..., c])
        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title(f'Prediction (IoU: {iou*100:.1f}%)', fontsize=10)
        axes[i, 2].axis('off')

        # Column 3: GT overlay on image (green)
        gt_overlay = img.copy()
        gt_color = np.array([0.0, 1.0, 0.0])  # Green
        gt_binary = (gt_mask > 0.5)
        for c in range(3):
            gt_overlay[..., c] = np.where(gt_binary,
                                          gt_overlay[..., c] * 0.5 + gt_color[c] * 0.5,
                                          gt_overlay[..., c])
        axes[i, 3].imshow(gt_overlay)
        gt_coverage = 100 * gt_mask.sum() / (H * W)
        axes[i, 3].set_title(f'GT ({gt_coverage:.1f}% coverage)', fontsize=10)
        axes[i, 3].axis('off')

        # Column 4: Side-by-side comparison (pred=blue, gt=green, overlap=cyan)
        comparison = img.copy()
        # Green where GT only
        gt_only = gt_binary & (pred_binary <= 0.5)
        # Red where pred only (false positive)
        pred_only = (pred_binary > 0.5) & ~gt_binary
        # Cyan where both (true positive)
        both = (pred_binary > 0.5) & gt_binary

        for c in range(3):
            comparison[..., c] = np.where(both, comparison[..., c] * 0.3 + np.array([0, 1, 1])[c] * 0.7, comparison[..., c])
            comparison[..., c] = np.where(gt_only, comparison[..., c] * 0.3 + np.array([0, 1, 0])[c] * 0.7, comparison[..., c])
            comparison[..., c] = np.where(pred_only, comparison[..., c] * 0.3 + np.array([1, 0, 0])[c] * 0.7, comparison[..., c])

        axes[i, 4].imshow(comparison)
        axes[i, 4].set_title('Cyan=TP, Green=FN, Red=FP', fontsize=9)
        axes[i, 4].axis('off')

    plt.tight_layout()
    # Handle both int epoch and string (e.g., "e1_b50")
    if isinstance(epoch, int):
        filename = f'epoch_{epoch:03d}.png'
    else:
        filename = f'{epoch}.png'
    plt.savefig(vis_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization to {vis_dir / filename}")


def main():
    # Parse config
    config = tyro.cli(TrainConfig)
    args = config.to_namespace()

    # Get parser defaults for detecting user overrides (used by --sam3-defaults and --resume)
    parser_defaults = TrainConfig.get_parser_defaults()

    # (argparse block removed -- all arguments are now defined in config.py)
    # --sam3-defaults: Apply SAM3-matching settings for any flag still at its default
    if args.sam3_defaults:
        sam3_overrides = {
            # Only entries that differ from argparse defaults:
            'pred_logits_source': 'text_scoring',   # default: 'mask_mean'
            'init_scoring_from_sam3': True,          # default: False
            'no_initial_text': True,                 # default: False
            'no_text_proj': True,                    # default: False — SAM3 has no text projection
            'clean_v': True,                         # default: False — SAM3 keeps V PE-free
            'per_layer_align': True,                 # default: False
            'init_text_crossattn_from_sam3': True,   # default: False (also implied by init_decoder)
            'init_decoder_from_sam3': True,           # default: False
        }
        sam3_applied = []
        for key, value in sam3_overrides.items():
            current = getattr(args, key, None)
            default = parser_defaults.get(key)
            if current == default and current != value:
                setattr(args, key, value)
                sam3_applied.append(f"{key}={value}")
        if sam3_applied:
            print(f"[--sam3-defaults] Applied: {', '.join(sam3_applied)}")
        else:
            print(f"[--sam3-defaults] All SAM3 settings already active or overridden by CLI")

    # pe_type=none implies no world PE
    if args.pe_type == 'none':
        args.use_world_pe = False

    # Auto-load config from resume checkpoint (with CLI override support)
    if args.resume:
        resume_path = Path(args.resume)
        # Config is in runs/train/{run_name}/, not checkpoints/
        # Extract run_name from checkpoint path (e.g., checkpoints/my_run/best.pt -> my_run)
        if resume_path.is_dir():
            run_name = resume_path.name
        else:
            run_name = resume_path.parent.name

        # Try multiple locations for config.json
        possible_config_paths = [
            resume_path / 'config.json' if resume_path.is_dir() else resume_path.parent / 'config.json',
            Path(__file__).parent.parent.parent / 'runs' / 'train' / run_name / 'config.json',
        ]
        config_path = None
        for p in possible_config_paths:
            if p.exists():
                config_path = p
                break

        if config_path:
            import json
            with open(config_path) as f:
                saved_config = json.load(f)

            # Apply saved config values ONLY for args that weren't explicitly set by user
            # (i.e., args that are still at their parser default)
            overridden = []
            loaded = []
            for key, value in saved_config.items():
                if hasattr(args, key):
                    current_value = getattr(args, key)
                    default_value = parser_defaults.get(key)

                    # If current value matches parser default, use saved config value
                    if current_value == default_value and value != default_value:
                        setattr(args, key, value)
                        loaded.append(key)
                    elif current_value != default_value and current_value != value:
                        overridden.append(f"{key}={current_value}")

            # Backward compat: if old config is missing keys that changed defaults,
            # use the OLD defaults to avoid shape mismatches when loading weights.
            # Only applies when these keys weren't in the saved config AND user didn't override.
            compat_defaults = {
                'dim_feedforward': 1024,    # Old default was 1024, new is 2048
                'post_norm': False,         # Old default was pre-norm, new is post-norm
                'ffn_fp32': False,          # Old runs used AMP in FFN, new default is FP32
                # use_query_pe default is False in both old and new, no compat needed
            }
            compat_applied = []
            for key, old_default in compat_defaults.items():
                if key not in saved_config and hasattr(args, key):
                    current = getattr(args, key)
                    parser_def = parser_defaults.get(key)
                    if current == parser_def and current != old_default:
                        setattr(args, key, old_default)
                        compat_applied.append(f"{key}={old_default}")

            print(f"[Resume] Loaded config from {config_path}")
            if compat_applied:
                print(f"[Resume]   Backward compat (old config missing keys): {', '.join(compat_applied)}")
            if loaded:
                print(f"[Resume]   Restored {len(loaded)} settings: {', '.join(loaded[:10])}" +
                      (f"... and {len(loaded)-10} more" if len(loaded) > 10 else ""))
            if overridden:
                print(f"[Resume]   CLI overrides: {', '.join(overridden)}")
        else:
            print(f"[Resume] Warning: No config.json found for run '{run_name}', using CLI args only")
            print(f"[Resume]   Searched: {[str(p) for p in possible_config_paths]}")

    # Initialize DDP (auto-detects if running via torchrun)
    ddp = DDPManager()
    ddp.init()

    if args.run_name is None:
        args.run_name = f"gasa_decoder_{datetime.now().strftime('%m%d_%H%M')}"

    set_seed(args.seed, ddp.rank)
    device = ddp.device

    # CUDA performance settings
    torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms (~5-15% speedup)

    ddp.print("TRIANGULANG: GASA DECODER (REPLACES SAM3's DECODER)")
    ddp.print(f"  World size: {ddp.world_size}, Rank: {ddp.rank}")

    # DA3 model capabilities
    da3_name = args.da3_model.split('/')[-1].upper()
    da3_has_pose = 'METRIC' not in da3_name and 'MONO' not in da3_name
    da3_has_metric = 'METRIC' in da3_name or 'NESTED' in da3_name
    ddp.print(f"  DA3 model: {da3_name}")
    ddp.print(f"    - Pose estimation capability: {da3_has_pose}")
    ddp.print(f"    - Metric depth: {da3_has_metric}")
    ddp.print(f"    - NOTE: Will use GT poses from dataset if available, otherwise identity pose")
    if args.use_sheaf_loss:
        ddp.print(f"  Sheaf loss: enabled (weight={args.sheaf_weight}, type={args.sheaf_type})")
        if args.sheaf_type == 'feature':
            ddp.print(f"    - NON-CONSTANT sheaf: learned restriction maps on R^256 -> R^{args.sheaf_d_edge}")
        else:
            ddp.print(f"    - Constant sheaf: identity restriction maps")
        ddp.print(f"    - Requires GT extrinsics from dataset for world-consistent pointmaps")
        ddp.print(f"    - If GT poses unavailable, falls back to camera-frame (less effective)")

    ddp.print(f"  GASA (geometric bias): {args.use_gasa}" + (" [ABLATION: disabled]" if not args.use_gasa else ""))
    if args.use_gt_poses_for_gasa:
        ddp.print(f"  GASA pointmaps: GT COLMAP poses (globally consistent)")
    elif args.use_da3_poses_for_gasa:
        ddp.print(f"  GASA pointmaps: DA3-NESTED estimated poses (chunk-consistent)")
    else:
        ddp.print(f"  GASA pointmaps: camera-frame (identity pose) [WARNING: no cross-view consistency]")
    kernel_desc = {'learned': f'learned MLP (dim={args.gasa_kernel_dim})', 'rbf': 'RBF exp(-d²/2σ²)', 'fixed': 'fixed φ(d) = -d'}
    bidir_str = " [BIDIRECTIONAL: boost+suppress]" if args.gasa_bidirectional else " [suppress-only]"
    ddp.print(f"  GASA kernel: {kernel_desc.get(args.gasa_kernel_type, args.gasa_kernel_type)}{bidir_str}")
    ddp.print(f"  Text proj: Linear(256→{args.d_model}) for cross-attention only (scoring uses raw text)")
    if args.pred_logits_source == 'text_scoring':
        sel_str = " (text-aware scoring, eval-only selection)"
    else:
        sel_str = " (text-agnostic mask mean)"
    ddp.print(f"  pred_logits source: {args.pred_logits_source}{sel_str}")
    ddp.print(f"  Depth cross-attention: {args.use_depth_crossattn}" + (" (queries attend to 3D positions)" if args.use_depth_crossattn else ""))
    ddp.print(f"  Iterative query positions: {args.use_iterative_pos}" + (" (P_Q = attn-weighted centroid)" if args.use_iterative_pos else " (P_Q = scene centroid)"))
    ddp.print(f"  Positional encoding: {args.pe_type}" + (" [ABLATION: disabled]" if args.pe_type == 'none' else ""))
    ddp.print(f"  Presence token: {args.use_presence_token} (weight={args.presence_weight}, focal={args.presence_focal}, α={args.presence_alpha}, γ={args.presence_gamma})")
    centroid_mode = " [triangulation]" if args.use_triangulation else (" [mask-based]" if args.mask_based_centroid else " [attention-based]")
    ddp.print(f"  Centroid head: {args.use_centroid_head}" + (f" (weight={args.centroid_weight}){centroid_mode}" if args.use_centroid_head else ""))
    if args.eval_localization and not args.use_centroid_head:
        ddp.print(f"  Eval localization: {args.eval_localization} (tracking Acc@m from mask+depth, no loss)")
    ddp.print(f"  Box prompts: {args.use_box_prompts}" + (f" (dropout={args.box_prompt_dropout})" if args.box_prompt_dropout > 0 else ""))
    ddp.print(f"  Point prompts: {args.use_point_prompts} ({args.num_pos_points} pos + {args.num_neg_points} neg)" + (f" (dropout={args.point_prompt_dropout})" if args.point_prompt_dropout > 0 else ""))
    ddp.print(f"  Prompt type: {args.prompt_type}")
    ddp.print(f"  Mask selection: {args.mask_selection}")
    if args.mask_selection == 'predicted_iou' and not args.use_iou_head:
        ddp.print("  WARNING: --mask-selection predicted_iou requires --use-iou-head! Falling back to confidence.")
        args.mask_selection = 'confidence'
    ddp.print(f"  Sheaf consistency loss: {args.use_sheaf_loss}" + (f" (type={args.sheaf_type}, weight={args.sheaf_weight}, threshold={args.sheaf_threshold}m)" if args.use_sheaf_loss else " [ABLATION: disabled]"))
    ddp.print(f"  Contrastive loss: {args.contrastive_weight > 0}" + (f" (weight={args.contrastive_weight}, margin={args.contrastive_margin}, source={args.contrastive_source})" if args.contrastive_weight > 0 else ""))
    ddp.print(f"  Align loss (SAM3-style): {args.align_weight > 0}" + (f" (weight={args.align_weight}, α={args.align_alpha}, γ={args.align_gamma}, τ={args.align_tau})" if args.align_weight > 0 else ""))
    ddp.print(f"  Lovász loss: {args.lovasz_weight > 0}" + (f" (weight={args.lovasz_weight})" if args.lovasz_weight > 0 else ""))
    ddp.print(f"  Point sampling: {args.use_point_sampling}" + (f" ({args.num_sample_points} points)" if args.use_point_sampling else ""))
    ddp.print(f"  Loss at native res: {args.loss_at_native_res}")
    ddp.print(f"  IoU head: {args.use_iou_head}" + (f" (MSE weight={args.iou_head_weight})" if args.use_iou_head else ""))
    ddp.print(f"  Semantic union GT: {args.semantic_union}" + (" (text='mug' matches ALL mugs)" if args.semantic_union else " [per-instance mode]"))
    ddp.print(f"  Class-balanced sampling: {args.class_balanced}" + (f" (power={args.class_balance_power})" if args.class_balanced else ""))
    if args.min_mask_coverage > 0:
        # Coverage is computed at ORIGINAL resolution (~1752×1168 = ~2M pixels)
        orig_pixels = int(args.min_mask_coverage * 1752 * 1168)
        ddp.print(f"  Min mask coverage: {args.min_mask_coverage*100:.2f}% (≈{orig_pixels} pixels at ~1752×1168 original)")
    else:
        ddp.print(f"  Min mask coverage: disabled (any non-empty mask is valid)")

    # Get dataset config defaults
    dataset_config = get_dataset_config(args.dataset)
    data_root = args.data_root or str(PROJECT_ROOT / dataset_config.get('data_root', f'data/{args.dataset}'))
    split = args.split or dataset_config.get('split', 'train')

    ddp.print(f"\nLoading dataset '{args.dataset}'...")
    ddp.print(f"  Data root: {data_root}")
    ddp.print(f"  Split: {split}")

    # SAM3 mask decoder output resolution = (input_size / 14) * 4
    native_mask_res = (args.resolution // 14) * 4
    ddp.print(f"  Mask size: {native_mask_res}x{native_mask_res} (native for resolution={args.resolution})")

    # --multi-object flag overrides --num-objects to dynamic mode (0)
    # MUST happen before dataset creation so the dataset sees the correct value
    if args.multi_object:
        args.num_objects = 0
        # Auto-set samples_per_scene for multi-object so dataset is large enough for batching.
        # scene_grouped mode: len(dataset) = num_scenes * samples_per_scene.
        # Need enough samples per GPU for batch_size to work with drop_last=True.
        if args.samples_per_scene == 1:
            # Aim for ~5 batches per GPU (enough for stable training)
            target_batches = 5
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            per_gpu_scenes = max(1, args.max_scenes // world_size)
            min_needed = args.batch_size * target_batches
            if per_gpu_scenes < min_needed:
                args.samples_per_scene = max(2, (min_needed + per_gpu_scenes - 1) // per_gpu_scenes)
                ddp.print(f"  Auto-set --samples-per-scene {args.samples_per_scene} for multi-object "
                         f"({args.max_scenes} scenes / {world_size} GPUs = {per_gpu_scenes}/GPU, "
                         f"target ≥{target_batches} batches of {args.batch_size})")

    if args.num_objects != 1:
        if args.num_objects == 0:
            ddp.print(f"  Multi-object training: DYNAMIC K (all visible objects per sample, like SAM3)")
        else:
            ddp.print(f"  Multi-object training: K={args.num_objects} objects per sample")
        ddp.print(f"  Hungarian matching enabled: {args.num_queries} queries competing for K GT objects")

    dataset = get_dataset(
        dataset_name=args.dataset,
        data_root=data_root,
        split=split,
        views_per_sample=args.views,
        image_size=(args.resolution, args.resolution),
        mask_size=(native_mask_res, native_mask_res),
        max_scenes=args.max_scenes,
        # ScanNetPP-specific options
        use_undistorted=True,
        supervised=True,
        semantic_union=args.semantic_union,
        sampling_strategy=args.sampling_strategy,  # View sampling: stratified, chunk_aware, etc.
        da3_chunk_size=args.da3_chunk_size,  # Chunk size for chunk_aware sampling
        use_cached_depth=args.use_cached_depth,  # 2-4x faster training
        da3_cache_name=args.da3_cache_name,  # 'da3_cache' or 'da3_nested_cache'
        min_category_samples=args.min_category_samples,  # Filter rare categories
        exclude_categories=args.exclude_categories,  # Exclude structural elements (wall, floor, ceiling)
        include_categories=args.include_categories,  # Whitelist specific categories
        # NVOS/SpinNeRF-specific options
        num_pos_points=args.num_pos_points,
        num_neg_points=args.num_neg_points,
        samples_per_scene=args.samples_per_scene,  # Augment small datasets
        # uCO3D-specific options
        frames_per_sequence=args.frames_per_sequence,
        samples_per_sequence=args.samples_per_sequence,
        # PartImageNet-specific options
        part_query_mode=args.part_query_mode,
        augment=args.augment,
        # Multi-object training
        num_objects_per_sample=args.num_objects,
    )

    # Log sampling strategy (only relevant for ScanNet++ with DA3-NESTED chunks)
    if args.dataset == 'scannetpp':
        ddp.print(f"  Sampling strategy: {args.sampling_strategy}")
        if args.sampling_strategy == 'chunk_aware':
            ddp.print(f"  DA3 chunk size: {args.da3_chunk_size}")
            ddp.print(f"  Note: Views will be sampled from same DA3-NESTED chunk for world-frame consistency")

    # Verify cached depth is available if requested
    if args.use_cached_depth:
        ddp.print(f"  da3_cache_dir: {dataset.da3_cache_dir} ({args.da3_cache_name})")
        ddp.print(f"  da3_cache_dir exists: {dataset.da3_cache_dir.exists() if dataset.da3_cache_dir else 'N/A'}")
        test_sample = dataset[0]
        ddp.print(f"  Sample keys: {list(test_sample.keys())}")
        if 'cached_depth' in test_sample:
            ddp.print(f"  Cached depth ENABLED: shape={test_sample['cached_depth'].shape}")
            # Check for cached poses (da3_nested_cache format)
            if 'cached_da3_extrinsics' in test_sample:
                ddp.print(f"  Cached DA3 poses ENABLED: extrinsics={test_sample['cached_da3_extrinsics'].shape}")
        else:
            ddp.print("  WARNING: --use-cached-depth set but cache not found! DA3 will run live.")

    # Compute class-balanced sample weights if enabled
    sample_weights = None
    if args.class_balanced and getattr(dataset, 'object_samples', None):
        ddp.print("Computing class-balanced sample weights...")
        # Count category frequencies
        category_counts = Counter()
        for sample in dataset.object_samples:
            category = sample.get('label', sample.get('category', 'unknown'))
            category_counts[category] += 1

        # Compute per-sample weights (inverse frequency raised to power)
        # power=1.0: pure inverse freq, power=0.5: sqrt (smoother), power=0: uniform
        total_samples = len(dataset.object_samples)
        num_categories = len(category_counts)
        sample_weights = []
        for sample in dataset.object_samples:
            category = sample.get('label', sample.get('category', 'unknown'))
            freq = category_counts[category] / total_samples
            # Weight = (1/freq)^power, normalized so mean=1
            weight = (1.0 / (freq * num_categories)) ** args.class_balance_power
            sample_weights.append(weight)

        # Normalize so weights sum to len(dataset)
        weight_sum = sum(sample_weights)
        sample_weights = [w * len(sample_weights) / weight_sum for w in sample_weights]

        # Log category distribution
        sorted_cats = category_counts.most_common()
        ddp.print(f"  Categories: {num_categories}, Samples: {total_samples}")
        ddp.print(f"  Most common: {sorted_cats[:3]}")
        ddp.print(f"  Least common: {sorted_cats[-3:]}")
        ddp.print(f"  Weight range: [{min(sample_weights):.2f}, {max(sample_weights):.2f}]")

    # Use partial to pass max_objects to collate_fn for multi-object K capping
    collate = partial(collate_fn, max_objects=args.max_objects) if args.max_objects > 0 else collate_fn
    if args.max_objects > 0 and args.num_objects != 1:
        ddp.print(f"  Max objects per sample: {args.max_objects} (capping K)")

    dataloader = ddp.wrap_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0 and not args.no_persistent_workers,  # Keep workers alive (disable with --no-persistent-workers)
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,  # Prefetch batches
        sample_weights=sample_weights,  # For class-balanced sampling
    )
    ddp.print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/gpu")

    # Note: --no-save-best-val is handled by tyro automatically (negation of save_best_val)

    # Load validation dataset if validation is enabled
    val_dataset = None
    val_dataloader = None
    if args.val_every > 0:
        ddp.print(f"\nLoading validation dataset (split='{args.val_split}')...")
        val_max_samples = args.val_max_samples

        # For PartImageNet, use 'all' mode for validation to evaluate all parts
        val_part_query_mode = 'all' if args.dataset == 'partimagenet' else args.part_query_mode

        val_dataset = get_dataset(
            dataset_name=args.dataset,
            data_root=data_root,
            split=args.val_split,
            views_per_sample=args.views,
            image_size=(args.resolution, args.resolution),
            mask_size=(native_mask_res, native_mask_res),
            max_scenes=val_max_samples,
            # ScanNetPP-specific options
            use_undistorted=True,
            supervised=True,
            semantic_union=args.semantic_union,
            sampling_strategy=args.sampling_strategy,
            da3_chunk_size=args.da3_chunk_size,
            use_cached_depth=args.use_cached_depth,
            da3_cache_name=args.da3_cache_name,
            min_category_samples=1,  # Don't filter rare categories for val
            exclude_categories=args.exclude_categories,  # Match training exclusions
            include_categories=args.include_categories,  # Match training whitelist
            # NVOS/SpinNeRF-specific options
            num_pos_points=args.num_pos_points,
            num_neg_points=args.num_neg_points,
            samples_per_scene=1,  # Single sample per scene for val
            # uCO3D-specific options
            frames_per_sequence=args.frames_per_sequence,
            samples_per_sequence=1,  # Single sample per sequence for val
            # PartImageNet-specific options
            part_query_mode=val_part_query_mode,
            augment=False,  # No augmentation for validation
        )

        val_dataloader = ddp.wrap_dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate, pin_memory=True, drop_last=False,
            persistent_workers=False,  # Don't keep val workers alive
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        ddp.print(f"Validation dataset: {len(val_dataset)} samples, {len(val_dataloader)} batches/gpu")
        ddp.print(f"  Validation every {args.val_every} epoch(s)")
        ddp.print(f"  Save best based on: {'validation mIoU' if args.save_best_val else 'training IoU'}")

    ddp.print("\nLoading models...")
    sam3_model = build_sam3_image_model(bpe_path=_BPE_PATH, img_size=args.resolution).to(device)

    # Skip DA3 loading when depth is not needed (no GASA, no world PE, no centroid)
    needs_depth = args.use_gasa or args.use_world_pe or (args.pe_type not in ('none', None)) or args.use_centroid_head or args.use_cached_depth
    if needs_depth:
        da3_model = DepthAnything3.from_pretrained(args.da3_model).to(device)
    else:
        ddp.print("  Skipping DA3 loading (no GASA, no PE, no centroid — depth not needed)")
        da3_model = None

    # Synchronize all ranks after model loading to prevent DDP hang
    # This ensures all ranks have completed model downloads/loading before DDP wrapping
    ddp.barrier()
    ddp.print("All ranks synchronized after model loading")
    print(f"[R{ddp.rank}] Creating TrianguLangModel...", flush=True)

    # Optional: torch.compile for faster inference (slower first epoch due to compilation)
    if args.torch_compile:
        ddp.print(f"Compiling SAM3 backbone with torch.compile(mode='{args.compile_mode}')...")
        ddp.print("  Note: First epoch will be slower due to JIT compilation")
        # Only compile SAM3 backbone (the heavy part) - DA3 has data-dependent control flow
        # that torch.compile can't handle (sky mask computation with if statements)
        sam3_model.backbone = torch.compile(sam3_model.backbone, mode=args.compile_mode, fullgraph=False)
        ddp.print("  SAM3 compilation registered (DA3 not compiled due to control flow)")

    # Handle prompt-type to determine which prompts to enable
    use_box = args.use_box_prompts
    use_point = args.use_point_prompts
    if args.prompt_type == 'text_only':
        use_box, use_point = False, False
    elif args.prompt_type == 'box_only':
        use_point = False
    elif args.prompt_type == 'point_only':
        use_box = False
    elif args.prompt_type == 'text_box':
        use_point = False
    elif args.prompt_type == 'text_point':
        use_box = False
    # 'all' and 'random' keep both enabled (random handled per-batch in forward)

    model = TrianguLangModel(
        sam3_model=sam3_model, da3_model=da3_model, d_model=args.d_model, n_heads=args.n_heads,
        num_decoder_layers=args.num_decoder_layers, num_queries=args.num_queries, train_seghead=args.train_seghead,
        attn_map_size=args.attn_map_size,
        use_presence_token=args.use_presence_token, use_box_prompts=use_box, use_point_prompts=use_point,
        mask_selection=args.mask_selection,
        use_world_pe=args.use_world_pe, use_gasa=args.use_gasa,
        use_centroid_head=args.use_centroid_head,
        box_prompt_dropout=args.box_prompt_dropout, point_prompt_dropout=args.point_prompt_dropout,
        num_pos_points=args.num_pos_points, num_neg_points=args.num_neg_points,
        use_iterative_pos=args.use_iterative_pos,
        cross_view=args.cross_view, pe_type=args.pe_type,
        da3_model_name=args.da3_model.split('/')[-1],  # Extract model name from path
        use_iou_head=args.use_iou_head,
        use_spatial_tokens=args.use_spatial_tokens,
        use_spatial_attn_bias=getattr(args, 'use_spatial_attn_bias', False),
        use_text_spatial_bias=getattr(args, 'use_text_spatial_bias', False),
        use_image_to_token=getattr(args, 'use_image_to_token', False),
        use_pos_refine=getattr(args, 'use_pos_refine', False),
        use_box_rpb=getattr(args, 'use_box_rpb', False),
        use_spatial_points=args.use_spatial_points,
        use_object_aware_spatial=args.use_object_aware_spatial,
        da3_resolution=args.da3_resolution,
        pointmap_normalize=not args.no_pointmap_normalize,
        resolution=args.resolution,
        gasa_beta_init=args.gasa_beta_init,
        use_da3_poses_for_gasa=args.use_da3_poses_for_gasa,
        use_gt_poses_for_gasa=args.use_gt_poses_for_gasa,
        sheaf_use_gt_poses=args.sheaf_use_gt_poses,
        gasa_kernel_dim=args.gasa_kernel_dim,
        gasa_fixed_kernel=args.gasa_fixed_kernel,
        gasa_kernel_type=args.gasa_kernel_type,
        use_depth_crossattn=args.use_depth_crossattn,
        per_layer_text=args.per_layer_text,
        pred_logits_source=args.pred_logits_source,
        gasa_bidirectional=args.gasa_bidirectional,
        query_proj_mlp=args.query_proj_mlp,
        no_query_proj=getattr(args, 'no_query_proj', False),
        train_mask_embed=args.train_mask_embed,
        use_mask_refiner=getattr(args, 'use_mask_refiner', False),
        dim_feedforward=args.dim_feedforward,
        post_norm=args.post_norm,
        use_query_pe=args.use_query_pe,
        ffn_fp32=args.ffn_fp32,
        no_initial_text=args.no_initial_text,
        no_text_proj=args.no_text_proj,
        clean_v=args.clean_v,
        additive_pe=args.additive_pe,
        grouped_text_attn=args.grouped_text_attn,
    ).to(device)
    # Set per-text decode mode (processes each text through its own decoder pass)
    model.per_text_decode = getattr(args, 'per_text_decode', False)
    model.sam3_multi_object = getattr(args, 'sam3_multi_object', False)
    # --init-decoder-from-sam3 implies the other two init flags
    if args.init_decoder_from_sam3:
        args.init_text_crossattn_from_sam3 = True
        args.init_scoring_from_sam3 = True

    # Initialize scoring heads from SAM3's pretrained DotProductScoring weights
    if args.init_scoring_from_sam3 and hasattr(model.sam3, 'dot_prod_scoring'):
        dps = model.sam3.dot_prod_scoring
        if dps.prompt_mlp is not None:
            model.gasa_decoder.scoring_prompt_mlp.load_state_dict(dps.prompt_mlp.state_dict())
            ddp.print("  Initialized scoring_prompt_mlp from SAM3 DotProductScoring")
        model.gasa_decoder.scoring_prompt_proj.load_state_dict(dps.prompt_proj.state_dict())
        model.gasa_decoder.scoring_hs_proj.load_state_dict(dps.hs_proj.state_dict())
        ddp.print("  Initialized scoring_prompt_proj + scoring_hs_proj from SAM3 DotProductScoring")
    elif args.init_scoring_from_sam3:
        ddp.print("  WARNING: --init-scoring-from-sam3 requested but SAM3 has no dot_prod_scoring module")

    # Initialize text cross-attention from SAM3's pretrained decoder
    if args.init_text_crossattn_from_sam3 and hasattr(model.sam3, 'transformer'):
        sam3_decoder_layers = model.sam3.transformer.decoder.layers
        num_sam3_layers = len(sam3_decoder_layers)
        transferred = 0

        # Per-layer text cross-attention
        if model.gasa_decoder.per_layer_text:
            for i, gasa_layer in enumerate(model.gasa_decoder.layers):
                if hasattr(gasa_layer, 'text_cross_attn') and i < num_sam3_layers:
                    sam3_ca = sam3_decoder_layers[i].ca_text
                    # Transfer MultiheadAttention weights (same shapes: d_model=256, n_heads=8)
                    gasa_layer.text_cross_attn.in_proj_weight.data.copy_(sam3_ca.in_proj_weight.data)
                    gasa_layer.text_cross_attn.in_proj_bias.data.copy_(sam3_ca.in_proj_bias.data)
                    gasa_layer.text_cross_attn.out_proj.weight.data.copy_(sam3_ca.out_proj.weight.data)
                    gasa_layer.text_cross_attn.out_proj.bias.data.copy_(sam3_ca.out_proj.bias.data)
                    # Transfer LayerNorm weights
                    if hasattr(gasa_layer, 'text_norm') and hasattr(sam3_decoder_layers[i], 'catext_norm'):
                        gasa_layer.text_norm.weight.data.copy_(sam3_decoder_layers[i].catext_norm.weight.data)
                        gasa_layer.text_norm.bias.data.copy_(sam3_decoder_layers[i].catext_norm.bias.data)
                    transferred += 1

        # Initial text cross-attention (GASADecoder level) — init from layer 0's weights
        if hasattr(model.gasa_decoder, 'text_cross_attn') and num_sam3_layers > 0:
            sam3_ca0 = sam3_decoder_layers[0].ca_text
            model.gasa_decoder.text_cross_attn.in_proj_weight.data.copy_(sam3_ca0.in_proj_weight.data)
            model.gasa_decoder.text_cross_attn.in_proj_bias.data.copy_(sam3_ca0.in_proj_bias.data)
            model.gasa_decoder.text_cross_attn.out_proj.weight.data.copy_(sam3_ca0.out_proj.weight.data)
            model.gasa_decoder.text_cross_attn.out_proj.bias.data.copy_(sam3_ca0.out_proj.bias.data)
            if hasattr(model.gasa_decoder, 'text_norm') and hasattr(sam3_decoder_layers[0], 'catext_norm'):
                model.gasa_decoder.text_norm.weight.data.copy_(sam3_decoder_layers[0].catext_norm.weight.data)
                model.gasa_decoder.text_norm.bias.data.copy_(sam3_decoder_layers[0].catext_norm.bias.data)
            transferred += 1

        # Initialize text_proj as identity so pretrained ca_text receives raw text
        # SAM3 feeds raw 256-dim text directly to ca_text with no extra projection.
        # Our text_proj (Linear 256→256) would transform text BEFORE it hits the
        # pretrained cross-attention, undermining the SAM3 init. Identity = passthrough.
        if hasattr(model.gasa_decoder, 'text_proj'):
            nn.init.eye_(model.gasa_decoder.text_proj.weight)
            nn.init.zeros_(model.gasa_decoder.text_proj.bias)
            ddp.print("  Initialized text_proj as identity (SAM3 uses no text projection)")

        ddp.print(f"  Initialized {transferred} text cross-attention modules from SAM3 decoder")
    elif args.init_text_crossattn_from_sam3:
        ddp.print("  WARNING: --init-text-crossattn-from-sam3 requested but SAM3 has no transformer module")

    # Initialize decoder self-attn, FFN, norms from SAM3's pretrained decoder
    if args.init_decoder_from_sam3 and hasattr(model.sam3, 'transformer'):
        sam3_decoder = model.sam3.transformer.decoder
        sam3_layers = sam3_decoder.layers
        num_sam3_layers = len(sam3_layers)
        num_gasa_layers = len(model.gasa_decoder.layers)
        transferred = []

        def _copy_mha(dst, src, name):
            """Copy MultiheadAttention weights. batch_first differs but weights are identical."""
            dst.in_proj_weight.data.copy_(src.in_proj_weight.data)
            dst.in_proj_bias.data.copy_(src.in_proj_bias.data)
            dst.out_proj.weight.data.copy_(src.out_proj.weight.data)
            dst.out_proj.bias.data.copy_(src.out_proj.bias.data)
            transferred.append(name)

        def _copy_ln(dst, src, name):
            """Copy LayerNorm weights."""
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)
            transferred.append(name)

        # Per-layer transfers
        for i in range(min(num_gasa_layers, num_sam3_layers)):
            gasa_l = model.gasa_decoder.layers[i]
            sam3_l = sam3_layers[i]

            # Self-attention (architecturally identical: d_model=256, n_heads=8)
            _copy_mha(gasa_l.self_attn, sam3_l.self_attn, f"layers[{i}].self_attn")

            # Norm mapping: SAM3 norm2 = self-attn norm → GASA norm1
            #               SAM3 norm1 = cross-attn norm → GASA norm2
            #               SAM3 norm3 = FFN norm → GASA norm3
            _copy_ln(gasa_l.norm1, sam3_l.norm2, f"layers[{i}].norm1←norm2 (self-attn)")
            _copy_ln(gasa_l.norm2, sam3_l.norm1, f"layers[{i}].norm2←norm1 (cross-attn)")
            _copy_ln(gasa_l.norm3, sam3_l.norm3, f"layers[{i}].norm3 (FFN)")

            # FFN: SAM3 has linear1/linear2, we have ffn[0]/ffn[3] (nn.Sequential)
            gasa_l.ffn[0].weight.data.copy_(sam3_l.linear1.weight.data)
            gasa_l.ffn[0].bias.data.copy_(sam3_l.linear1.bias.data)
            gasa_l.ffn[3].weight.data.copy_(sam3_l.linear2.weight.data)
            gasa_l.ffn[3].bias.data.copy_(sam3_l.linear2.bias.data)
            transferred.append(f"layers[{i}].ffn (linear1→ffn[0], linear2→ffn[3])")

        # Output norm
        if hasattr(sam3_decoder, 'norm') and hasattr(model.gasa_decoder, 'norm'):
            _copy_ln(model.gasa_decoder.norm, sam3_decoder.norm, "output norm")

        # Presence token embedding + norm + head
        if (hasattr(model.gasa_decoder, 'presence_token') and
                hasattr(sam3_decoder, 'presence_token') and
                sam3_decoder.presence_token is not None):
            model.gasa_decoder.presence_token.weight.data.copy_(sam3_decoder.presence_token.weight.data)
            transferred.append("presence_token")
            if hasattr(model.gasa_decoder, 'presence_norm') and hasattr(sam3_decoder, 'presence_token_out_norm'):
                _copy_ln(model.gasa_decoder.presence_norm, sam3_decoder.presence_token_out_norm, "presence_norm")
            # Presence head: SAM3 MLP(256,256,1,3) → layers[0,1,2]
            # Ours: Sequential([0]=Linear, [1]=ReLU, [2]=Linear, [3]=ReLU, [4]=Linear)
            if hasattr(model.gasa_decoder, 'presence_head') and hasattr(sam3_decoder, 'presence_token_head'):
                sam3_ph = sam3_decoder.presence_token_head
                our_ph = model.gasa_decoder.presence_head
                # SAM3 MLP.layers[i] → our Sequential[i*2] (interleaved with ReLU)
                for j, sam3_linear in enumerate(sam3_ph.layers):
                    our_linear = our_ph[j * 2]  # 0→0, 1→2, 2→4
                    our_linear.weight.data.copy_(sam3_linear.weight.data)
                    our_linear.bias.data.copy_(sam3_linear.bias.data)
                transferred.append("presence_head (3 layers)")

        # Query embeddings: transfer first min(ours, SAM3's) queries
        if hasattr(sam3_decoder, 'query_embed') and hasattr(model.gasa_decoder, 'query_embed'):
            sam3_nq = sam3_decoder.query_embed.weight.shape[0]
            our_nq = model.gasa_decoder.query_embed.weight.shape[0]
            n_transfer = min(our_nq, sam3_nq)
            model.gasa_decoder.query_embed.weight.data[:n_transfer].copy_(
                sam3_decoder.query_embed.weight.data[:n_transfer]
            )
            transferred.append(f"query_embed ({n_transfer}/{our_nq} queries from SAM3's {sam3_nq})")

        ddp.print(f"  --init-decoder-from-sam3: Transferred {len(transferred)} module groups:")
        for t in transferred:
            ddp.print(f"    {t}")
    elif args.init_decoder_from_sam3:
        ddp.print("  WARNING: --init-decoder-from-sam3 requested but SAM3 has no transformer module")

    print(f"[R{ddp.rank}] TrianguLangModel created and moved to device", flush=True)

    # Enable profiling if requested
    if args.profile:
        ddp.print("\n  Profiling enabled - will print timing summary after first epoch")

    # Wrap model with DDP
    # find_unused_parameters=False is safe because GASADecoder.forward() now explicitly
    # connects all conditional params (text_proj, world_pe, etc.) to the output.
    # This avoids the ~10-20% performance overhead of find_unused_parameters=True.
    # Note: cross-view mode now goes through forward() with cross_view_mode=True, which
    # dispatches to forward_multiview() INSIDE the DDP forward pass, ensuring proper gradient sync.
    use_find_unused = False  # Safe: all params connected, cross-view goes through DDP properly
    print(f"[R{ddp.rank}] Wrapping with DDP (find_unused_parameters={use_find_unused})...", flush=True)
    model = ddp.wrap_model(model, find_unused_parameters=use_find_unused)
    print(f"[R{ddp.rank}] DDP wrap complete", flush=True)
    base_model = ddp.get_model(model)  # Get underlying model for state_dict access

    # Enable profiling on base model
    if args.profile:
        base_model.set_profile(True)

    # Setup spatial augmentation (adds "nearest", "leftmost", etc. to labels)
    spatial_augmentor = None
    gt_aware_spatial = None  # GT-aware augmentor (uses actual mask positions)
    if args.spatial_augment_prob > 0:
        if args.spatial_gt_aware:
            # GT-aware: only adds qualifiers based on actual mask positions
            gt_aware_spatial = GTAwareSpatialAugmentor(
                augment_prob=args.spatial_augment_prob,
                relational_prob=args.spatial_relational_prob,
                multi_instance_only=args.spatial_multi_instance_only,
                qualifier_diversity=True
            )
            ddp.print(f"GT-aware spatial augmentation enabled (prob={args.spatial_augment_prob}, "
                      f"relational={args.spatial_relational_prob})")
            if not args.use_cached_depth:
                ddp.print(f"  WARNING: --spatial-gt-aware requires --use-cached-depth for accurate depth!")
        else:
            # Legacy: random qualifiers (may be incorrect!)
            spatial_augmentor = SpatialAugmentor(augment_prob=args.spatial_augment_prob)
            ddp.print(f"Spatial augmentation enabled (prob={args.spatial_augment_prob}) [RANDOM - labels may be wrong!]")

    # Log spatial reasoning options
    if args.use_spatial_tokens or args.use_spatial_points or args.use_object_aware_spatial or args.spatial_augment_prob > 0:
        ddp.print(f"\n  Spatial Reasoning:")
        ddp.print(f"    Spatial tokens: {args.use_spatial_tokens}")
        ddp.print(f"    Spatial-as-points: {args.use_spatial_points}")
        ddp.print(f"    Object-aware spatial: {args.use_object_aware_spatial}" + (" (uses mask+depth for 'nearest chair')" if args.use_object_aware_spatial else ""))
        ddp.print(f"    Spatial augmentation: {args.spatial_augment_prob > 0} (prob={args.spatial_augment_prob})")
        ddp.print(f"    GT-aware augmentation: {args.spatial_gt_aware}" + (" (uses qualifiers from GT masks)" if args.spatial_gt_aware else ""))
        if args.spatial_gt_aware and args.spatial_relational_prob > 0:
            ddp.print(f"    Relational queries: {args.spatial_relational_prob} (e.g., 'chair next to table')")
        if args.spatial_ranking_weight > 0:
            ddp.print(f"    Spatial ranking loss: weight={args.spatial_ranking_weight}, margin={args.spatial_ranking_margin}")
    if args.mask_smooth_kernel > 0:
        ddp.print(f"  Mask smoothing: {args.mask_smooth_kernel}x{args.mask_smooth_kernel} avg_pool (matches eval-time LangSplat)")

    # Setup sheaf consistency loss
    sheaf_loss_fn = None
    feature_sheaf_loss_fn = None  # Non-constant sheaf on encoder features
    if args.use_sheaf_loss and args.sheaf_weight > 0:
        if args.sheaf_type == 'feature':
            # Non-constant sheaf: learned restriction maps on SAM3 encoder features
            feature_sheaf_loss_fn = FeatureSheafLoss(
                d_stalk=256,  # SAM3 encoder feature dim
                d_edge=args.sheaf_d_edge,
                context_dim=5,  # depth, corr_dist, displacement, viewing_angle, depth_edge
            ).to(device)
            n_sheaf_params = sum(p.numel() for p in feature_sheaf_loss_fn.parameters())
            ddp.print(f"Feature sheaf loss initialized (non-constant, learned restriction maps)")
            ddp.print(f"  Stalks: R^256 (SAM3 features), Edge space: R^{args.sheaf_d_edge}")
            ddp.print(f"  Trainable params: {n_sheaf_params:,}")
            ddp.print(f"  Threshold: {args.sheaf_threshold}m, max_frame_distance: {args.sheaf_max_frame_distance}")
        else:
            # Constant sheaf: identity restriction maps (original behavior)
            sheaf_loss_fn = SheafConsistencyLoss(
                threshold=args.sheaf_threshold,
                use_soft_correspondences=args.sheaf_soft_correspondences,
                sigma=args.sheaf_sigma,
                detach_target=args.sheaf_detach_target,
                max_frame_distance=args.sheaf_max_frame_distance,
                symmetric_detach=args.sheaf_symmetric_detach,
                mutual_nn=args.sheaf_mutual_nn,
            )
            if args.sheaf_soft_correspondences:
                soft_str = f", Gaussian sigma={args.sheaf_sigma}m (cutoff at 3σ={3*args.sheaf_sigma:.2f}m)"
            else:
                soft_str = f", hard NN (threshold={args.sheaf_threshold}m)"
            detach_str = ", detach_target=True" if args.sheaf_detach_target else ", detach_target=False (CAUTION: may cause fighting)"
            sym_str = " (symmetric)" if args.sheaf_symmetric_detach and args.sheaf_detach_target else ""
            mutual_str = ", mutual_nn=True" if args.sheaf_mutual_nn else ""
            ddp.print(f"Sheaf consistency loss initialized (constant sheaf){soft_str}{detach_str}{sym_str}{mutual_str}")

    # Setup LoRA adapters for frozen backbones (optional)
    lora_manager = None
    if args.use_lora or args.lora_mask_embed:
        if not args.use_lora:
            args.use_lora = True  # Auto-enable when lora_mask_embed is set
        lora_manager = LoRAManager(rank=args.lora_rank, alpha=args.lora_alpha)
        ddp.print(f"\nLoRA enabled (rank={args.lora_rank}, alpha={args.lora_alpha}):")
        if args.lora_sam3:
            sam3_count = lora_manager.add_lora_to_model(base_model.sam3, "sam3")
            ddp.print(f"  SAM3: {sam3_count} adapters")
        if args.lora_da3:
            da3_count = lora_manager.add_lora_to_model(base_model.da3, "da3")
            ddp.print(f"  DA3: {da3_count} adapters")
        if args.lora_mask_embed:
            # Apply LoRA specifically to mask_embed's Linear layers
            mask_pred = base_model.sam3.segmentation_head.mask_predictor
            me_count = 0
            for i, layer in enumerate(mask_pred.mask_embed.layers):
                if isinstance(layer, nn.Linear):
                    adapter = LoRALayer(layer.in_features, layer.out_features,
                                        rank=args.lora_rank, alpha=args.lora_alpha)
                    adapter_name = f"mask_embed_layer{i}"
                    lora_manager.adapters[adapter_name] = adapter
                    hook = lora_manager._create_hook(adapter)
                    handle = layer.register_forward_hook(hook)
                    lora_manager.hooks.append(handle)
                    me_count += 1
            lora_manager._adapter_count += me_count
            ddp.print(f"  mask_embed: {me_count} adapters")
        lora_manager.to(device)
        ddp.print(f"  Total LoRA params: {lora_manager.num_parameters:,}")

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    gasa_params = sum(p.numel() for p in base_model.gasa_decoder.parameters())
    lora_params = lora_manager.num_parameters if lora_manager else 0
    ddp.print(f"\nParameters: Total={total_params:,}, Trainable={trainable_params + lora_params:,} ({100*(trainable_params + lora_params)/total_params:.2f}%), GASA={gasa_params:,}" + (f", LoRA={lora_params:,}" if lora_params > 0 else ""))

    # NOTE: Not scaling LR by world_size - DDP averages gradients, which is like larger batch
    # Linear scaling (LR * world_size) often hurts for small batches. Keep LR same as single-GPU.
    # Combine model params, LoRA params, and sheaf embedding params for optimizer
    trainable_model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    all_params = trainable_model_params
    if lora_manager:
        all_params = all_params + list(lora_manager.parameters())
    if feature_sheaf_loss_fn is not None:
        feature_sheaf_params = list(feature_sheaf_loss_fn.parameters())
        all_params = all_params + feature_sheaf_params
        ddp.print(f"Feature sheaf params: {sum(p.numel() for p in feature_sheaf_params):,}")
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
    ddp.print(f"Learning rate: {args.lr} (no scaling, world_size={ddp.world_size})")
    scaler = GradScaler()

    # LR Scheduler setup
    scheduler = None
    if args.lr_scheduler == 'cosine':
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_epochs = min(args.lr_warmup_epochs, args.epochs - 1)  # Ensure at least 1 epoch for cosine
        cosine_epochs = max(1, args.epochs - warmup_epochs)  # T_max must be >= 1
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.lr_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        ddp.print(f"LR Scheduler: Cosine annealing with {warmup_epochs} warmup epochs, min_lr={args.lr_min}")
    elif args.lr_scheduler == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        ddp.print(f"LR Scheduler: Step decay every {args.lr_step_size} epochs, gamma={args.lr_gamma}")
    else:
        ddp.print("LR Scheduler: None (flat LR)")

    # Setup directories: runs/train for config/summary/vis, checkpoints for weights
    run_dir = PROJECT_ROOT / 'runs' / 'train' / args.run_name
    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name
    best_iou = 0.0
    best_val_miou = 0.0  # Track best validation mIoU for checkpoint selection
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        # If path is a directory, prefer last.pt (most recent), fall back to best.pt
        if resume_path.is_dir():
            if (resume_path / 'last.pt').exists():
                resume_path = resume_path / 'last.pt'
            else:
                resume_path = resume_path / 'best.pt'
        if resume_path.exists():
            ddp.print(f"Resuming from checkpoint: {resume_path}")
            # Load to CPU first, then load_state_dict handles device placement.
            # Loading directly to GPU causes memory fragmentation (duplicate tensors on GPU
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            # Use compatibility loader to handle old checkpoint formats
            base_model.gasa_decoder.load_state_dict_compat(checkpoint['gasa_decoder'], strict=False)
            base_model.query_proj.load_state_dict(checkpoint['query_proj'])
            start_epoch = checkpoint.get('epoch', -1) + 1  # Start from next epoch
            best_iou = checkpoint.get('best_iou', 0.0)
            best_val_miou = checkpoint.get('best_val_miou', 0.0)

            # Restore optimizer state
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    ddp.print(f"  Restored optimizer state")
                except (ValueError, RuntimeError) as e:
                    ddp.print(f"  WARNING: Could not restore optimizer state ({e}). "
                              f"Re-initializing optimizer (new params added since checkpoint).")

            # Restore scheduler state
            if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
                ddp.print(f"  Restored scheduler state")

            # Restore GradScaler state (important for AMP stability on resume)
            if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                scaler.load_state_dict(checkpoint['scaler'])
                ddp.print(f"  Restored scaler state (scale={scaler.get_scale():.0f})")

            # Restore LoRA state if available
            if lora_manager is not None and 'lora' in checkpoint and checkpoint['lora'] is not None:
                lora_manager.load_state_dict(checkpoint['lora'])
                ddp.print(f"  Restored LoRA state ({lora_manager.num_adapters} adapters)")

            # Restore SAM3 seghead if it was trained and saved
            if 'sam3_seghead' in checkpoint and checkpoint['sam3_seghead'] is not None:
                base_model.sam3.segmentation_head.load_state_dict(checkpoint['sam3_seghead'])
                ddp.print(f"  Restored SAM3 seghead state")

            # Restore mask_embed if it was trained and saved
            if 'mask_embed' in checkpoint and checkpoint['mask_embed'] is not None:
                base_model.sam3.segmentation_head.mask_predictor.mask_embed.load_state_dict(checkpoint['mask_embed'])
                ddp.print(f"  Restored mask_embed state")

            # Restore RNG states for reproducible resume (best-effort, not critical)
            try:
                if 'rng_state' in checkpoint:
                    random.setstate(checkpoint['rng_state'])
                if 'np_rng_state' in checkpoint:
                    np.random.set_state(checkpoint['np_rng_state'])
                if 'torch_rng_state' in checkpoint:
                    rng_state = checkpoint['torch_rng_state']
                    if not isinstance(rng_state, torch.ByteTensor):
                        rng_state = torch.ByteTensor(rng_state)
                    torch.set_rng_state(rng_state)
                if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None and torch.cuda.is_available():
                    cuda_states = [torch.ByteTensor(s) if not isinstance(s, torch.ByteTensor) else s for s in checkpoint['cuda_rng_state']]
                    torch.cuda.set_rng_state_all(cuda_states)
                ddp.print("  Restored RNG states")
            except Exception as e:
                ddp.print(f"  Skipping RNG restore (non-critical): {e}")

            val_str = f", best_val_miou={100*best_val_miou:.2f}%" if best_val_miou > 0 else ""
            ddp.print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', -1) + 1}, best_iou={100*best_iou:.2f}%{val_str}")
            ddp.print(f"  Resuming training from epoch {start_epoch + 1}")
            del checkpoint
        else:
            ddp.print(f"WARNING: Checkpoint not found at {resume_path}, starting fresh")

    # Load weights only (no optimizer/scheduler, fresh start from epoch 0)
    elif args.load_weights:
        weights_path = Path(args.load_weights)
        if weights_path.is_dir():
            weights_path = weights_path / 'best.pt'
        if weights_path.exists():
            ddp.print(f"Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            # Use compatibility loader with strict=False to handle old checkpoint formats
            missing, unexpected = base_model.gasa_decoder.load_state_dict_compat(checkpoint['gasa_decoder'], strict=False)
            if missing:
                ddp.print(f"  Missing keys (will be randomly initialized): {len(missing)} keys")
            if unexpected:
                ddp.print(f"  Unexpected keys (ignored): {len(unexpected)} keys")
            base_model.query_proj.load_state_dict(checkpoint['query_proj'], strict=False)
            # Load SAM3 seghead if available in checkpoint
            if 'sam3_seghead' in checkpoint and checkpoint['sam3_seghead'] is not None:
                base_model.sam3.segmentation_head.load_state_dict(checkpoint['sam3_seghead'])
                ddp.print(f"  Loaded SAM3 seghead weights")
            # Load mask_embed if available in checkpoint
            if 'mask_embed' in checkpoint and checkpoint['mask_embed'] is not None:
                base_model.sam3.segmentation_head.mask_predictor.mask_embed.load_state_dict(checkpoint['mask_embed'])
                ddp.print(f"  Loaded mask_embed weights")
            ddp.print(f"  Loaded model weights (epoch {checkpoint.get('epoch', '?')}, iou={100*checkpoint.get('best_iou', 0):.2f}%)")
            ddp.print(f"  Starting fresh from epoch 0 with new optimizer/scheduler")
            # Keep start_epoch=0 and best_iou=0 for fresh training
        else:
            ddp.print(f"WARNING: Weights not found at {weights_path}, starting fresh")

    # Only create dirs and save config on main process
    if ddp.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        import json
        config = vars(args).copy()
        config['world_size'] = ddp.world_size
        config['resumed_from'] = str(args.resume) if args.resume else None
        # Save actual values used (prompt_type overrides these)
        config['use_box_prompts'] = use_box
        config['use_point_prompts'] = use_point

        # Save to run_dir (runs/train/{run_name}/)
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {run_dir / 'config.json'}")

        # Also save to checkpoint_dir (checkpoints/{run_name}/) for easier eval loading
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config also saved to {checkpoint_dir / 'config.json'}")

    ddp.barrier()  # Sync before training
    ddp.print(f"\nTraining for {args.epochs} epochs (starting from epoch {start_epoch + 1})...")

    # Category metrics tracker for proper mIoU calculation
    cat_metrics = CategoryMetricsTracker()

    # Per-epoch history tracking (separate from summary.json)
    history = []
    if ddp.is_main:
        history_path = run_dir / 'history.json'
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                print(f"Loaded existing history with {len(history)} epochs")
            except Exception as e:
                print(f"Warning: Could not load history.json: {e}")
                history = []

    # Initialize metrics in case training exits early (e.g., --stop-at-epoch)
    num_samples = 0
    avg_loss, avg_iou, avg_macc, avg_recall = 0.0, 0.0, 0.0, 0.0

    # Diagnostic logging via environment variable
    TRAIN_DEBUG = os.environ.get('TRAIN_DEBUG', '0') == '1'

    for epoch in range(start_epoch, args.epochs):
        if TRAIN_DEBUG:
            print(f"[DEBUG] Epoch {epoch+1}: Starting epoch loop", flush=True)

        # Early stopping check
        if args.stop_at_epoch > 0 and epoch >= args.stop_at_epoch:
            ddp.print(f"\n[Early Stop] Reached --stop-at-epoch {args.stop_at_epoch}, stopping training.")
            break

        if TRAIN_DEBUG:
            print(f"[DEBUG] Epoch {epoch+1}: Calling ddp.set_epoch()", flush=True)
        ddp.set_epoch(epoch)  # Important for proper shuffling in DDP
        model.train()
        base_model.sam3.eval()
        if base_model.da3 is not None:
            base_model.da3.eval()

        # Curriculum learning: switch or ramp mask selection strategy based on epoch
        epoch_loss, epoch_iou, epoch_macc, epoch_recall, num_samples = 0.0, 0.0, 0.0, 0.0, 0
        epoch_sheaf_loss = 0.0  # Track sheaf loss separately
        epoch_centroid_errors = []  # Track centroid distance errors for Acc@m metrics
        accum_valid = 0
        last_vis_data = None
        cat_metrics.reset()  # Reset per epoch

        if TRAIN_DEBUG:
            print(f"[DEBUG] Epoch {epoch+1}: Creating dataloader iterator (tqdm)", flush=True)
        pbar = tqdm(dataloader, desc=f"[R{ddp.rank}] Epoch {epoch+1}/{args.epochs}")
        if TRAIN_DEBUG:
            print(f"[DEBUG] Epoch {epoch+1}: Starting batch loop", flush=True)
        for batch_idx, batch in enumerate(pbar):
            # With drop_last=True and DistributedSampler, batch should never be None
            # and all ranks get same number of batches. This is just defensive.
            if batch is None:
                # Still need to do dummy backward for DDP sync if other ranks have data
                if ddp.is_distributed:
                    dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    scaler.scale(dummy_loss).backward()
                continue

            images = batch['images'].to(device, non_blocking=True)
            gt_masks = batch['gt_masks'].to(device, non_blocking=True)
            prompts = batch['prompts']
            B, N = images.shape[:2]

            # Get GT camera parameters if available (for world-consistent pointmaps / sheaf loss)
            # If --no-gt-poses is set, suppress GT poses to force using DA3-NESTED estimated poses
            if args.no_gt_poses:
                gt_extrinsics = None  # Force using DA3 poses instead
                gt_intrinsics = None
            else:
                gt_extrinsics = batch.get('extrinsics', None)  # [B, N, 4, 4]
                gt_intrinsics = batch.get('intrinsics', None)  # [B, N, 3, 3]
            intrinsics_orig_hw = batch.get('orig_hw', None)  # (H, W) original resolution for intrinsics
            if gt_extrinsics is not None:
                gt_extrinsics = gt_extrinsics.to(device, non_blocking=True)
            if gt_intrinsics is not None:
                gt_intrinsics = gt_intrinsics.to(device, non_blocking=True)

            # Get cached depth if available (2-4x faster training)
            cached_depth = batch.get('cached_depth', None)  # [B, N, 1, H, W] or None
            if cached_depth is not None:
                cached_depth = cached_depth.to(device, non_blocking=True)

            # Get cached DA3-NESTED poses if available (for world-frame GASA)
            cached_da3_extrinsics = batch.get('cached_da3_extrinsics', None)  # [B, N, 4, 4] or None
            cached_da3_intrinsics = batch.get('cached_da3_intrinsics', None)  # [B, N, 3, 3] or None
            if cached_da3_extrinsics is not None:
                cached_da3_extrinsics = cached_da3_extrinsics.to(device, non_blocking=True)
                # NOTE: Cache already stores c2w (camera-to-world) after Feb 2026 fix.
                # preprocess_da3_nested.py inverts w2c→c2w via extract_c2w_from_extrinsics().
                # DO NOT invert again here - that was causing double-inversion bug!
            if cached_da3_intrinsics is not None:
                cached_da3_intrinsics = cached_da3_intrinsics.to(device, non_blocking=True)

            # Log cached depth status (first batch of each epoch)
            if batch_idx == 0 and ddp.is_main:
                if cached_depth is not None:
                    ddp.print(f"  [Epoch {epoch}] cached_depth: {cached_depth.shape} ✓ DA3 BYPASSED")
                else:
                    ddp.print(f"  [Epoch {epoch}] cached_depth: None ✗ DA3 RUNNING LIVE")

            # Log pose configuration (first batch only)
            if batch_idx == 0 and epoch == start_epoch and ddp.is_main:
                if args.no_gt_poses:
                    ddp.print(f"  ⚠ GT poses suppressed (--no-gt-poses) → using DA3-NESTED estimated poses")
                if args.use_da3_poses_for_gasa and cached_da3_extrinsics is not None:
                    ddp.print(f"  ✓ World PE / GASA: Using DA3-NESTED estimated poses → world-frame pointmaps")
                else:
                    ddp.print(f"  ✓ World PE / GASA: Using camera-frame pointmaps (train/eval consistent)")
                if sheaf_loss_fn is not None:
                    if gt_extrinsics is not None:
                        ddp.print(f"  ✓ Sheaf loss: Using GT extrinsics → world-frame pointmaps")
                    elif args.no_gt_poses and cached_da3_extrinsics is not None:
                        ddp.print(f"  ✓ Sheaf loss: Using DA3-NESTED estimated poses (calibration-free mode)")
                    elif model.da3_has_pose_estimation if hasattr(model, 'da3_has_pose_estimation') else False:
                        ddp.print(f"  ⚠ Sheaf loss: Using DA3-estimated poses → world-frame pointmaps")
                    else:
                        ddp.print(f"  ⚠ Sheaf loss: No world-frame poses available → camera-frame (less effective)")

            # Apply spatial augmentation if enabled (adds "nearest", "leftmost", etc.)
            # GT-aware mode uses actual mask positions; otherwise random qualifiers
            if gt_aware_spatial is not None:
                # GT-AWARE SPATIAL AUGMENTATION
                # Uses spatial_context from dataloader to determine qualifiers
                augmented_prompts = []
                augmented_spatial_indices = []

                # Get spatial contexts (may be list, tuple, or None depending on batching)
                spatial_contexts_raw = batch.get('spatial_context', None)

                # Handle different batching scenarios
                if spatial_contexts_raw is None:
                    spatial_contexts = [None] * len(prompts)
                elif isinstance(spatial_contexts_raw, (list, tuple)):
                    spatial_contexts = list(spatial_contexts_raw)
                    # Pad with None if needed
                    while len(spatial_contexts) < len(prompts):
                        spatial_contexts.append(None)
                else:
                    # Single context (shouldn't happen in batched mode, but handle it)
                    spatial_contexts = [spatial_contexts_raw] + [None] * (len(prompts) - 1)

                for b_idx, p in enumerate(prompts):
                    ctx = spatial_contexts[b_idx] if b_idx < len(spatial_contexts) else None

                    # Augment with GT-aware method
                    aug_prompt, qualifier_type, spatial_idx = gt_aware_spatial.augment(p, ctx)
                    augmented_prompts.append(aug_prompt)
                    augmented_spatial_indices.append(spatial_idx)

                prompts = augmented_prompts

                # Log stats periodically (end of first epoch)
                if batch_idx == 0 and epoch > start_epoch and ddp.is_main:
                    ddp.print(f"  [Spatial] {gt_aware_spatial.get_stats_summary()}")

            elif spatial_augmentor is not None:
                # Random spatial augmentation
                spatial_qualifiers = ['nearest', 'farthest', 'leftmost', 'rightmost', 'topmost', 'bottommost']
                augmented_prompts = []

                # If multi-instance-only, check if this label appears multiple times in batch
                # (approximation: in real scenes, multi-instance objects span multiple frames)
                if args.spatial_multi_instance_only:
                    label_counts = {}
                    for p in prompts:
                        label_counts[p] = label_counts.get(p, 0) + 1

                for p in prompts:
                    should_augment = random.random() < args.spatial_augment_prob

                    # Skip augmentation if multi-instance-only and this is a singleton
                    if args.spatial_multi_instance_only and label_counts.get(p, 1) < 2:
                        should_augment = False

                    if should_augment:
                        qual = random.choice(spatial_qualifiers)
                        augmented_prompts.append(f"{qual} {p}")
                    else:
                        augmented_prompts.append(p)
                prompts = augmented_prompts

            # Parse spatial qualifiers from prompts and create index tensor
            # This is used for spatial token conditioning
            spatial_qualifier_idx = None
            if args.use_spatial_tokens or args.use_spatial_points:
                sq_indices = []
                base_prompts = []
                for p in prompts:
                    qualifier_type, base = parse_spatial_qualifier(p)
                    sq_indices.append(get_spatial_qualifier_idx(qualifier_type))
                    base_prompts.append(base)
                spatial_qualifier_idx = torch.tensor(sq_indices, device=device, dtype=torch.long)
                # Use base prompts for text encoding (without spatial qualifiers)
                # The spatial qualifier is handled via spatial tokens or pseudo-points
                if args.use_spatial_tokens:
                    prompts = base_prompts  # Strip spatial qualifiers for cleaner text embedding

            # Use tensors for accumulation to avoid GPU-CPU sync per view
            batch_loss_tensor = torch.tensor(0.0, device=device)
            batch_iou_tensor = torch.tensor(0.0, device=device)
            batch_macc_tensor = torch.tensor(0.0, device=device)
            batch_recall_tensor = torch.tensor(0.0, device=device)
            batch_sheaf_loss_tensor = torch.tensor(0.0, device=device)
            valid = 0
            accumulated_loss = None  # Accumulate loss across views, single backward per batch

            # Collect predictions and pointmaps for sheaf consistency loss
            sheaf_preds = []  # Will be [N_valid, B, H, W]
            sheaf_pointmaps = []  # Will be [N_valid, B, H, W, 3]
            sheaf_embeddings = []  # Will be [N_valid, B, H, W, D] for feature sheaf loss

            # Use no_sync for view loop to avoid DDP gradient sync per view
            # This prevents deadlock when different ranks skip different views
            sync_context = model.no_sync if ddp.is_distributed and hasattr(model, 'no_sync') else nullcontext

            if args.cross_view:
                # Use forward_multiview for cross-view attention
                # This concatenates memories from all views and lets GASA attend across views
                B, N_views = images.shape[:2]

                # Check if we have extrinsics (required for world-frame PE)
                if gt_extrinsics is None and cached_da3_extrinsics is None:
                    raise ValueError("--cross-view requires gt_extrinsics or da3_extrinsics for world-frame pointmaps")

                try:
                    with autocast('cuda'):
                        # Call through DDP wrapper (cross_view_mode dispatches to forward_multiview)
                        # This ensures DDP's gradient sync hooks are properly triggered
                        outputs = model(
                            images, prompts, gt_masks.float(),
                            gt_extrinsics=gt_extrinsics,
                            gt_intrinsics=gt_intrinsics,
                            intrinsics_orig_hw=intrinsics_orig_hw,
                            cached_depth=cached_depth,
                            da3_extrinsics=cached_da3_extrinsics,
                            da3_intrinsics=cached_da3_intrinsics,
                            cross_view_mode=True
                        )
                        # pred_masks: [B, N, H, W]
                        pred = outputs['pred_masks']
                        all_view_masks = outputs['all_masks']  # [B, N, Q, H, W]

                        # Flatten for processing: [B*N, ...]
                        pred_flat = pred.view(B * N_views, *pred.shape[2:])  # [B*N, H, W]
                        gt_flat = gt_masks.view(B * N_views, *gt_masks.shape[2:]).float()  # [B*N, H, W]
                        all_masks_flat = all_view_masks.view(B * N_views, *all_view_masks.shape[2:])  # [B*N, Q, H, W]

                        # Resize GT if needed
                        if gt_flat.shape[-2:] != pred_flat.shape[-2:]:
                            gt_flat = F.interpolate(gt_flat.unsqueeze(1), size=pred_flat.shape[-2:],
                                                   mode='nearest').squeeze(1)

                        # Determine valid views
                        valid_mask = gt_flat.sum(dim=(-2, -1)) > 0  # [B*N]

                        # Compute loss for all views
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        n_valid = 0

                        for i in range(B * N_views):
                            view_pred = pred_flat[i:i+1]  # [1, H, W]
                            view_gt = gt_flat[i:i+1]  # [1, H, W]

                            view_loss = (args.focal_weight * focal_loss(view_pred, view_gt, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                        args.dice_weight * dice_loss(view_pred.unsqueeze(1), view_gt.unsqueeze(1)))

                            if not valid_mask[i]:
                                view_loss = view_loss * 0.0
                            loss = loss + view_loss

                            if valid_mask[i]:
                                batch_iou_tensor = batch_iou_tensor + compute_iou(view_pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                                batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(view_pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                                batch_recall_tensor = batch_recall_tensor + compute_recall(view_pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                                n_valid += 1

                                # Track per-category metrics
                                prompt_idx = i // N_views  # Map back to original batch index
                                category = prompts[prompt_idx] if prompt_idx < len(prompts) else "unknown"
                                cat_metrics.update(view_pred, view_gt, category)

                        # IoU prediction loss (cross-view path)
                        # Note: iou_pred is [B, Q] (per-scene), not [B*N, Q] (per-view)
                        # We compute average IoU across valid views for each scene as the target
                        if n_valid > 0 and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                            for b in range(B):
                                # Gather IoUs from all valid views of this scene
                                scene_ious = []
                                for v in range(N_views):
                                    idx = b * N_views + v
                                    if valid_mask[idx]:
                                        actual_ious = compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1])
                                        scene_ious.append(actual_ious)
                                if len(scene_ious) > 0:
                                    # Average IoUs across valid views
                                    avg_scene_ious = torch.stack(scene_ious, dim=0).mean(dim=0)  # [1, Q]
                                    iou_pred_loss = F.mse_loss(outputs['iou_pred'][b:b+1], avg_scene_ious.detach())
                                    loss = loss + args.iou_head_weight * iou_pred_loss / B

                        # Contrastive loss (cross-view path)
                        # Note: pred_logits and iou_pred are [B, Q] (per-scene)
                        # Use average IoU across valid views to determine best query per scene
                        if n_valid > 0 and args.contrastive_weight > 0:
                            for b in range(B):
                                # Gather IoUs from all valid views of this scene
                                scene_ious = []
                                for v in range(N_views):
                                    idx = b * N_views + v
                                    if valid_mask[idx]:
                                        actual_ious = compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1])
                                        scene_ious.append(actual_ious)
                                if len(scene_ious) > 0:
                                    avg_scene_ious = torch.stack(scene_ious, dim=0).mean(dim=0)  # [1, Q]
                                    best_idx = avg_scene_ious.argmax(dim=1)
                                    if args.contrastive_source == 'logits':
                                        scores = outputs['pred_logits'][b:b+1]
                                    elif args.contrastive_source == 'iou_pred' and 'iou_pred' in outputs:
                                        scores = outputs['iou_pred'][b:b+1]
                                    else:
                                        scores = None
                                    if scores is not None:
                                        contrast_loss = contrastive_mask_loss(scores, best_idx, margin=args.contrastive_margin)
                                        loss = loss + args.contrastive_weight * contrast_loss / B

                        # Align loss (cross-view path)
                        # Note: pred_logits is [B, Q] (per-scene)
                        # Use average IoU across valid views as the target
                        if n_valid > 0 and args.align_weight > 0:
                            for b in range(B):
                                # Gather IoUs from all valid views of this scene
                                scene_ious = []
                                for v in range(N_views):
                                    idx = b * N_views + v
                                    if valid_mask[idx]:
                                        actual_ious = compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1])
                                        scene_ious.append(actual_ious)
                                if len(scene_ious) > 0:
                                    avg_scene_ious = torch.stack(scene_ious, dim=0).mean(dim=0)  # [1, Q]
                                    logits = outputs['pred_logits'][b:b+1]
                                    align_l = align_loss(logits, avg_scene_ious,
                                                        alpha=args.align_alpha,
                                                        gamma=args.align_gamma,
                                                        tau=args.align_tau)
                                    loss = loss + args.align_weight * align_l / B

                        # PER-LAYER AUXILIARY ALIGN LOSS (SAM3-style)
                        # Compute align loss on intermediate decoder layer outputs to give
                        # intermediate layers direct gradient signal for scoring.
                        # Uses same IoU targets as final layer (masks only computed from final layer).
                        if args.per_layer_align and args.align_weight > 0 and 'aux_queries' in outputs and outputs['aux_queries'] is not None:
                            aux_align_weight = args.per_layer_align_weight if args.per_layer_align_weight is not None else args.align_weight
                            num_aux_layers = len(outputs['aux_queries'])
                            # Pre-compute per-scene avg IoUs (reuse across aux layers)
                            cached_avg_ious = {}
                            for b in range(B):
                                scene_ious = []
                                for v in range(N_views):
                                    idx = b * N_views + v
                                    if valid_mask[idx]:
                                        actual_ious = compute_per_mask_ious(all_masks_flat[idx:idx+1], gt_flat[idx:idx+1])
                                        scene_ious.append(actual_ious)
                                if len(scene_ious) > 0:
                                    cached_avg_ious[b] = torch.stack(scene_ious, dim=0).mean(dim=0)
                            # Apply align loss to each intermediate layer
                            for layer_idx, aux_q in enumerate(outputs['aux_queries']):
                                aux_text_scores = base_model.gasa_decoder.compute_scores_for_queries(aux_q)
                                if aux_text_scores is None:
                                    continue
                                for b, avg_ious in cached_avg_ious.items():
                                    aux_logits = aux_text_scores[b:b+1]
                                    aux_align_l = align_loss(aux_logits, avg_ious,
                                                             alpha=args.align_alpha,
                                                             gamma=args.align_gamma,
                                                             tau=args.align_tau)
                                    loss = loss + aux_align_weight * aux_align_l / (B * num_aux_layers)

                        # Presence loss (cross-view path)
                        # Note: presence_logit is [B, 1] (per-scene)
                        # Target is 1 if any view in the scene has valid GT, 0 otherwise
                        if args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                            # Check if each scene has at least one valid view
                            scene_has_object = torch.zeros(B, 1, device=device)
                            for b in range(B):
                                if valid_mask[b * N_views:(b + 1) * N_views].any():
                                    scene_has_object[b, 0] = 1.0
                            if args.presence_focal:
                                presence_loss = focal_loss(outputs['presence_logit'], scene_has_object,
                                                           alpha=args.presence_alpha, gamma=args.presence_gamma)
                            else:
                                presence_loss = F.binary_cross_entropy_with_logits(
                                    outputs['presence_logit'], scene_has_object
                                )
                            loss = loss + args.presence_weight * presence_loss

                        # Centroid loss (cross-view path)
                        # Note: per_query_centroids is [B, Q, 3] (per-scene, in world coords)
                        # We compute GT centroid per view and average for the scene target
                        if n_valid > 0 and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                            pointmaps_full = outputs['pointmaps_full']  # [B, N, H_da3, W_da3, 3]
                            pointmaps_full_flat = pointmaps_full.view(B * N_views, *pointmaps_full.shape[2:])  # [B*N, H, W, 3]
                            pm_h, pm_w = pointmaps_full_flat.shape[1:3]

                            # Resize GT masks to match pointmaps resolution
                            gt_resized = F.interpolate(
                                gt_flat.unsqueeze(1).float(),
                                size=(pm_h, pm_w),
                                mode='nearest'
                            ).squeeze(1)  # [B*N, H_da3, W_da3]

                            # Resize pred masks for mask-based centroid
                            pred_resized = F.interpolate(
                                pred_flat.unsqueeze(1),
                                size=(pm_h, pm_w),
                                mode='bilinear', align_corners=False
                            ).squeeze(1)  # [B*N, H, W]

                            per_query_cents = outputs['per_query_centroids']  # [B, Q, 3]
                            best_idx_flat = outputs['best_idx']  # [B*N]

                            for b in range(B):
                                # Gather GT centroids from valid views and average
                                gt_cents = []
                                pred_cents = []
                                for v in range(N_views):
                                    idx = b * N_views + v
                                    if valid_mask[idx]:
                                        gt_cent = compute_gt_centroid(gt_resized[idx], pointmaps_full_flat[idx])
                                        gt_cents.append(gt_cent)
                                        if args.mask_based_centroid:
                                            pred_cent = compute_gt_centroid(pred_resized[idx], pointmaps_full_flat[idx])
                                            pred_cents.append(pred_cent)

                                if len(gt_cents) > 0:
                                    avg_gt_cent = torch.stack(gt_cents, dim=0).mean(dim=0)  # [3]
                                    if args.mask_based_centroid and len(pred_cents) > 0:
                                        selected_cent = torch.stack(pred_cents, dim=0).mean(dim=0)  # [3]
                                    else:
                                        # Attention-based: use centroid from best query (use first valid view's best_idx)
                                        first_valid_idx = b * N_views + [v for v in range(N_views) if valid_mask[b * N_views + v]][0]
                                        selected_cent = per_query_cents[b, best_idx_flat[first_valid_idx]]  # [3]
                                    cent_loss = centroid_loss(selected_cent.unsqueeze(0), avg_gt_cent.unsqueeze(0))
                                    loss = loss + args.centroid_weight * cent_loss / B

                        # Centroid error tracking for Acc@m metrics (cross-view path)
                        if n_valid > 0 and (args.use_centroid_head or args.eval_localization) and 'pointmaps_full' in outputs:
                            with torch.no_grad():
                                pointmaps_full = outputs['pointmaps_full']  # [B, N, H_da3, W_da3, 3]
                                pointmaps_full_flat = pointmaps_full.view(B * N_views, *pointmaps_full.shape[2:])
                                pm_h, pm_w = pointmaps_full_flat.shape[1:3]

                                # Resize GT masks to pointmap resolution
                                gt_resized = F.interpolate(
                                    gt_flat.unsqueeze(1).float(),
                                    size=(pm_h, pm_w),
                                    mode='nearest'
                                ).squeeze(1)

                                # Resize pred masks to pointmap resolution
                                pred_resized = F.interpolate(
                                    pred_flat.unsqueeze(1),
                                    size=(pm_h, pm_w),
                                    mode='bilinear', align_corners=False
                                ).squeeze(1)

                                # Get normalization scale to convert back to meters
                                norm_params = outputs.get('norm_params', None)
                                scale = norm_params['scale'].item() if norm_params and 'scale' in norm_params else 1.0

                                for i in range(B * N_views):
                                    if valid_mask[i]:
                                        pred_cent = compute_gt_centroid(pred_resized[i], pointmaps_full_flat[i])
                                        gt_cent = compute_gt_centroid(gt_resized[i], pointmaps_full_flat[i])
                                        dist_error = torch.norm(pred_cent - gt_cent).item() * scale
                                        epoch_centroid_errors.append(dist_error)

                        # Connect ALL auxiliary heads to graph to ensure DDP gradient sync
                        if 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                            loss = loss + outputs['presence_logit'].sum() * 0.0
                        if 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                            loss = loss + outputs['iou_pred'].sum() * 0.0
                        if 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                            loss = loss + outputs['per_query_centroids'].sum() * 0.0
                        if 'text_scores' in outputs and outputs['text_scores'] is not None:
                            loss = loss + outputs['text_scores'].sum() * 0.0
                        if 'joint_scores' in outputs and outputs['joint_scores'] is not None:
                            loss = loss + outputs['joint_scores'].sum() * 0.0

                        # Nuclear option: connect ALL trainable GASA decoder params
                        base_module = model.module if hasattr(model, 'module') else model
                        for p in base_module.gasa_decoder.parameters():
                            if p.requires_grad:
                                loss = loss + p.sum() * 0.0
                        for p in base_module.query_proj.parameters():
                            if p.requires_grad:
                                loss = loss + p.sum() * 0.0

                        if n_valid > 0:
                            batch_loss_tensor = batch_loss_tensor + loss.detach()
                        accumulated_loss = loss / args.grad_accum
                        valid = n_valid

                except Exception as e:
                    ddp.print(f"Error in cross-view forward: {e}")
                    import traceback
                    traceback.print_exc()

            elif args.batch_views:
                # Batch all views together in a single forward pass
                # This is much faster than sequential processing
                B, N_views = images.shape[:2]

                # Detect multi-object mode from batch
                multi_object_K = 1
                multi_object_prompts_list = None
                all_gt_multi = None  # [B*N, K, H, W] for multi-object
                num_objects_per_item = None  # [B] actual K per item (may vary if padded)

                if 'num_objects' in batch and batch['num_objects'] is not None:
                    num_objects_per_item = batch['num_objects'].to(device)  # [B]
                    multi_object_K = int(num_objects_per_item.max().item())

                if multi_object_K > 1:
                    # Multi-object: gt_masks is [B, K, N, H, W]
                    # Reshape to [B*N, K, H, W] by permuting N and K
                    gt_multi_raw = gt_masks  # [B, K, N, H, W]
                    all_gt_multi = gt_multi_raw.permute(0, 2, 1, 3, 4).reshape(
                        B * N_views, multi_object_K, *gt_multi_raw.shape[3:]
                    ).float()  # [B*N, K, H, W]
                    # For valid_mask: use ANY object's coverage (not just primary)
                    # In scene_grouped mode, the primary object may only be visible in a few views,
                    # but other objects ARE visible → must not zero out their loss
                    all_gt = all_gt_multi.max(dim=1).values  # [B*N, H, W] union of all objects
                    # Flatten multi-object prompts: each view gets K prompts
                    multi_object_prompts_list = batch['multi_object_prompts']  # List[List[str]] [B][K]
                else:
                    all_gt = gt_masks.reshape(B * N_views, *gt_masks.shape[2:]).float()

                # Reshape: [B, N, C, H, W] -> [B*N, C, H, W]
                all_views = images.reshape(B * N_views, *images.shape[2:])

                # SAM3-style multi-object mode flag
                sam3_mo = False  # Will be set True after forward if sam3_mo_K in outputs

                # Build prompts for model forward
                if multi_object_K > 1:
                    # Multi-object: flatten K*B*N prompts for text encoding
                    # For each view, repeat the batch item's K prompts
                    # all_prompts_flat = K*B*N strings, ordered as:
                    #   [b0_text0, b0_text1, ..., b0_textK-1, b1_text0, ..., bB-1_textK-1] * N_views
                    flat_prompts_per_batch = []
                    for b_idx in range(B):
                        for k in range(multi_object_K):
                            flat_prompts_per_batch.append(multi_object_prompts_list[b_idx][k])
                    all_prompts = flat_prompts_per_batch * N_views  # Repeat for each view
                else:
                    # Single-object: repeat prompts for each view
                    all_prompts = prompts * N_views  # Repeat N times

                # Check which views have valid GT masks (non-empty with sufficient coverage)
                if 'gt_mask_coverage' in batch:
                    # Coverage computed at original resolution (e.g., 1752x1168)
                    mask_coverage = batch['gt_mask_coverage'].to(device).reshape(B * N_views)  # [B*N]
                    if args.min_mask_coverage > 0:
                        valid_mask = mask_coverage >= args.min_mask_coverage
                    else:
                        mask_pixels = all_gt.sum(dim=(-2, -1))
                        valid_mask = mask_pixels > 0
                else:
                    # Fallback: compute on resized mask (less accurate for small objects)
                    mask_pixels = all_gt.sum(dim=(-2, -1))  # [B*N]
                    mask_coverage = mask_pixels / all_gt[0].numel()  # fraction
                    if args.min_mask_coverage > 0:
                        valid_mask = mask_coverage >= args.min_mask_coverage
                    else:
                        valid_mask = mask_pixels > 0
                # NOTE: Don't use 'continue' even if all GTs empty - backward() must be
                # called on ALL ranks for DDP gradient sync. Skipping causes deadlock!
                # The loss will be zero but all ranks must participate in all_reduce.

                # Get extrinsics/intrinsics for all views
                all_extrinsics = gt_extrinsics.reshape(B * N_views, 4, 4) if gt_extrinsics is not None else None
                all_intrinsics = gt_intrinsics.reshape(B * N_views, 3, 3) if gt_intrinsics is not None else None

                # Reshape cached depth for all views
                all_cached_depth = cached_depth.reshape(B * N_views, *cached_depth.shape[2:]) if cached_depth is not None else None

                # Reshape cached DA3-NESTED poses for all views (for world-frame GASA)
                all_da3_extrinsics = cached_da3_extrinsics.reshape(B * N_views, 4, 4) if cached_da3_extrinsics is not None else None
                all_da3_intrinsics = cached_da3_intrinsics.reshape(B * N_views, 3, 3) if cached_da3_intrinsics is not None else None

                # Repeat spatial qualifiers if used
                all_spatial_idx = None
                if spatial_qualifier_idx is not None:
                    all_spatial_idx = spatial_qualifier_idx.repeat(N_views)  # [B] -> [B*N]


                try:
                    with autocast('cuda'):
                        # For per-text decode, pass multi-object GT [B*N, K, H, W] so each
                        # text gets oracle mask selection against its own GT
                        fwd_gt = all_gt_multi if ((args.per_text_decode or getattr(args, 'sam3_multi_object', False)) and all_gt_multi is not None) else all_gt
                        outputs = model(all_views, all_prompts, fwd_gt,
                                      gt_extrinsics=all_extrinsics,
                                      gt_intrinsics=all_intrinsics,
                                      spatial_qualifier_idx=all_spatial_idx,
                                      intrinsics_orig_hw=intrinsics_orig_hw,
                                      cached_depth=all_cached_depth,
                                      da3_extrinsics=all_da3_extrinsics,
                                      da3_intrinsics=all_da3_intrinsics,
                                      num_texts=multi_object_K)

                        # SAM3-MO: outputs are [B*N*K, ...], reshape GT and valid_mask
                        if outputs.get('sam3_mo_K') is not None:
                            sam3_K = outputs['sam3_mo_K']
                            # Reshape GT: [B*N, K, H, W] → [B*N*K, H, W]
                            all_gt = all_gt_multi.reshape(-1, *all_gt_multi.shape[2:])
                            # Per-object valid mask
                            valid_mask = (all_gt.sum(dim=(-2, -1)) > 0)
                            # Override to single-object loss path
                            multi_object_K = 1
                            sam3_mo = True
                            # Build per-item prompts for category tracking
                            sam3_mo_prompts = []
                            for v_idx in range(N_views):
                                for b_idx in range(B):
                                    K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else sam3_K
                                    for k in range(sam3_K):
                                        if k < K_i and multi_object_prompts_list is not None:
                                            sam3_mo_prompts.append(multi_object_prompts_list[b_idx][k])
                                        else:
                                            sam3_mo_prompts.append("padding")

                        # Compute loss for ALL views (but multiply by 0 for invalid ones to keep graph connected)
                        # This ensures all trainable params are used for DDP gradient sync
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        n_valid = 0

                        if multi_object_K > 1 and 'per_text_masks' in outputs:
                            per_text_masks = outputs['per_text_masks']  # [B*N, K, H, W]
                            grad_text_indices = outputs.get('grad_text_indices', list(range(multi_object_K)))
                            if per_text_masks.shape[-2:] != all_gt_multi.shape[-2:]:
                                per_text_masks = F.interpolate(per_text_masks, size=all_gt_multi.shape[-2:],
                                                              mode='bilinear', align_corners=False)

                            for i in range(B * N_views):
                                b_idx = i % B
                                K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                                view_loss = torch.tensor(0.0, device=device, requires_grad=True)
                                n_k = 0
                                for k_idx in range(K_i):
                                    gt_k = all_gt_multi[i, k_idx:k_idx+1]  # [1, H, W]
                                    if gt_k.sum() > 0:
                                        pred_k = per_text_masks[i, k_idx:k_idx+1]  # [1, H, W]
                                        # Only compute loss for texts with gradients
                                        if k_idx in grad_text_indices:
                                            pair_loss = (
                                                args.focal_weight * focal_loss(pred_k, gt_k, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                                args.dice_weight * dice_loss(pred_k.unsqueeze(1), gt_k.unsqueeze(1))
                                            )
                                            view_loss = view_loss + pair_loss
                                            n_k += 1

                                        # Track metrics for ALL texts (no gradients needed)
                                        with torch.no_grad():
                                            batch_iou_tensor = batch_iou_tensor + compute_iou(pred_k.unsqueeze(1), gt_k.unsqueeze(1), return_tensor=True)
                                            batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(pred_k.unsqueeze(1), gt_k.unsqueeze(1), return_tensor=True)
                                            batch_recall_tensor = batch_recall_tensor + compute_recall(pred_k.unsqueeze(1), gt_k.unsqueeze(1), return_tensor=True)
                                        n_valid += 1
                                        mo_prompts = multi_object_prompts_list[b_idx] if multi_object_prompts_list else None
                                        category = mo_prompts[k_idx] if mo_prompts and k_idx < len(mo_prompts) else "unknown"
                                        cat_metrics.update(pred_k.detach(), gt_k, category)

                                if n_k > 0:
                                    view_loss = view_loss / n_k
                                if not valid_mask[i]:
                                    view_loss = view_loss * 0.0
                                loss = loss + view_loss

                        elif multi_object_K > 1:
                            all_masks = outputs['all_masks']  # [B*N, Q, H, W]
                            text_scores_multi = outputs.get('text_scores', None)  # [B*N, Q, K] or None

                            # Resize all_masks to match GT if needed
                            if all_masks.shape[-2:] != all_gt_multi.shape[-2:]:
                                all_masks_resized = F.interpolate(
                                    all_masks, size=all_gt_multi.shape[-2:],
                                    mode='bilinear', align_corners=False
                                )
                            else:
                                all_masks_resized = all_masks

                            # Pre-compute matching ONCE per batch item (consistent across views)
                            batch_matched_pairs = {}  # b_idx -> (matched_pairs, unmatched)
                            for b_idx in range(B):
                                K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K

                                if args.match_strategy == 'text_greedy' and text_scores_multi is not None and text_scores_multi.dim() == 3:
                                    # Text-greedy: stable assignment based on text scoring head
                                    first_valid = next((v_idx * B + b_idx for v_idx in range(N_views) if valid_mask[v_idx * B + b_idx]), 0)
                                    ts = text_scores_multi[first_valid, :, :K_i]
                                    matched, unmatched = text_greedy_match(ts, K_i)
                                    batch_matched_pairs[b_idx] = (matched, unmatched)
                                else:
                                    # Hungarian: IoU-based bipartite matching averaged across views
                                    avg_cost = torch.zeros(all_masks_resized.shape[1], K_i, device=device)
                                    n_views_for_match = 0
                                    for v_idx in range(N_views):
                                        i = v_idx * B + b_idx
                                        if valid_mask[i]:
                                            view_masks = all_masks_resized[i]
                                            view_gt = all_gt_multi[i, :K_i]
                                            pred_binary = (torch.sigmoid(view_masks) > 0.5).float()
                                            for k in range(K_i):
                                                gt_k = (view_gt[k] > 0.5).float()
                                                inter = (pred_binary * gt_k.unsqueeze(0)).sum(dim=(-2, -1))
                                                union = pred_binary.sum(dim=(-2, -1)) + gt_k.sum() - inter
                                                avg_cost[:, k] += -(inter / union.clamp(min=1.0))
                                            n_views_for_match += 1
                                    if n_views_for_match > 0:
                                        avg_cost /= n_views_for_match
                                    if text_scores_multi is not None and text_scores_multi.dim() == 3:
                                        first_valid = next((v_idx * B + b_idx for v_idx in range(N_views) if valid_mask[v_idx * B + b_idx]), 0)
                                        ts = text_scores_multi[first_valid, :, :K_i]
                                        avg_cost = avg_cost + 0.5 * (-ts.sigmoid())
                                    from scipy.optimize import linear_sum_assignment
                                    row_ind, col_ind = linear_sum_assignment(avg_cost.detach().cpu().numpy())
                                    matched = list(zip(row_ind.tolist(), col_ind.tolist()))
                                    unmatched = [q for q in range(all_masks_resized.shape[1]) if q not in set(row_ind.tolist())]
                                    batch_matched_pairs[b_idx] = (matched, unmatched)

                            for i in range(B * N_views):
                                b_idx = i % B
                                K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                                view_gt_k = all_gt_multi[i, :K_i]  # [K_i, H, W]

                                # Use pre-computed consistent matching
                                matched_pairs, unmatched = batch_matched_pairs[b_idx]

                                # Per-matched-pair loss
                                view_loss = torch.tensor(0.0, device=device, requires_grad=True)
                                n_matched = 0
                                for q_idx, k_idx in matched_pairs:
                                    if k_idx < K_i and view_gt_k[k_idx].sum() > 0:
                                        pred_k = all_masks_resized[i, q_idx:q_idx+1]  # [1, H, W]
                                        gt_k = view_gt_k[k_idx:k_idx+1]  # [1, H, W]
                                        pair_loss = (
                                            args.focal_weight * focal_loss(pred_k, gt_k, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                            args.dice_weight * dice_loss(pred_k.unsqueeze(1), gt_k.unsqueeze(1))
                                        )
                                        view_loss = view_loss + pair_loss
                                        n_matched += 1

                                if n_matched > 0:
                                    view_loss = view_loss / n_matched

                                # No-object loss: force unmatched queries to predict empty masks
                                if args.no_object_weight > 0 and len(unmatched) > 0:
                                    empty_gt = torch.zeros(1, all_masks_resized.shape[-2], all_masks_resized.shape[-1],
                                                          device=device)
                                    no_obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
                                    for q_idx in unmatched:
                                        pred_q = all_masks_resized[i, q_idx:q_idx+1]
                                        # Sigmoid BCE against empty target (penalize any positive predictions)
                                        no_obj_loss = no_obj_loss + F.binary_cross_entropy_with_logits(
                                            pred_q, empty_gt, reduction='mean')
                                    no_obj_loss = no_obj_loss / len(unmatched)
                                    view_loss = view_loss + args.no_object_weight * no_obj_loss

                                # For invalid views, zero out loss but keep graph connected
                                if not valid_mask[i]:
                                    view_loss = view_loss * 0.0

                                loss = loss + view_loss

                                # Metrics: track ALL matched objects (not just primary)
                                if valid_mask[i] and matched_pairs:
                                    b_idx_metric = i % B
                                    mo_prompts = multi_object_prompts_list[b_idx_metric] if multi_object_prompts_list else None
                                    for q_idx, k_idx in matched_pairs:
                                        if k_idx < K_i and all_gt_multi[i, k_idx].sum() > 0:
                                            view_pred = all_masks_resized[i, q_idx:q_idx+1]
                                            view_gt_k = all_gt_multi[i, k_idx:k_idx+1]
                                            batch_iou_tensor = batch_iou_tensor + compute_iou(view_pred.unsqueeze(1), view_gt_k.unsqueeze(1), return_tensor=True)
                                            batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(view_pred.unsqueeze(1), view_gt_k.unsqueeze(1), return_tensor=True)
                                            batch_recall_tensor = batch_recall_tensor + compute_recall(view_pred.unsqueeze(1), view_gt_k.unsqueeze(1), return_tensor=True)
                                            n_valid += 1
                                            category = mo_prompts[k_idx] if mo_prompts and k_idx < len(mo_prompts) else "unknown"
                                            cat_metrics.update(view_pred, view_gt_k, category)

                            # IoU head loss for multi-object
                            if n_valid > 0 and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs:
                                # Compute IoU targets for ALL queries against their matched GTs
                                # Unmatched queries get target IoU = 0
                                for i in range(B * N_views):
                                    if valid_mask[i]:
                                        b_idx = i % B
                                        K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                                        view_masks = all_masks_resized[i]  # [Q, H, W]
                                        view_gt_k = all_gt_multi[i, :K_i]
                                        view_text_scores = None
                                        if text_scores_multi is not None and text_scores_multi.dim() == 3:
                                            view_text_scores = text_scores_multi[i, :, :K_i]
                                        matched_pairs_iou, _ = hungarian_match(view_masks, view_gt_k, K_i, view_text_scores)
                                        iou_targets = torch.zeros(view_masks.shape[0], device=device)
                                        for q_idx, k_idx in matched_pairs_iou:
                                            if k_idx < K_i:
                                                pred_bin = (torch.sigmoid(view_masks[q_idx]) > 0.5).float()
                                                gt_bin = (view_gt_k[k_idx] > 0.5).float()
                                                inter = (pred_bin * gt_bin).sum()
                                                union = pred_bin.sum() + gt_bin.sum() - inter
                                                iou_targets[q_idx] = inter / union.clamp(min=1.0)
                                        iou_pred_loss = F.mse_loss(outputs['iou_pred'][i], iou_targets.detach())
                                        loss = loss + args.iou_head_weight * iou_pred_loss / n_valid

                            # Align loss for multi-object
                            if n_valid > 0 and args.align_weight > 0:
                                for i in range(B * N_views):
                                    if valid_mask[i]:
                                        b_idx = i % B
                                        K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                                        view_masks = all_masks_resized[i]  # [Q, H, W]
                                        view_gt_k = all_gt_multi[i, :K_i]
                                        # Compute IoU of each query against ALL GT objects, take max
                                        actual_ious = torch.zeros(1, view_masks.shape[0], device=device)
                                        for k in range(K_i):
                                            per_mask_ious = compute_per_mask_ious(view_masks.unsqueeze(0), view_gt_k[k:k+1])
                                            actual_ious = torch.max(actual_ious, per_mask_ious)
                                        logits = outputs['pred_logits'][i:i+1]
                                        align_l = align_loss(logits, actual_ious,
                                                            alpha=args.align_alpha,
                                                            gamma=args.align_gamma,
                                                            tau=args.align_tau)
                                        loss = loss + args.align_weight * align_l / n_valid

                            # Text scoring loss: train text_scores to predict query-text assignment
                            # This is essential for text_greedy matching to work
                            if n_valid > 0 and text_scores_multi is not None and text_scores_multi.dim() == 3:
                                text_score_loss = torch.tensor(0.0, device=device, requires_grad=True)
                                n_ts_valid = 0
                                for i in range(B * N_views):
                                    if valid_mask[i]:
                                        b_idx = i % B
                                        K_i = int(num_objects_per_item[b_idx].item()) if num_objects_per_item is not None else multi_object_K
                                        matched_pairs, _ = batch_matched_pairs[b_idx]
                                        # Target: 1.0 for matched (query, text) pairs, 0.0 for rest
                                        ts_target = torch.zeros(all_masks_resized.shape[1], K_i, device=device)
                                        for q_idx, k_idx in matched_pairs:
                                            if k_idx < K_i and all_gt_multi[i, k_idx].sum() > 0:
                                                ts_target[q_idx, k_idx] = 1.0
                                        ts_pred = text_scores_multi[i, :, :K_i]
                                        text_score_loss = text_score_loss + F.binary_cross_entropy_with_logits(
                                            ts_pred, ts_target, reduction='mean')
                                        n_ts_valid += 1
                                if n_ts_valid > 0:
                                    loss = loss + text_score_loss / n_ts_valid

                        else:
                            # For SAM3-MO: batch is expanded to B*N*K, each item = 1 object
                            pred = outputs['pred_masks'][:, 0] if outputs['pred_masks'].dim() == 4 else outputs['pred_masks']

                            if args.loss_at_native_res:
                                # Downsample GT to native mask resolution (288x288)
                                # Avoids blurring gradients through bilinear upsampling
                                if all_gt.shape[-2:] != pred.shape[-2:]:
                                    all_gt_for_loss = F.interpolate(
                                        all_gt.unsqueeze(1).float(), size=pred.shape[-2:],
                                        mode='nearest'
                                    ).squeeze(1)
                                else:
                                    all_gt_for_loss = all_gt
                                pred_for_loss = pred
                            else:
                                # Original: upsample pred to GT resolution
                                if pred.shape[-2:] != all_gt.shape[-2:]:
                                    pred = F.interpolate(pred.unsqueeze(1), size=all_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
                                all_gt_for_loss = all_gt
                                pred_for_loss = pred

                            # Mask smoothing (29x29 avg pool, matches eval-time LangSplat protocol)
                            if args.mask_smooth_kernel > 0:
                                sk = args.mask_smooth_kernel
                                sp = sk // 2
                                pred_for_loss = F.avg_pool2d(
                                    pred_for_loss.unsqueeze(1), kernel_size=sk, stride=1, padding=sp,
                                    count_include_pad=False
                                ).squeeze(1)

                            n_items = all_gt_for_loss.shape[0]  # B*N for single-obj, B*N*K for SAM3-MO

                            if args.use_point_sampling:
                                # SAM3-style: compute loss on sampled uncertain points
                                # Only on valid views
                                valid_pred = pred_for_loss[valid_mask[:n_items]]
                                valid_gt = all_gt_for_loss[valid_mask[:n_items]]
                                if valid_pred.shape[0] > 0:
                                    view_loss = point_sampled_loss(
                                        valid_pred, valid_gt,
                                        focal_fn=focal_loss, dice_fn=dice_loss,
                                        focal_weight=args.focal_weight, dice_weight=args.dice_weight,
                                        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                                        num_points=args.num_sample_points,
                                    )
                                    if args.lovasz_weight > 0:
                                        view_loss = view_loss + args.lovasz_weight * lovasz_loss(valid_pred, valid_gt)
                                    loss = loss + view_loss

                                # Metrics (use original resolution pred for accurate IoU)
                                for i in range(n_items):
                                    if valid_mask[i]:
                                        vp = pred_for_loss[i:i+1]
                                        vg = all_gt_for_loss[i:i+1]
                                        batch_iou_tensor = batch_iou_tensor + compute_iou(vp.unsqueeze(1), vg.unsqueeze(1), return_tensor=True)
                                        batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(vp.unsqueeze(1), vg.unsqueeze(1), return_tensor=True)
                                        batch_recall_tensor = batch_recall_tensor + compute_recall(vp.unsqueeze(1), vg.unsqueeze(1), return_tensor=True)
                                        n_valid += 1
                                        if sam3_mo and 'sam3_mo_prompts' in dir():
                                            category = sam3_mo_prompts[i] if i < len(sam3_mo_prompts) else "unknown"
                                        else:
                                            prompt_idx = i % B
                                            category = prompts[prompt_idx] if prompt_idx < len(prompts) else "unknown"
                                        cat_metrics.update(vp, vg, category)
                            else:
                                for i in range(n_items):
                                    view_pred = pred_for_loss[i:i+1]
                                    view_gt_single = all_gt_for_loss[i:i+1]

                                    view_loss = (args.focal_weight * focal_loss(view_pred, view_gt_single, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                                args.dice_weight * dice_loss(view_pred.unsqueeze(1), view_gt_single.unsqueeze(1)))

                                    # Lovász loss: directly optimizes IoU for sharper boundaries
                                    if args.lovasz_weight > 0:
                                        view_loss = view_loss + args.lovasz_weight * lovasz_loss(view_pred, view_gt_single)

                                    # For invalid views (empty GT), multiply loss by 0 to zero gradients
                                    # but keep graph connected so all trainable params are used
                                    if not valid_mask[i]:
                                        view_loss = view_loss * 0.0

                                    loss = loss + view_loss

                                # Only accumulate metrics for valid views (non-empty GT)
                                if valid_mask[i]:
                                    batch_iou_tensor = batch_iou_tensor + compute_iou(view_pred.unsqueeze(1), view_gt_single.unsqueeze(1), return_tensor=True)
                                    batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(view_pred.unsqueeze(1), view_gt_single.unsqueeze(1), return_tensor=True)
                                    batch_recall_tensor = batch_recall_tensor + compute_recall(view_pred.unsqueeze(1), view_gt_single.unsqueeze(1), return_tensor=True)
                                    n_valid += 1

                                    # Track per-category metrics
                                    if sam3_mo and 'sam3_mo_prompts' in dir():
                                        category = sam3_mo_prompts[i] if i < len(sam3_mo_prompts) else "unknown"
                                    else:
                                        prompt_idx = i % B  # Map back to original batch index
                                        category = prompts[prompt_idx] if prompt_idx < len(prompts) else "unknown"
                                    cat_metrics.update(view_pred, view_gt_single, category)

                        # Pre-compute per-view IoU cache (reused by IoU head, contrastive, align losses)
                        _iou_cache = {}
                        if multi_object_K == 1 and n_valid > 0:
                            _need_ious = (
                                (args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs) or
                                (args.contrastive_weight > 0) or
                                (args.align_weight > 0)
                            )
                            if _need_ious:
                                all_masks = outputs['all_masks']  # [B*N, Q, H, W] or [B*N*K, Q, H, W]
                                for i in range(n_items):
                                    if valid_mask[i]:
                                        _iou_cache[i] = compute_per_mask_ious(all_masks[i:i+1], all_gt[i:i+1])

                        # IoU prediction loss (only for valid views, single-object only)
                        # Multi-object IoU loss is handled inside the multi-object block above
                        if multi_object_K == 1 and n_valid > 0 and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs:
                            for i in range(n_items):
                                if i in _iou_cache:
                                    iou_pred_loss = F.mse_loss(outputs['iou_pred'][i:i+1], _iou_cache[i].detach())
                                    loss = loss + args.iou_head_weight * iou_pred_loss / n_valid

                        # Contrastive loss (single-object only)
                        if multi_object_K == 1 and n_valid > 0 and args.contrastive_weight > 0:
                            for i in range(n_items):
                                if i in _iou_cache:
                                    best_idx = _iou_cache[i].argmax(dim=1)
                                    if args.contrastive_source == 'logits':
                                        scores = outputs['pred_logits'][i:i+1]
                                    elif args.contrastive_source == 'iou_pred' and 'iou_pred' in outputs:
                                        scores = outputs['iou_pred'][i:i+1]
                                    else:
                                        scores = None
                                    if scores is not None:
                                        contrast_loss = contrastive_mask_loss(scores, best_idx, margin=args.contrastive_margin)
                                        loss = loss + args.contrastive_weight * contrast_loss / n_valid

                        # Text scoring loss: REMOVED — pred_logits now comes from DotProductScoring head,
                        # so the existing align loss trains text-query matching end-to-end (SAM3-style).
                        # The separate cross-entropy text scoring loss was redundant.

                        # Align loss (single-object only; multi-object handled above)
                        if multi_object_K == 1 and n_valid > 0 and args.align_weight > 0:
                            for i in range(n_items):
                                if i in _iou_cache:
                                    logits = outputs['pred_logits'][i:i+1]
                                    align_l = align_loss(logits, _iou_cache[i],
                                                        alpha=args.align_alpha,
                                                        gamma=args.align_gamma,
                                                        tau=args.align_tau)
                                    loss = loss + args.align_weight * align_l / n_valid

                            # PER-LAYER AUXILIARY ALIGN LOSS (single-object per-view path)
                            if args.per_layer_align and 'aux_queries' in outputs and outputs['aux_queries'] is not None:
                                aux_align_weight = args.per_layer_align_weight if args.per_layer_align_weight is not None else args.align_weight
                                num_aux_layers = len(outputs['aux_queries'])
                                for aux_q in outputs['aux_queries']:
                                    aux_text_scores = base_model.gasa_decoder.compute_scores_for_queries(aux_q)
                                    if aux_text_scores is None:
                                        continue
                                    for i in range(n_items):
                                        if i in _iou_cache:
                                            aux_logits = aux_text_scores[i:i+1]
                                            aux_align_l = align_loss(aux_logits, _iou_cache[i],
                                                                     alpha=args.align_alpha,
                                                                     gamma=args.align_gamma,
                                                                     tau=args.align_tau)
                                            loss = loss + aux_align_weight * aux_align_l / (n_valid * num_aux_layers)

                        # Presence loss: predict 1.0 when object exists, 0.0 when empty
                        # This ALWAYS runs (even for empty views) to train presence detection
                        if args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                            presence_targets = valid_mask.float().unsqueeze(1)  # [B*N, 1]
                            if args.presence_focal:
                                presence_loss = focal_loss(outputs['presence_logit'], presence_targets,
                                                           alpha=args.presence_alpha, gamma=args.presence_gamma)
                            else:
                                presence_loss = F.binary_cross_entropy_with_logits(
                                    outputs['presence_logit'], presence_targets
                                )
                            loss = loss + args.presence_weight * presence_loss

                        # Centroid loss (batched path) - only for valid views
                        if n_valid > 0 and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                            pointmaps_full = outputs['pointmaps_full']  # [B*N, H_da3, W_da3, 3]
                            pm_h, pm_w = pointmaps_full.shape[1:3]
                            # Resize GT masks to match pointmaps resolution
                            all_gt_resized = F.interpolate(
                                all_gt.unsqueeze(1).float(),
                                size=(pm_h, pm_w),
                                mode='nearest'
                            ).squeeze(1)  # [B*N, H_da3, W_da3]

                            per_query_cents = outputs['per_query_centroids']  # [B*N, Q, 3]
                            best_idx = outputs['best_idx']  # [B*N]

                            # Resize pred masks for mask-based or triangulation centroid
                            if args.mask_based_centroid or args.use_triangulation:
                                all_pred_resized = F.interpolate(
                                    all_pred.unsqueeze(1),
                                    size=(pm_h, pm_w),
                                    mode='bilinear', align_corners=False
                                ).squeeze(1)  # [B*N, H_da3, W_da3]

                            # TRIANGULATION: Multi-view ray intersection for 3D centroid
                            if args.use_triangulation and all_da3_extrinsics is not None and all_da3_intrinsics is not None and N_views > 1 and not sam3_mo:
                                # Reshape to [B, N, ...] for per-scene triangulation
                                pred_resized_bv = all_pred_resized.reshape(B, N_views, pm_h, pm_w)
                                gt_resized_bv = all_gt_resized.reshape(B, N_views, pm_h, pm_w)
                                ext_bv = all_da3_extrinsics.reshape(B, N_views, 4, 4)
                                int_bv = all_da3_intrinsics.reshape(B, N_views, 3, 3)
                                pointmaps_bv = pointmaps_full.reshape(B, N_views, pm_h, pm_w, 3)

                                for b_idx in range(B):
                                    # Check if this scene has any valid views
                                    scene_valid = valid_mask[b_idx * N_views:(b_idx + 1) * N_views]
                                    if scene_valid.sum() < 2:
                                        continue  # Need at least 2 views for triangulation

                                    # Triangulate predicted centroid
                                    pred_tri, pred_valid = triangulate_centroid(
                                        pred_resized_bv[b_idx], ext_bv[b_idx], int_bv[b_idx]
                                    )

                                    # Triangulate GT centroid (for supervision target)
                                    gt_tri, gt_valid = triangulate_centroid(
                                        gt_resized_bv[b_idx], ext_bv[b_idx], int_bv[b_idx]
                                    )

                                    if pred_valid and gt_valid:
                                        cent_loss = centroid_loss(pred_tri.unsqueeze(0), gt_tri.unsqueeze(0))
                                        loss = loss + args.centroid_weight * cent_loss / B
                            else:
                                # Original per-view centroid computation
                                for i in range(n_items):
                                    if valid_mask[i]:
                                        gt_cent = compute_gt_centroid(all_gt_resized[i], pointmaps_full[i])
                                        if args.mask_based_centroid:
                                            # MASK-BASED: Compute centroid from predicted mask + depth
                                            selected_cent = compute_gt_centroid(all_pred_resized[i], pointmaps_full[i])
                                        else:
                                            # ATTENTION-BASED: Use centroid from selected query
                                            selected_cent = per_query_cents[i, best_idx[i]]  # [3]
                                        cent_loss = centroid_loss(selected_cent.unsqueeze(0), gt_cent.unsqueeze(0))
                                        loss = loss + args.centroid_weight * cent_loss / n_valid

                        # Connect ALL auxiliary heads to graph to ensure DDP gradient sync
                        # This must happen ALWAYS to avoid "unused parameters" error with DDP
                        # Multiply by 0 to connect without affecting gradients

                        # Connect presence_logit (from presence_token + presence_head)
                        # Even if presence_weight > 0, we already added it above, but * 0.0 is safe
                        if 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                            loss = loss + outputs['presence_logit'].sum() * 0.0

                        # Connect iou_pred (from iou_head)
                        if 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                            loss = loss + outputs['iou_pred'].sum() * 0.0

                        # Connect per_query_centroids (from centroid_proj)
                        if 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                            loss = loss + outputs['per_query_centroids'].sum() * 0.0

                        # Connect text_scores (from scoring head)
                        if 'text_scores' in outputs and outputs['text_scores'] is not None:
                            loss = loss + outputs['text_scores'].sum() * 0.0
                        if 'joint_scores' in outputs and outputs['joint_scores'] is not None:
                            loss = loss + outputs['joint_scores'].sum() * 0.0

                        # Nuclear option: Connect ALL trainable GASA decoder params to loss
                        # This ensures DDP gradient sync never fails due to unused params
                        # The * 0.0 means no actual gradient contribution
                        base_module = model.module if hasattr(model, 'module') else model
                        for p in base_module.gasa_decoder.parameters():
                            if p.requires_grad:
                                loss = loss + p.sum() * 0.0
                        # Also connect query_proj
                        for p in base_module.query_proj.parameters():
                            if p.requires_grad:
                                loss = loss + p.sum() * 0.0

                        # ALWAYS set accumulated_loss (even if all views invalid) to ensure
                        # backward() runs and keeps DDP gradient sync working
                        if n_valid > 0:
                            batch_loss_tensor = batch_loss_tensor + loss.detach()
                        accumulated_loss = loss / args.grad_accum
                        valid = n_valid

                    # Save visualization data
                    if last_vis_data is None and ddp.is_main:
                        last_vis_data = {'images': all_views[:B].detach().cpu(), 'gt_masks': all_gt[:B].detach().cpu(),
                                         'outputs': {k: v[:B].detach().cpu() if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in outputs.items()},
                                         'prompts': prompts}

                except Exception as e:
                    ddp.print(f"Error in batched forward: {e}")
                    import traceback
                    traceback.print_exc()

            else:
                # Get pre-computed coverage at ORIGINAL resolution if available
                gt_mask_coverage = batch.get('gt_mask_coverage')  # (B, N) or None
                if gt_mask_coverage is not None:
                    gt_mask_coverage = gt_mask_coverage.to(device)

                # Detect multi-object mode
                seq_multi_object = gt_masks.dim() == 5  # [B, K, N, H, W]
                seq_multi_K = 1
                seq_multi_prompts = None  # List[List[str]] [B][K]
                seq_num_objects = None  # [B] tensor of actual K per item
                if seq_multi_object:
                    seq_num_objects = batch.get('num_objects', None)
                    if seq_num_objects is not None:
                        seq_num_objects = seq_num_objects.to(device)
                        seq_multi_K = int(seq_num_objects.max().item())
                    else:
                        seq_multi_K = gt_masks.shape[1]
                    seq_multi_prompts = batch.get('multi_object_prompts', None)  # [B][K]

                    # Apply spatial augmentation to multi-object prompts
                    # Each object gets independently augmented (e.g., "chair" → "nearest chair")
                    # Only applies to multi-instance objects (same label appears 2+ times)
                    if seq_multi_prompts is not None and gt_aware_spatial is not None:
                        spatial_contexts_raw = batch.get('spatial_context', None)
                        spatial_contexts = list(spatial_contexts_raw) if isinstance(spatial_contexts_raw, (list, tuple)) else [None] * B
                        augmented_multi_prompts = []
                        augmented_spatial_indices_mo = []
                        for b_idx in range(B):
                            ctx = spatial_contexts[b_idx] if b_idx < len(spatial_contexts) else None
                            b_prompts = seq_multi_prompts[b_idx]
                            aug_b = []
                            aug_b_idx = []
                            for k, p in enumerate(b_prompts):
                                aug_p, _, s_idx = gt_aware_spatial.augment(p, ctx)
                                aug_b.append(aug_p)
                                aug_b_idx.append(s_idx)
                            augmented_multi_prompts.append(aug_b)
                            augmented_spatial_indices_mo.append(aug_b_idx)
                        seq_multi_prompts = augmented_multi_prompts
                        # Build per-object spatial qualifier indices [K*B]
                        spatial_qualifier_idx = torch.tensor(
                            [augmented_spatial_indices_mo[b][k]
                             for b in range(B) for k in range(seq_multi_K)],
                            device=device, dtype=torch.long
                        )

                    # Build flat prompt list: [b0_t0, b0_t1, ..., b0_tK, b1_t0, ...] = K*B strings
                    if seq_multi_prompts is not None:
                        flat_prompts = []
                        for b_idx in range(B):
                            for k in range(seq_multi_K):
                                flat_prompts.append(seq_multi_prompts[b_idx][k])
                    else:
                        flat_prompts = prompts  # Fallback to single prompts

                with sync_context():
                    for v in range(N):
                        view_img = images[:, v]
                        if seq_multi_object:
                            # Multi-object: use ANY object coverage for validity, all K for loss
                            view_gt_all = gt_masks[:, :, v].float()  # [B, K, H, W] all objects for Hungarian
                            view_gt = view_gt_all.max(dim=1).values  # [B, H, W] union for coverage check
                        else:
                            view_gt = gt_masks[:, v].float()  # [B, H, W]
                            view_gt_all = None
                        # Check if view has valid GT (non-empty mask with sufficient coverage)
                        # IMPORTANT: Do NOT skip empty views! Process them with loss masked to 0.
                        # Skipping causes DDP straggler problem and requires find_unused_parameters=True.
                        if gt_mask_coverage is not None:
                            # Use pre-computed coverage at ORIGINAL resolution
                            mask_coverage = gt_mask_coverage[:, v].mean().item()  # avg across batch
                        else:
                            # Fallback: compute on resized mask (less accurate)
                            mask_coverage = view_gt.sum() / view_gt.numel()
                        is_valid_view = (mask_coverage >= args.min_mask_coverage) if args.min_mask_coverage > 0 else (view_gt.sum() > 0)

                        # Get view-specific extrinsics/intrinsics/cached_depth
                        view_extrinsics = gt_extrinsics[:, v] if gt_extrinsics is not None else None
                        view_intrinsics = gt_intrinsics[:, v] if gt_intrinsics is not None else None
                        view_cached_depth = cached_depth[:, v] if cached_depth is not None else None

                        # Get view-specific cached DA3-NESTED poses (for world-frame GASA)
                        view_da3_extrinsics = cached_da3_extrinsics[:, v] if cached_da3_extrinsics is not None else None
                        view_da3_intrinsics = cached_da3_intrinsics[:, v] if cached_da3_intrinsics is not None else None

                        try:
                            with autocast('cuda'):
                                # Use multi-object prompts if available
                                fwd_prompts = flat_prompts if seq_multi_object and seq_multi_prompts is not None else prompts
                                fwd_num_texts = seq_multi_K if seq_multi_object else 1

                                # SAM3-MO: pass per-object GT for oracle mask selection
                                fwd_gt = view_gt_all if (getattr(args, 'sam3_multi_object', False) and view_gt_all is not None) else view_gt

                                outputs = model(view_img, fwd_prompts, fwd_gt,
                                              gt_extrinsics=view_extrinsics,
                                              gt_intrinsics=view_intrinsics,
                                              spatial_qualifier_idx=spatial_qualifier_idx,
                                              intrinsics_orig_hw=intrinsics_orig_hw,
                                              cached_depth=view_cached_depth,
                                              da3_extrinsics=view_da3_extrinsics,
                                              da3_intrinsics=view_da3_intrinsics,
                                              num_texts=fwd_num_texts)

                                # SAM3-MO: model expanded batch by K. Each item = 1 object.
                                # Use simple per-object single-object loss (no matching needed).
                                if outputs.get('sam3_mo_K') is not None:
                                    sam3_K = outputs['sam3_mo_K']
                                    # GT: [B, K, H, W] → [B*K, H, W]
                                    per_obj_gt = view_gt_all.reshape(-1, *view_gt_all.shape[2:]).float()  # [B*K, H, W]
                                    per_obj_valid = (per_obj_gt.sum(dim=(-2, -1)) > 0)

                                    # Predictions: [B*K, ...]
                                    pred = outputs['pred_masks'][:, 0] if outputs['pred_masks'].dim() == 4 else outputs['pred_masks']
                                    if pred.shape[-2:] != per_obj_gt.shape[-2:]:
                                        pred = F.interpolate(pred.unsqueeze(1), size=per_obj_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

                                    # Mask smoothing (29x29 avg pool, matches eval-time LangSplat protocol)
                                    pred_for_loss = pred
                                    if args.mask_smooth_kernel > 0:
                                        sk = args.mask_smooth_kernel
                                        sp = sk // 2
                                        pred_for_loss = F.avg_pool2d(
                                            pred.unsqueeze(1), kernel_size=sk, stride=1, padding=sp,
                                            count_include_pad=False
                                        ).squeeze(1)

                                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                                    n_obj_valid = 0
                                    for oi in range(pred.shape[0]):
                                        obj_loss = (
                                            args.focal_weight * focal_loss(pred_for_loss[oi:oi+1], per_obj_gt[oi:oi+1], alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                            args.dice_weight * dice_loss(pred_for_loss[oi:oi+1].unsqueeze(1), per_obj_gt[oi:oi+1].unsqueeze(1))
                                        )
                                        # Boundary loss: penalize predictions far from GT boundary
                                        if getattr(args, 'boundary_weight', 0) > 0 and per_obj_valid[oi]:
                                            obj_loss = obj_loss + args.boundary_weight * boundary_loss(
                                                pred_for_loss[oi:oi+1], per_obj_gt[oi:oi+1])
                                        if not per_obj_valid[oi]:
                                            obj_loss = obj_loss * 0.0
                                        loss = loss + obj_loss

                                        if per_obj_valid[oi]:
                                            n_obj_valid += 1
                                            batch_iou_tensor = batch_iou_tensor + compute_iou(pred[oi:oi+1].unsqueeze(1), per_obj_gt[oi:oi+1].unsqueeze(1), return_tensor=True)
                                            batch_recall_tensor = batch_recall_tensor + compute_recall(pred[oi:oi+1].unsqueeze(1), per_obj_gt[oi:oi+1].unsqueeze(1), return_tensor=True)
                                            # Track per-category
                                            b_idx = oi // sam3_K
                                            k_idx = oi % sam3_K
                                            if seq_multi_prompts and b_idx < len(seq_multi_prompts) and k_idx < len(seq_multi_prompts[b_idx]):
                                                cat_name = seq_multi_prompts[b_idx][k_idx]
                                            else:
                                                cat_name = "unknown"
                                            cat_metrics.update(pred[oi:oi+1], per_obj_gt[oi:oi+1], cat_name)

                                    # Align loss for SAM3-MO
                                    if n_obj_valid > 0 and args.align_weight > 0:
                                        all_masks_mo = outputs['all_masks']  # [B*K, Q, H, W]
                                        for oi in range(pred.shape[0]):
                                            if per_obj_valid[oi]:
                                                oi_ious = compute_per_mask_ious(all_masks_mo[oi:oi+1], per_obj_gt[oi:oi+1])
                                                logits = outputs['pred_logits'][oi:oi+1]
                                                al = align_loss(logits, oi_ious, alpha=args.align_alpha, gamma=args.align_gamma, tau=args.align_tau)
                                                loss = loss + args.align_weight * al / n_obj_valid

                                    # Presence loss for SAM3-MO
                                    if args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                                        presence_targets = per_obj_valid.float().unsqueeze(1)  # [B*K, 1]
                                        if args.presence_focal:
                                            presence_loss = focal_loss(outputs['presence_logit'], presence_targets,
                                                                       alpha=args.presence_alpha, gamma=args.presence_gamma)
                                        else:
                                            presence_loss = F.binary_cross_entropy_with_logits(
                                                outputs['presence_logit'], presence_targets)
                                        loss = loss + args.presence_weight * presence_loss

                                    # Centroid loss for SAM3-MO
                                    if n_obj_valid > 0 and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs:
                                        pointmaps_full = outputs['pointmaps_full']  # [B*K, H, W, 3]
                                        pm_h, pm_w = pointmaps_full.shape[1:3]
                                        gt_resized = F.interpolate(per_obj_gt.unsqueeze(1).float(), size=(pm_h, pm_w), mode='nearest').squeeze(1)
                                        per_query_cents = outputs['per_query_centroids']
                                        best_idx = outputs['best_idx']
                                        for oi in range(pred.shape[0]):
                                            if per_obj_valid[oi]:
                                                gt_cent = compute_gt_centroid(gt_resized[oi], pointmaps_full[oi])
                                                selected_cent = per_query_cents[oi, best_idx[oi]]
                                                cent_loss = centroid_loss(selected_cent.unsqueeze(0), gt_cent.unsqueeze(0))
                                                loss = loss + args.centroid_weight * cent_loss / n_obj_valid

                                    # Spatial ranking loss for SAM3-MO
                                    # Teaches model to produce masks whose centroids respect spatial ordering
                                    if n_obj_valid > 0 and args.spatial_ranking_weight > 0 and view_cached_depth is not None:
                                        for b_idx in range(B):
                                            K_b = int(seq_num_objects[b_idx].item()) if seq_num_objects is not None else sam3_K
                                            if K_b < 2:
                                                continue
                                            # Get this batch item's predictions and depth
                                            b_start = b_idx * sam3_K
                                            b_pred = pred[b_start:b_start + K_b]  # [K_b, H, W]
                                            b_depth = view_cached_depth[b_idx]  # [1, H, W]
                                            if b_depth.shape[-2:] != b_pred.shape[-2:]:
                                                b_depth = F.interpolate(b_depth.unsqueeze(0), size=b_pred.shape[-2:],
                                                                        mode='bilinear', align_corners=False).squeeze(0)
                                            # Get labels for this batch item
                                            b_labels = []
                                            for k in range(K_b):
                                                if seq_multi_prompts and b_idx < len(seq_multi_prompts) and k < len(seq_multi_prompts[b_idx]):
                                                    b_labels.append(seq_multi_prompts[b_idx][k])
                                                else:
                                                    b_labels.append(f"obj_{k}")
                                            # Only include valid objects
                                            b_valid = per_obj_valid[b_start:b_start + K_b]
                                            if b_valid.sum() >= 2:
                                                sr_loss = spatial_ranking_loss(
                                                    b_pred, b_depth, b_labels,
                                                    margin=args.spatial_ranking_margin
                                                )
                                                loss = loss + args.spatial_ranking_weight * sr_loss / B

                                            # Spatial selection loss: trains spatial tokens to pick correct instance
                                            if spatial_qualifier_idx is not None and (spatial_qualifier_idx > 0).any():
                                                b_sq = spatial_qualifier_idx[b_start:b_start + K_b]
                                                b_gt = per_obj_gt[b_start:b_start + K_b]
                                                ss_loss = spatial_selection_loss(
                                                    b_pred, b_gt, b_depth, b_labels, b_sq
                                                )
                                                loss = loss + args.spatial_ranking_weight * ss_loss / B

                                    # Connect unused params for DDP
                                    base_module = model.module if hasattr(model, 'module') else model
                                    for p in base_module.gasa_decoder.parameters():
                                        if p.requires_grad:
                                            loss = loss + p.sum() * 0.0

                                # Multi-object: Hungarian matching per batch item (non-SAM3-MO path)
                                elif seq_multi_object and seq_multi_K > 1 and 'all_masks' in outputs:
                                    all_masks = outputs['all_masks']  # [B, Q, H, W]
                                    text_scores_multi = outputs.get('text_scores', None)  # [B, Q, K] or None

                                    # Resize to GT resolution if needed
                                    if all_masks.shape[-2:] != view_gt_all.shape[-2:]:
                                        all_masks = F.interpolate(all_masks, size=view_gt_all.shape[-2:],
                                                                  mode='bilinear', align_corners=False)

                                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                                    n_matched_total = 0
                                    for b_idx in range(B):
                                        K_b = int(seq_num_objects[b_idx].item()) if seq_num_objects is not None else seq_multi_K
                                        view_preds = all_masks[b_idx]  # [Q, H, W]
                                        view_gts = view_gt_all[b_idx, :K_b]  # [K_b, H, W]

                                        view_text_scores = None
                                        if text_scores_multi is not None and text_scores_multi.dim() == 3:
                                            view_text_scores = text_scores_multi[b_idx, :, :K_b]

                                        if args.match_strategy == 'text_greedy' and view_text_scores is not None:
                                            matched_pairs, unmatched_q = text_greedy_match(view_text_scores, K_b)
                                        else:
                                            matched_pairs, unmatched_q = hungarian_match(view_preds, view_gts, K_b, view_text_scores)

                                        for q_idx, k_idx in matched_pairs:
                                            if k_idx < K_b and view_gts[k_idx].sum() > 0:
                                                pred_k = all_masks[b_idx, q_idx]  # [H, W]
                                                gt_k = view_gts[k_idx]  # [H, W]
                                                pair_loss = (
                                                    args.focal_weight * focal_loss(pred_k, gt_k, alpha=args.focal_alpha, gamma=args.focal_gamma) +
                                                    args.dice_weight * dice_loss(pred_k.unsqueeze(0).unsqueeze(0), gt_k.unsqueeze(0).unsqueeze(0))
                                                )
                                                loss = loss + pair_loss
                                                n_matched_total += 1

                                                # Track per-category metrics
                                                cat_name = seq_multi_prompts[b_idx][k_idx] if seq_multi_prompts else prompts[b_idx]
                                                cat_metrics.update(pred_k.unsqueeze(0), gt_k.unsqueeze(0), cat_name)

                                        # No-object loss for unmatched queries (sequential path)
                                        if args.no_object_weight > 0 and len(unmatched_q) > 0:
                                            empty_gt = torch.zeros(1, all_masks.shape[-2], all_masks.shape[-1], device=device)
                                            no_obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
                                            for q_idx in unmatched_q:
                                                pred_q = all_masks[b_idx, q_idx:q_idx+1]
                                                no_obj_loss = no_obj_loss + F.binary_cross_entropy_with_logits(
                                                    pred_q, empty_gt, reduction='mean')
                                            no_obj_loss = no_obj_loss / len(unmatched_q)
                                            loss = loss + args.no_object_weight * no_obj_loss

                                    if n_matched_total > 0:
                                        loss = loss / n_matched_total

                                    # Use best-matched primary object mask for IoU/metrics tracking
                                    pred = outputs['pred_masks'][:, 0] if outputs['pred_masks'].dim() == 4 else outputs['pred_masks']
                                    if pred.shape[-2:] != view_gt.shape[-2:]:
                                        pred = F.interpolate(pred.unsqueeze(1), size=view_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

                                else:
                                    # Single-object path (unchanged)
                                    pred = outputs['pred_masks'][:, 0] if outputs['pred_masks'].dim() == 4 else outputs['pred_masks']
                                    if pred.shape[-2:] != view_gt.shape[-2:]:
                                        pred = F.interpolate(pred.unsqueeze(1), size=view_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

                                    # Main losses: focal + dice
                                    loss = args.focal_weight * focal_loss(pred, view_gt, alpha=args.focal_alpha, gamma=args.focal_gamma) + args.dice_weight * dice_loss(pred.unsqueeze(1), view_gt.unsqueeze(1))

                                # Presence loss: predict 1.0 when object exists, 0.0 when empty
                                # This always runs (even for empty views) to train presence detection
                                # SAM3-MO handles presence inside its own block above
                                if outputs.get('sam3_mo_K') is None and args.presence_weight > 0 and 'presence_logit' in outputs and outputs['presence_logit'] is not None:
                                    presence_target = torch.full((view_img.shape[0], 1), float(is_valid_view), device=view_img.device)
                                    if args.presence_focal:
                                        presence_loss = focal_loss(outputs['presence_logit'], presence_target,
                                                                   alpha=args.presence_alpha, gamma=args.presence_gamma)
                                    else:
                                        presence_loss = F.binary_cross_entropy_with_logits(
                                            outputs['presence_logit'], presence_target
                                        )
                                    loss = loss + args.presence_weight * presence_loss

                                # Centroid loss: predict 3D object centroid (only for valid views)
                                # SAM3-MO handles centroid inside its own block above
                                if outputs.get('sam3_mo_K') is None and is_valid_view and args.use_centroid_head and args.centroid_weight > 0 and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                                    # Compute GT centroid from mask and full-res pointmaps
                                    # pointmaps_full is at DA3 resolution (e.g., 518x518)
                                    # view_gt is at training resolution (e.g., 1008x1008)
                                    pointmaps_full = outputs['pointmaps_full']  # [B, H_da3, W_da3, 3]
                                    pm_h, pm_w = pointmaps_full.shape[1:3]


                                    # Resize masks to match pointmaps resolution
                                    view_gt_resized = F.interpolate(
                                        view_gt.unsqueeze(1).float(),  # [B, 1, H, W]
                                        size=(pm_h, pm_w),
                                        mode='nearest'
                                    ).squeeze(1)  # [B, H_da3, W_da3]

                                    gt_centroids = []
                                    for b_idx in range(view_gt_resized.shape[0]):
                                        gt_cent = compute_gt_centroid(view_gt_resized[b_idx], pointmaps_full[b_idx])
                                        gt_centroids.append(gt_cent)
                                    gt_centroids = torch.stack(gt_centroids, dim=0)  # [B, 3]

                                    # Choose centroid computation method
                                    if args.mask_based_centroid:
                                        # MASK-BASED: Compute centroid directly from predicted mask + depth
                                        # This ties centroid quality directly to mask quality
                                        pred_mask_resized = F.interpolate(
                                            pred.unsqueeze(1),  # [B, 1, H, W]
                                            size=(pm_h, pm_w),
                                            mode='bilinear', align_corners=False
                                        ).squeeze(1)  # [B, H_pm, W_pm]

                                        pred_centroids = []
                                        for b_idx in range(pred_mask_resized.shape[0]):
                                            pred_cent = compute_gt_centroid(pred_mask_resized[b_idx], pointmaps_full[b_idx])
                                            pred_centroids.append(pred_cent)
                                        selected_centroids = torch.stack(pred_centroids, dim=0)  # [B, 3]
                                    else:
                                        # ATTENTION-BASED: Use centroid from selected query (attention-weighted + residual)
                                        per_query_cents = outputs['per_query_centroids']  # [B, Q, 3]
                                        best_idx = outputs['best_idx']  # [B]
                                        B_cent = per_query_cents.shape[0]
                                        b_indices = torch.arange(B_cent, device=per_query_cents.device)
                                        selected_centroids = per_query_cents[b_indices, best_idx]  # [B, 3]

                                    cent_loss = centroid_loss(selected_centroids, gt_centroids)
                                    loss = loss + args.centroid_weight * cent_loss

                                    # Track distance errors for Acc@m metrics (in REAL meters)
                                    # Use MASK-BASED centroid (like evaluation) for comparable Acc@m
                                    with torch.no_grad():
                                        # Compute pred centroid from predicted mask (matching evaluate_gasa.py)
                                        pred_mask_resized = F.interpolate(
                                            pred.unsqueeze(1),  # [B, 1, H, W]
                                            size=(pm_h, pm_w),
                                            mode='bilinear', align_corners=False
                                        ).squeeze(1)  # [B, H_pm, W_pm]

                                        # Get normalization scale to convert back to meters
                                        norm_params = outputs.get('norm_params', None)
                                        scale = norm_params['scale'].item() if norm_params and 'scale' in norm_params else 1.0

                                        for b_idx in range(pred_mask_resized.shape[0]):
                                            # Compute centroid from predicted mask
                                            pred_cent = compute_gt_centroid(pred_mask_resized[b_idx], pointmaps_full[b_idx])
                                            gt_cent = gt_centroids[b_idx]
                                            # Error: multiply by scale to convert normalized units to meters
                                            dist_error = torch.norm(pred_cent - gt_cent).item() * scale
                                            epoch_centroid_errors.append(dist_error)

                                # Eval-only localization metrics: compute Acc@m from mask+depth WITHOUT centroid head
                                # This tracks 3D localization quality during training without adding any loss
                                elif outputs.get('sam3_mo_K') is None and is_valid_view and args.eval_localization and 'pointmaps_full' in outputs:
                                    with torch.no_grad():
                                        pointmaps_full = outputs['pointmaps_full']  # [B, H_da3, W_da3, 3]
                                        pm_h, pm_w = pointmaps_full.shape[1:3]

                                        # Resize GT mask to pointmap resolution
                                        view_gt_resized = F.interpolate(
                                            view_gt.unsqueeze(1).float(),
                                            size=(pm_h, pm_w),
                                            mode='nearest'
                                        ).squeeze(1)

                                        # Resize pred mask to pointmap resolution
                                        pred_mask_resized = F.interpolate(
                                            pred.unsqueeze(1),
                                            size=(pm_h, pm_w),
                                            mode='bilinear', align_corners=False
                                        ).squeeze(1)

                                        # Get normalization scale to convert back to meters
                                        norm_params = outputs.get('norm_params', None)
                                        scale = norm_params['scale'].item() if norm_params and 'scale' in norm_params else 1.0

                                        for b_idx in range(pred_mask_resized.shape[0]):
                                            # Compute centroids from masks + depth
                                            pred_cent = compute_gt_centroid(pred_mask_resized[b_idx], pointmaps_full[b_idx])
                                            gt_cent = compute_gt_centroid(view_gt_resized[b_idx], pointmaps_full[b_idx])
                                            # Error: multiply by scale to convert normalized units to meters
                                            dist_error = torch.norm(pred_cent - gt_cent).item() * scale
                                            epoch_centroid_errors.append(dist_error)

                                # IoU prediction loss: teach model to predict mask quality (only for valid views)
                                if outputs.get('sam3_mo_K') is None and is_valid_view and args.use_iou_head and args.iou_head_weight > 0 and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                                    all_masks = outputs['all_masks']  # [B, Q, H, W]
                                    actual_ious = compute_per_mask_ious(all_masks, view_gt)  # [B, Q]
                                    iou_pred_loss = F.mse_loss(outputs['iou_pred'], actual_ious.detach())
                                    loss = loss + args.iou_head_weight * iou_pred_loss

                                # Contrastive mask loss: push best mask's score above others
                                # Can use pred_logits (no IoU head) or iou_pred based on --contrastive-source
                                if outputs.get('sam3_mo_K') is None and is_valid_view and args.contrastive_weight > 0:
                                    all_masks = outputs['all_masks']  # [B, Q, H, W]
                                    actual_ious = compute_per_mask_ious(all_masks, view_gt)
                                    best_idx = actual_ious.argmax(dim=1)  # [B]

                                    # Choose score source based on flag
                                    if args.contrastive_source == 'logits':
                                        # Use mean mask logits - no IoU head needed!
                                        scores = outputs['pred_logits']  # [B, Q]
                                    elif args.contrastive_source == 'iou_pred' and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                                        # Use IoU predictions (requires --use-iou-head)
                                        scores = outputs['iou_pred']  # [B, Q]
                                    else:
                                        scores = None

                                    if scores is not None:
                                        contrast_loss = contrastive_mask_loss(scores, best_idx, margin=args.contrastive_margin)
                                        loss = loss + args.contrastive_weight * contrast_loss

                                # Text scoring loss: REMOVED — pred_logits now comes from DotProductScoring head,
                                # so the existing align loss trains text-query matching end-to-end (SAM3-style).

                                # Align loss: IoU-aware focal loss on pred_logits (SAM3/AlignDETR style)
                                # Trains the classification logits to directly predict mask quality
                                if outputs.get('sam3_mo_K') is None and is_valid_view and args.align_weight > 0:
                                    all_masks = outputs['all_masks']  # [B, Q, H, W]
                                    actual_ious = compute_per_mask_ious(all_masks, view_gt)  # [B, Q]
                                    logits = outputs['pred_logits']  # [B, Q] - mean mask logits
                                    align_l = align_loss(logits, actual_ious,
                                                        alpha=args.align_alpha,
                                                        gamma=args.align_gamma,
                                                        tau=args.align_tau)
                                    loss = loss + args.align_weight * align_l

                                # Collect predictions and pointmaps for sheaf consistency loss
                                # IMPORTANT: Use world_pointmaps for sheaf loss (world-frame consistency)
                                # Fall back to camera-frame pointmaps if world_pointmaps not available
                                # Only collect for valid views (sheaf loss requires non-empty masks)
                                if outputs.get('sam3_mo_K') is None and is_valid_view and (sheaf_loss_fn is not None or feature_sheaf_loss_fn is not None):
                                    sheaf_preds.append(pred)  # [B, H, W] - keep gradients!
                                    if 'world_pointmaps' in outputs:
                                        sheaf_pointmaps.append(outputs['world_pointmaps'].detach())  # [B, H, W, 3] world-frame
                                        # Debug: check if world pointmaps are being used
                                        if batch_idx == 0 and v == 0 and epoch == start_epoch:
                                            ddp.print(f"  [Sheaf] Using WORLD pointmaps for view {v}")
                                    elif 'pointmaps' in outputs:
                                        sheaf_pointmaps.append(outputs['pointmaps'].detach())  # [B, H, W, 3] camera-frame fallback
                                        if batch_idx == 0 and v == 0 and epoch == start_epoch:
                                            ddp.print(f"  [Sheaf] WARNING: Using CAMERA-FRAME pointmaps for view {v} (world_pointmaps not in outputs)")

                                    # Collect embeddings for feature sheaf loss
                                    if feature_sheaf_loss_fn is not None:
                                        if 'encoder_features' in outputs:
                                            emb = outputs['encoder_features']  # [B, H, W, D]
                                        else:
                                            emb = pred.unsqueeze(-1)  # [B, H, W, 1]
                                        sheaf_embeddings.append(emb)

                            # For invalid views (empty GT), multiply loss by 0 to zero gradients
                            # BUT keep the graph connected so all trainable params are used (DDP requirement)
                            # SAM3-MO handles invalid objects and DDP param connection internally
                            if outputs.get('sam3_mo_K') is None and not is_valid_view:
                                loss = loss * 0.0
                                # Also connect auxiliary heads (iou_head, centroid_head) to the graph
                                # with 0-weighted dummy terms to ensure DDP gradient sync works
                                if args.use_iou_head and 'iou_pred' in outputs and outputs['iou_pred'] is not None:
                                    loss = loss + outputs['iou_pred'].sum() * 0.0
                                if args.use_centroid_head and 'per_query_centroids' in outputs and outputs['per_query_centroids'] is not None:
                                    loss = loss + outputs['per_query_centroids'].sum() * 0.0
                                if 'text_scores' in outputs and outputs['text_scores'] is not None:
                                    loss = loss + outputs['text_scores'].sum() * 0.0
                                if 'joint_scores' in outputs and outputs['joint_scores'] is not None:
                                    loss = loss + outputs['joint_scores'].sum() * 0.0

                            # Accumulate loss instead of backward per view
                            # Divide by grad_accum only (not N) to match original gradient magnitude
                            if accumulated_loss is None:
                                accumulated_loss = loss / args.grad_accum
                            else:
                                accumulated_loss = accumulated_loss + loss / args.grad_accum

                            # Only accumulate metrics for valid views (non-empty GT)
                            # SAM3-MO: metrics already accumulated in the SAM3-MO block
                            if outputs.get('sam3_mo_K') is not None:
                                batch_loss_tensor = batch_loss_tensor + loss.detach()
                                valid += n_obj_valid  # Match IoU accumulation count
                            elif is_valid_view:
                                # Use tensor accumulation to avoid GPU-CPU sync per view
                                batch_loss_tensor = batch_loss_tensor + loss.detach()
                                batch_iou_tensor = batch_iou_tensor + compute_iou(pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                                batch_macc_tensor = batch_macc_tensor + compute_mean_accuracy(pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                                batch_recall_tensor = batch_recall_tensor + compute_recall(pred.unsqueeze(1), view_gt.unsqueeze(1), return_tensor=True)
                                valid += 1

                                # Track per-category metrics for proper mIoU
                                for b_idx in range(pred.shape[0]):
                                    category = prompts[b_idx] if b_idx < len(prompts) else "unknown"
                                    cat_metrics.update(pred[b_idx:b_idx+1], view_gt[b_idx:b_idx+1], category)

                                if last_vis_data is None and ddp.is_main:
                                    last_vis_data = {'images': view_img.detach().cpu(), 'gt_masks': view_gt.detach().cpu(),
                                                     'outputs': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()},
                                                     'prompts': prompts}
                        except Exception as e:
                            ddp.print(f"Error: {e}")
                            import traceback
                            traceback.print_exc()

                # Sheaf consistency loss: penalize inconsistent predictions across views
                if sheaf_loss_fn is not None and len(sheaf_preds) >= 2:
                    try:
                        with autocast('cuda'):
                            # Stack predictions and pointmaps across views: [N_valid, B, H, W] -> [B, N_valid, H, W]
                            stacked_preds = torch.stack(sheaf_preds, dim=1)  # [B, N_valid, H, W]
                            stacked_pts = torch.stack(sheaf_pointmaps, dim=1)  # [B, N_valid, H, W, 3]

                            # Ensure preds and pointmaps have matching spatial resolution
                            _, _, H_pred, W_pred = stacked_preds.shape
                            _, _, H_pts, W_pts, _ = stacked_pts.shape
                            if H_pred != H_pts or W_pred != W_pts:
                                # Resize preds to match pointmap resolution (higher res usually)
                                stacked_preds = F.interpolate(
                                    stacked_preds, size=(H_pts, W_pts), mode='bilinear', align_corners=False
                                )

                            sheaf_loss = sheaf_loss_fn(stacked_preds, stacked_pts)

                            # Add to accumulated loss (scaled by sheaf_weight and grad_accum)
                            if accumulated_loss is None:
                                accumulated_loss = args.sheaf_weight * sheaf_loss / args.grad_accum
                            else:
                                accumulated_loss = accumulated_loss + args.sheaf_weight * sheaf_loss / args.grad_accum

                            # Tensor accumulation (no GPU-CPU sync)
                            batch_loss_tensor = batch_loss_tensor + args.sheaf_weight * sheaf_loss.detach()
                            batch_sheaf_loss_tensor = sheaf_loss.detach()  # Raw sheaf loss (before weight)
                    except Exception as e:
                        sheaf_loss_fn._failure_count += 1
                        if ddp.is_main:
                            print(f"  [SHEAF WARNING] Loss computation failed ({sheaf_loss_fn._failure_count} total failures): {e}")
                            if sheaf_loss_fn._failure_count <= 3:
                                import traceback
                                traceback.print_exc()
                            if sheaf_loss_fn._failure_count == 10:
                                print(f"  [SHEAF ERROR] 10 consecutive failures — sheaf loss may not be working. "
                                      f"Check world_pointmaps and correspondence quality.")

                # Feature sheaf loss: non-constant sheaf on encoder features
                if feature_sheaf_loss_fn is not None and len(sheaf_embeddings) >= 2 and len(sheaf_pointmaps) >= 2:
                    try:
                        with autocast('cuda'):
                            stacked_embs = torch.stack(sheaf_embeddings, dim=1)  # [B, N_valid, H, W, D]
                            stacked_pts = torch.stack(sheaf_pointmaps, dim=1)  # [B, N_valid, H, W, 3]

                            # Ensure spatial dimensions match
                            B_e, V_e, H_e, W_e, D_e = stacked_embs.shape
                            _, _, H_p, W_p, _ = stacked_pts.shape
                            if H_e != H_p or W_e != W_p:
                                stacked_embs = stacked_embs.reshape(B_e * V_e, H_e, W_e, D_e).permute(0, 3, 1, 2)
                                stacked_embs = F.interpolate(stacked_embs, size=(H_p, W_p), mode='bilinear', align_corners=False)
                                stacked_embs = stacked_embs.permute(0, 2, 3, 1).reshape(B_e, V_e, H_p, W_p, D_e)

                            # Compute feature sheaf loss across view pairs
                            feature_sheaf_total = torch.tensor(0.0, device=device)
                            n_pairs = 0
                            for vi in range(V_e):
                                for vj in range(vi + 1, V_e):
                                    if args.sheaf_max_frame_distance > 0 and (vj - vi) > args.sheaf_max_frame_distance:
                                        continue
                                    for b in range(B_e):
                                        pts_i = stacked_pts[b, vi].reshape(-1, 3)  # [H*W, 3]
                                        pts_j = stacked_pts[b, vj].reshape(-1, 3)
                                        feats_i = stacked_embs[b, vi].reshape(-1, D_e)  # [H*W, D]
                                        feats_j = stacked_embs[b, vj].reshape(-1, D_e)

                                        # Subsample for memory
                                        n_sub = min(1024, pts_i.shape[0])
                                        idx_i = torch.randperm(pts_i.shape[0], device=device)[:n_sub]
                                        pts_i_s, feats_i_s = pts_i[idx_i], feats_i[idx_i]

                                        idx_j = torch.randperm(pts_j.shape[0], device=device)[:n_sub]
                                        pts_j_s, feats_j_s = pts_j[idx_j], feats_j[idx_j]

                                        # Find NN correspondences
                                        dists = torch.cdist(pts_i_s, pts_j_s)  # [n_sub, n_sub]
                                        min_dists, nn_idx = dists.min(dim=-1)
                                        valid_corresp = min_dists < args.sheaf_threshold
                                        if valid_corresp.sum() < 5:
                                            continue

                                        # Matched features
                                        fi_matched = feats_i_s[valid_corresp]
                                        fj_matched = feats_j_s[nn_idx[valid_corresp]]
                                        pi_matched = pts_i_s[valid_corresp]
                                        pj_matched = pts_j_s[nn_idx[valid_corresp]]
                                        di_matched = min_dists[valid_corresp]

                                        # Compute geometric context
                                        ctx_i = AsymmetricRestrictionSheaf.compute_context(pi_matched, pj_matched, di_matched)
                                        ctx_j = AsymmetricRestrictionSheaf.compute_context(pj_matched, pi_matched, di_matched)

                                        # Feature sheaf loss
                                        pair_loss = feature_sheaf_loss_fn(fi_matched, fj_matched, ctx_i, ctx_j)
                                        feature_sheaf_total = feature_sheaf_total + pair_loss
                                        n_pairs += 1

                            if n_pairs > 0:
                                feature_sheaf_loss = feature_sheaf_total / n_pairs
                                if accumulated_loss is None:
                                    accumulated_loss = args.sheaf_weight * feature_sheaf_loss / args.grad_accum
                                else:
                                    accumulated_loss = accumulated_loss + args.sheaf_weight * feature_sheaf_loss / args.grad_accum
                                batch_loss_tensor = batch_loss_tensor + args.sheaf_weight * feature_sheaf_loss.detach()
                                batch_sheaf_loss_tensor = feature_sheaf_loss.detach()
                    except Exception as e:
                        if not hasattr(feature_sheaf_loss_fn, '_failure_count'):
                            feature_sheaf_loss_fn._failure_count = 0
                        feature_sheaf_loss_fn._failure_count += 1
                        if ddp.is_main:
                            print(f"  [FEATURE-SHEAF WARNING] Loss failed ({feature_sheaf_loss_fn._failure_count}): {e}")
                            if feature_sheaf_loss_fn._failure_count <= 3:
                                import traceback
                                traceback.print_exc()

            # Single backward per batch (outside no_sync, so DDP syncs gradients here)
            # All ranks must call backward for DDP gradient sync
            # If some ranks skip backward() while others call it, NCCL will hang waiting for sync.
            if accumulated_loss is None:
                # Create dummy zero loss CONNECTED TO MODEL to ensure gradient sync
                # IMPORTANT: With find_unused_parameters=False, we must flow gradients through
                # model params or DDP will hang waiting for gradient hooks that never fire.
                # Sum all trainable params with 0 multiplier to connect graph without affecting gradients.
                dummy_loss = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
                accumulated_loss = dummy_loss if dummy_loss != 0 else torch.tensor(0.0, device=device, requires_grad=True)
            scaler.scale(accumulated_loss).backward()
            if accumulated_loss.item() > 0:  # Only count valid accumulations
                accum_valid += 1

            if (batch_idx + 1) % args.grad_accum == 0 and accum_valid > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_valid = 0

            if valid > 0:
                # Convert tensors to floats only once per batch (single GPU-CPU sync per batch, not per view)
                batch_loss = batch_loss_tensor.item() / valid
                batch_iou = batch_iou_tensor.item() / valid
                batch_macc = batch_macc_tensor.item() / valid
                batch_recall = batch_recall_tensor.item() / valid
                batch_sheaf_loss = batch_sheaf_loss_tensor.item()

                epoch_loss += batch_loss
                epoch_iou += batch_iou
                epoch_macc += batch_macc
                epoch_recall += batch_recall
                epoch_sheaf_loss += batch_sheaf_loss
                num_samples += 1

                # Batch-level visualization (if enabled, main only)
                if args.vis_every_batches > 0 and (batch_idx + 1) % args.vis_every_batches == 0 and last_vis_data and ddp.is_main:
                    try:
                        visualize_predictions(run_dir, f"e{epoch+1}_b{batch_idx+1}", last_vis_data['images'],
                                              last_vis_data['gt_masks'], last_vis_data['outputs'], last_vis_data['prompts'])
                    except Exception as e:
                        print(f"  Batch vis failed: {e}")

            if num_samples > 0:
                cur_miou = cat_metrics.get_miou()
                pbar.set_postfix({'loss': f'{epoch_loss/num_samples:.4f}', 'IoU': f'{100*epoch_iou/num_samples:.1f}%', 'mIoU': f'{100*cur_miou:.1f}%', 'mAcc': f'{100*epoch_macc/num_samples:.1f}%'})

        # Compute local metrics (use zeros if no samples to avoid div-by-zero)
        if num_samples > 0:
            avg_loss, avg_iou, avg_macc, avg_recall = epoch_loss / num_samples, epoch_iou / num_samples, epoch_macc / num_samples, epoch_recall / num_samples
            avg_sheaf_loss = epoch_sheaf_loss / num_samples
        else:
            avg_loss, avg_iou, avg_macc, avg_recall = 0.0, 0.0, 0.0, 0.0
            avg_sheaf_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # Aggregate metrics across GPUs - ALL ranks must participate in collectives!
        if ddp.is_distributed:
            # Debug: Log before each sync point to help identify deadlocks
            if os.environ.get('DDP_DEBUG'):
                print(f"[R{ddp.rank}] Epoch {epoch+1}: Starting sync (cuda_sync)...", flush=True)

            # Ensure all GPU work is done before CPU-side collectives
            torch.cuda.synchronize()

            if os.environ.get('DDP_DEBUG'):
                print(f"[R{ddp.rank}] Epoch {epoch+1}: cuda_sync done, starting all_reduce...", flush=True)

            # Sync scalar metrics (weighted average by num_samples across ranks)
            metrics_tensor = torch.tensor([avg_loss * num_samples, avg_iou * num_samples,
                                           avg_macc * num_samples, avg_recall * num_samples,
                                           avg_sheaf_loss * num_samples,
                                           float(num_samples)], device=device)
            metrics_tensor = ddp.all_reduce(metrics_tensor, op="sum")
            total_samples = metrics_tensor[5].item()
            if total_samples > 0:
                avg_loss = metrics_tensor[0].item() / total_samples
                avg_iou = metrics_tensor[1].item() / total_samples
                avg_macc = metrics_tensor[2].item() / total_samples
                avg_recall = metrics_tensor[3].item() / total_samples
                avg_sheaf_loss = metrics_tensor[4].item() / total_samples
            num_samples = int(total_samples)  # Update to global count

            if os.environ.get('DDP_DEBUG'):
                print(f"[R{ddp.rank}] Epoch {epoch+1}: all_reduce done, starting cat_metrics sync...", flush=True)

            # Sync category metrics by aggregating raw intersection/union (mathematically correct)
            cat_metrics.sync_across_ranks(ddp)

        # Get mIoU from (now synced) category metrics
        miou = cat_metrics.get_miou()
        num_cats = len(cat_metrics.get_per_category_iou())

        # Debug: Show all ranks see the same category count after sync
        if os.environ.get('DDP_DEBUG'):
            print(f"[R{ddp.rank}] Epoch {epoch+1}: Final num_cats={num_cats}, mIoU={100*miou:.2f}%", flush=True)

        # Compute Acc@m metrics for 3D localization (if centroid head OR eval-localization is enabled)
        acc_5cm, acc_10cm, acc_50cm, mean_dist_error = 0.0, 0.0, 0.0, 0.0
        if (args.use_centroid_head or args.eval_localization) and len(epoch_centroid_errors) > 0:
            # np is imported globally at top of file
            errors = np.array(epoch_centroid_errors)
            acc_5cm = (errors < 0.05).mean() * 100  # % within 5cm
            acc_10cm = (errors < 0.10).mean() * 100  # % within 10cm
            acc_50cm = (errors < 0.50).mean() * 100  # % within 50cm
            mean_dist_error = errors.mean()  # Mean distance error in meters

        # Only log/save if we have samples (globally in DDP case)
        if num_samples > 0:

            sheaf_str = f", Sheaf={avg_sheaf_loss:.4f}" if args.use_sheaf_loss else ""
            acc_str = f", Acc@5cm={acc_5cm:.1f}%, Acc@10cm={acc_10cm:.1f}%, MDE={mean_dist_error*100:.1f}cm" if (args.use_centroid_head or args.eval_localization) and len(epoch_centroid_errors) > 0 else ""
            ddp.print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}{sheaf_str}, IoU={100*avg_iou:.2f}%, mIoU={100*miou:.2f}% ({num_cats} cats), mAcc={100*avg_macc:.2f}%, Recall={100*avg_recall:.2f}%{acc_str}, LR={current_lr:.2e}")

            # Print profiling summary after first epoch
            if args.profile and epoch == start_epoch and ddp.is_main:
                ddp.print(f"\n{base_model.get_profile_summary()}\n")

            # Epoch-level visualization (if enabled, main only)
            if args.vis_every_epochs > 0 and (epoch + 1) % args.vis_every_epochs == 0 and last_vis_data and ddp.is_main:
                try:
                    visualize_predictions(run_dir, epoch + 1, last_vis_data['images'], last_vis_data['gt_masks'],
                                          last_vis_data['outputs'], last_vis_data['prompts'])
                except Exception as e:
                    print(f"  Visualization failed: {e}")

            # Run validation if enabled
            val_metrics = None
            if args.val_every > 0 and val_dataloader is not None and (epoch + 1) % args.val_every == 0:
                ddp.print(f"  Running validation...")
                val_metrics = run_validation(model, val_dataloader, device, ddp, args, scaler)
                val_str = f"  Val: Loss={val_metrics['val_loss']:.4f}, IoU={100*val_metrics['val_iou']:.2f}%, mIoU={100*val_metrics['val_miou']:.2f}% ({val_metrics['val_num_categories']} cats)"
                ddp.print(val_str)

                # Check for new best validation mIoU
                if val_metrics['val_miou'] > best_val_miou:
                    best_val_miou = val_metrics['val_miou']
                    if args.save_best_val and ddp.is_main:
                        ckpt = {
                            'epoch': epoch,
                            'gasa_decoder': base_model.gasa_decoder.state_dict(),
                            'query_proj': base_model.query_proj.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict() if scheduler else None,
                            'scaler': scaler.state_dict(),
                            'lora': lora_manager.state_dict() if lora_manager else None,
                            'best_iou': best_iou,
                            'best_val_miou': best_val_miou,
                            # Save SAM3 seghead if it was trained
                            'sam3_seghead': base_model.sam3.segmentation_head.state_dict() if args.train_seghead else None,
                            'mask_embed': base_model.sam3.segmentation_head.mask_predictor.mask_embed.state_dict() if args.train_mask_embed else None,
                            'mask_refiner': base_model.mask_refiner.state_dict() if getattr(base_model, 'use_mask_refiner', False) else None,
                        }
                        torch.save(ckpt, checkpoint_dir / 'best.pt')
                        print(f"  -> New best val mIoU! Saved to {checkpoint_dir / 'best.pt'}")

            # Save best checkpoint based on training IoU (if not using val or val not run this epoch)
            if not args.save_best_val or val_metrics is None:
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    if ddp.is_main:
                        ckpt = {
                            'epoch': epoch,
                            'gasa_decoder': base_model.gasa_decoder.state_dict(),
                            'query_proj': base_model.query_proj.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict() if scheduler else None,
                            'scaler': scaler.state_dict(),
                            'lora': lora_manager.state_dict() if lora_manager else None,
                            'best_iou': best_iou,
                            'best_val_miou': best_val_miou,
                            # Save SAM3 seghead if it was trained
                            'sam3_seghead': base_model.sam3.segmentation_head.state_dict() if args.train_seghead else None,
                            'mask_embed': base_model.sam3.segmentation_head.mask_predictor.mask_embed.state_dict() if args.train_mask_embed else None,
                            'mask_refiner': base_model.mask_refiner.state_dict() if getattr(base_model, 'use_mask_refiner', False) else None,
                        }
                        torch.save(ckpt, checkpoint_dir / 'best.pt')
                        print(f"  -> New best train IoU! Saved to {checkpoint_dir / 'best.pt'}")
            else:
                # Still track best_iou even when saving based on val
                if avg_iou > best_iou:
                    best_iou = avg_iou

            # Always save last.pt for resume (even if not best)
            if ddp.is_main:
                ckpt = {
                    'epoch': epoch,
                    'gasa_decoder': base_model.gasa_decoder.state_dict(),
                    'query_proj': base_model.query_proj.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'scaler': scaler.state_dict(),
                    'lora': lora_manager.state_dict() if lora_manager else None,
                    'best_iou': best_iou,
                    'best_val_miou': best_val_miou,
                    # Save SAM3 seghead if it was trained
                    'sam3_seghead': base_model.sam3.segmentation_head.state_dict() if args.train_seghead else None,
                    'mask_embed': base_model.sam3.segmentation_head.mask_predictor.mask_embed.state_dict() if args.train_mask_embed else None,
                    # RNG states for reproducible resume
                    'rng_state': random.getstate(),
                    'np_rng_state': np.random.get_state(),
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }
                torch.save(ckpt, checkpoint_dir / 'last.pt')

            # Save summary every epoch (so we don't lose progress if interrupted)
            if ddp.is_main:
                cat_summary = cat_metrics.summary()  # Uses synced intersection/union in DDP
                epoch_summary = {
                    'best_iou': best_iou,
                    'best_val_miou': best_val_miou,
                    'current_epoch': epoch + 1,
                    'total_epochs': args.epochs,
                    'current_loss': avg_loss,
                    'current_iou': avg_iou,
                    'current_miou': miou,
                    'current_mAcc': avg_macc,
                    'num_categories': cat_summary['num_categories'],
                    'per_category_iou': cat_summary['per_category_iou'],  # Now synced in DDP
                }
                # Add validation metrics if available
                if val_metrics is not None:
                    epoch_summary['val_loss'] = val_metrics['val_loss']
                    epoch_summary['val_iou'] = val_metrics['val_iou']
                    epoch_summary['val_miou'] = val_metrics['val_miou']
                    epoch_summary['val_mAcc'] = val_metrics['val_mAcc']
                    epoch_summary['val_num_categories'] = val_metrics['val_num_categories']
                # Add Acc@m metrics if centroid head OR eval-localization is enabled
                if (args.use_centroid_head or args.eval_localization) and len(epoch_centroid_errors) > 0:
                    epoch_summary['acc_5cm'] = acc_5cm
                    epoch_summary['acc_10cm'] = acc_10cm
                    epoch_summary['acc_50cm'] = acc_50cm
                    epoch_summary['mean_dist_error_m'] = mean_dist_error
                with open(run_dir / 'summary.json', 'w') as f:
                    json.dump(epoch_summary, f, indent=2)
                # Also save to checkpoint dir for consistency
                with open(checkpoint_dir / 'summary.json', 'w') as f:
                    json.dump(epoch_summary, f, indent=2)

                # Update or append to history (handles resume correctly)
                history_entry = {
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'iou': avg_iou,
                    'miou': miou,
                    'mAcc': avg_macc,
                    'recall': avg_recall,
                    'lr': current_lr,
                    'num_categories': cat_summary['num_categories'],
                }
                # Add validation metrics to history if available
                if val_metrics is not None:
                    history_entry['val_loss'] = val_metrics['val_loss']
                    history_entry['val_iou'] = val_metrics['val_iou']
                    history_entry['val_miou'] = val_metrics['val_miou']
                    history_entry['val_mAcc'] = val_metrics['val_mAcc']
                    history_entry['val_num_categories'] = val_metrics['val_num_categories']
                if args.use_sheaf_loss:
                    history_entry['sheaf_loss'] = avg_sheaf_loss
                if (args.use_centroid_head or args.eval_localization) and len(epoch_centroid_errors) > 0:
                    history_entry['acc_5cm'] = acc_5cm
                    history_entry['acc_10cm'] = acc_10cm
                    history_entry['acc_50cm'] = acc_50cm
                    history_entry['mean_dist_error_m'] = mean_dist_error

                # Find if this epoch already exists in history (for resume)
                existing_idx = None
                for i, h in enumerate(history):
                    if h.get('epoch') == epoch + 1:
                        existing_idx = i
                        break

                if existing_idx is not None:
                    history[existing_idx] = history_entry
                else:
                    history.append(history_entry)
                with open(run_dir / 'history.json', 'w') as f:
                    json.dump(history, f, indent=2)

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Free memory between epochs
        # This reclaims leaked references that Python's refcount missed
        if TRAIN_DEBUG:
            print(f"[DEBUG] Epoch {epoch+1}: Starting gc.collect()", flush=True)
        gc.collect()
        if TRAIN_DEBUG:
            print(f"[DEBUG] Epoch {epoch+1}: Starting torch.cuda.empty_cache()", flush=True)
        torch.cuda.empty_cache()
        if TRAIN_DEBUG:
            print(f"[DEBUG] Epoch {epoch+1}: Finished cleanup", flush=True)

        # RAM check - stop gracefully if available RAM drops below threshold
        if args.min_ram_gb > 0:
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            # In DDP, rank 0 decides and broadcasts to all ranks
            should_stop = torch.tensor([available_ram_gb < args.min_ram_gb], dtype=torch.bool, device=device)
            if ddp.world_size > 1:
                torch.distributed.broadcast(should_stop, src=0)
            if should_stop.item():
                ddp.print(f"\n[Low RAM] Available: {available_ram_gb:.1f}GB < threshold {args.min_ram_gb}GB")
                ddp.print(f"[Low RAM] Stopping gracefully. Checkpoint saved at epoch {epoch+1}.")
                ddp.print(f"[Low RAM] Resume with: --resume {checkpoint_dir}")
                break

        # IMPORTANT: Add barrier to ensure all ranks complete the epoch before starting the next one
        # This prevents race conditions where some ranks race ahead while others are still doing I/O
        # The barrier was previously removed for "faster epoch transitions" but this caused deadlocks
        if os.environ.get('DDP_DEBUG') or TRAIN_DEBUG:
            print(f"[DEBUG R{ddp.rank}] Epoch {epoch+1}: Entering epoch barrier...", flush=True)
        ddp.barrier()
        if os.environ.get('DDP_DEBUG') or TRAIN_DEBUG:
            print(f"[DEBUG R{ddp.rank}] Epoch {epoch+1}: Exited epoch barrier, epoch complete", flush=True)

    # Get final category metrics
    final_miou = cat_metrics.get_miou()

    # Save final summary (main only) - overwrites per-epoch summary with final stats
    if ddp.is_main:
        cat_summary = cat_metrics.summary()
        summary = {
            'best_iou': best_iou,
            'best_val_miou': best_val_miou,
            'final_loss': avg_loss if num_samples > 0 else 0.0,
            'final_iou': avg_iou if num_samples > 0 else 0.0,
            'final_miou': cat_summary['mIoU'],
            # mAcc = mean class accuracy (FG_acc + BG_acc) / 2 - balanced metric
            'final_mAcc': avg_macc if num_samples > 0 else 0.0,
            # Mean class recall = TP/(TP+FN) per sample
            'final_mean_class_recall': avg_recall if num_samples > 0 else 0.0,
            'num_categories': cat_summary['num_categories'],
            'per_category_iou': cat_summary['per_category_iou'],
            'epochs': args.epochs,
            'world_size': ddp.world_size,
            'status': 'completed',
        }
        with open(run_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        with open(checkpoint_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Print per-category breakdown
        if cat_summary['per_category_iou']:
            sorted_cats = sorted(cat_summary['per_category_iou'].items(), key=lambda x: x[1], reverse=True)
            ddp.print(f"\nPer-category IoU (top 10):")
            for cat, iou in sorted_cats[:10]:
                ddp.print(f"  {cat}: {100*iou:.1f}%")

    val_str = f", Best Val mIoU: {100*best_val_miou:.1f}%" if best_val_miou > 0 else ""
    ddp.print(f"\nTraining complete! Best IoU: {100*best_iou:.1f}%{val_str}, Final mIoU: {100*final_miou:.1f}%")
    ddp.print(f"Summary saved to {run_dir / 'summary.json'}")
    ddp.print(f"Per-epoch history saved to {run_dir / 'history.json'}")
    ddp.cleanup()


if __name__ == '__main__':
    main()
