"""Helper functions for TrianguLang training: setup, validation, model building."""
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
from triangulang.losses.spatial_losses import spatial_ranking_loss, spatial_selection_loss
from triangulang.utils.matching import hungarian_match, text_greedy_match
from triangulang.utils.geometry import triangulate_centroid
from triangulang.training.forward_passes import (
    _forward_cross_view, _forward_batch_views, _forward_sequential,
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

    # Optional: cached PI3X world-frame pointmaps (bypasses DA3 for pointmaps)
    if all('cached_pi3x_pointmaps' in b and b['cached_pi3x_pointmaps'] is not None for b in valid_batch):
        result['cached_pi3x_pointmaps'] = torch.stack([b['cached_pi3x_pointmaps'] for b in valid_batch])

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

                    # Compute metrics (using shared metric functions)
                    iou = compute_iou(view_pred, view_gt)
                    macc = compute_mean_accuracy(view_pred, view_gt)
                    recall = compute_recall(view_pred, view_gt)

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
