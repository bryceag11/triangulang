"""
TrianguLang Benchmark Evaluation

Evaluates the trained GASA decoder on ScanNet++ validation scenes.
Uses the same protocol as MV-SAM:
- 100 uniformly sampled DSLR frames per scene
- Objects with ≥0.1% pixel coverage
- 5 objects per validation scene
- Metrics: mIoU and mAcc (per-category recall)

Data structure:
- Images: data/{scene_id}/dslr/resized_undistorted_images/*.JPG
- GT masks: semantics_2d_val/{scene_id}/{frame}.JPG.pth (numpy int32 arrays)
- Labels: data/{scene_id}/scans/segments_anno.json
"""
import sys
import argparse
import json
import time
import math
import tyro
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import random

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'sam3'))
sys.path.insert(0, str(PROJECT_ROOT / 'depth_anything_v3' / 'src'))

from sam3 import build_sam3_image_model
from sam3.model.geometry_encoders import Prompt
from sam3.model.data_misc import FindStage
from depth_anything_3.api import DepthAnything3
from triangulang.models.triangulang_model import TrianguLangModel
from triangulang.utils.lora import LoRAManager, LoRALayer
from triangulang.utils.metrics import compute_gt_centroid
from triangulang.utils.prompt_augmentor import PromptAugmentor
from triangulang.data.dataset_factory import get_dataset, get_dataset_config
from triangulang.utils.scannetpp_loader import normalize_label, is_excluded_frame
from triangulang.utils.spatial_reasoning import (
    get_mask_centroid, get_depth_at_centroid,
    parse_spatial_qualifier, get_spatial_qualifier_idx,
)
from triangulang.models.sheaf_embeddings import compute_3d_localization, format_localization_text
from triangulang.utils.ddp_utils import DDPManager
from triangulang.evaluation.config import BenchmarkConfig

from triangulang import BPE_PATH as _BPE_PATH

# SAM3-style colors for visualization
MASK_COLORS = [
    [30, 144, 255],   # dodgerblue (GT)
    [50, 205, 50],    # limegreen (Pred)
    [255, 99, 71],    # tomato
    [255, 215, 0],    # gold
    [138, 43, 226],   # blueviolet
    [0, 206, 209],    # darkturquoise
    [255, 105, 180],  # hotpink
    [60, 179, 113],   # mediumseagreen
    [255, 140, 0],    # darkorange
    [70, 130, 180],   # steelblue
    [220, 20, 60],    # crimson
    [154, 205, 50],   # yellowgreen
    [199, 21, 133],   # mediumvioletred
    [0, 191, 255],    # deepskyblue
    [218, 165, 32],   # goldenrod
]

# Spatial qualifiers for spatial language queries
# Maps qualifier words to their canonical form (used by compute_spatial_gt)
SPATIAL_QUALIFIERS = {
    # Depth-based
    'nearest': 'nearest',
    'closest': 'nearest',
    'near': 'nearest',
    'close': 'nearest',
    'farthest': 'farthest',
    'far': 'farthest',
    'furthest': 'farthest',
    # X-coordinate (left/right)
    'leftmost': 'leftmost',
    'left': 'leftmost',
    'rightmost': 'rightmost',
    'right': 'rightmost',
    # Y-coordinate (top/bottom)
    'topmost': 'topmost',
    'top': 'topmost',
    'upper': 'topmost',
    'bottommost': 'bottommost',
    'bottom': 'bottommost',
    'lower': 'bottommost',
    # Size-based (for future extension)
    'largest': 'largest',
    'biggest': 'largest',
    'smallest': 'smallest',
}


def parse_spatial_query(prompt: str) -> Tuple[Optional[str], str]:
    """Parse a spatial query into qualifier and base object name.

    Args:
        prompt: Full query like "leftmost towel" or "nearest chair"

    Returns:
        Tuple of (canonical_qualifier, base_prompt)
        If no spatial qualifier found, returns (None, original_prompt)

    Examples:
        "leftmost towel" -> ("leftmost", "towel")
        "nearest red chair" -> ("nearest", "red chair")
        "kitchen towel" -> (None, "kitchen towel")
        "the left chair" -> ("leftmost", "chair")
    """
    prompt_lower = prompt.lower().strip()
    words = prompt_lower.split()

    if len(words) < 2:
        return None, prompt

    # Check first word for qualifier
    first_word = words[0]

    # Handle "the leftmost X" pattern
    if first_word == 'the' and len(words) >= 3:
        second_word = words[1]
        if second_word in SPATIAL_QUALIFIERS:
            canonical = SPATIAL_QUALIFIERS[second_word]
            base = ' '.join(words[2:])
            return canonical, base

    # Check if first word is a spatial qualifier
    if first_word in SPATIAL_QUALIFIERS:
        canonical = SPATIAL_QUALIFIERS[first_word]
        base = ' '.join(words[1:])
        return canonical, base

    return None, prompt


class BaselineSAM3Wrapper(torch.nn.Module):
    """Wrapper around native SAM3 for baseline comparison.

    Runs SAM3's own text-prompted segmentation (encoder + decoder + seghead)
    without GASA, depth, or cross-view fusion. Matches TrianguLangModel's
    forward interface so the eval loop works unchanged.
    """

    def __init__(self, sam3_model, resolution=1008):
        super().__init__()
        self.sam3 = sam3_model
        self.resolution = resolution
        self.mask_selection = 'confidence'  # For eval loop compat

    @torch.no_grad()
    def forward(self, images, text_prompts, gt_masks=None,
                gt_intrinsics=None, gt_extrinsics=None, **kwargs):
        device = images.device
        B = images.shape[0]

        # Normalize from [0, 1] to [-1, 1] (SAM3 expects this)
        sam3_images = (images - 0.5) / 0.5

        # Resize to the resolution SAM3 was built with (must be square, divisible by 14)
        if sam3_images.shape[-2:] != (self.resolution, self.resolution):
            sam3_images = F.interpolate(sam3_images, size=(self.resolution, self.resolution),
                                        mode='bilinear', align_corners=False)

        # Build backbone_out
        backbone_out = {"img_batch_all_stages": sam3_images}
        backbone_out.update(self.sam3.backbone.forward_image(sam3_images))

        # Encode text
        text_out = self.sam3.backbone.forward_text(text_prompts, device=device)
        backbone_out.update(text_out)

        # Create find_input and geometric_prompt
        find_input = FindStage(
            img_ids=torch.arange(B, device=device, dtype=torch.long),
            text_ids=torch.arange(B, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=B)

        # Run SAM3's full pipeline (encoder + decoder + seghead)
        outputs = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )

        # Select best mask by confidence (logits * presence)
        pred_logits = outputs['pred_logits']  # [B, Q, 1]
        pred_masks = outputs['pred_masks']    # [B, Q, H, W]

        scores = pred_logits.sigmoid()  # [B, Q, 1]
        if 'presence_logit_dec' in outputs:
            presence = outputs['presence_logit_dec'].sigmoid()  # [B, Q]
            scores = scores.squeeze(-1) * presence  # [B, Q]
        else:
            scores = scores.squeeze(-1)  # [B, Q]

        # Select best mask per batch element
        best_idx = scores.argmax(dim=-1)  # [B]
        batch_idx = torch.arange(B, device=device)
        best_masks = pred_masks[batch_idx, best_idx]  # [B, H, W]

        return {
            'pred_masks': best_masks.unsqueeze(1),  # [B, 1, H, W] to match TrianguLangModel format
            'all_masks': pred_masks,  # [B, Q, H, W] for oracle IoU
        }


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_prompts_from_gt(
    gt_mask: torch.Tensor,
    prompt_type: str = 'text_only',
    num_pos_points: int = 10,
    num_neg_points: int = 2,
    device: str = 'cuda'
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Create point/box prompts from GT mask based on prompt type.

    Args:
        gt_mask: [H, W] binary ground truth mask
        prompt_type: One of 'text_only', 'text_box', 'text_point', 'text_box_point', 'all',
                     'point_only', 'box_only', 'box_point_only'
        num_pos_points: Number of positive points to sample
        num_neg_points: Number of negative points to sample
        device: Device for tensors

    Returns:
        Dict with 'box_prompts', 'box_labels', 'point_prompts', 'point_labels' (None if not used)
              and 'use_text' (bool) indicating whether text should be used
    """
    result = {
        'box_prompts': None,
        'box_labels': None,
        'point_prompts': None,
        'point_labels': None,
        'use_text': prompt_type not in ('point_only', 'box_only', 'box_point_only'),
    }

    if prompt_type == 'text_only':
        return result

    # Create PromptAugmentor with no jitter for eval (we want exact prompts)
    augmentor = PromptAugmentor(
        point_jitter_px=0,  # No jitter for evaluation
        bbox_jitter_ratio=0.0,  # No jitter for evaluation
        bbox_expand_ratio=0.05,  # Small expansion for robustness
    )

    # Box prompts
    if prompt_type in ['text_box', 'text_box_point', 'all', 'box_only', 'box_point_only']:
        bbox = augmentor.augment_bbox(gt_mask, jitter_ratio=0.0, expand_ratio=0.05)
        H, W = gt_mask.shape
        # Normalize to [0, 1]
        bbox_norm = bbox.clone()
        bbox_norm[0] /= W
        bbox_norm[1] /= H
        bbox_norm[2] /= W
        bbox_norm[3] /= H
        result['box_prompts'] = bbox_norm.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 4]
        result['box_labels'] = torch.ones(1, 1, device=device, dtype=torch.long)  # [1, 1]

    # Point prompts
    if prompt_type in ['text_point', 'text_box_point', 'all', 'point_only', 'box_point_only']:
        points, labels = augmentor.augment_points(
            gt_mask,
            num_points=num_pos_points,
            jitter_px=0,
            include_negative=True,
            negative_ratio=num_neg_points / max(num_pos_points, 1),
        )
        H, W = gt_mask.shape
        # Normalize to [0, 1]
        points_norm = points.clone()
        points_norm[:, 0] /= W
        points_norm[:, 1] /= H
        result['point_prompts'] = points_norm.unsqueeze(0).to(device)  # [1, N, 2]
        result['point_labels'] = labels.unsqueeze(0).to(device)  # [1, N]

    return result


def overlay_mask_sam3_style(image: np.ndarray, mask: np.ndarray, color: List[int], alpha: float = 0.5) -> np.ndarray:
    """Overlay mask on image using SAM3's visualization style."""
    overlay = image.copy()
    mask_bool = mask > 0.5

    # Colored overlay
    for c in range(3):
        overlay[:, :, c] = np.where(mask_bool,
                                     overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                                     overlay[:, :, c])

    # Add contour
    from scipy import ndimage
    contour = ndimage.binary_dilation(mask_bool) ^ mask_bool
    for c in range(3):
        overlay[:, :, c] = np.where(contour, color[c], overlay[:, :, c])

    return overlay.astype(np.uint8)


def render_mask_standalone(
    mask: np.ndarray,
    color: str = 'white',
    background: str = 'black',
) -> np.ndarray:
    """Render a binary mask as a standalone RGB image for paper visualization.

    Args:
        mask: [H, W] float32 binary mask (values in {0, 1} or continuous).
        color: Mask foreground color name or 'colored' (uses MASK_COLORS).
        background: Background color: 'black' or 'white'.

    Returns:
        [H, W, 3] uint8 RGB image.
    """
    COLOR_MAP = {
        'white': [255, 255, 255],
        'blue': MASK_COLORS[0],   # [30, 144, 255]
        'green': MASK_COLORS[1],  # [50, 205, 50]
    }
    BG_MAP = {'black': [0, 0, 0], 'white': [255, 255, 255]}

    fg_rgb = np.array(COLOR_MAP.get(color, [255, 255, 255]), dtype=np.uint8)
    bg_rgb = np.array(BG_MAP.get(background, [0, 0, 0]), dtype=np.uint8)

    H, W = mask.shape[:2]
    result = np.full((H, W, 3), bg_rgb, dtype=np.uint8)
    mask_bool = mask > 0.5
    result[mask_bool] = fg_rgb
    return result


# 3D Metrics: Cross-View Consistency and Centroid Accuracy

def compute_3d_centroid(mask: torch.Tensor, pointmaps: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Compute 3D centroid from mask and pointmaps.

    Args:
        mask: [H, W] binary mask or logits
        pointmaps: [H, W, 3] world coordinates

    Returns:
        [3] centroid in world coordinates, or None if mask is empty
    """
    if mask.shape != pointmaps.shape[:2]:
        # Resize mask to match pointmaps
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=pointmaps.shape[:2],
            mode='nearest'
        ).squeeze()

    mask_binary = mask > 0.5
    if mask_binary.sum() < 10:
        return None

    masked_points = pointmaps[mask_binary]  # [N, 3]
    centroid = masked_points.mean(dim=0)  # [3]

    return centroid


def compute_centroid_error(pred_centroid: torch.Tensor, gt_centroid: torch.Tensor) -> float:
    """Compute Euclidean distance between predicted and GT centroids in meters."""
    return torch.norm(pred_centroid - gt_centroid).item()


# Procrustes Alignment for True 3D Localization

def umeyama_alignment(src_points: np.ndarray, dst_points: np.ndarray, with_scale: bool = True, allow_reflection: bool = True):
    """
    Umeyama alignment: find optimal rotation, translation, and scale
    that aligns src_points to dst_points.

    This is used to align DA3's estimated camera trajectory to GT poses,
    enabling true 3D localization evaluation.

    Args:
        src_points: [N, 3] source points (DA3 estimated camera positions)
        dst_points: [N, 3] destination points (GT camera positions)
        with_scale: whether to estimate scale (7-DoF) or just rotation+translation (6-DoF)
        allow_reflection: if True, allow reflections (det(R) < 0) which is needed
                         when source and destination use different coordinate conventions

    Returns:
        R: [3, 3] rotation/reflection matrix
        t: [3] translation vector
        s: scale factor (1.0 if with_scale=False)
    """
    assert src_points.shape == dst_points.shape
    n, dim = src_points.shape

    # Compute centroids
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)

    # Center the points
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    # Compute covariance matrix
    H = src_centered.T @ dst_centered / n

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation (or rotation + reflection)
    R = Vt.T @ U.T

    # Handle reflection case - only force positive determinant if reflections not allowed
    d = np.linalg.det(R)
    if not allow_reflection and d < 0:
        Vt[-1, :] *= -1
        S_copy = S.copy()
        S_copy[-1] *= -1  # Also negate the corresponding singular value
        R = Vt.T @ U.T
        S = S_copy

    # Scale
    if with_scale:
        src_var = (src_centered ** 2).sum() / n
        # Proper scale: sum of singular values divided by source variance
        s = S.sum() / src_var if src_var > 1e-8 else 1.0
    else:
        s = 1.0

    # Translation
    t = dst_mean - s * R @ src_mean

    return R, t, s


def load_gt_centroids(data_root: Path) -> Dict:
    """Load GT 3D centroids from centroid_cache.json (computed from mesh)."""
    centroid_path = data_root / 'centroid_cache.json'
    if not centroid_path.exists():
        return {}

    with open(centroid_path) as f:
        return json.load(f)


def load_gt_poses_for_scene(data_root: Path, scene_id: str) -> Optional[Dict]:
    """Load GT poses from nerfstudio transforms for a scene."""
    nerf_path = data_root / 'data' / scene_id / 'dslr' / 'nerfstudio' / 'transforms_undistorted.json'

    if not nerf_path.exists():
        return None

    with open(nerf_path) as f:
        data = json.load(f)

    # nerfstudio to mesh coordinate transform
    T_nerf_to_mesh = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    poses = {}
    for frame in data['frames']:
        frame_name = Path(frame['file_path']).stem
        T = np.array(frame['transform_matrix'])
        T_mesh = T_nerf_to_mesh @ T
        poses[frame_name] = T_mesh

    return poses


def compute_cross_view_consistency(
    pred_masks: List[torch.Tensor],
    pointmaps: torch.Tensor,
    threshold: float = 0.05,
    subsample: int = 1024,
) -> Dict:
    """
    Compute cross-view consistency: do corresponding 3D points get same prediction?

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

    # Convert masks to probabilities
    probs = []
    for m in pred_masks:
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear').squeeze()
        probs.append(torch.sigmoid(m) if m.min() < 0 else m)  # Handle logits vs probs

    # Resize pointmaps if needed
    if pointmaps.shape[1:3] != (H, W):
        pts = pointmaps.permute(0, 3, 1, 2)  # [N, 3, H, W]
        pts = F.interpolate(pts, size=(H, W), mode='bilinear', align_corners=False)
        pointmaps = pts.permute(0, 2, 3, 1)  # [N, H, W, 3]

    total_agreements = 0
    total_correspondences = 0

    # Compare all pairs of views
    for i in range(N):
        for j in range(i + 1, N):
            pts_i = pointmaps[i].reshape(-1, 3)  # [H*W, 3]
            pts_j = pointmaps[j].reshape(-1, 3)
            prob_i = probs[i].reshape(-1)  # [H*W]
            prob_j = probs[j].reshape(-1)

            # Get valid points (non-zero depth)
            valid_i = pts_i[:, 2] > 0.01
            valid_j = pts_j[:, 2] > 0.01

            pts_i_valid = pts_i[valid_i]
            pts_j_valid = pts_j[valid_j]
            prob_i_valid = prob_i[valid_i]
            prob_j_valid = prob_j[valid_j]

            if pts_i_valid.shape[0] < 10 or pts_j_valid.shape[0] < 10:
                continue

            # Subsample for memory efficiency
            if pts_i_valid.shape[0] > subsample:
                idx = torch.randperm(pts_i_valid.shape[0], device=device)[:subsample]
                pts_i_valid = pts_i_valid[idx]
                prob_i_valid = prob_i_valid[idx]

            # Find nearest neighbors in 3D
            dists = torch.cdist(pts_i_valid, pts_j_valid)  # [subsample, N_j]
            min_dists, min_indices = dists.min(dim=-1)  # [subsample]

            # Valid correspondences within threshold
            valid_corresp = min_dists < threshold
            if valid_corresp.sum() < 5:
                continue

            # Get corresponding predictions
            pred_i = (prob_i_valid[valid_corresp] > 0.5).float()
            pred_j = (prob_j_valid[min_indices[valid_corresp]] > 0.5).float()

            # Count agreements
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


def create_comparison_grid(
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    labels: List[str],
    ious: List[float],
    scene_id: str,
    max_cols: int = 5,
) -> plt.Figure:
    """Create a grid visualization comparing GT vs predicted masks."""
    n_samples = len(images)
    n_cols = min(n_samples, max_cols)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4 * n_cols, 6 * n_rows),
                             squeeze=False)  # Always return 2D array

    for idx in range(n_samples):
        row = (idx // n_cols) * 2
        col = idx % n_cols

        img = images[idx]
        gt = gt_masks[idx]
        pred = pred_masks[idx]
        label = labels[idx]
        iou = ious[idx]

        # GT row
        gt_overlay = overlay_mask_sam3_style(img, gt, MASK_COLORS[0])
        axes[row, col].imshow(gt_overlay)
        axes[row, col].set_title(f'GT: {label}', fontsize=10, fontweight='bold')
        axes[row, col].axis('off')

        # Pred row
        pred_overlay = overlay_mask_sam3_style(img, pred, MASK_COLORS[1])
        axes[row + 1, col].imshow(pred_overlay)
        axes[row + 1, col].set_title(f'Pred IoU: {iou*100:.1f}%', fontsize=10)
        axes[row + 1, col].axis('off')

    # Hide unused axes
    for idx in range(n_samples, n_rows * n_cols):
        row = (idx // n_cols) * 2
        col = idx % n_cols
        if row < len(axes) and col < len(axes[row]):
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')

    # Legend
    gt_patch = patches.Patch(color=np.array(MASK_COLORS[0])/255, label='Ground Truth')
    pred_patch = patches.Patch(color=np.array(MASK_COLORS[1])/255, label='Prediction')
    fig.legend(handles=[gt_patch, pred_patch], loc='upper center', ncol=2, fontsize=12)

    fig.suptitle(f'Scene: {scene_id}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def create_multi_object_viz(
    frame_groups: Dict[str, dict],
    scene_id: str,
    max_frames: int = 5,
) -> plt.Figure:
    """Create multi-object visualization: all objects overlaid on same frame.

    Each frame shows two panels:
      Left: GT masks (each object a different color, labeled)
      Right: Pred masks (same colors, with per-object IoU)

    Args:
        frame_groups: {frame_name: {'image': np.array, 'objects': [
            {'label': str, 'gt_mask': np.array, 'pred_mask': np.array, 'iou': float}, ...
        ]}}
        scene_id: Scene identifier for title
        max_frames: Maximum frames to show
    """
    frames = list(frame_groups.items())[:max_frames]
    n_frames = len(frames)
    if n_frames == 0:
        return None

    fig, axes = plt.subplots(n_frames, 2, figsize=(16, 5 * n_frames), squeeze=False)

    for row, (frame_name, frame_data) in enumerate(frames):
        img = frame_data['image']
        objects = frame_data['objects']

        # GT overlay: all objects with different colors
        gt_overlay = img.copy().astype(np.float32)
        pred_overlay = img.copy().astype(np.float32)
        legend_entries = []

        for oi, obj in enumerate(objects):
            color = MASK_COLORS[oi % len(MASK_COLORS)]
            gt_m = obj['gt_mask']
            pred_m = obj['pred_mask']

            # Overlay GT mask
            if gt_m is not None:
                mask_bool = gt_m > 0.5
                color_arr = np.array(color, dtype=np.float32)
                gt_overlay[mask_bool] = gt_overlay[mask_bool] * 0.5 + color_arr * 0.5

            # Overlay pred mask
            if pred_m is not None:
                mask_bool = pred_m > 0.5
                color_arr = np.array(color, dtype=np.float32)
                pred_overlay[mask_bool] = pred_overlay[mask_bool] * 0.5 + color_arr * 0.5

            legend_entries.append((obj['label'], obj['iou'], color))

        axes[row, 0].imshow(gt_overlay.astype(np.uint8))
        axes[row, 0].set_title(f'GT — {frame_name}', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(pred_overlay.astype(np.uint8))
        # Build label string with per-object IoU
        iou_str = ', '.join(f'{l}: {iou*100:.0f}%' for l, iou, _ in legend_entries)
        axes[row, 1].set_title(f'Pred — {iou_str}', fontsize=9)
        axes[row, 1].axis('off')

        # Add colored legend patches
        for label, iou, color in legend_entries:
            axes[row, 0].plot([], [], 's', color=np.array(color)/255,
                              markersize=10, label=f'{label} ({iou*100:.0f}%)')
        axes[row, 0].legend(loc='lower left', fontsize=7, framealpha=0.7)

    fig.suptitle(f'Multi-Object: {scene_id}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_category_iou(per_category_iou: Dict[str, float], output_path: Path, top_n: int = 30):
    """Create bar chart of per-category IoU."""
    sorted_cats = sorted(per_category_iou.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_cats) > top_n:
        sorted_cats = sorted_cats[:top_n]

    categories = [c[0] for c in sorted_cats]
    ious = [c[1] * 100 for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.RdYlGn(np.array(ious) / 100)
    bars = ax.barh(range(len(categories)), ious, color=colors)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel('IoU (%)', fontsize=12)
    ax.set_title('Per-Category IoU (Top Categories)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    for i, (bar, iou) in enumerate(zip(bars, ious)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{iou:.1f}%', va='center', fontsize=8)

    mean_iou = np.mean(ious)
    ax.axvline(mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_iou:.1f}%')
    ax.legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_scene_metrics(scene_results: List[Dict], output_path: Path):
    """Create bar chart of per-scene mIoU and mAcc."""
    scene_names = [r.get('scene_id', f"Scene {i+1}")[:12] for i, r in enumerate(scene_results)]
    mious = [r['miou'] * 100 for r in scene_results]
    recalls = [r['recall'] * 100 for r in scene_results]

    x = np.arange(len(scene_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(scene_names) * 0.8), 6))

    bars1 = ax.bar(x - width/2, mious, width, label='mIoU', color='steelblue')
    bars2 = ax.bar(x + width/2, recalls, width, label='Recall (mAcc)', color='coral')

    ax.set_xlabel('Scene', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Scene mIoU and Recall', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scene_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, 100)

    mean_miou = np.mean(mious)
    mean_recall = np.mean(recalls)
    ax.axhline(mean_miou, color='steelblue', linestyle='--', alpha=0.7)
    ax.axhline(mean_recall, color='coral', linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_summary(results_dict: Dict, output_path: Path):
    """Create summary metrics visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ['Sample IoU', 'Scene mIoU', 'Global mIoU', 'mAcc', 'Mean Recall', 'F1']
    values = [
        results_dict['sample_iou'] * 100,
        results_dict['scene_miou'] * 100,
        results_dict['global_miou'] * 100,
        results_dict['mAcc'] * 100,
        results_dict['mean_class_recall'] * 100,
        results_dict['f1'] * 100,
    ]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#E91E63', '#00BCD4']

    bars = axes[0].bar(metrics, values, color=colors)
    axes[0].set_ylabel('Score (%)', fontsize=12)
    axes[0].set_title('Overall Metrics', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    axes[1].axis('off')
    checkpoint_name = Path(results_dict['checkpoint']).name if results_dict.get('checkpoint') else "SAM3 Zero-Shot"
    info_text = f"""
    Evaluation Summary
    {'='*35}

    Checkpoint: {checkpoint_name}
    Split: {results_dict['split']}

    Scenes Evaluated: {results_dict['num_scenes']}
    Total Samples: {results_dict['total_samples']}
    Categories: {len(results_dict.get('per_category_iou', {}))}

    {'-'*35}
    Sample-avg IoU:    {results_dict['sample_iou']*100:.2f}%
    Scene-avg mIoU:    {results_dict['scene_miou']*100:.2f}%
    Global mIoU:       {results_dict['global_miou']*100:.2f}%

    mAcc (Pixel Acc):  {results_dict['mAcc']*100:.2f}%
    Mean Class Recall: {results_dict['mean_class_recall']*100:.2f}%
    Precision:         {results_dict['precision']*100:.2f}%
    F1 Score:          {results_dict['f1']*100:.2f}%
    {'-'*35}

    Avg Preprocess:    {results_dict.get('avg_preprocess_ms', 0):.1f} ms
    Avg Inference:     {results_dict.get('avg_inference_ms', 0):.1f} ms
    Consistency IoU:   {results_dict.get('consistency_iou', 'N/A') if results_dict.get('consistency_iou') is None else f"{results_dict['consistency_iou']*100:.2f}%"}
    """
    axes[1].text(0.1, 0.5, info_text, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def load_model(checkpoint_path: str, device: str = 'cuda', da3_resolution: int = None, num_queries: int = None, skip_trained_seghead: bool = False, train_config_path: str = None, resolution: int = None) -> TrianguLangModel:
    """Load trained GASA decoder model.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        da3_resolution: Override DA3 resolution (default: use config or 504)
        resolution: Override model resolution/image size (default: use config or 1008)
        train_config_path: Optional explicit path to training config.json
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_dir = Path(checkpoint_path).parent

    # Try explicit config path first, then default locations
    config = {}
    config_path = None
    if train_config_path:
        config_path = Path(train_config_path)
    else:
        # Try checkpoint dir first (new location - checkpoints/{run_name}/config.json)
        config_path = ckpt_dir / 'config.json'
        if not config_path.exists():
            # Try runs/train/ (standard location)
            config_path = ckpt_dir.parent.parent / 'runs' / 'train' / ckpt_dir.name / 'config.json'
        if not config_path.exists():
            # Try runs/ablations/ (ablation runs)
            config_path = ckpt_dir.parent.parent / 'runs' / 'ablations' / ckpt_dir.name / 'config.json'

    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"WARNING: Config not found, using defaults!")

    # Print key config values for debugging
    effective_da3_res = da3_resolution if da3_resolution is not None else (config.get('da3_resolution') or 504)
    print(f"Config: use_box_prompts={config.get('use_box_prompts', False)}, "
          f"use_point_prompts={config.get('use_point_prompts', False)}, "
          f"use_world_pe={config.get('use_world_pe', True)}, "
          f"use_gasa={config.get('use_gasa', True)}, "
          f"mask_selection={config.get('mask_selection', 'iou_match')}, "
          f"use_iou_head={config.get('use_iou_head', False)}, "
          f"use_spatial_tokens={config.get('use_spatial_tokens', False)}, "
          f"pe_type={config.get('pe_type', 'world')}, "
          f"per_layer_text={config.get('per_layer_text', False)}, "
          f"da3_resolution={effective_da3_res}")

    print("Loading SAM3...")
    sam3_resolution = resolution if resolution is not None else config.get('resolution', 1008)
    res_source = f"--image-size override ({resolution})" if resolution is not None else "config"
    print(f"  SAM3 img_size={sam3_resolution} ({res_source})")
    sam3_model = build_sam3_image_model(bpe_path=_BPE_PATH, img_size=sam3_resolution).to(device)

    print("Loading DA3...")
    da3_model = DepthAnything3.from_pretrained(
        config.get('da3_model', 'depth-anything/DA3METRIC-LARGE'),
        device=device
    )

    model = TrianguLangModel(
        sam3_model=sam3_model,
        da3_model=da3_model,
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        num_queries=num_queries if num_queries is not None else config.get('num_queries', 100),
        train_seghead=config.get('train_seghead', False),
        attn_map_size=config.get('attn_map_size', 64),
        use_presence_token=config.get('use_presence_token', True),
        use_box_prompts=config.get('use_box_prompts', False),
        use_point_prompts=config.get('use_point_prompts', False),  # Default False for old checkpoints
        num_pos_points=config.get('num_pos_points', 10),
        num_neg_points=config.get('num_neg_points', 2),
        use_world_pe=config.get('use_world_pe', True),
        use_gasa=config.get('use_gasa', True),
        gasa_beta_init=config.get('gasa_beta_init', 1.0),
        gasa_kernel_dim=config.get('gasa_kernel_dim', 32),
        gasa_fixed_kernel=config.get('gasa_fixed_kernel', False),
        gasa_kernel_type=config.get('gasa_kernel_type', 'learned'),
        mask_selection=config.get('mask_selection', 'iou_match'),
        use_iou_head=config.get('use_iou_head', False),
        use_spatial_tokens=config.get('use_spatial_tokens', False),
        use_spatial_points=config.get('use_spatial_points', False),
        use_object_aware_spatial=config.get('use_object_aware_spatial', False),
        use_centroid_head=config.get('use_centroid_head', False),
        use_iterative_pos=config.get('use_iterative_pos', False),
        cross_view=config.get('cross_view', True),
        pe_type=config.get('pe_type', 'world'),
        pointmap_normalize=config.get('pointmap_normalize', True),
        resolution=sam3_resolution,
        da3_resolution=da3_resolution if da3_resolution is not None else (config.get('da3_resolution') or 504),
        per_layer_text=config.get('per_layer_text', False),
        pred_logits_source=config.get('pred_logits_source', 'mask_mean'),
        use_da3_poses_for_gasa=config.get('use_da3_poses_for_gasa', False),
        use_gt_poses_for_gasa=config.get('use_gt_poses_for_gasa', False),
        da3_model_name=config.get('da3_model', 'depth-anything/DA3METRIC-LARGE').split('/')[-1],
        query_proj_mlp=config.get('query_proj_mlp', False),
        no_query_proj=config.get('no_query_proj', False),
        train_mask_embed=config.get('train_mask_embed', False),
        use_mask_refiner=config.get('use_mask_refiner', False),
        dim_feedforward=config.get('dim_feedforward', 2048),
        post_norm=config.get('post_norm', True),
        use_query_pe=config.get('use_query_pe', False),
        ffn_fp32=config.get('ffn_fp32', True),
        no_initial_text=config.get('no_initial_text', False),
        no_text_proj=config.get('no_text_proj', False),
        clean_v=config.get('clean_v', False),
        additive_pe=config.get('additive_pe', False),
        gasa_bidirectional=config.get('gasa_bidirectional', False),
        use_image_to_token=config.get('use_image_to_token', False),
        use_pos_refine=config.get('use_pos_refine', False),
        use_box_rpb=config.get('use_box_rpb', False),
        use_spatial_attn_bias=config.get('use_spatial_attn_bias', False),
        use_text_spatial_bias=config.get('use_text_spatial_bias', False),
        use_depth_crossattn=config.get('use_depth_crossattn', False),
        grouped_text_attn=config.get('grouped_text_attn', False),
        per_text_decode=config.get('per_text_decode', False),
    ).to(device)

    # Load checkpoint with compatibility layer to handle old formats
    missing, unexpected = model.gasa_decoder.load_state_dict_compat(checkpoint['gasa_decoder'], strict=False)
    if missing:
        print(f"WARNING: Missing keys in gasa_decoder: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys in gasa_decoder: {unexpected}")
    if not config.get('no_query_proj', False):
        model.query_proj.load_state_dict(checkpoint['query_proj'], strict=False)

    # Load mask refiner if present
    if 'mask_refiner' in checkpoint and checkpoint['mask_refiner'] is not None:
        model.mask_refiner.load_state_dict(checkpoint['mask_refiner'])
        print("  Loaded mask_refiner weights")

    # Load SAM3 seghead if it was trained and saved
    if 'sam3_seghead' in checkpoint and checkpoint['sam3_seghead'] is not None:
        if skip_trained_seghead:
            print("Skipping trained SAM3 seghead (--skip-trained-seghead), using default SAM3 weights")
        else:
            model.sam3.segmentation_head.load_state_dict(checkpoint['sam3_seghead'])
            print(f"Loaded trained SAM3 seghead from checkpoint")

    # Load mask_embed if it was trained and saved
    if 'mask_embed' in checkpoint and checkpoint['mask_embed'] is not None:
        model.sam3.segmentation_head.mask_predictor.mask_embed.load_state_dict(checkpoint['mask_embed'])
        print(f"Loaded trained mask_embed from checkpoint")

    # Load LoRA adapters if available (needed for mask_embed LoRA at eval)
    if 'lora' in checkpoint and checkpoint['lora'] is not None:
        import torch.nn as nn
        lora_rank = config.get('lora_rank', 8)
        lora_alpha = config.get('lora_alpha', 16.0)
        lora_manager = LoRAManager(rank=lora_rank, alpha=lora_alpha)
        # Re-create adapters based on config
        if config.get('lora_sam3', False):
            lora_manager.add_lora_to_model(model.sam3, "sam3")
        if config.get('lora_da3', False):
            lora_manager.add_lora_to_model(model.da3, "da3")
        if config.get('lora_mask_embed', False):
            mask_pred = model.sam3.segmentation_head.mask_predictor
            for i, layer in enumerate(mask_pred.mask_embed.layers):
                if isinstance(layer, nn.Linear):
                    adapter = LoRALayer(layer.in_features, layer.out_features,
                                        rank=lora_rank, alpha=lora_alpha)
                    adapter_name = f"mask_embed_layer{i}"
                    lora_manager.adapters[adapter_name] = adapter
                    hook = lora_manager._create_hook(adapter)
                    handle = layer.register_forward_hook(hook)
                    lora_manager.hooks.append(handle)
                    lora_manager._adapter_count += 1
        lora_manager.load_state_dict(checkpoint['lora'])
        lora_manager.to(device)
        print(f"Loaded LoRA state ({lora_manager.num_adapters} adapters, {lora_manager.num_parameters:,} params)")

    # Set multi-object mode from config (needed for SAM3-style batch expansion in forward)
    model.sam3_multi_object = config.get('sam3_multi_object', False)
    model.multi_object = config.get('multi_object', False)
    if model.sam3_multi_object:
        print(f"SAM3 multi-object mode: ENABLED (batch expansion)")

    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, best_iou={checkpoint.get('best_iou', 0)*100:.2f}%")

    return model


def load_scene_data(scene_path: Path, semantics_dir: Path) -> Tuple[List[Path], Dict, List[str]]:
    """Load scene images and annotations."""
    image_dir = scene_path / "dslr" / "resized_undistorted_images"
    if not image_dir.exists():
        image_dir = scene_path / "dslr" / "resized_images"

    images = sorted(image_dir.glob("*.JPG")) + sorted(image_dir.glob("*.jpg"))

    # Filter out excluded frames (DA3 depth errors)
    scene_id = scene_path.name
    before = len(images)
    images = [img for img in images if not is_excluded_frame(scene_id, img.stem)]
    if len(images) < before:
        print(f"  Excluded {before - len(images)} bad frames from {scene_id}")

    anno_path = scene_path / "scans" / "segments_anno.json"
    objects = {}

    if anno_path.exists():
        with open(anno_path) as f:
            anno_data = json.load(f)

        for group in anno_data.get('segGroups', []):
            obj_id = group.get('objectId') or group.get('id')
            # Normalize label to fix typos and inconsistencies (match training)

            label = normalize_label(group.get('label', 'unknown')).lower()
            if obj_id is not None:
                objects[obj_id] = {
                    'label': label,
                    'segments': group.get('segments', []),
                }

    available_frames = []
    if semantics_dir.exists():
        for pth_file in semantics_dir.glob("*.pth"):
            frame_name = pth_file.stem
            if not is_excluded_frame(scene_id, frame_name):
                available_frames.append(frame_name)

    return images, objects, available_frames


def load_gt_masks(semantics_dir: Path, frame_name: str) -> Dict[int, np.ndarray]:
    """Load ground truth masks for a frame."""
    mask_path = semantics_dir / f"{frame_name}.pth"

    if mask_path.exists():
        try:
            instance_map = torch.load(mask_path, weights_only=False)
            if isinstance(instance_map, torch.Tensor):
                instance_map = instance_map.numpy()

            masks = {}
            for obj_id in np.unique(instance_map):
                if obj_id > 0:
                    masks[int(obj_id)] = (instance_map == obj_id).astype(np.float32)
            return masks
        except Exception as e:
            print(f"Error loading {mask_path}: {e}")
            return {}

    return {}


def load_gt_poses(scene_path: Path) -> Tuple[Optional[Dict], Optional[torch.Tensor]]:
    """
    Load ground truth camera poses from transforms.json (NeRFStudio format).

    Returns:
        (transforms_dict, intrinsics_tensor)
        - transforms_dict: dict with 'frames' containing per-frame transforms
        - intrinsics_tensor: [3, 3] intrinsics matrix (shared across all frames)
    """
    transforms_path = scene_path / 'dslr' / 'nerfstudio' / 'transforms_undistorted.json'
    if not transforms_path.exists():
        transforms_path = scene_path / 'dslr' / 'nerfstudio' / 'transforms.json'

    if not transforms_path.exists():
        return None, None

    try:
        with open(transforms_path) as f:
            transforms = json.load(f)

        # Build intrinsics matrix from transforms
        fx = transforms.get('fl_x', 500)
        fy = transforms.get('fl_y', 500)
        cx = transforms.get('cx', 256)
        cy = transforms.get('cy', 256)

        intrinsics = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        return transforms, intrinsics

    except Exception as e:
        print(f"Error loading transforms from {transforms_path}: {e}")
        return None, None


def get_frame_extrinsics(transforms: Dict, frame_name: str) -> Optional[torch.Tensor]:
    """
    Get extrinsics (4x4 camera-to-world transform) for a specific frame.

    Args:
        transforms: transforms dict from load_gt_poses
        frame_name: image filename (e.g., 'DSC00001.JPG')

    Returns:
        [4, 4] extrinsics tensor, or None if frame not found
    """
    if transforms is None:
        return None

    # Build frame lookup
    for frame in transforms.get('frames', []):
        file_path = frame.get('file_path', '')
        if Path(file_path).name == frame_name:
            transform_matrix = frame.get('transform_matrix')
            if transform_matrix:
                return torch.tensor(transform_matrix, dtype=torch.float32)

    return None


def load_cached_da3_nested(
    cache_dir: Path,
    scene_id: str,
    frame_names: List[str],
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load cached DA3-NESTED outputs for a scene.

    Args:
        cache_dir: Path to DA3-NESTED cache directory
        scene_id: Scene identifier
        frame_names: List of frame names to load

    Returns:
        dict with:
            'depths': [N, H, W] metric depth in meters
            'extrinsics': [N, 4, 4] estimated camera poses
            'intrinsics': [N, 3, 3] estimated intrinsics
        Or None if cache not found
    """
    scene_cache_dir = cache_dir / scene_id

    if not scene_cache_dir.exists():
        return None

    # Check manifest
    manifest_path = scene_cache_dir / 'manifest.json'
    if not manifest_path.exists():
        return None

    depths = []
    extrinsics = []
    intrinsics = []

    for frame_name in frame_names:
        stem = Path(frame_name).stem
        cache_path = scene_cache_dir / f'{stem}.pt'

        if not cache_path.exists():
            return None  # Missing frame, can't use this cache

        try:
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            # Handle both torch tensor and numpy array cache formats
            d = data['depth']
            depth_t = d.float() if isinstance(d, torch.Tensor) else torch.from_numpy(d.astype(np.float32))
            depths.append(depth_t)

            e = data['extrinsics']
            ext_t = e.float() if isinstance(e, torch.Tensor) else torch.from_numpy(e).float()
            extrinsics.append(ext_t)

            i = data['intrinsics']
            int_t = i.float() if isinstance(i, torch.Tensor) else torch.from_numpy(i).float()
            intrinsics.append(int_t)
        except Exception as e:
            print(f"Error loading cached data for {scene_id}/{stem}: {e}")
            return None

    return {
        'depths': torch.stack(depths),  # [N, H, W]
        'extrinsics': torch.stack(extrinsics),  # [N, 4, 4]
        'intrinsics': torch.stack(intrinsics),  # [N, 3, 3]
    }


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute IoU and accuracy metrics.

    Returns:
        iou: Intersection over Union = TP / (TP + FP + FN)
        pixel_acc: (TP + TN) / total_pixels - global pixel accuracy (mAcc in most papers)
        recall: TP / (TP + FN) - what % of GT object did we find (mean class recall)
        precision: TP / (TP + FP) - what % of our prediction is correct
        f1: Harmonic mean of precision and recall
        tp, fp, fn, tn: Raw counts for aggregation
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    gt_binary = (gt > 0.5).float()

    # Compute confusion matrix elements
    tp = (pred_binary * gt_binary).sum()
    fp = (pred_binary * (1 - gt_binary)).sum()
    fn = ((1 - pred_binary) * gt_binary).sum()
    tn = ((1 - pred_binary) * (1 - gt_binary)).sum()
    total_pixels = tp + fp + fn + tn

    # IoU = TP / (TP + FP + FN)
    union = tp + fp + fn
    iou = (tp / union).item() if union > 0 else 1.0

    # Pixel Accuracy = (TP + TN) / total - standard mAcc in most papers
    pixel_acc = ((tp + tn) / total_pixels).item() if total_pixels > 0 else 1.0

    # Recall = TP / (TP + FN) - mean class recall metric
    recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 1.0

    # Precision = TP / (TP + FP)
    precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 1.0

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'iou': iou,
        'pixel_acc': pixel_acc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'tp': tp.item(),
        'fp': fp.item(),
        'fn': fn.item(),
        'tn': tn.item()
    }


def compute_oracle_iou(all_masks: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute oracle IoU by selecting the mask with best match to GT.

    Args:
        all_masks: [B, Q, H, W] or [Q, H, W] - all mask predictions
        gt: [H, W] or [B, H, W] - ground truth mask
        threshold: Threshold for binarization

    Returns:
        dict with oracle_iou (best possible IoU) and best_mask_idx
    """
    if all_masks.dim() == 4:
        all_masks = all_masks[0]  # [Q, H, W]
    if gt.dim() == 3:
        gt = gt[0]  # [H, W]

    # Upsample masks to GT resolution (not downsample GT) to match selected mask behavior
    # The selected mask in pred_masks is upsampled to image resolution, so we should do the same
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

    return {
        'oracle_iou': best_iou,
        'best_mask_idx': best_idx,
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

    # Depth-based (smaller depth = nearer)
    result['nearest'] = int(np.argmin(depths_at_centroid))
    result['closest'] = result['nearest']
    result['farthest'] = int(np.argmax(depths_at_centroid))

    # X-coordinate (smaller x = more left)
    all_x = [c[0] for c in centroids]
    result['leftmost'] = int(np.argmin(all_x))
    result['left'] = result['leftmost']
    result['rightmost'] = int(np.argmax(all_x))
    result['right'] = result['rightmost']

    # Y-coordinate (smaller y = higher/top)
    all_y = [c[1] for c in centroids]
    result['topmost'] = int(np.argmin(all_y))
    result['top'] = result['topmost']
    result['bottommost'] = int(np.argmax(all_y))
    result['bottom'] = result['bottommost']

    return result




@torch.no_grad()
def evaluate_multiview_single_prompt(
    model: TrianguLangModel,
    images: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    label: str,
    prompt_view: int = 0,
    device: str = 'cuda',
    world_extrinsics: Optional[torch.Tensor] = None,
    world_intrinsics: Optional[torch.Tensor] = None,
    gt_extrinsics: Optional[torch.Tensor] = None,
    gt_intrinsics: Optional[torch.Tensor] = None,
    use_world_poses: bool = False,
) -> Dict:
    """
    Multi-view single-prompt evaluation - Our key differentiator from MV-SAM.

    Process N views together, prompt only view 0, measure IoU on views 1-N.
    This tests whether GASA can propagate object understanding across views
    using geometric attention.

    Args:
        model: TrianguLangModel
        images: List of [C, H, W] tensors for each view
        gt_masks: List of [H, W] tensors for each view
        label: Text prompt (applied only to prompt_view)
        prompt_view: Which view to prompt (default 0)
        device: cuda/cpu
        world_extrinsics: [N, 4, 4] camera-to-world transforms for world PE (estimated or GT)
        world_intrinsics: [N, 3, 3] or [3, 3] intrinsics for world PE
        gt_extrinsics: [N, 4, 4] GT camera-to-world transforms (for GT 3D centroid computation)
        gt_intrinsics: [N, 3, 3] or [3, 3] GT intrinsics (for GT 3D centroid computation)
        use_world_poses: If True, use world-frame pointmaps for GASA and localization

    Returns:
        dict with prompted_iou, unprompted_iou, per_view_ious, and 3D metrics
    """
    N = len(images)
    if N < 2:
        return {'error': 'Need at least 2 views for single-prompt eval'}

    resolution = model.resolution

    # 1. Preprocess all images
    img_tensors = []
    for img in images:
        if img.shape[-2:] != (resolution, resolution):
            img = F.interpolate(img.unsqueeze(0), size=(resolution, resolution),
                               mode='bilinear', align_corners=False).squeeze(0)
        img_tensors.append(img)

    # Stack into batch [N, C, H, W]
    img_batch = torch.stack(img_tensors, dim=0).to(device)

    # 2. Get depth and pointmaps for all views
    with autocast('cuda', dtype=torch.float16):
        depth, pose, intrinsics = model.get_depth_and_pose(img_batch)

    # Compute pointmaps - either in camera frame or world frame
    if use_world_poses and world_extrinsics is not None:
        # Use provided world-frame poses for cross-view consistency
        world_extrinsics = world_extrinsics.to(device=device, dtype=depth.dtype)
        if world_intrinsics is not None:
            world_intrinsics = world_intrinsics.to(device=device, dtype=depth.dtype)
            if world_intrinsics.dim() == 2:
                world_intrinsics = world_intrinsics.unsqueeze(0).expand(N, -1, -1)
        else:
            world_intrinsics = intrinsics

        # world_intrinsics should already be scaled to model resolution by caller
        # (e.g. cached intrinsics scaled from 336x504 -> 504x504 at load time)
        pointmaps, _ = model.pointmap_computer(depth, world_extrinsics, world_intrinsics, normalize=True)
    else:
        # Default: camera-frame pointmaps (identity pose)
        pointmaps, _ = model.pointmap_computer(depth, pose, intrinsics, normalize=True)

    pointmaps = pointmaps.squeeze(1)

    # Downsample pointmaps
    pts = pointmaps.permute(0, 3, 1, 2)
    pts = F.adaptive_avg_pool2d(pts, (model.attn_map_size, model.attn_map_size))
    pointmaps_small = pts.permute(0, 2, 3, 1)  # [N, H', W', 3]

    # 3. Run SAM3 backbone + encoder for all views (batched, matching forward_multiview)
    with autocast('cuda', dtype=torch.float16):
        backbone_out = {"img_batch_all_stages": img_batch}
        backbone_out.update(model.sam3.backbone.forward_image(img_batch))  # [N, ...] batched

        # Text encoding - repeat for each view (encoder expects 1:1 img:text)
        text_prompts_expanded = [label] * N
        text_out = model.sam3.backbone.forward_text(text_prompts_expanded, device=device)
        backbone_out.update(text_out)

        # Run encoder batched for all N views at once
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

        prompt, prompt_mask, backbone_out = model.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )
        backbone_out, encoder_out, _ = model.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

    # Extract encoder memories: [N, L, D]
    encoder_memory = encoder_out["encoder_hidden_states"].transpose(0, 1)  # [N, L, D]
    L_per_view = encoder_memory.shape[1]

    # Extract per-view FPN features for seghead (needs per-view spatial maps)
    fpn_features_list = []
    for i in range(N):
        fpn_features_list.append([f[i:i+1] for f in backbone_out['backbone_fpn']])

    # 4. Concatenate all encoder memories for cross-view attention
    # This is the key: GASA attends to features from ALL views
    all_memory = encoder_memory.unsqueeze(0).view(1, N * L_per_view, -1)  # [1, N*L, D]

    # Concatenate pointmaps (flatten spatial dims)
    H_pts, W_pts = pointmaps_small.shape[1:3]

    # Reshape pointmaps to match encoder memory spatial size
    all_pointmaps_list = []
    for i in range(N):
        pts_flat = pointmaps_small[i].reshape(-1, 3)  # [H'*W', 3]
        # Resize to match L_per_view
        if pts_flat.shape[0] != L_per_view:
            pts_2d = pointmaps_small[i].permute(2, 0, 1).unsqueeze(0)  # [1, 3, H', W']
            target_size = int(math.sqrt(L_per_view))
            pts_2d = F.adaptive_avg_pool2d(pts_2d, (target_size, target_size))
            pts_flat = pts_2d.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            if pts_flat.shape[0] > L_per_view:
                pts_flat = pts_flat[:L_per_view]
            elif pts_flat.shape[0] < L_per_view:
                pad = torch.zeros(L_per_view - pts_flat.shape[0], 3, device=device)
                pts_flat = torch.cat([pts_flat, pad], dim=0)
        all_pointmaps_list.append(pts_flat)

    all_pointmaps = torch.cat(all_pointmaps_list, dim=0).unsqueeze(0)  # [1, N*L, 3]

    # 5. Get text embedding (all N copies are identical, take first)
    text_embedding = backbone_out.get('language_features', None)
    if text_embedding is not None:
        text_embedding = text_embedding.transpose(0, 1)  # [N, T, D]
        text_embedding = text_embedding[:1]  # [1, T, D] - all copies identical

    # 6. Run GASA decoder on concatenated memory (cross-view attention!)
    with autocast('cuda', dtype=torch.float16):
        queries, presence_logit, centroid_pred, iou_pred, per_query_centroids, text_scores, joint_scores, aux_outputs = model.gasa_decoder(
            all_memory,
            pointmaps_small[0],  # Use view 0's pointmap structure
            text_embedding,
            box_prompts=None,
            box_labels=None
        )
        queries = model.query_proj(queries)  # [1, Q, D]

    # 7. Compute GT pointmaps if GT poses provided (for accurate GT centroid measurement)
    gt_pointmaps = None
    if gt_extrinsics is not None:
        gt_extrinsics_dev = gt_extrinsics.to(device=device, dtype=depth.dtype)
        if gt_intrinsics is not None:
            gt_intrinsics_dev = gt_intrinsics.to(device=device, dtype=depth.dtype)
            if gt_intrinsics_dev.dim() == 2:
                gt_intrinsics_dev = gt_intrinsics_dev.unsqueeze(0).expand(N, -1, -1)
        else:
            gt_intrinsics_dev = intrinsics
        gt_pointmaps, _ = model.pointmap_computer(depth, gt_extrinsics_dev, gt_intrinsics_dev, normalize=True)
        gt_pointmaps = gt_pointmaps.squeeze(1)

    # 8. Generate masks for each view using shared queries
    per_view_results = []
    all_pred_masks = []  # Collect for cross-view consistency
    centroid_errors = []  # Collect for Acc@Xcm metrics
    centroid_errors_world = []  # Centroid errors in world frame (if GT poses available)

    for i in range(N):
        # Get pixel embeddings for this view
        fpn_features = fpn_features_list[i]
        with autocast('cuda', dtype=torch.float16):
            pixel_embed = model.sam3.segmentation_head.pixel_decoder(fpn_features)
            instance_embeds = model.sam3.segmentation_head.instance_seg_head(pixel_embed)

            # Predict mask using shared queries
            mask_preds = model.sam3.segmentation_head.mask_predictor(queries, instance_embeds)  # [1, Q, H, W]

        # Select best mask by confidence
        pred_logits = mask_preds.mean(dim=(-2, -1))
        best_idx = pred_logits.argmax(dim=1)
        pred_mask = mask_preds[0, best_idx[0]]  # [H, W]

        # Compute IoU with GT
        gt = gt_masks[i]
        if gt.shape != pred_mask.shape:
            gt = F.interpolate(gt.unsqueeze(0).unsqueeze(0).float(),
                              size=pred_mask.shape, mode='nearest').squeeze()

        pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
        gt_binary = (gt > 0.5).float()

        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        iou = (intersection / union.clamp(min=1.0)).item()

        per_view_results.append({
            'view_idx': i,
            'iou': iou,
            'is_prompted': i == prompt_view,
        })

        # Collect pred mask for consistency metric
        all_pred_masks.append(pred_mask.detach())

        # Compute 3D centroid error using the pointmaps used for prediction
        pred_centroid = compute_3d_centroid(pred_mask.detach(), pointmaps[i])
        gt_centroid = compute_3d_centroid(gt.to(device), pointmaps[i])
        if pred_centroid is not None and gt_centroid is not None:
            error = compute_centroid_error(pred_centroid, gt_centroid)
            centroid_errors.append(error)

        # Also compute world-frame centroid error if GT pointmaps available
        # This is the "true" localization error using GT coordinate system
        if gt_pointmaps is not None:
            pred_centroid_world = compute_3d_centroid(pred_mask.detach(), gt_pointmaps[i])
            gt_centroid_world = compute_3d_centroid(gt.to(device), gt_pointmaps[i])
            if pred_centroid_world is not None and gt_centroid_world is not None:
                error_world = compute_centroid_error(pred_centroid_world, gt_centroid_world)
                centroid_errors_world.append(error_world)

    # 9. Compute aggregate metrics
    prompted_ious = [r['iou'] for r in per_view_results if r['is_prompted']]
    unprompted_ious = [r['iou'] for r in per_view_results if not r['is_prompted']]

    # 10. Compute cross-view consistency
    consistency_result = compute_cross_view_consistency(all_pred_masks, pointmaps)

    # 11. Compute centroid accuracy metrics
    acc_5cm = sum(1 for e in centroid_errors if e < 0.05) / max(len(centroid_errors), 1)
    acc_10cm = sum(1 for e in centroid_errors if e < 0.10) / max(len(centroid_errors), 1)
    mean_centroid_error = np.mean(centroid_errors) if centroid_errors else float('inf')

    # World-frame accuracy (using GT poses for reference)
    acc_5cm_world = None
    acc_10cm_world = None
    mean_centroid_error_world = None
    if centroid_errors_world:
        acc_5cm_world = sum(1 for e in centroid_errors_world if e < 0.05) / max(len(centroid_errors_world), 1)
        acc_10cm_world = sum(1 for e in centroid_errors_world if e < 0.10) / max(len(centroid_errors_world), 1)
        mean_centroid_error_world = np.mean(centroid_errors_world)

    result = {
        'prompted_iou': np.mean(prompted_ious) if prompted_ious else 0.0,
        'unprompted_iou': np.mean(unprompted_ious) if unprompted_ious else 0.0,
        'all_views_iou': np.mean([r['iou'] for r in per_view_results]),
        'per_view_results': per_view_results,
        'propagation_ratio': np.mean(unprompted_ious) / max(np.mean(prompted_ious), 0.01) if prompted_ious else 0.0,
        # 3D metrics (camera-frame or world-frame depending on input)
        'cross_view_consistency': consistency_result['consistency'],
        'num_correspondences': consistency_result['num_correspondences'],
        'acc_5cm': acc_5cm,
        'acc_10cm': acc_10cm,
        'mean_centroid_error_m': mean_centroid_error,
    }

    # Add world-frame metrics if GT poses were provided
    if acc_5cm_world is not None:
        result['acc_5cm_world'] = acc_5cm_world
        result['acc_10cm_world'] = acc_10cm_world
        result['mean_centroid_error_world_m'] = mean_centroid_error_world

    return result


@torch.no_grad()
def evaluate_scene_single_prompt(
    model: TrianguLangModel,
    scene_path: Path,
    semantics_dir: Path,
    device: str = 'cuda',
    num_views: int = 4,
    objects_per_scene: int = 5,
    min_pixel_fraction: float = 0.001,
    image_size: Tuple[int, int] = (1008, 1008),
    prompt_view: int = 0,
    use_world_poses: bool = False,
    use_estimated_poses: bool = False,
    da3_nested_cache_dir: Optional[Path] = None,
    allowed_categories: Optional[set] = None,
) -> Dict:
    """
    Evaluate single-prompt propagation on a scene.

    For each object, find N views where it's visible, prompt view 0,
    measure how well the model propagates to views 1-N.

    Args:
        use_world_poses: If True, use world-frame poses for pointmaps
        use_estimated_poses: If True, use DA3-NESTED estimated poses (else GT)
        da3_nested_cache_dir: Path to DA3-NESTED cache directory
    """
    from triangulang.evaluation.evaluate_gasa import load_scene_data, load_gt_masks

    images, objects, available_frames = load_scene_data(scene_path, semantics_dir)
    available_frames_set = set(available_frames)
    images = [img for img in images if img.name in available_frames_set]

    if len(images) < num_views:
        return {'error': f'Need at least {num_views} images, found {len(images)}'}

    # Match training skip_labels + structural elements
    skip_labels = {
        # From training (scannetpp_loader.py)
        'remove', 'split', 'object', 'objects', 'stuff', 'unknown',
        # Structural elements (too easy/not interesting)
        'wall', 'floor', 'ceiling', 'door', 'window', 'doorframe', 'window frame',
        # Annotation artifacts
        'reflection', 'mirror', 'structure', 'shoes', 'book', 'shoe'
    }
    valid_objects = [(obj_id, obj) for obj_id, obj in objects.items()
                     if obj['label'] and obj['label'].lower() not in skip_labels]

    # Filter by allowed categories if specified (supports substring matching)
    if allowed_categories is not None:
        def matches_allowed(label):
            label_lower = label.lower()
            for allowed in allowed_categories:
                # Exact match or substring match (e.g., "towel" matches "kitchen towel")
                if allowed == label_lower or allowed in label_lower or label_lower in allowed:
                    return True
            return False
        valid_objects = [(obj_id, obj) for obj_id, obj in valid_objects
                         if matches_allowed(obj['label'])]

    if len(valid_objects) == 0:
        return {'error': 'No valid objects'}

    random.shuffle(valid_objects)

    results = {
        'prompted_ious': [],
        'unprompted_ious': [],
        'propagation_ratios': [],
        'objects_evaluated': 0,
        # 3D metrics
        'cross_view_consistencies': [],
        'acc_5cm_list': [],
        'acc_10cm_list': [],
        'centroid_errors': [],
    }

    # Load GT poses if using world poses
    gt_transforms = None
    gt_intrinsics_shared = None
    if use_world_poses or use_estimated_poses:
        gt_transforms, gt_intrinsics_shared = load_gt_poses(scene_path)

    # Load estimated poses cache if using estimated poses
    scene_id = scene_path.name
    estimated_cache = None
    if use_estimated_poses and da3_nested_cache_dir:
        # Will load per-object frame selection below
        pass

    for obj_id, obj_data in valid_objects[:objects_per_scene]:
        label = obj_data['label']

        # Find images where this object is visible with sufficient coverage
        obj_images = []
        obj_gt_masks = []
        obj_frame_names = []  # Track frame names for pose lookup

        for img_path in images:
            frame_name = img_path.name
            gt_masks_dict = load_gt_masks(semantics_dir, frame_name)

            if obj_id not in gt_masks_dict:
                continue

            gt_mask = gt_masks_dict[obj_id]
            pixel_fraction = gt_mask.sum() / gt_mask.size

            if pixel_fraction >= min_pixel_fraction:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size, Image.BILINEAR)
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

                # Resize GT mask
                gt_tensor = torch.from_numpy(gt_mask).float()
                gt_tensor = F.interpolate(gt_tensor.unsqueeze(0).unsqueeze(0),
                                         size=image_size, mode='nearest').squeeze()

                obj_images.append(img_tensor)
                obj_gt_masks.append(gt_tensor)
                obj_frame_names.append(frame_name)

                if len(obj_images) >= num_views:
                    break

        if len(obj_images) < num_views:
            continue

        # Prepare pose tensors for this object's views
        world_extrinsics = None
        world_intrinsics = None
        gt_extrinsics = None
        gt_intrinsics = None

        if use_world_poses or use_estimated_poses:
            selected_frame_names = obj_frame_names[:num_views]

            # Always load GT extrinsics for ground truth centroid comparison
            if gt_transforms is not None:
                gt_ext_list = []
                for fname in selected_frame_names:
                    ext = get_frame_extrinsics(gt_transforms, fname)
                    if ext is not None:
                        gt_ext_list.append(ext)
                    else:
                        gt_ext_list.append(torch.eye(4))
                if gt_ext_list:
                    gt_extrinsics = torch.stack(gt_ext_list)
                    gt_intrinsics = gt_intrinsics_shared

            # Load world extrinsics (either GT or estimated)
            if use_estimated_poses and da3_nested_cache_dir:
                # Load from DA3-NESTED cache
                estimated_cache = load_cached_da3_nested(
                    da3_nested_cache_dir, scene_id, selected_frame_names
                )
                if estimated_cache is not None:
                    world_extrinsics = estimated_cache['extrinsics']
                    world_intrinsics = estimated_cache['intrinsics']
                    # Scale intrinsics from cache resolution to model resolution
                    # Cache stores intrinsics at processing resolution (e.g. 336x504)
                    # but model computes pointmaps at model resolution (e.g. 504x504)
                    cache_h, cache_w = estimated_cache['depths'].shape[-2:]
                    model_res = model.resolution
                    if cache_h != model_res or cache_w != model_res:
                        scale_x = model_res / cache_w
                        scale_y = model_res / cache_h
                        world_intrinsics = world_intrinsics.clone()
                        world_intrinsics[:, 0, 0] *= scale_x  # fx
                        world_intrinsics[:, 1, 1] *= scale_y  # fy
                        world_intrinsics[:, 0, 2] *= scale_x  # cx
                        world_intrinsics[:, 1, 2] *= scale_y  # cy
                else:
                    # Fall back to GT poses if cache not available
                    world_extrinsics = gt_extrinsics
                    world_intrinsics = gt_intrinsics
            else:
                # Use GT poses for world coordinates
                world_extrinsics = gt_extrinsics
                world_intrinsics = gt_intrinsics

        # Scale GT intrinsics from original resolution to model resolution if needed
        # (cached DA3 intrinsics are already scaled above; this handles GT fallback paths)
        if world_intrinsics is not None and gt_transforms is not None:
            orig_w = gt_transforms.get('w', None)
            orig_h = gt_transforms.get('h', None)
            model_res = model.resolution
            # Only scale if these are GT intrinsics (at original resolution),
            # not already-scaled cache intrinsics
            if orig_w and orig_h and world_intrinsics is gt_intrinsics:
                scale_x = model_res / orig_w
                scale_y = model_res / orig_h
                world_intrinsics = world_intrinsics.clone()
                if world_intrinsics.dim() == 2:
                    world_intrinsics[0, 0] *= scale_x  # fx
                    world_intrinsics[1, 1] *= scale_y  # fy
                    world_intrinsics[0, 2] *= scale_x  # cx
                    world_intrinsics[1, 2] *= scale_y  # cy
                else:
                    world_intrinsics[:, 0, 0] *= scale_x
                    world_intrinsics[:, 1, 1] *= scale_y
                    world_intrinsics[:, 0, 2] *= scale_x
                    world_intrinsics[:, 1, 2] *= scale_y

        # Scale GT intrinsics to model resolution for accurate GT centroid computation
        gt_intrinsics_scaled = gt_intrinsics
        if gt_intrinsics is not None and gt_transforms is not None:
            orig_w = gt_transforms.get('w', None)
            orig_h = gt_transforms.get('h', None)
            model_res = model.resolution
            if orig_w and orig_h and (orig_w != model_res or orig_h != model_res):
                scale_x = model_res / orig_w
                scale_y = model_res / orig_h
                gt_intrinsics_scaled = gt_intrinsics.clone()
                if gt_intrinsics_scaled.dim() == 2:
                    gt_intrinsics_scaled[0, 0] *= scale_x
                    gt_intrinsics_scaled[1, 1] *= scale_y
                    gt_intrinsics_scaled[0, 2] *= scale_x
                    gt_intrinsics_scaled[1, 2] *= scale_y
                else:
                    gt_intrinsics_scaled[:, 0, 0] *= scale_x
                    gt_intrinsics_scaled[:, 1, 1] *= scale_y
                    gt_intrinsics_scaled[:, 0, 2] *= scale_x
                    gt_intrinsics_scaled[:, 1, 2] *= scale_y

        # Run multi-view single-prompt evaluation
        mv_result = evaluate_multiview_single_prompt(
            model,
            obj_images[:num_views],
            obj_gt_masks[:num_views],
            label,
            prompt_view=prompt_view,
            device=device,
            world_extrinsics=world_extrinsics,
            world_intrinsics=world_intrinsics,
            gt_extrinsics=gt_extrinsics,
            gt_intrinsics=gt_intrinsics_scaled,
            use_world_poses=use_world_poses or use_estimated_poses,
        )

        if 'error' not in mv_result:
            results['prompted_ious'].append(mv_result['prompted_iou'])
            results['unprompted_ious'].append(mv_result['unprompted_iou'])
            results['propagation_ratios'].append(mv_result['propagation_ratio'])
            results['objects_evaluated'] += 1
            # 3D metrics
            results['cross_view_consistencies'].append(mv_result['cross_view_consistency'])
            results['acc_5cm_list'].append(mv_result['acc_5cm'])
            results['acc_10cm_list'].append(mv_result['acc_10cm'])
            if mv_result['mean_centroid_error_m'] != float('inf'):
                results['centroid_errors'].append(mv_result['mean_centroid_error_m'])

            # World-frame metrics (if GT poses were provided)
            if 'acc_5cm_world' in mv_result:
                if 'acc_5cm_world_list' not in results:
                    results['acc_5cm_world_list'] = []
                    results['acc_10cm_world_list'] = []
                    results['centroid_errors_world'] = []
                results['acc_5cm_world_list'].append(mv_result['acc_5cm_world'])
                results['acc_10cm_world_list'].append(mv_result['acc_10cm_world'])
                if mv_result.get('mean_centroid_error_world_m') and mv_result['mean_centroid_error_world_m'] != float('inf'):
                    results['centroid_errors_world'].append(mv_result['mean_centroid_error_world_m'])

    if results['objects_evaluated'] == 0:
        return {'error': 'No objects with sufficient views'}

    scene_result = {
        'scene_id': scene_path.name,
        'objects_evaluated': results['objects_evaluated'],
        'mean_prompted_iou': np.mean(results['prompted_ious']),
        'mean_unprompted_iou': np.mean(results['unprompted_ious']),
        'mean_propagation_ratio': np.mean(results['propagation_ratios']),
        'std_propagation_ratio': np.std(results['propagation_ratios']),
        # 3D metrics
        'cross_view_consistency': np.mean(results['cross_view_consistencies']) if results['cross_view_consistencies'] else 0.0,
        'acc_5cm': np.mean(results['acc_5cm_list']) if results['acc_5cm_list'] else 0.0,
        'acc_10cm': np.mean(results['acc_10cm_list']) if results['acc_10cm_list'] else 0.0,
        'mean_centroid_error_m': np.mean(results['centroid_errors']) if results['centroid_errors'] else float('inf'),
    }

    # Add world-frame metrics if available
    if 'acc_5cm_world_list' in results:
        scene_result['acc_5cm_world'] = np.mean(results['acc_5cm_world_list']) if results['acc_5cm_world_list'] else 0.0
        scene_result['acc_10cm_world'] = np.mean(results['acc_10cm_world_list']) if results['acc_10cm_world_list'] else 0.0
        scene_result['mean_centroid_error_world_m'] = np.mean(results['centroid_errors_world']) if results['centroid_errors_world'] else float('inf')

    return scene_result


@torch.no_grad()
def _evaluate_scene_multi_object(
    model, scene_path, semantics_dir, device, images, image_size,
    eval_items, min_pixel_fraction, da3_cache_dir, precomputed_backbone,
    save_viz=False, viz_dir=None, viz_samples=5,
    temporal_smooth_alpha=0.0,
    use_crf=False,
):
    """Multi-object eval: batch all K objects per frame in a single forward pass.

    Instead of K separate forward passes per frame (one per object),
    passes all K text prompts at once. The model produces Q masks and
    text scores [Q, K], allowing us to assign the best query to each object.

    Speedup: ~K× on decoder time (backbone already cached).
    """
    from scipy.optimize import linear_sum_assignment

    results = defaultdict(list)
    category_metrics = defaultdict(lambda: {'iou': [], 'oracle_iou': [], 'pixel_acc': [], 'recall': [], 'precision': [], 'f1': []})
    preprocess_times = []
    inference_times = []
    viz_data = []

    # Temporal EMA smoothing: maintain per-object logit history across frames
    prev_logits = {}  # k_idx -> [H, W] tensor on CPU
    use_temporal_smooth = temporal_smooth_alpha > 0.0

    # Build list of (label, obj_ids, prompt, cat_label) — use spatial override as model prompt when available
    # cat_label: for spatial items, use the spatial prompt (e.g. "nearest chair") so reporting can find them;
    # for regular items, use the base label.
    object_list = []
    for label, oids, spatial_override in eval_items:
        prompt = spatial_override if spatial_override else label
        cat_label = spatial_override if spatial_override else label
        object_list.append((label, oids, prompt, cat_label))
    K = len(object_list)
    if K == 0:
        return {'error': 'no valid objects'}

    all_labels = [prompt for _, _, prompt, _ in object_list]

    for frame_idx, img_path in enumerate(images):
        frame_name = img_path.name
        frame_stem = img_path.stem
        gt_masks_dict = load_gt_masks(semantics_dir, frame_name)

        # Build GT masks for all K objects on this frame
        gt_per_object = []  # K entries: numpy mask or None
        valid_objects = []   # indices of objects with valid GT on this frame
        for k, (label, obj_ids, _prompt, _cat) in enumerate(object_list):
            gt_mask = None
            for oid in obj_ids:
                if oid in gt_masks_dict:
                    if gt_mask is None:
                        gt_mask = gt_masks_dict[oid].copy()
                    else:
                        gt_mask = np.maximum(gt_mask, gt_masks_dict[oid])
            if gt_mask is not None and gt_mask.sum() / gt_mask.size >= min_pixel_fraction:
                gt_per_object.append(gt_mask)
                valid_objects.append(k)
            else:
                gt_per_object.append(None)

        if not valid_objects:
            continue

        # Load image
        t_preprocess_start = time.perf_counter()
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size, Image.BILINEAR)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

        # Load cached depth
        cached_depth = None
        da3_intrinsics = None
        da3_extrinsics = None
        if da3_cache_dir is not None:
            scene_id = scene_path.name
            cache_path = da3_cache_dir / scene_id / f"{frame_stem}.pt"
            if cache_path.exists():
                try:
                    cache_data = torch.load(cache_path, map_location='cpu', weights_only=True)
                    depth = cache_data['depth'].float()
                    if depth.dim() == 4:
                        depth = depth.squeeze(0)
                    elif depth.dim() == 2:
                        depth = depth.unsqueeze(0)
                    cached_depth = depth.unsqueeze(0).to(device)
                    if 'intrinsics' in cache_data:
                        da3_intrinsics = cache_data['intrinsics'].float().unsqueeze(0).to(device)
                    if 'extrinsics' in cache_data:
                        da3_extrinsics = cache_data['extrinsics'].float().unsqueeze(0).to(device)
                except Exception:
                    pass

        # Get precomputed backbone
        _frame_backbone = precomputed_backbone.get(frame_stem)

        # Build K text prompts (for all valid objects)
        # We pass ALL K labels so query-text alignment works correctly
        text_prompts_k = all_labels  # K prompts

        # Build per-object GT tensor [1, K, H, W] for oracle mask selection (iou_match)
        # Without this, the model falls back to confidence selection which is much worse.
        # This matches single-object eval which also passes GT for oracle selection.
        gt_for_model = None
        sam3_mo = getattr(model, 'sam3_multi_object', False)
        if sam3_mo:
            gt_list = []
            for k in range(K):
                if gt_per_object[k] is not None:
                    gt_t = torch.from_numpy(gt_per_object[k]).float()
                    gt_t = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), size=image_size,
                                         mode='nearest').squeeze(0).squeeze(0)
                else:
                    gt_t = torch.zeros(image_size, dtype=torch.float32)
                gt_list.append(gt_t)
            gt_for_model = torch.stack(gt_list).unsqueeze(0).to(device)  # [1, K, H, W]

        t_preprocess_end = time.perf_counter()
        preprocess_times.append(t_preprocess_end - t_preprocess_start)

        t_start = time.perf_counter()
        with autocast('cuda', dtype=torch.float16):
            outputs = model(
                img_tensor, text_prompts_k, gt_for_model,
                cached_depth=cached_depth,
                da3_intrinsics=da3_intrinsics,
                da3_extrinsics=da3_extrinsics,
                precomputed_sam3=_frame_backbone,
                num_texts=K,
            )
        t_end = time.perf_counter()
        inference_times.append(t_end - t_start)

        # SAM3-MO mode: pred_masks is [K, 1, H, W] — one mask per object, already matched
        # Non-SAM3: all_masks is [1, Q, H, W] — need Hungarian matching
        sam3_mo_K = outputs.get('sam3_mo_K')
        if sam3_mo and sam3_mo_K is not None:
            # SAM3-style: pred_masks[k] is the prediction for object k
            pred_masks_k = outputs['pred_masks']  # [K, 1, H, W]
            pred_masks_k = pred_masks_k.squeeze(1)  # [K, H, W]

            # Get all candidate masks for oracle computation: [K, Q, H_mask, W_mask]
            all_masks_mo = outputs.get('all_masks')  # [K, Q, H_mask, W_mask] in SAM3-MO mode

            # Resize to GT resolution
            gt_h, gt_w = gt_per_object[valid_objects[0]].shape
            mask_h, mask_w = pred_masks_k.shape[-2:]
            if (mask_h, mask_w) != (gt_h, gt_w):
                pred_masks_k = F.interpolate(
                    pred_masks_k.unsqueeze(1).float(), size=(gt_h, gt_w),
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            # Temporal EMA smoothing on logits before thresholding
            if use_temporal_smooth:
                alpha = temporal_smooth_alpha
                for k_idx in range(pred_masks_k.shape[0]):
                    logit_cpu = pred_masks_k[k_idx].cpu().float()
                    if k_idx in prev_logits:
                        logit_cpu = alpha * logit_cpu + (1 - alpha) * prev_logits[k_idx]
                    prev_logits[k_idx] = logit_cpu
                    pred_masks_k[k_idx] = logit_cpu.to(device)

            # CRF / morphological post-processing to refine mask boundaries
            if use_crf:
                from triangulang.utils.crf_postprocess import morphological_smooth
                for k_idx in range(pred_masks_k.shape[0]):
                    mask_binary = (torch.sigmoid(pred_masks_k[k_idx]) > 0.5).cpu().numpy().astype(np.float32)
                    refined = morphological_smooth(mask_binary, kernel_size=7)
                    refined_logit = torch.from_numpy(refined * 10.0 - 5.0).to(device)
                    pred_masks_k[k_idx] = refined_logit

            # Direct 1:1 matching — object k's mask is pred_masks_k[k]
            matched_pairs = [(k_idx, k_idx) for k_idx in valid_objects]  # (obj_idx, pred_idx)
            pred_source = pred_masks_k
            gt_source = gt_per_object
        else:
            # Original path: Hungarian matching on IoU
            all_masks = outputs.get('all_masks')  # [1, Q, H, W]
            if all_masks is None:
                continue
            all_masks = all_masks.squeeze(0)  # [Q, H, W]
            Q = all_masks.shape[0]

            # Resize masks to GT resolution
            gt_h, gt_w = gt_per_object[valid_objects[0]].shape
            mask_h, mask_w = all_masks.shape[-2:]
            if (mask_h, mask_w) != (gt_h, gt_w):
                all_masks = F.interpolate(
                    all_masks.unsqueeze(1).float(), size=(gt_h, gt_w),
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            # Build GT tensor for valid objects
            gt_tensors = []
            for k_idx in valid_objects:
                gt_t = torch.from_numpy(gt_per_object[k_idx]).float().to(device)
                gt_tensors.append(gt_t)
            gt_stack = torch.stack(gt_tensors)  # [K_valid, H, W]
            K_valid = len(valid_objects)

            # IoU cost matrix [Q, K_valid]
            pred_binary = (torch.sigmoid(all_masks) > 0.5).float()
            cost_matrix = torch.zeros(Q, K_valid, device=device)
            for ki, _ in enumerate(valid_objects):
                gt_k = (gt_stack[ki] > 0.5).float()
                intersection = (pred_binary * gt_k.unsqueeze(0)).sum(dim=(-2, -1))
                union = pred_binary.sum(dim=(-2, -1)) + gt_k.sum() - intersection
                ious = intersection / union.clamp(min=1.0)
                cost_matrix[:, ki] = -ious

            # Add text score cost if available
            text_scores = outputs.get('text_scores')
            if text_scores is not None:
                ts = text_scores.squeeze(0)  # [Q, K]
                if ts.shape[-1] >= K:
                    valid_ts = ts[:, valid_objects]
                    cost_matrix = cost_matrix + 0.3 * (-valid_ts.sigmoid())

            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            # Build matched pairs: (obj_idx, query_idx)
            matched_pairs = [(valid_objects[ki], qi) for qi, ki in zip(row_ind.tolist(), col_ind.tolist())]
            pred_source = all_masks
            gt_source = None  # Use gt_stack indexed by col_ind
            # Remap: for Hungarian path, pred is query idx, gt is from gt_stack
            matched_pairs_hungarian = list(zip(row_ind.tolist(), col_ind.tolist()))

        # Spatial postprocessing: after getting all masks, assign spatial labels via geometry
        # This uses depth at predicted mask centroids to determine nearest/farthest/left/right
        # among same-label predicted masks — leverages good segmentation + geometric reasoning
        spatial_postprocess_labels = {}  # obj_k -> list of spatial cat_labels
        if sam3_mo and sam3_mo_K is not None and cached_depth is not None:
            # Group predictions by base label
            from collections import defaultdict as _dd
            label_groups = _dd(list)
            for obj_k, pred_k in matched_pairs:
                base_label = object_list[obj_k][0]  # base label (not spatial prompt)
                label_groups[base_label].append((obj_k, pred_k))

            for base_label, group in label_groups.items():
                if len(group) < 2:
                    continue
                # Compute centroids and depth for predicted masks
                pred_centroids = []
                for obj_k, pred_k in group:
                    pm = torch.sigmoid(pred_source[pred_k])
                    if pm.sum() < 1:
                        pred_centroids.append((0.5, 0.5, 0.0))
                        continue
                    ys, xs = torch.where(pm > 0.5)
                    if len(xs) == 0:
                        pred_centroids.append((0.5, 0.5, 0.0))
                        continue
                    cx, cy = float(xs.float().mean()), float(ys.float().mean())
                    # Get depth at centroid
                    depth_map = cached_depth.squeeze()  # [H, W]
                    if depth_map.shape != pm.shape:
                        depth_map = F.interpolate(depth_map.unsqueeze(0).unsqueeze(0).float(),
                                                   size=pm.shape, mode='bilinear',
                                                   align_corners=False).squeeze()
                    cy_int = min(max(int(cy), 0), depth_map.shape[0] - 1)
                    cx_int = min(max(int(cx), 0), depth_map.shape[1] - 1)
                    d = float(depth_map[cy_int, cx_int])
                    pred_centroids.append((cx, cy, d))

                # Assign spatial labels
                depths = [c[2] for c in pred_centroids]
                xs = [c[0] for c in pred_centroids]
                for qualifier, values, fn in [
                    ('nearest', depths, min), ('farthest', depths, max),
                    ('leftmost', xs, min), ('rightmost', xs, max),
                ]:
                    if all(v == 0 for v in values):
                        continue
                    best_idx = values.index(fn(v for v in values if v != 0) if any(v != 0 for v in values) else 0)
                    obj_k = group[best_idx][0]
                    spatial_cat = f"{qualifier} {base_label}"
                    if obj_k not in spatial_postprocess_labels:
                        spatial_postprocess_labels[obj_k] = []
                    spatial_postprocess_labels[obj_k].append(spatial_cat)

        # Compute metrics for matched pairs
        if sam3_mo and sam3_mo_K is not None:
            for obj_k, pred_k in matched_pairs:
                cat_label = object_list[obj_k][3]  # cat_label: spatial prompt for spatial items, base label otherwise
                pred_mask = pred_source[pred_k]  # [H, W]
                gt_mask = torch.from_numpy(gt_per_object[obj_k]).float().to(device)

                metrics = compute_metrics(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
                results['iou'].append(metrics['iou'])

                # Compute true oracle IoU from all Q candidate masks
                oracle_iou = metrics['iou']  # fallback
                if all_masks_mo is not None and obj_k < all_masks_mo.shape[0]:
                    oracle_result = compute_oracle_iou(all_masks_mo[obj_k], gt_mask)
                    oracle_iou = oracle_result['oracle_iou']
                results['oracle_iou'].append(oracle_iou)

                results['pixel_acc'].append(metrics['pixel_acc'])
                results['recall'].append(metrics['recall'])
                results['precision'].append(metrics['precision'])
                results['f1'].append(metrics['f1'])
                results['tp'].append(metrics['tp'])
                results['fp'].append(metrics['fp'])
                results['fn'].append(metrics['fn'])
                results['tn'].append(metrics['tn'])
                category_metrics[cat_label]['iou'].append(metrics['iou'])
                category_metrics[cat_label]['oracle_iou'].append(oracle_iou)
                category_metrics[cat_label]['pixel_acc'].append(metrics['pixel_acc'])
                category_metrics[cat_label]['recall'].append(metrics['recall'])

                # Spatial postprocess: also record this mask's IoU under geometric spatial labels
                if obj_k in spatial_postprocess_labels:
                    for sp_cat in spatial_postprocess_labels[obj_k]:
                        sp_cat_pp = f"pp_{sp_cat}"  # prefix with pp_ to distinguish from model-predicted spatial
                        category_metrics[sp_cat_pp]['iou'].append(metrics['iou'])
                        category_metrics[sp_cat_pp]['oracle_iou'].append(oracle_iou)
                        category_metrics[sp_cat_pp]['recall'].append(metrics['recall'])
        else:
            for qi, ki in matched_pairs_hungarian:
                obj_k = valid_objects[ki]
                cat_label = object_list[obj_k][3]  # cat_label: spatial prompt for spatial items, base label otherwise
                pred_mask = all_masks[qi]
                gt_mask = gt_stack[ki]

                metrics = compute_metrics(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
                results['iou'].append(metrics['iou'])

                # Oracle: find best of Q masks for this GT object
                oracle_result = compute_oracle_iou(all_masks.unsqueeze(0), gt_mask)
                oracle_iou = oracle_result['oracle_iou']
                results['oracle_iou'].append(oracle_iou)

                results['pixel_acc'].append(metrics['pixel_acc'])
                results['recall'].append(metrics['recall'])
                results['precision'].append(metrics['precision'])
                results['f1'].append(metrics['f1'])
                results['tp'].append(metrics['tp'])
                results['fp'].append(metrics['fp'])
                results['fn'].append(metrics['fn'])
                results['tn'].append(metrics['tn'])
                category_metrics[cat_label]['iou'].append(metrics['iou'])
                category_metrics[cat_label]['oracle_iou'].append(oracle_iou)
                category_metrics[cat_label]['pixel_acc'].append(metrics['pixel_acc'])
                category_metrics[cat_label]['recall'].append(metrics['recall'])

        # Collect viz data: group all objects per frame for multi-object overlay
        if save_viz and len(viz_data) < viz_samples:
            frame_objects = []
            img_h, img_w = img_np.shape[:2]  # image_size resolution
            if sam3_mo and sam3_mo_K is not None:
                for obj_k, pred_k in matched_pairs:
                    # Resize pred mask to image resolution
                    pred_t = torch.sigmoid(pred_source[pred_k]).unsqueeze(0).unsqueeze(0).float()
                    pred_resized = F.interpolate(pred_t, size=(img_h, img_w), mode='bilinear', align_corners=False)
                    pred_np = (pred_resized.squeeze().cpu().numpy() > 0.5).astype(np.float32)
                    # Resize GT mask to image resolution
                    gt_raw = gt_per_object[obj_k]
                    if gt_raw.shape[:2] != (img_h, img_w):
                        gt_t = torch.from_numpy(gt_raw).unsqueeze(0).unsqueeze(0).float()
                        gt_resized = F.interpolate(gt_t, size=(img_h, img_w), mode='nearest')
                        gt_np_viz = gt_resized.squeeze().numpy()
                    else:
                        gt_np_viz = gt_raw
                    frame_objects.append({
                        'label': all_labels[obj_k],
                        'gt_mask': gt_np_viz,
                        'pred_mask': pred_np,
                        'iou': results['iou'][-1] if results['iou'] else 0,
                    })
            if frame_objects:
                viz_data.append({
                    'frame_name': frame_name,
                    'image': img_np,
                    'objects': frame_objects,
                })

    # Aggregate results (same format as single-object eval)
    if not results['iou']:
        return {'error': 'no valid predictions'}

    total_tp = sum(results['tp'])
    total_fp = sum(results['fp'])
    total_fn = sum(results['fn'])
    total_tn = sum(results['tn'])
    total_pixels = total_tp + total_fp + total_fn + total_tn

    scene_result = {
        'scene_id': scene_path.name,
        'miou': np.mean(results['iou']),
        'mean_iou': np.mean(results['iou']),
        'sample_iou': np.mean(results['iou']),
        'oracle_miou': np.mean(results['oracle_iou']),
        'oracle_iou': np.mean(results['oracle_iou']),
        'pixel_acc': np.mean(results['pixel_acc']),
        'recall': np.mean(results['recall']),
        'precision': np.mean(results['precision']),
        'f1': np.mean(results['f1']),
        'num_samples': len(results['iou']),
        'avg_preprocess_ms': np.mean(preprocess_times) * 1000 if preprocess_times else 0,
        'avg_inference_ms': np.mean(inference_times) * 1000 if inference_times else 0,
        'multi_object_eval': True,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_tn': total_tn,
        'global_pixel_acc': (total_tp + total_tn) / max(total_pixels, 1),
    }

    # Per-category metrics
    scene_result['per_category_iou'] = {cat: np.mean(m['iou']) for cat, m in category_metrics.items()}
    scene_result['per_category_oracle_iou'] = {cat: np.mean(m['oracle_iou']) for cat, m in category_metrics.items()}
    scene_result['per_category_pixel_acc'] = {cat: np.mean(m['pixel_acc']) for cat, m in category_metrics.items()}
    scene_result['per_category_recall'] = {cat: np.mean(m['recall']) for cat, m in category_metrics.items()}

    # Spatial eval metrics: count items whose cat_label starts with a spatial qualifier
    spatial_qualifiers = {'nearest', 'farthest', 'leftmost', 'rightmost'}
    spatial_cats = {cat for cat in category_metrics if cat.split()[0].lower() in spatial_qualifiers}
    scene_result['spatial_num_queries'] = sum(len(category_metrics[c]['iou']) for c in spatial_cats)
    if spatial_cats:
        scene_result['spatial_miou'] = np.mean([np.mean(category_metrics[c]['iou']) for c in spatial_cats])
    else:
        scene_result['spatial_miou'] = None

    # Cross-view consistency (mean IoU of same object across frames)
    scene_result['consistency_iou'] = 0.0

    # Save multi-object visualization: all objects overlaid per frame
    if save_viz and viz_data and viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        frame_groups = {v['frame_name']: v for v in viz_data}
        fig = create_multi_object_viz(frame_groups, scene_path.name, max_frames=viz_samples)
        if fig is not None:
            fig.savefig(viz_dir / f"{scene_path.name}_multi_obj.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    return scene_result


@torch.no_grad()
def evaluate_scene(
    model: TrianguLangModel,
    scene_path: Path,
    semantics_dir: Path,
    device: str = 'cuda',
    num_frames: int = 100,
    objects_per_scene: int = 5,
    min_pixel_fraction: float = 0.001,
    image_size: Tuple[int, int] = (1008, 1008),
    save_viz: bool = False,
    viz_dir: Optional[Path] = None,
    viz_samples: int = 10,
    prompt_type: str = 'text_only',
    num_pos_points: int = 10,
    num_neg_points: int = 2,
    sparse_prompts: bool = True,
    num_prompted_frames: int = 3,
    output_localization: bool = False,
    output_depth: bool = False,
    prompt_augmentor: Optional[PromptAugmentor] = None,
    semantic_union: bool = False,
    da3_cache_dir: Optional[Path] = None,
    # Procrustes evaluation parameters
    procrustes: bool = False,
    procrustes_with_scale: bool = True,
    gt_centroids_cache: Optional[Dict] = None,
    data_root: Optional[Path] = None,
    # Category filtering
    allowed_categories: Optional[set] = None,
    # Spatial query filtering: maps prompt -> (qualifier, base_prompt)
    spatial_query_map: Optional[Dict[str, Tuple[Optional[str], str]]] = None,
    # Automatic spatial evaluation for multi-instance labels
    spatial_eval: bool = False,
    # Paper visualization collector
    paper_viz_collector: Optional[List] = None,
    # Frame selection
    frame_names: Optional[List[str]] = None,
    eval_sampling: str = 'stratified',
    # Multi-object eval: batch all objects per frame in one forward pass
    multi_object_eval: bool = False,
    # Temporal EMA smoothing: blend mask logits across consecutive frames
    temporal_smooth_alpha: float = 0.0,
    # CRF post-processing for mask boundary refinement
    use_crf: bool = False,
) -> Dict:
    """Evaluate model on a single scene.

    Args:
        prompt_type: One of 'text_only', 'text_box', 'text_point', 'text_box_point', 'all'
        num_pos_points: Total positive points (distributed across prompted frames in sparse mode)
        num_neg_points: Total negative points (distributed across prompted frames in sparse mode)
        sparse_prompts: If True, distribute points across num_prompted_frames (MV-SAM protocol).
                       If False, give all points to every frame (dense prompting).
        num_prompted_frames: Number of frames to receive prompts in sparse mode.
        output_localization: If True, compute 3D localization for each prediction.
        output_depth: If True, save depth maps for each prediction.
        semantic_union: If True, merge all instances of same label into one GT mask.
                       This matches training behavior when --semantic-union is used.
        da3_cache_dir: Path to DA3 cache directory (da3_cache or da3_nested_cache).
        spatial_query_map: Dict mapping original prompt -> (qualifier, base_prompt) for
                          spatial queries like "leftmost towel" -> ("leftmost", "towel").
                      If provided, loads cached depth instead of running DA3 live.
        paper_viz_collector: If not None, append paper viz data dicts here for grid generation.
    """

    images, objects, available_frames = load_scene_data(scene_path, semantics_dir)
    available_frames_set = set(available_frames)

    if len(images) == 0:
        return {'error': 'No images found'}

    images = [img for img in images if img.name in available_frames_set]

    if len(images) == 0:
        return {'error': 'No images with GT masks'}

    # Filter to specific frames if requested
    if frame_names is not None:
        frame_names_set = set(frame_names)
        # Also try with .JPG extension if not specified
        frame_names_set.update(f + '.JPG' for f in frame_names if not f.endswith('.JPG'))
        frame_names_set.update(f.replace('.JPG', '') for f in frame_names if f.endswith('.JPG'))
        images = [img for img in images if img.name in frame_names_set or img.stem in frame_names_set]
        if len(images) == 0:
            return {'error': f'No matching frames found for: {frame_names}'}
    elif len(images) > num_frames:
        # Apply sampling strategy to match training distribution
        if eval_sampling == 'sequential':
            # First N consecutive frames (matches sequential training)
            images = images[:num_frames]
        elif eval_sampling == 'random':
            # Random sampling
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            indices = rng.choice(len(images), size=num_frames, replace=False)
            indices = sorted(indices)
            images = [images[i] for i in indices]
        elif eval_sampling == 'overlap':
            # Dense consecutive frames from middle of sequence (matches overlap training)
            # Take frames from middle to simulate high-overlap training views
            start_idx = max(0, (len(images) - num_frames) // 2)
            images = images[start_idx:start_idx + num_frames]
        else:  # stratified (default)
            # Uniform spacing across all frames
            indices = np.linspace(0, len(images) - 1, num_frames, dtype=int)
            images = [images[i] for i in indices]

    # Match training skip_labels + structural elements
    skip_labels = {
        # From training (scannetpp_loader.py)
        'remove', 'split', 'object', 'objects', 'stuff', 'unknown',
        # Structural elements (too easy/not interesting)
        'wall', 'floor', 'ceiling', 'door', 'window', 'doorframe', 'windowframe', 'window frame',
        # Annotation artifacts
        'reflection', 'mirror', 'structure',
    }
    valid_objects = [(obj_id, obj) for obj_id, obj in objects.items()
                     if obj['label'] and obj['label'].lower() not in skip_labels]

    # Filter by allowed categories if specified (supports substring matching)
    if allowed_categories is not None:
        def matches_allowed(label):
            label_lower = label.lower()
            for allowed in allowed_categories:
                # Exact match or substring match (e.g., "towel" matches "kitchen towel")
                if allowed == label_lower or allowed in label_lower or label_lower in allowed:
                    return True
            return False
        valid_objects = [(obj_id, obj) for obj_id, obj in valid_objects
                         if matches_allowed(obj['label'])]

    if len(valid_objects) == 0:
        return {'error': 'No valid objects'}

    # Pre-filter to objects visible in the sampled frames (avoids "No valid predictions"
    # when random selection picks objects not present in the evaluation frame subset)
    if len(images) > 0:
        # Check a few frames for visibility (first, middle, last) for efficiency
        check_indices = [0, len(images) // 2, len(images) - 1]
        check_indices = sorted(set(min(i, len(images) - 1) for i in check_indices))
        visible_obj_ids = set()
        for idx in check_indices:
            frame_masks = load_gt_masks(semantics_dir, images[idx].name)
            for obj_id in frame_masks:
                mask = frame_masks[obj_id]
                if mask.sum() / mask.size >= min_pixel_fraction:
                    visible_obj_ids.add(obj_id)
        # Keep only objects visible in at least one checked frame
        visible_objects = [(oid, od) for oid, od in valid_objects if oid in visible_obj_ids]
        if visible_objects:
            valid_objects = visible_objects

    # Pre-filter objects using spatial queries if provided
    # This must happen before eval_items creation to select the right instance
    if spatial_query_map and da3_cache_dir is not None:
        # Group objects by label first
        label_to_objects = defaultdict(list)
        for obj_id, obj_data in valid_objects:
            label_to_objects[obj_data['label'].lower()].append((obj_id, obj_data))

        # Apply spatial filtering for each label that has a qualifier
        filtered_valid_objects = []
        for label_lower, obj_list in label_to_objects.items():
            # Check if this label has a spatial qualifier
            spatial_qualifier = None
            for orig_prompt, (qualifier, base) in spatial_query_map.items():
                if qualifier and (base.lower() == label_lower or
                                  base.lower() in label_lower or
                                  label_lower in base.lower()):
                    spatial_qualifier = qualifier
                    break

            if spatial_qualifier and len(obj_list) > 1:
                # Load depth from reference frame for spatial filtering
                reference_frame = images[0]
                ref_frame_name = reference_frame.name
                ref_gt_masks = load_gt_masks(semantics_dir, ref_frame_name)

                # Collect masks for each candidate
                candidate_masks = []
                candidate_objs = []
                for oid, obj_data in obj_list:
                    if oid in ref_gt_masks and ref_gt_masks[oid].sum() > 0:
                        candidate_masks.append(ref_gt_masks[oid])
                        candidate_objs.append((oid, obj_data))

                if len(candidate_masks) > 1:
                    # Load depth
                    ref_depth = None
                    scene_id_local = scene_path.name
                    frame_stem = reference_frame.stem
                    cache_path = da3_cache_dir / scene_id_local / f"{frame_stem}.pt"
                    if cache_path.exists():
                        try:
                            cache_data = torch.load(cache_path, map_location='cpu', weights_only=True)
                            ref_depth = cache_data['depth'].numpy()
                            if ref_depth.ndim == 4:
                                ref_depth = ref_depth.squeeze()
                            elif ref_depth.ndim == 3:
                                ref_depth = ref_depth.squeeze(0)
                        except Exception:
                            pass

                    if ref_depth is not None:
                        # Resize depth to match mask resolution
                        mask_h, mask_w = candidate_masks[0].shape
                        if ref_depth.shape != (mask_h, mask_w):
                            # Convert to float32 for PIL compatibility
                            ref_depth = ref_depth.astype(np.float32)
                            ref_depth = np.array(Image.fromarray(ref_depth).resize(
                                (mask_w, mask_h), Image.BILINEAR))

                        # Compute spatial mapping
                        spatial_gt = compute_spatial_gt(candidate_masks, ref_depth)

                        if spatial_qualifier in spatial_gt:
                            selected_idx = spatial_gt[spatial_qualifier]
                            selected_obj = candidate_objs[selected_idx]
                            filtered_valid_objects.append(selected_obj)
                            # Print debug info
                            print(f"    🎯 Spatial filter '{spatial_qualifier}' for '{label_lower}': "
                                  f"selected obj {selected_obj[0]} (idx {selected_idx}/{len(candidate_masks)})")
                            continue

                # Fallback: if spatial filtering failed, keep all
                filtered_valid_objects.extend(obj_list)
            else:
                # No spatial qualifier or single object - keep all
                filtered_valid_objects.extend(obj_list)

        valid_objects = filtered_valid_objects

    # Semantic union: group objects by label and evaluate each label once
    # Per-instance: evaluate each object instance separately
    # Seed per-scene so the same objects are selected regardless of which DDP rank
    # processes this scene (rank-dependent seeds cause different object selection,
    # leading to "No valid predictions" when unlucky objects have no visible frames)
    scene_rng = random.Random(int.from_bytes(scene_path.name.encode(), 'big') % (2**31))
    if semantic_union:
        # Group objects by label
        label_to_obj_ids = defaultdict(list)
        for obj_id, obj_data in valid_objects:
            label_to_obj_ids[obj_data['label']].append(obj_id)

        # Create eval items: (label, obj_ids_list)
        eval_items = list(label_to_obj_ids.items())
        scene_rng.shuffle(eval_items)
        eval_items = eval_items[:objects_per_scene]
    else:
        # Original behavior: per-instance evaluation
        scene_rng.shuffle(valid_objects)
        eval_items = [(obj_data['label'], [obj_id]) for obj_id, obj_data in valid_objects[:objects_per_scene]]

    # --spatial-eval: auto-generate spatial queries for multi-instance labels
    # Each spatial query (e.g., "nearest chair") targets a specific instance as GT
    # eval_items entries are extended to (label, obj_ids, spatial_prompt_override)
    # Capped to avoid blowing up eval time (each query = full multi-view inference)
    MAX_SPATIAL_ITEMS_PER_SCENE = 8  # 2 labels × 4 qualifiers, or 4 labels × 2 qualifiers
    spatial_eval_items = []
    if spatial_eval and da3_cache_dir is not None:
        SPATIAL_QUALIFIERS_TO_TEST = ['nearest', 'farthest', 'leftmost', 'rightmost']

        # Group all valid objects by label
        label_to_all_objs = defaultdict(list)
        for obj_id, obj_data in valid_objects:
            label_to_all_objs[obj_data['label']].append(obj_id)

        # Find multi-instance labels
        multi_instance_labels = {lab: oids for lab, oids in label_to_all_objs.items()
                                 if len(oids) >= 2}

        if multi_instance_labels:
            # Load depth from reference frame
            reference_frame = images[0]
            ref_frame_name = reference_frame.name
            ref_gt_masks = load_gt_masks(semantics_dir, ref_frame_name)

            ref_depth = None
            scene_id_local = scene_path.name
            frame_stem = reference_frame.stem
            cache_path = da3_cache_dir / scene_id_local / f"{frame_stem}.pt"
            if cache_path.exists():
                try:
                    cache_data = torch.load(cache_path, map_location='cpu', weights_only=True)
                    ref_depth = cache_data['depth'].numpy()
                    if ref_depth.ndim == 4:
                        ref_depth = ref_depth.squeeze()
                    elif ref_depth.ndim == 3:
                        ref_depth = ref_depth.squeeze(0)
                except Exception:
                    pass

            if ref_depth is not None:
                for label, obj_id_list in multi_instance_labels.items():
                    if len(spatial_eval_items) >= MAX_SPATIAL_ITEMS_PER_SCENE:
                        break

                    # Get masks visible in reference frame
                    candidate_masks = []
                    candidate_oids = []
                    for oid in obj_id_list:
                        if oid in ref_gt_masks and ref_gt_masks[oid].sum() > 0:
                            candidate_masks.append(ref_gt_masks[oid])
                            candidate_oids.append(oid)

                    if len(candidate_masks) < 2:
                        continue

                    # Resize depth to mask resolution if needed
                    mask_h, mask_w = candidate_masks[0].shape
                    depth_for_spatial = ref_depth
                    if ref_depth.shape != (mask_h, mask_w):
                        depth_for_spatial = ref_depth.astype(np.float32)
                        depth_for_spatial = np.array(Image.fromarray(depth_for_spatial).resize(
                            (mask_w, mask_h), Image.BILINEAR))

                    spatial_gt = compute_spatial_gt(candidate_masks, depth_for_spatial)

                    for qualifier in SPATIAL_QUALIFIERS_TO_TEST:
                        if len(spatial_eval_items) >= MAX_SPATIAL_ITEMS_PER_SCENE:
                            break
                        if qualifier in spatial_gt:
                            selected_oid = candidate_oids[spatial_gt[qualifier]]
                            spatial_prompt = f"{qualifier} {label}"
                            spatial_eval_items.append((label, [selected_oid], spatial_prompt))

    # Normalize eval_items to 3-tuples: (label, obj_ids, spatial_prompt_override)
    eval_items = [(label, oids, None) for label, oids in eval_items] + spatial_eval_items

    # Pre-compute SAM3 backbone (ViT) features for all frames in batches.
    # The backbone is text-independent and the most expensive part (~37ms/frame).
    # By batching 4 frames and caching, we avoid re-running the ViT for every
    # object×frame combination. E.g., 5 objects × 100 frames = 500 → 100 backbone calls.
    # Text encoder + SAM3 encoder still run per-object (text-conditioned).
    _precomputed_backbone = {}  # frame_stem -> dict of backbone output tensors (on CPU)
    _precompute_batch_size = 4  # 4 frames at 1008 res fits comfortably on A100 80GB
    _all_frame_stems = [img_path.stem for img_path in images]
    _all_frame_imgs = []

    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size, Image.BILINEAR)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        _all_frame_imgs.append(img_tensor)

    with torch.no_grad():
        for batch_start in range(0, len(_all_frame_imgs), _precompute_batch_size):
            batch_end = min(batch_start + _precompute_batch_size, len(_all_frame_imgs))
            batch_imgs = torch.stack(_all_frame_imgs[batch_start:batch_end]).to(device)

            if batch_imgs.shape[-2:] != (model.resolution, model.resolution):
                batch_imgs = F.interpolate(batch_imgs, size=(model.resolution, model.resolution),
                                           mode='bilinear', align_corners=False)

            with autocast('cuda', dtype=torch.float16):
                bb_out = model.sam3.backbone.forward_image(batch_imgs)

            # Slice per frame — keep on GPU (plenty of VRAM, avoids CPU↔GPU transfer)
            for j in range(batch_end - batch_start):
                frame_stem = _all_frame_stems[batch_start + j]
                frame_data = {
                    'backbone_fpn': [f[j:j+1] for f in bb_out['backbone_fpn']],
                    'vision_features': bb_out['vision_features'][j:j+1],
                    'vision_pos_enc': [p[j:j+1] for p in bb_out['vision_pos_enc']],
                }
                sam2_out = bb_out.get('sam2_backbone_out')
                if sam2_out is not None:
                    frame_data['sam2_backbone_out'] = {
                        'vision_features': sam2_out['vision_features'][j:j+1],
                        'vision_pos_enc': [p[j:j+1] for p in sam2_out['vision_pos_enc']],
                        'backbone_fpn': [f[j:j+1] for f in sam2_out['backbone_fpn']],
                    }
                else:
                    frame_data['sam2_backbone_out'] = None
                _precomputed_backbone[frame_stem] = frame_data

    del _all_frame_imgs  # Free memory

    if multi_object_eval:
        return _evaluate_scene_multi_object(
            model=model, scene_path=scene_path, semantics_dir=semantics_dir,
            device=device, images=images, image_size=image_size,
            eval_items=eval_items, min_pixel_fraction=min_pixel_fraction,
            da3_cache_dir=da3_cache_dir,
            precomputed_backbone=_precomputed_backbone,
            save_viz=save_viz, viz_dir=viz_dir, viz_samples=viz_samples,
            temporal_smooth_alpha=temporal_smooth_alpha,
            use_crf=use_crf,
        )

    results = defaultdict(list)
    category_metrics = defaultdict(lambda: {'iou': [], 'oracle_iou': [], 'pixel_acc': [], 'recall': [], 'precision': [], 'f1': []})
    preprocess_times = []
    inference_times = []
    viz_data = []
    consistency_ious = []  # Per-object cross-view consistency

    # Procrustes evaluation: store raw centroids for alignment
    # These are transformed from camera-frame to DA3 world frame using c2w extrinsics
    raw_centroids_per_object = defaultdict(list)  # obj_id -> list of 3D centroids in DA3 world frame

    # Load GT poses and DA3 poses for Procrustes alignment (if enabled)
    scene_id = scene_path.name
    gt_poses = None
    scene_da3_extrinsics = {}  # Scene-level: frame_name -> c2w matrix (for Procrustes)
    procrustes_alignment = None  # Will store (R, t, s) if computed

    if procrustes and gt_centroids_cache and data_root and scene_id in gt_centroids_cache:
        # Load GT poses
        gt_poses = load_gt_poses_for_scene(data_root, scene_id)
        if gt_poses is None:
            print(f"    [Procrustes] WARNING: GT poses not found for {scene_id}")

        # Load DA3 extrinsics from cache (.pt files)
        # NOTE: Cache stores C2W (camera-to-world) after Feb 2026 fix.
        # preprocess_da3_nested.py inverts DA3's w2c output via extract_c2w_from_extrinsics()
        # before saving, so we can directly use [:3, 3] as camera position.
        if da3_cache_dir is not None:
            scene_cache_dir = da3_cache_dir / scene_id
            if scene_cache_dir.exists():
                pt_files = sorted(scene_cache_dir.glob('*.pt'))
                # Cap Procrustes frame loading — 300 frames is plenty for robust Sim(3) alignment
                # This avoids loading ~2400 .pt files per scene from val_allframes cache
                max_procrustes_frames = 300
                if len(pt_files) > max_procrustes_frames:
                    import random as _rng
                    _rng_state = _rng.getstate()
                    _rng.seed(42)  # Deterministic subset
                    pt_files = sorted(_rng.sample(pt_files, max_procrustes_frames))
                    _rng.setstate(_rng_state)
                for pt_file in pt_files:
                    try:
                        cache_data = torch.load(pt_file, map_location='cpu', weights_only=True)
                        if 'extrinsics' in cache_data:
                            frame_name = pt_file.stem
                            # Cache already stores c2w (camera-to-world) after Feb 2026 fix
                            # See preprocess_da3_nested.py: extract_c2w_from_extrinsics() inverts w2c→c2w
                            # DO NOT invert again - that was causing double-inversion bug!
                            c2w = cache_data['extrinsics'].numpy()
                            scene_da3_extrinsics[frame_name] = c2w
                    except Exception:
                        pass
            else:
                print(f"    [Procrustes] WARNING: scene cache dir not found: {scene_cache_dir}")
        else:
            print(f"    [Procrustes] WARNING: da3_cache_dir is None")

        # Pre-compute Procrustes alignment using ALL cached frames (not just sampled ones)
        # This gives a more robust alignment since DA3 cache only has frames with GT masks
        #
        # ORIENTATION-AWARE ALIGNMENT: We include both camera positions AND
        # orientation "virtual points" (position + offset * forward_direction).
        # Without orientations, Umeyama can find rotations that align positions (~5cm)
        # but flip view directions 178°, causing 3D points to be meters off.
        if gt_poses and scene_da3_extrinsics:
            gt_points = []
            da3_points = []
            ORIENT_OFFSET = 0.5  # meters ahead along view direction
            # Use ALL frames that are in BOTH gt_poses and scene_da3_extrinsics
            for frame_name in scene_da3_extrinsics.keys():
                if frame_name in gt_poses:
                    gt_c2w = gt_poses[frame_name]
                    da3_c2w = scene_da3_extrinsics[frame_name]
                    # Camera position
                    gt_pos = gt_c2w[:3, 3]
                    da3_pos = da3_c2w[:3, 3]
                    gt_points.append(gt_pos)
                    da3_points.append(da3_pos)
                    # Orientation virtual point: position + offset * forward direction
                    # Forward direction is -Z in camera frame = -column 2 of rotation
                    # But for c2w, column 2 is the camera Z axis in world coords
                    # Camera looks down -Z, so forward = -c2w[:3, 2]
                    gt_fwd = -gt_c2w[:3, 2]
                    da3_fwd = -da3_c2w[:3, 2]
                    gt_points.append(gt_pos + ORIENT_OFFSET * gt_fwd)
                    da3_points.append(da3_pos + ORIENT_OFFSET * da3_fwd)

            n_frames = len(gt_points) // 2  # Each frame contributes 2 points
            # Debug: log matching stats
            image_stems = {img.stem for img in images}
            gt_pose_frames = set(gt_poses.keys())
            da3_frames = set(scene_da3_extrinsics.keys())
            in_gt = len(image_stems & gt_pose_frames)
            in_da3 = len(image_stems & da3_frames)
            in_both = len(image_stems & gt_pose_frames & da3_frames)
            print(f"    [Procrustes] eval_images={len(images)}, in_gt_poses={in_gt}, in_da3_cache={in_da3}, "
                  f"in_both={in_both}, frames_for_alignment={n_frames} (pts={len(gt_points)})")

            if n_frames >= 3:
                try:
                    R, t, s = umeyama_alignment(
                        np.array(da3_points),
                        np.array(gt_points),
                        with_scale=procrustes_with_scale
                    )
                    procrustes_alignment = (R, t, s)
                    # Debug: check orientation preservation
                    # Pick first frame and check if forward directions align after transform
                    if n_frames >= 1:
                        da3_pos0 = da3_points[0]
                        da3_fwd_pt0 = da3_points[1]
                        da3_fwd0 = da3_fwd_pt0 - da3_pos0
                        transformed_fwd0 = s * R @ da3_fwd0  # Scale+rotate forward vector
                        gt_fwd0 = gt_points[1] - gt_points[0]
                        cos_sim = np.dot(transformed_fwd0, gt_fwd0) / (np.linalg.norm(transformed_fwd0) * np.linalg.norm(gt_fwd0) + 1e-8)
                        print(f"    [Procrustes] Orientation check: cos_sim(DA3_fwd_aligned, GT_fwd)={cos_sim:.4f} "
                              f"(should be ~+1.0, was ~-1.0 before fix)")
                except Exception as e:
                    print(f"    [Procrustes] Alignment failed: {e}")
                    pass

    # MV-SAM sparse prompting: select which frames get point prompts
    # For text_point mode with sparse_prompts=True:
    #   - Only num_prompted_frames frames get point prompts
    #   - Total points (num_pos_points + num_neg_points) are distributed across those frames
    #   - Other frames get text_only prompts
    use_sparse = sparse_prompts and prompt_type in ['text_point', 'text_box_point', 'all']
    if use_sparse:
        # Select evenly spaced frames to receive prompts
        n_prompted = min(num_prompted_frames, len(images))
        prompted_indices = set(np.linspace(0, len(images) - 1, n_prompted, dtype=int).tolist())

        # Distribute points across prompted frames
        total_pos = num_pos_points
        total_neg = num_neg_points
        # Points per frame (distribute evenly, with remainder to first frames)
        pos_per_frame = [total_pos // n_prompted] * n_prompted
        neg_per_frame = [total_neg // n_prompted] * n_prompted
        for i in range(total_pos % n_prompted):
            pos_per_frame[i] += 1
        for i in range(total_neg % n_prompted):
            neg_per_frame[i] += 1
        # Map frame index to point counts
        prompted_frame_list = sorted(prompted_indices)
        points_per_prompted_frame = {
            idx: (pos_per_frame[i], neg_per_frame[i])
            for i, idx in enumerate(prompted_frame_list)
        }
    else:
        prompted_indices = set()

    for label, obj_ids, item_spatial_override in tqdm(eval_items, desc=f"Objects in {scene_path.name}", leave=False):
        eval_label = item_spatial_override if item_spatial_override else label  # Use spatial prompt as metric key

        # Check if this label has a spatial qualifier for the model prompt
        # item_spatial_override comes from --spatial-eval auto-generation
        # spatial_query_map comes from --custom-prompts
        spatial_prompt_override = item_spatial_override
        if not spatial_prompt_override and spatial_query_map:
            for orig_prompt, (qualifier, base) in spatial_query_map.items():
                if qualifier and (base.lower() == label.lower() or
                                  base.lower() in label.lower() or
                                  label.lower() in base.lower()):
                    # Construct spatial prompt: "leftmost kitchen towel"
                    spatial_prompt_override = f"{qualifier} {label}"
                    break

        # Apply synonym augmentation if enabled
        if prompt_augmentor is not None:
            label = prompt_augmentor.augment_language(label)

        # Collect per-object predictions and pointmaps for cross-view consistency
        obj_pred_masks = []
        obj_pointmaps = []

        for frame_idx, img_path in enumerate(images):
            frame_name = img_path.name
            gt_masks_dict = load_gt_masks(semantics_dir, frame_name)

            # Union all instance masks for this label (semantic_union mode)
            # or use single instance mask (per-instance mode)
            gt_mask = None
            for oid in obj_ids:
                if oid in gt_masks_dict:
                    if gt_mask is None:
                        gt_mask = gt_masks_dict[oid].copy()
                    else:
                        gt_mask = np.maximum(gt_mask, gt_masks_dict[oid])

            if gt_mask is None:
                continue

            pixel_fraction = gt_mask.sum() / gt_mask.size
            if pixel_fraction < min_pixel_fraction:
                continue

            t_preprocess_start = time.perf_counter()

            img = Image.open(img_path).convert('RGB')
            img = img.resize(image_size, Image.BILINEAR)
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)
            gt_tensor = F.interpolate(gt_tensor, size=image_size, mode='nearest').squeeze()
            gt_tensor = gt_tensor.to(device)

            t_preprocess_end = time.perf_counter()
            preprocess_times.append(t_preprocess_end - t_preprocess_start)

            try:
                t_inference_start = time.perf_counter()

                # Create prompts based on prompt_type and sparse prompting mode
                if use_sparse:
                    if frame_idx in prompted_indices:
                        # This frame gets point prompts
                        frame_pos, frame_neg = points_per_prompted_frame[frame_idx]
                        prompts = create_prompts_from_gt(
                            gt_tensor, prompt_type, frame_pos, frame_neg, device
                        )
                    else:
                        # This frame gets text_only (no point prompts)
                        prompts = create_prompts_from_gt(
                            gt_tensor, 'text_only', 0, 0, device
                        )
                else:
                    # Dense prompting: every frame gets all points
                    prompts = create_prompts_from_gt(
                        gt_tensor, prompt_type, num_pos_points, num_neg_points, device
                    )

                # Load cached depth if available
                cached_depth = None
                da3_intrinsics = None
                da3_extrinsics = None
                depth_source = 'live'  # Track depth source for debugging
                if da3_cache_dir is not None:
                    scene_id = scene_path.name
                    frame_stem = img_path.stem
                    cache_path = da3_cache_dir / scene_id / f"{frame_stem}.pt"
                    if cache_path.exists():
                        try:
                            cache_data = torch.load(cache_path, map_location='cpu', weights_only=True)
                            depth = cache_data['depth'].float()
                            # Handle both formats: [1, 1, H, W] (old) or [H, W] (new)
                            if depth.dim() == 4:
                                depth = depth.squeeze(0)  # [1, 1, H, W] -> [1, H, W]
                            elif depth.dim() == 2:
                                depth = depth.unsqueeze(0)  # [H, W] -> [1, H, W]
                            cached_depth = depth.unsqueeze(0).to(device)  # [1, 1, H, W]
                            depth_source = f'cache:{frame_stem}'
                            # Load intrinsics if available (da3_nested_cache format)
                            if 'intrinsics' in cache_data:
                                da3_intrinsics = cache_data['intrinsics'].float().unsqueeze(0).to(device)  # [1, 3, 3]
                            # Load extrinsics if available (da3_nested_cache format) - needed for CV2 models
                            if 'extrinsics' in cache_data:
                                da3_extrinsics = cache_data['extrinsics'].float().unsqueeze(0).to(device)  # [1, 4, 4]
                        except Exception as e:
                            depth_source = f'live (cache error: {e})'
                    else:
                        depth_source = f'live (cache not found: {cache_path})'
                else:
                    depth_source = 'live (no --da3-nested-cache)'

                # Use empty text for point/box-only modes (MV-SAM comparison)
                # Use spatial prompt override if available (e.g., "leftmost kitchen towel")
                model_prompt = spatial_prompt_override if spatial_prompt_override else label
                text_input = [model_prompt] if prompts.get('use_text', True) else ['']

                # Parse spatial qualifier for spatial token conditioning
                sq_type, _ = parse_spatial_qualifier(model_prompt)
                sq_idx = get_spatial_qualifier_idx(sq_type)
                sq_tensor = torch.tensor([sq_idx], device=device, dtype=torch.long) if sq_idx > 0 else None

                # Look up precomputed backbone features for this frame
                _frame_backbone = _precomputed_backbone.get(img_path.stem)

                with autocast('cuda', dtype=torch.float16):
                    outputs = model(
                        img_tensor, text_input, gt_tensor.unsqueeze(0),
                        box_prompts=prompts['box_prompts'],
                        box_labels=prompts['box_labels'],
                        point_prompts=prompts['point_prompts'],
                        point_labels=prompts['point_labels'],
                        cached_depth=cached_depth,
                        da3_intrinsics=da3_intrinsics,
                        da3_extrinsics=da3_extrinsics,
                        precomputed_sam3=_frame_backbone,
                        spatial_qualifier_idx=sq_tensor,
                    )
                pred = outputs['pred_masks'][:, 0]

                t_inference_end = time.perf_counter()
                inference_times.append(t_inference_end - t_inference_start)

                if pred.shape[-2:] != gt_tensor.shape[-2:]:
                    pred = F.interpolate(pred.unsqueeze(1), size=gt_tensor.shape[-2:],
                                        mode='bilinear', align_corners=False).squeeze(1)

                metrics = compute_metrics(pred, gt_tensor.unsqueeze(0))

                # Compute oracle IoU (best possible mask selection)
                if 'all_masks' in outputs:
                    oracle_result = compute_oracle_iou(outputs['all_masks'], gt_tensor)
                    oracle_iou = oracle_result['oracle_iou']
                else:
                    oracle_iou = metrics['iou']  # Fallback to selected mask IoU

                results['iou'].append(metrics['iou'])
                results['oracle_iou'].append(oracle_iou)
                results['pixel_acc'].append(metrics['pixel_acc'])
                results['recall'].append(metrics['recall'])
                results['precision'].append(metrics['precision'])
                results['f1'].append(metrics['f1'])
                # Track raw counts for global pixel accuracy computation
                results['tp'].append(metrics['tp'])
                results['fp'].append(metrics['fp'])
                results['fn'].append(metrics['fn'])
                results['tn'].append(metrics['tn'])

                category_metrics[eval_label]['iou'].append(metrics['iou'])
                category_metrics[eval_label]['oracle_iou'].append(oracle_iou)
                category_metrics[eval_label]['pixel_acc'].append(metrics['pixel_acc'])
                category_metrics[eval_label]['recall'].append(metrics['recall'])
                category_metrics[eval_label]['precision'].append(metrics['precision'])
                category_metrics[eval_label]['f1'].append(metrics['f1'])

                # Compute 3D centroid error for Acc@m metrics
                # Use pointmaps_full like training does (more robust than depth+intrinsics)
                if 'pointmaps_full' in outputs:
                    try:
                        pointmaps_full = outputs['pointmaps_full']  # [B, H, W, 3]
                        # Log first successful centroid computation
                        if len(results['centroid_errors']) == 0:
                            print(f"    📍 Centroid method: pointmaps_full [{pointmaps_full.shape}]")
                        # Resize pred mask to match pointmaps resolution if needed
                        pm_H, pm_W = pointmaps_full.shape[1:3]
                        if pred.shape[-2:] != (pm_H, pm_W):
                            pred_resized = F.interpolate(pred.unsqueeze(1), size=(pm_H, pm_W),
                                                        mode='bilinear', align_corners=False).squeeze(1)
                        else:
                            pred_resized = pred
                        if gt_tensor.shape[-2:] != (pm_H, pm_W):
                            gt_resized = F.interpolate(gt_tensor.unsqueeze(0).unsqueeze(0).float(),
                                                      size=(pm_H, pm_W),
                                                      mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                        else:
                            gt_resized = gt_tensor.float()
                        # Compute centroids using pointmaps (same as training)
                        pred_cent = compute_gt_centroid(pred_resized[0], pointmaps_full[0])
                        gt_cent = compute_gt_centroid(gt_resized, pointmaps_full[0])
                        centroid_error = torch.norm(pred_cent - gt_cent).item()
                        results['centroid_errors'].append(centroid_error)

                        # Store raw centroid for Procrustes evaluation
                        # IMPORTANT: pointmaps_full may be NORMALIZED (centered+scaled).
                        # Must denormalize back to meters before transforming to world frame.
                        if procrustes and pred_cent is not None:
                            frame_name = img_path.stem
                            if frame_name in scene_da3_extrinsics:
                                # Step 1: Denormalize centroid from unitless back to meters
                                norm_params = outputs.get('norm_params', None)
                                if norm_params is not None and 'scale' in norm_params and 'centroid' in norm_params:
                                    cent_offset = norm_params['centroid']
                                    if cent_offset.dim() > 1:
                                        cent_offset = cent_offset[0]  # First batch item
                                    denorm_cent = (pred_cent * norm_params['scale'] + cent_offset).cpu().numpy()
                                    # Debug: log first Procrustes centroid computation per scene
                                    if len(raw_centroids_per_object) == 0:
                                        print(f"    [Procrustes DEBUG] norm_params: scale={norm_params['scale'].item():.4f}, "
                                              f"centroid={cent_offset.cpu().numpy()}")
                                        print(f"    [Procrustes DEBUG] pred_cent (normalized): {pred_cent.cpu().numpy()}")
                                        print(f"    [Procrustes DEBUG] denorm_cent (meters): {denorm_cent}")
                                        print(f"    [Procrustes DEBUG] pointmaps_in_world_frame: {outputs.get('pointmaps_in_world_frame', False)}")
                                else:
                                    denorm_cent = pred_cent.cpu().numpy()  # Already in meters (no normalization)
                                    if len(raw_centroids_per_object) == 0:
                                        print(f"    [Procrustes DEBUG] norm_params is None — using raw pred_cent: {denorm_cent}")

                                # Step 2: Transform to DA3 world frame if needed
                                # When use_da3_poses_for_gasa=True, pointmaps are already in DA3 world frame
                                # When False (identity pose), pointmaps are in camera frame → need c2w
                                if outputs.get('pointmaps_in_world_frame', False):
                                    world_cent = denorm_cent  # Already in DA3 world frame
                                else:
                                    c2w = scene_da3_extrinsics[frame_name]  # [4, 4] numpy array
                                    world_cent = c2w[:3, :3] @ denorm_cent + c2w[:3, 3]
                                for obj_id in obj_ids:
                                    raw_centroids_per_object[obj_id].append(world_cent)
                            else:
                                # No extrinsic available - skip this frame for Procrustes
                                pass
                    except Exception as e:
                        pass  # Silently skip if centroid computation fails
                # Fallback to depth+intrinsics method if pointmaps not available
                elif 'depth' in outputs and 'intrinsics' in outputs:
                    intrinsics = outputs['intrinsics']
                    # Validate intrinsics shape - must be [B, 3, 3]
                    if intrinsics.dim() == 3 and intrinsics.shape[-2:] == (3, 3):
                        try:
                            # Compute predicted and GT 3D centroids using compute_3d_localization
                            pred_loc = compute_3d_localization(
                                pred_masks=pred,
                                depth=outputs['depth'],
                                intrinsics=intrinsics,
                                threshold=0.5,
                            )
                            gt_loc = compute_3d_localization(
                                pred_masks=gt_tensor.unsqueeze(0).float(),
                                depth=outputs['depth'],
                                intrinsics=intrinsics,
                                threshold=0.5,
                            )
                            if pred_loc['centroid_3d'] is not None and gt_loc['centroid_3d'] is not None:
                                centroid_error = torch.norm(pred_loc['centroid_3d'] - gt_loc['centroid_3d']).item()
                                results['centroid_errors'].append(centroid_error)
                        except Exception as e:
                            pass  # Silently skip if centroid computation fails

                # Compute 3D localization if requested
                localization_data = None
                if output_localization:
                    if 'depth' in outputs and 'intrinsics' in outputs:
                        try:
                            loc_result = compute_3d_localization(
                                pred_masks=pred,
                                depth=outputs['depth'],
                                intrinsics=outputs['intrinsics'],
                                threshold=0.5,
                            )
                            loc_text = format_localization_text(loc_result['centroid_3d'])[0]
                            localization_data = {
                                'centroid_3d': loc_result['centroid_3d'][0].cpu().tolist(),
                                'centroid_2d': loc_result['centroid_2d'][0].cpu().tolist(),
                                'mean_depth': loc_result['mean_depth'][0].item(),
                                'description': loc_text,
                            }
                            results.setdefault('localizations', []).append({
                                'label': label,
                                'frame': img_path.name,
                                **localization_data
                            })
                        except Exception as e:
                            print(f"    Localization error: {e}")

                if save_viz and len(viz_data) < viz_samples:
                    pred_np = (torch.sigmoid(pred[0]).cpu().numpy() > 0.5).astype(np.float32)
                    gt_np = gt_tensor.cpu().numpy()
                    viz_data.append({
                        'image': img_np,
                        'gt_mask': gt_np,
                        'pred_mask': pred_np,
                        'label': label,
                        'iou': metrics['iou'],
                        'localization': localization_data,
                    })

                # Collect paper viz data (independent of --visualize flag)
                if paper_viz_collector is not None:
                    pv_pred_np = (torch.sigmoid(pred[0]).cpu().numpy() > 0.5).astype(np.float32)
                    pv_gt_np = gt_tensor.cpu().numpy()
                    pv_depth = None
                    if 'depth' in outputs:
                        pv_depth = outputs['depth'][0].cpu().float().numpy()  # [H, W]
                    paper_viz_collector.append({
                        'image': img_np.copy(),
                        'gt_mask': pv_gt_np,
                        'pred_mask': pv_pred_np,
                        'depth': pv_depth,
                        'label': label,
                        'scene_id': scene_path.name,
                        'frame_name': img_path.stem,  # For debugging depth alignment
                        'depth_source': depth_source,  # 'cache:xxx' or 'live (reason)'
                        'iou': metrics['iou'],
                    })

                # Collect pred mask + pointmaps for cross-view consistency
                if 'pointmaps_full' in outputs:
                    obj_pred_masks.append(pred[0].detach())  # [H, W]
                    obj_pointmaps.append(outputs['pointmaps_full'][0].detach())  # [H, W, 3]

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue

        # Compute per-object cross-view consistency after all frames
        if len(obj_pred_masks) >= 2:
            try:
                pointmaps_stack = torch.stack(obj_pointmaps)  # [V, H, W, 3]
                consistency_result = compute_cross_view_consistency(
                    obj_pred_masks, pointmaps_stack, threshold=0.05, subsample=1024
                )
                if consistency_result['num_correspondences'] > 0:
                    consistency_ious.append(consistency_result['consistency'])
            except Exception:
                pass

    if not results['iou']:
        return {'error': 'No valid predictions'}

    # Procrustes-aligned 3D Localization (localization metric)
    procrustes_errors = []
    procrustes_scale = None

    if procrustes and procrustes_alignment is not None and gt_centroids_cache and scene_id in gt_centroids_cache:
        R, t, s = procrustes_alignment
        procrustes_scale = s

        # Compute Procrustes-aligned centroid errors for each object
        for label, obj_ids, *_ in eval_items:
            for obj_id in obj_ids:
                obj_id_str = str(obj_id)
                if obj_id_str in gt_centroids_cache[scene_id] and obj_id in raw_centroids_per_object:
                    gt_cent_mesh = np.array(gt_centroids_cache[scene_id][obj_id_str])

                    # Average predicted centroids across frames (in DA3 frame)
                    raw_cents = raw_centroids_per_object[obj_id]
                    if raw_cents:
                        pred_cent_da3 = np.mean(raw_cents, axis=0)

                        # Apply Procrustes alignment: DA3 → GT mesh frame
                        pred_cent_aligned = s * R @ pred_cent_da3 + t

                        # Compute error against GT mesh centroid
                        error = np.linalg.norm(pred_cent_aligned - gt_cent_mesh)
                        procrustes_errors.append(error)

    # Compute Procrustes Acc@m if we have errors
    procrustes_acc_5cm = None
    procrustes_acc_10cm = None
    procrustes_mean_error = None
    if procrustes_errors:
        procrustes_acc_5cm = sum(1 for e in procrustes_errors if e < 0.05) / len(procrustes_errors)
        procrustes_acc_10cm = sum(1 for e in procrustes_errors if e < 0.10) / len(procrustes_errors)
        procrustes_mean_error = np.mean(procrustes_errors)

    # Save visualization immediately after each scene
    if save_viz and viz_data and viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        fig = create_comparison_grid(
            [v['image'] for v in viz_data],
            [v['gt_mask'] for v in viz_data],
            [v['pred_mask'] for v in viz_data],
            [v['label'] for v in viz_data],
            [v['iou'] for v in viz_data],
            scene_path.name,
        )
        viz_path = viz_dir / f'{scene_path.name}_comparison.png'
        fig.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved viz: {viz_path.name}")

    # Per-category metrics
    per_cat_iou = {cat: np.mean(m['iou']) for cat, m in category_metrics.items() if m['iou']}
    per_cat_oracle_iou = {cat: np.mean(m['oracle_iou']) for cat, m in category_metrics.items() if m.get('oracle_iou')}
    per_cat_pixel_acc = {cat: np.mean(m['pixel_acc']) for cat, m in category_metrics.items() if m['pixel_acc']}
    per_cat_recall = {cat: np.mean(m['recall']) for cat, m in category_metrics.items() if m['recall']}
    miou = np.mean(list(per_cat_iou.values())) if per_cat_iou else 0.0
    oracle_miou = np.mean(list(per_cat_oracle_iou.values())) if per_cat_oracle_iou else 0.0
    mrecall = np.mean(list(per_cat_recall.values())) if per_cat_recall else 0.0

    # Separate spatial eval metrics (categories with spatial qualifiers like "nearest chair")
    spatial_qualifiers_set = {'nearest', 'farthest', 'leftmost', 'rightmost', 'topmost', 'bottommost',
                              'closest', 'left', 'right', 'top', 'bottom'}
    spatial_cat_iou = {cat: v for cat, v in per_cat_iou.items()
                       if cat.split()[0].lower() in spatial_qualifiers_set}
    spatial_miou = np.mean(list(spatial_cat_iou.values())) if spatial_cat_iou else None

    # Compute global pixel accuracy from raw counts (more accurate than averaging)
    total_tp = sum(results['tp'])
    total_fp = sum(results['fp'])
    total_fn = sum(results['fn'])
    total_tn = sum(results['tn'])
    total_pixels = total_tp + total_fp + total_fn + total_tn
    global_pixel_acc = (total_tp + total_tn) / total_pixels if total_pixels > 0 else 0.0

    avg_preprocess_ms = np.mean(preprocess_times) * 1000 if preprocess_times else 0
    avg_inference_ms = np.mean(inference_times) * 1000 if inference_times else 0

    # Compute Acc@m metrics for 3D localization
    centroid_errors = results.get('centroid_errors', [])
    acc_5cm = sum(1 for e in centroid_errors if e < 0.05) / max(len(centroid_errors), 1)
    acc_10cm = sum(1 for e in centroid_errors if e < 0.10) / max(len(centroid_errors), 1)
    acc_50cm = sum(1 for e in centroid_errors if e < 0.50) / max(len(centroid_errors), 1)
    mean_centroid_error = np.mean(centroid_errors) if centroid_errors else float('inf')

    return {
        'scene_id': scene_path.name,
        'sample_iou': np.mean(results['iou']),
        'oracle_iou': np.mean(results.get('oracle_iou', results['iou'])),
        'miou': miou,
        'oracle_miou': oracle_miou,
        'pixel_acc': np.mean(results['pixel_acc']),  # Sample-averaged pixel accuracy
        'global_pixel_acc': global_pixel_acc,  # Global pixel accuracy from raw counts
        'recall': np.mean(results['recall']),  # Mean class recall
        'mrecall': mrecall,
        'precision': np.mean(results['precision']),
        'f1': np.mean(results['f1']),
        'num_samples': len(results['iou']),
        'num_categories': len(per_cat_iou),
        'per_category_iou': per_cat_iou,
        'per_category_oracle_iou': per_cat_oracle_iou,
        'per_category_pixel_acc': per_cat_pixel_acc,
        'per_category_recall': per_cat_recall,
        'avg_preprocess_ms': avg_preprocess_ms,
        'avg_inference_ms': avg_inference_ms,
        # Raw counts for global aggregation
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_tn': total_tn,
        # 3D localization metrics (IoU-based, same pointmap for pred & GT)
        'acc_5cm': acc_5cm,
        'acc_10cm': acc_10cm,
        'acc_50cm': acc_50cm,
        'mean_centroid_error_m': mean_centroid_error,
        'num_centroid_samples': len(centroid_errors),
        # Spatial eval metrics (auto-generated spatial queries)
        'spatial_miou': spatial_miou,
        'spatial_num_queries': len(spatial_cat_iou),
        'spatial_per_category_iou': spatial_cat_iou,
        # Procrustes-aligned localization 
        'procrustes_acc_5cm': procrustes_acc_5cm,
        'procrustes_acc_10cm': procrustes_acc_10cm,
        'procrustes_mean_error_m': procrustes_mean_error,
        'procrustes_scale': procrustes_scale,
        'procrustes_num_samples': len(procrustes_errors) if procrustes_errors else 0,
        # Cross-view consistency (do corresponding 3D points get same prediction?)
        'consistency_iou': np.mean(consistency_ious) if consistency_ious else None,
        'num_consistency_objects': len(consistency_ious),
    }


# Generic Dataset Evaluation (for uCO3D and other dataset-based benchmarks)

def evaluate_with_dataset(
    model: TrianguLangModel,
    dataset: Dataset,
    device: str,
    ddp: DDPManager,
    args,
    output_dir: Path,
    viz_dir: Optional[Path] = None,
    paper_viz_collector: Optional[List] = None,
) -> Dict:
    """
    Evaluate model on a dataset object (used for uCO3D and other benchmarks).

    Args:
        model: TrianguLangModel instance
        dataset: Dataset returning samples with 'images', 'gt_masks', 'prompt', etc.
        device: Device for computation
        ddp: DDP manager for distributed evaluation
        args: Command line arguments
        paper_viz_collector: If not None, append paper viz data dicts here.
        output_dir: Directory for results
        viz_dir: Directory for visualizations (None to skip)

    Returns:
        Dictionary with evaluation metrics
    """
    from torch.utils.data import DataLoader

    model.eval()

    # Create dataloader
    # For distributed: each rank gets a subset
    if ddp.is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Full metrics tracking (matching ScanNet++ evaluation)
    all_ious = []
    all_oracle_ious = []
    all_categories = []
    category_ious = defaultdict(list)
    category_oracle_ious = defaultdict(list)
    category_recalls = defaultdict(list)
    category_precisions = defaultdict(list)
    category_f1s = defaultdict(list)
    all_recalls = []
    all_precisions = []
    all_f1s = []
    centroid_errors = []
    centroid_errors_world = []

    pbar = tqdm(dataloader, desc="Evaluating", disable=not ddp.is_main)

    for batch_idx, batch in enumerate(pbar):
        try:
            images = batch['images'].to(device)  # [1, N, 3, H, W]
            gt_masks = batch['gt_masks'].to(device)  # [1, N, H, W]
            intrinsics = batch.get('intrinsics', None)
            extrinsics = batch.get('extrinsics', None)
            if intrinsics is not None:
                intrinsics = intrinsics.to(device)
            if extrinsics is not None:
                extrinsics = extrinsics.to(device)
            prompt = batch['prompt'][0] if isinstance(batch['prompt'], list) else batch['prompt']
            category = batch.get('category', [prompt])[0] if 'category' in batch else prompt
            scene_id = batch.get('scene_id', ['unknown'])[0]

            B, N, C, H, W = images.shape
            images = images.squeeze(0)  # [N, 3, H, W]
            gt_masks = gt_masks.squeeze(0)  # [N, H, W]
            if intrinsics is not None:
                intrinsics = intrinsics.squeeze(0)  # [N, 3, 3]
            if extrinsics is not None:
                extrinsics = extrinsics.squeeze(0)  # [N, 4, 4]

            # Generate point/box prompts from GT masks if needed
            prompt_type = getattr(args, 'prompt_type', 'text_only')
            num_pos = getattr(args, 'num_pos_points', 10)
            num_neg = getattr(args, 'num_neg_points', 2)

            # For multi-view, we need prompts per view
            all_point_prompts = []
            all_point_labels = []
            all_box_prompts = []
            all_box_labels = []

            # Determine if text should be used based on prompt type
            no_text_types = ('point_only', 'box_only', 'box_point_only')
            use_text = prompt_type not in no_text_types
            text_for_model = prompt if use_text else ''

            if prompt_type != 'text_only':
                for view_idx in range(N):
                    view_gt = gt_masks[view_idx]  # [H, W]
                    prompts = create_prompts_from_gt(
                        view_gt, prompt_type, num_pos, num_neg, device
                    )
                    if prompts['point_prompts'] is not None:
                        all_point_prompts.append(prompts['point_prompts'].squeeze(0))  # [N_pts, 2]
                        all_point_labels.append(prompts['point_labels'].squeeze(0))  # [N_pts]
                    if prompts['box_prompts'] is not None:
                        all_box_prompts.append(prompts['box_prompts'].squeeze(0))  # [1, 4]
                        all_box_labels.append(prompts['box_labels'].squeeze(0))  # [1]

                # Stack into batched tensors
                if all_point_prompts:
                    all_point_prompts = torch.stack(all_point_prompts, dim=0)  # [N, N_pts, 2]
                    all_point_labels = torch.stack(all_point_labels, dim=0)  # [N, N_pts]
                else:
                    all_point_prompts = None
                    all_point_labels = None

                if all_box_prompts:
                    all_box_prompts = torch.stack(all_box_prompts, dim=0)  # [N, 1, 4]
                    all_box_labels = torch.stack(all_box_labels, dim=0)  # [N, 1]
                else:
                    all_box_prompts = None
                    all_box_labels = None
            else:
                all_point_prompts = None
                all_point_labels = None
                all_box_prompts = None
                all_box_labels = None

            # Run inference
            all_frame_masks = []
            all_frame_all_masks = []  # For oracle IoU: all candidate masks per frame
            first_frame_outputs = None  # Store first frame outputs for paper_viz

            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    if args.per_frame:
                        # Per-frame processing using model.forward()
                        # Option 1: --batch-da3 - batch DA3 for cross-view depth, then per-frame segmentation
                        # Option 2: default - run DA3 independently per frame (no cross-view depth consistency)

                        # Pre-compute batched DA3 depth if requested
                        batched_depths = None
                        batched_da3_intrinsics = None
                        if getattr(args, 'batch_da3', False):
                            # Batch all views for DA3 (in chunks to avoid OOM)
                            da3_chunk_size = args.view_chunk_size if args.view_chunk_size > 0 else 16
                            all_depths = []
                            all_da3_intrinsics = []

                            for chunk_start in range(0, N, da3_chunk_size):
                                chunk_end = min(chunk_start + da3_chunk_size, N)
                                chunk_images = images[chunk_start:chunk_end]  # [chunk, 3, H, W]

                                # Run DA3 on chunk - model expects [B, N, C, H, W] for multi-view
                                # Reshape to [1, chunk, C, H, W] so DA3 sees all views together
                                chunk_batch = chunk_images.unsqueeze(0)  # [1, chunk, 3, H, W]

                                # Get depth using DA3's batch processing
                                # Note: This calls DA3 with proper multi-view batching
                                da3_res = model.da3_resolution
                                da3_H = da3_W = (da3_res // 14) * 14
                                chunk_resized = F.interpolate(chunk_images, size=(da3_H, da3_W), mode='bilinear', align_corners=False)

                                # DA3 forward with multi-view batch (proper cross-view attention)
                                da3_output = model.da3.model.forward(
                                    chunk_resized.unsqueeze(0),  # [1, chunk, 3, H, W]
                                    extrinsics=None, intrinsics=None,
                                    export_feat_layers=[], infer_gs=False
                                )

                                chunk_depth = da3_output.depth  # [1, chunk, H, W] or [1, chunk, 1, H, W]
                                if chunk_depth.dim() == 5:
                                    chunk_depth = chunk_depth.squeeze(2)  # [1, chunk, H, W]
                                chunk_depth = chunk_depth.squeeze(0)  # [chunk, H, W]

                                # Resize depth to model resolution
                                if chunk_depth.shape[-2:] != (model.resolution, model.resolution):
                                    chunk_depth = F.interpolate(
                                        chunk_depth.unsqueeze(1),
                                        size=(model.resolution, model.resolution),
                                        mode='bilinear', align_corners=False
                                    ).squeeze(1)

                                all_depths.append(chunk_depth)

                                # Get intrinsics from DA3 if available
                                if hasattr(da3_output, 'intrinsics') and da3_output.intrinsics is not None:
                                    chunk_intr = da3_output.intrinsics
                                    if chunk_intr.dim() == 4:  # [1, chunk, 3, 3]
                                        chunk_intr = chunk_intr.squeeze(0)  # [chunk, 3, 3]
                                    elif chunk_intr.dim() == 2:  # [3, 3]
                                        chunk_intr = chunk_intr.unsqueeze(0).expand(chunk_end - chunk_start, -1, -1)
                                    all_da3_intrinsics.append(chunk_intr)

                            batched_depths = torch.cat(all_depths, dim=0)  # [N, H, W]
                            if all_da3_intrinsics:
                                batched_da3_intrinsics = torch.cat(all_da3_intrinsics, dim=0)  # [N, 3, 3]

                        # Now run per-frame segmentation
                        for frame_idx in range(N):
                            frame_img = images[frame_idx:frame_idx+1]  # [1, 3, H, W]
                            frame_intrinsics = intrinsics[frame_idx:frame_idx+1] if intrinsics is not None else None
                            frame_extrinsics = extrinsics[frame_idx:frame_idx+1] if extrinsics is not None else None

                            # Get prompts for this frame
                            frame_point_prompts = all_point_prompts[frame_idx:frame_idx+1] if all_point_prompts is not None else None
                            frame_point_labels = all_point_labels[frame_idx:frame_idx+1] if all_point_labels is not None else None
                            frame_box_prompts = all_box_prompts[frame_idx:frame_idx+1] if all_box_prompts is not None else None
                            frame_box_labels = all_box_labels[frame_idx:frame_idx+1] if all_box_labels is not None else None

                            # Get pre-computed depth if using batched DA3
                            frame_cached_depth = None
                            frame_da3_intrinsics = None
                            if batched_depths is not None:
                                frame_cached_depth = batched_depths[frame_idx:frame_idx+1].unsqueeze(1)  # [1, 1, H, W]
                                if batched_da3_intrinsics is not None:
                                    frame_da3_intrinsics = batched_da3_intrinsics[frame_idx:frame_idx+1]

                            # Use model.forward() for single-frame inference
                            # Pass cached_depth if we have batched DA3 depth
                            outputs = model.forward(
                                images=frame_img,  # [1, 3, H, W]
                                text_prompts=[text_for_model],
                                gt_masks=None,
                                gt_intrinsics=frame_intrinsics,
                                gt_extrinsics=frame_extrinsics,
                                da3_extrinsics=frame_extrinsics,
                                da3_intrinsics=frame_da3_intrinsics if frame_da3_intrinsics is not None else frame_intrinsics,
                                cached_depth=frame_cached_depth,
                                point_prompts=frame_point_prompts,
                                point_labels=frame_point_labels,
                                box_prompts=frame_box_prompts,
                                box_labels=frame_box_labels,
                            )

                            # Save first frame outputs for paper_viz (has depth from DA3)
                            if frame_idx == 0:
                                first_frame_outputs = outputs

                            frame_mask = outputs.get('pred_masks')
                            if frame_mask is not None:
                                # pred_masks from forward() is [B, H, W] or [B, 1, H, W]
                                if frame_mask.dim() == 4:
                                    frame_mask = frame_mask[:, 0]  # [B, H, W]
                                all_frame_masks.append(frame_mask)
                            # Collect all candidate masks for oracle IoU
                            frame_all_masks = outputs.get('all_masks')
                            if frame_all_masks is not None:
                                all_frame_all_masks.append(frame_all_masks)
                    else:
                        # Chunked processing with cross-view attention within chunks
                        chunk_size = args.view_chunk_size if args.view_chunk_size > 0 else N
                        for chunk_start in range(0, N, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, N)
                            chunk_images = images[chunk_start:chunk_end]  # [chunk, 3, H, W]
                            chunk_intrinsics = intrinsics[chunk_start:chunk_end] if intrinsics is not None else None
                            chunk_extrinsics = extrinsics[chunk_start:chunk_end] if extrinsics is not None else None

                            # Get prompts for this chunk
                            chunk_point_prompts = all_point_prompts[chunk_start:chunk_end] if all_point_prompts is not None else None
                            chunk_point_labels = all_point_labels[chunk_start:chunk_end] if all_point_labels is not None else None
                            chunk_box_prompts = all_box_prompts[chunk_start:chunk_end] if all_box_prompts is not None else None
                            chunk_box_labels = all_box_labels[chunk_start:chunk_end] if all_box_labels is not None else None

                            # Pass extrinsics as both gt and da3 (model uses da3 when use_da3_poses_for_gasa=True)
                            outputs = model.forward_multiview(
                                images=chunk_images.unsqueeze(0),  # [1, chunk, 3, H, W]
                                text_prompts=[text_for_model],
                                gt_masks=None,
                                gt_intrinsics=chunk_intrinsics.unsqueeze(0) if chunk_intrinsics is not None else None,
                                gt_extrinsics=chunk_extrinsics.unsqueeze(0) if chunk_extrinsics is not None else None,
                                da3_extrinsics=chunk_extrinsics.unsqueeze(0) if chunk_extrinsics is not None else None,
                                da3_intrinsics=chunk_intrinsics.unsqueeze(0) if chunk_intrinsics is not None else None,
                                point_prompts=chunk_point_prompts.unsqueeze(0) if chunk_point_prompts is not None else None,
                                point_labels=chunk_point_labels.unsqueeze(0) if chunk_point_labels is not None else None,
                                box_prompts=chunk_box_prompts.unsqueeze(0) if chunk_box_prompts is not None else None,
                                box_labels=chunk_box_labels.unsqueeze(0) if chunk_box_labels is not None else None,
                            )

                            # Save first chunk outputs for paper_viz (has depth from DA3)
                            if chunk_start == 0:
                                first_frame_outputs = outputs

                            chunk_masks = outputs.get('masks') or outputs.get('pred_masks')
                            if chunk_masks is not None:
                                # Normalize to [chunk, H, W]
                                if chunk_masks.dim() == 6:
                                    chunk_masks = chunk_masks[0, :, 0, 0]  # [chunk, H, W]
                                elif chunk_masks.dim() == 5:
                                    chunk_masks = chunk_masks[0, :, 0]  # [chunk, H, W]
                                elif chunk_masks.dim() == 4:
                                    chunk_masks = chunk_masks[0]  # [chunk, H, W]
                                all_frame_masks.append(chunk_masks)
                            # Collect all candidate masks for oracle IoU
                            chunk_all_masks = outputs.get('all_masks')
                            if chunk_all_masks is not None:
                                all_frame_all_masks.append(chunk_all_masks)

            # Concatenate all frames/chunks
            if not all_frame_masks:
                continue
            pred_masks = torch.cat(all_frame_masks, dim=0)  # [N, H, W]

            # Resize pred masks to match GT
            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks.unsqueeze(1).float(),
                    size=gt_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            # Compute full metrics per view (IoU, recall, precision, F1)
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            gt_binary = (gt_masks > 0.5).float()

            # Per-view metrics
            intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))
            pred_sum = pred_binary.sum(dim=(-2, -1))
            gt_sum = gt_binary.sum(dim=(-2, -1))
            union = pred_sum + gt_sum - intersection

            # IoU = TP / (TP + FP + FN)
            iou_per_view = (intersection / (union + 1e-6)).mean().item()

            # Recall = TP / (TP + FN) = intersection / gt_sum
            recall_per_view = (intersection / (gt_sum + 1e-6)).mean().item()

            # Precision = TP / (TP + FP) = intersection / pred_sum
            precision_per_view = (intersection / (pred_sum + 1e-6)).mean().item()

            # F1 = 2 * precision * recall / (precision + recall)
            f1_per_view = 2 * precision_per_view * recall_per_view / (precision_per_view + recall_per_view + 1e-6)

            # Oracle IoU: compute best possible mask from all candidates
            oracle_iou_per_view = iou_per_view  # Fallback to selected
            if all_frame_all_masks:
                try:
                    oracle_ious_views = []
                    for vi in range(gt_masks.shape[0]):
                        gt_v = gt_masks[vi]  # [H, W]
                        # Find the all_masks tensor for this view
                        # Per-frame: each entry is [1, Q, H, W] for one frame
                        # Chunked: each entry is [1, N_chunk, Q, H, W] or [chunk, Q, H, W]
                        if args.per_frame and vi < len(all_frame_all_masks):
                            masks_v = all_frame_all_masks[vi]
                            # Shape: [1, Q, H, W] from model.forward()
                            if masks_v.dim() == 4:
                                masks_v = masks_v[0]  # [Q, H, W]
                            elif masks_v.dim() == 5:
                                masks_v = masks_v[0, 0]  # [Q, H, W]
                        else:
                            break  # Can't match views in chunked mode easily
                        # Resize if needed
                        if masks_v.shape[-2:] != gt_v.shape[-2:]:
                            masks_v = F.interpolate(
                                masks_v.unsqueeze(0).float(), size=gt_v.shape[-2:],
                                mode='bilinear', align_corners=False
                            ).squeeze(0)
                        oracle_result = compute_oracle_iou(masks_v.unsqueeze(0), gt_v)
                        oracle_ious_views.append(oracle_result['oracle_iou'])
                    if oracle_ious_views:
                        oracle_iou_per_view = np.mean(oracle_ious_views)
                except Exception:
                    pass  # Fallback to selected IoU

            all_ious.append(iou_per_view)
            all_oracle_ious.append(oracle_iou_per_view)
            all_recalls.append(recall_per_view)
            all_precisions.append(precision_per_view)
            all_f1s.append(f1_per_view)
            all_categories.append(category)
            category_ious[category].append(iou_per_view)
            category_oracle_ious[category].append(oracle_iou_per_view)
            category_recalls[category].append(recall_per_view)
            category_precisions[category].append(precision_per_view)
            category_f1s[category].append(f1_per_view)

            # Compute 3D centroid error if depth/pointmaps available
            # Get pointmaps from model outputs (if model ran DA3 internally)
            try:
                pointmaps = outputs.get('pointmaps')
                gt_centroid = batch.get('gt_centroid')

                if pointmaps is not None and gt_centroid is not None:
                    # pointmaps: [N, H, W, 3] or [1, N, H, W, 3]
                    if pointmaps.dim() == 5:
                        pointmaps = pointmaps.squeeze(0)  # [N, H, W, 3]

                    # Resize pointmaps to match mask size if needed
                    pts_h, pts_w = pointmaps.shape[1:3]
                    mask_h, mask_w = pred_binary.shape[-2:]
                    if pts_h != mask_h or pts_w != mask_w:
                        pointmaps = F.interpolate(
                            pointmaps.permute(0, 3, 1, 2),  # [N, 3, H, W]
                            size=(mask_h, mask_w),
                            mode='bilinear',
                            align_corners=False
                        ).permute(0, 2, 3, 1)  # [N, H, W, 3]

                    # Compute predicted centroid from mask + pointmaps (mean over all views)
                    pred_centroid_list = []
                    for v in range(pred_binary.shape[0]):
                        mask_v = pred_binary[v]  # [H, W]
                        pts_v = pointmaps[v]  # [H, W, 3]
                        if mask_v.sum() > 0:
                            # Weighted mean of 3D points
                            pts_flat = pts_v.view(-1, 3)  # [HW, 3]
                            mask_flat = mask_v.view(-1)  # [HW]
                            centroid_v = (pts_flat * mask_flat.unsqueeze(-1)).sum(0) / mask_flat.sum()
                            pred_centroid_list.append(centroid_v)

                    if pred_centroid_list:
                        pred_centroid = torch.stack(pred_centroid_list).mean(0)  # [3]
                        gt_centroid_tensor = gt_centroid.to(device).squeeze()  # [3]
                        error = torch.norm(pred_centroid - gt_centroid_tensor).item()
                        centroid_errors.append(error)
            except Exception as e:
                pass  # Skip centroid error if data not available

            # Update progress bar
            pbar.set_postfix({'mIoU': f'{np.mean(all_ious)*100:.1f}%', 'Recall': f'{np.mean(all_recalls)*100:.1f}%'})

            # Visualization
            if viz_dir and batch_idx < args.viz_samples:
                # Save first few samples
                save_visualization(
                    images=images.cpu(),
                    gt_masks=gt_binary.cpu(),
                    pred_masks=pred_binary.cpu(),
                    prompt=prompt,
                    iou=iou_per_view,
                    output_path=viz_dir / f'{scene_id.replace("/", "_")}_{batch_idx}.png',
                )

            # Collect paper viz data (one frame per sequence, first view)
            if paper_viz_collector is not None:
                pv_img = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pv_gt = gt_binary[0].cpu().numpy().astype(np.float32)
                pv_pred = pred_binary[0].cpu().numpy().astype(np.float32)
                pv_depth = None
                # Use first_frame_outputs if available (per-frame mode), else fall back to outputs
                viz_outputs = first_frame_outputs if first_frame_outputs is not None else outputs
                if viz_outputs is not None and 'depth' in viz_outputs:
                    depth_tensor = viz_outputs['depth']
                    # Handle various depth tensor shapes
                    if depth_tensor.dim() == 4:  # [B, 1, H, W] or [B, N, H, W]
                        pv_depth = depth_tensor[0, 0].cpu().float().numpy()
                    elif depth_tensor.dim() == 3:  # [B, H, W] or [1, H, W]
                        pv_depth = depth_tensor[0].cpu().float().numpy()
                    elif depth_tensor.dim() == 2:  # [H, W]
                        pv_depth = depth_tensor.cpu().float().numpy()
                # Get frame name from batch if available
                frame_names = batch.get('frame_names', batch.get('frame_ids', None))
                if frame_names is not None:
                    pv_frame_name = frame_names[0] if isinstance(frame_names, (list, tuple)) else str(frame_names)
                else:
                    pv_frame_name = f"frame_{batch_idx}"

                paper_viz_collector.append({
                    'image': pv_img,
                    'gt_mask': pv_gt,
                    'pred_mask': pv_pred,
                    'depth': pv_depth,
                    'label': prompt if isinstance(prompt, str) else str(prompt),
                    'scene_id': scene_id,
                    'frame_name': pv_frame_name,
                    'depth_source': 'live (DA3 per-frame)',  # uCO3D always runs DA3 live
                    'iou': iou_per_view,
                })

        except Exception as e:
            if ddp.is_main:
                print(f"Error processing batch {batch_idx}: {e}")
            continue

    # Gather results across ranks
    if ddp.is_distributed:
        # Gather sample-level metrics (tensors)
        all_ious_tensor = torch.tensor(all_ious, device=device)
        gathered_ious = [torch.zeros_like(all_ious_tensor) for _ in range(ddp.world_size)]
        dist.all_gather(gathered_ious, all_ious_tensor)
        all_ious = torch.cat(gathered_ious).cpu().numpy().tolist()

        all_oracle_ious_tensor = torch.tensor(all_oracle_ious, device=device)
        gathered_oracle_ious = [torch.zeros_like(all_oracle_ious_tensor) for _ in range(ddp.world_size)]
        dist.all_gather(gathered_oracle_ious, all_oracle_ious_tensor)
        all_oracle_ious = torch.cat(gathered_oracle_ious).cpu().numpy().tolist()

        # Gather per-category metrics (Python dicts) using all_gather_object
        all_category_ious_list = [None] * ddp.world_size
        all_category_recalls_list = [None] * ddp.world_size
        all_category_oracle_ious_list = [None] * ddp.world_size
        dist.all_gather_object(all_category_ious_list, dict(category_ious))
        dist.all_gather_object(all_category_recalls_list, dict(category_recalls))
        dist.all_gather_object(all_category_oracle_ious_list, dict(category_oracle_ious))

        # Merge category metrics from all ranks
        merged_category_ious = defaultdict(list)
        merged_category_recalls = defaultdict(list)
        merged_category_oracle_ious = defaultdict(list)
        for rank_cat_ious in all_category_ious_list:
            for cat, ious in rank_cat_ious.items():
                merged_category_ious[cat].extend(ious)
        for rank_cat_recalls in all_category_recalls_list:
            for cat, recalls in rank_cat_recalls.items():
                merged_category_recalls[cat].extend(recalls)
        for rank_cat_oracle_ious in all_category_oracle_ious_list:
            for cat, ious in rank_cat_oracle_ious.items():
                merged_category_oracle_ious[cat].extend(ious)

        category_ious = merged_category_ious
        category_recalls = merged_category_recalls
        category_oracle_ious = merged_category_oracle_ious

    # Compute metrics
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_oracle_iou = np.mean(all_oracle_ious) if all_oracle_ious else 0.0
    mean_recall = np.mean(all_recalls) if all_recalls else 0.0
    mean_precision = np.mean(all_precisions) if all_precisions else 0.0
    mean_f1 = np.mean(all_f1s) if all_f1s else 0.0

    # Per-category metrics (now from all ranks in DDP mode)
    category_miou = {}
    category_oracle_miou = {}
    category_mrecall = {}
    for cat, ious in category_ious.items():
        category_miou[cat] = np.mean(ious)
    for cat, ious in category_oracle_ious.items():
        category_oracle_miou[cat] = np.mean(ious)
    for cat, recalls in category_recalls.items():
        category_mrecall[cat] = np.mean(recalls)

    # Global mIoU (mean of per-category mIoU)
    global_miou = np.mean(list(category_miou.values())) if category_miou else 0.0
    global_oracle_miou = np.mean(list(category_oracle_miou.values())) if category_oracle_miou else 0.0
    global_mrecall = np.mean(list(category_mrecall.values())) if category_mrecall else 0.0

    # Compute Acc@m metrics if centroid errors are available
    acc_5cm = sum(1 for e in centroid_errors if e < 0.05) / max(len(centroid_errors), 1) if centroid_errors else None
    acc_10cm = sum(1 for e in centroid_errors if e < 0.10) / max(len(centroid_errors), 1) if centroid_errors else None
    acc_50cm = sum(1 for e in centroid_errors if e < 0.50) / max(len(centroid_errors), 1) if centroid_errors else None
    mean_centroid_error = np.mean(centroid_errors) if centroid_errors else None

    results = {
        'dataset': 'uco3d',
        'num_samples': len(all_ious),
        # Sample-averaged metrics
        'sample_iou': mean_iou,
        'oracle_iou': mean_oracle_iou,
        'sample_recall': mean_recall,
        'sample_precision': mean_precision,
        'sample_f1': mean_f1,
        # Category-averaged metrics (mIoU, mRecall)
        'scene_miou': global_miou,  # Named scene_miou for compatibility with ScanNet++ format
        'global_miou': global_miou,
        'oracle_miou': global_oracle_miou,
        'mean_class_recall': global_mrecall,
        # Per-category breakdowns
        'per_category_iou': category_miou,
        'per_category_oracle_iou': category_oracle_miou,
        'per_category_recall': category_mrecall,
        'num_categories': len(category_miou),
        # Oracle gap
        'iou_gap': mean_oracle_iou - mean_iou,
        'miou_gap': global_oracle_miou - global_miou,
    }

    # Add Acc@m metrics if available
    if acc_5cm is not None:
        results['acc_5cm'] = acc_5cm
        results['acc_10cm'] = acc_10cm
        results['acc_50cm'] = acc_50cm
        results['mean_centroid_error_m'] = float(mean_centroid_error) if mean_centroid_error is not None else None
        results['num_centroid_samples'] = len(centroid_errors)

    if ddp.is_main:
        ddp.print("\n" + "="*70)
        ddp.print("EVALUATION RESULTS (uCO3D)")
        ddp.print("="*70)
        ddp.print(f"Samples evaluated: {results['num_samples']}")
        ddp.print(f"Categories: {results['num_categories']}")
        ddp.print("-"*70)
        ddp.print(f"{'Metric':<25} {'Selected':<15} {'Oracle':<15} {'Gap':<10}")
        ddp.print("-"*70)
        ddp.print(f"{'Sample-avg IoU:':<25} {100*mean_iou:>13.2f}%  {100*mean_oracle_iou:>13.2f}%  {100*(mean_oracle_iou-mean_iou):>+8.2f}%")
        ddp.print(f"{'Global mIoU:':<25} {100*global_miou:>13.2f}%  {100*global_oracle_miou:>13.2f}%  {100*(global_oracle_miou-global_miou):>+8.2f}%")
        ddp.print("-"*70)
        ddp.print(f"Mean Class Recall:{100*global_mrecall:.2f}%  (sample-avg: {100*mean_recall:.2f}%)")
        ddp.print(f"Precision:        {100*mean_precision:.2f}%")
        ddp.print(f"F1 Score:         {100*mean_f1:.2f}%")
        ddp.print("-"*70)

        if acc_5cm is not None:
            ddp.print(f"3D Localization (IoU-based, same pointmap):")
            ddp.print(f"  Acc@5cm:        {100*acc_5cm:.2f}%")
            ddp.print(f"  Acc@10cm:       {100*acc_10cm:.2f}%")
            ddp.print(f"  Acc@50cm:       {100*acc_50cm:.2f}%")
            if mean_centroid_error is not None:
                ddp.print(f"  Mean Error:     {mean_centroid_error*100:.1f} cm")
            ddp.print(f"  Samples:        {len(centroid_errors)}")
            ddp.print("-"*60)

        # Top/bottom categories
        sorted_cats = sorted(category_miou.items(), key=lambda x: x[1], reverse=True)
        ddp.print(f"\nTop 5 categories:")
        for cat, iou in sorted_cats[:5]:
            recall = category_mrecall.get(cat, 0)
            ddp.print(f"  {cat}: IoU={100*iou:.1f}%, Recall={100*recall:.1f}%")
        ddp.print(f"\nBottom 5 categories:")
        for cat, iou in sorted_cats[-5:]:
            recall = category_mrecall.get(cat, 0)
            ddp.print(f"  {cat}: IoU={100*iou:.1f}%, Recall={100*recall:.1f}%")

    # Tests whether spatial prefixes ("nearest X", "leftmost X") degrade performance.
    # For single-instance datasets like uCO3D, spatial queries should be no-ops.
    GENERIC_SPATIAL_QUALIFIERS = ['nearest', 'farthest', 'leftmost', 'rightmost']
    spatial_eval_enabled = getattr(args, 'spatial_eval', False)

    if spatial_eval_enabled and ddp.is_main:
        ddp.print(f"\nSpatial Eval Pass (single-instance robustness)")

    spatial_ious_generic = []
    spatial_details_generic = defaultdict(list)

    if spatial_eval_enabled:
        # Distribute samples across DDP ranks
        sample_indices = list(range(len(dataset)))
        if ddp.is_distributed:
            rank_indices = [i for i in sample_indices if i % ddp.world_size == ddp.rank]
        else:
            rank_indices = sample_indices

        spatial_pbar = tqdm(rank_indices, desc="Spatial Eval", disable=not ddp.is_main)
        for sample_idx in spatial_pbar:
          for qualifier in GENERIC_SPATIAL_QUALIFIERS:
            try:
                batch_s = dataset[sample_idx]
                images_s = batch_s['images'].to(device)  # [N, 3, H, W]
                gt_masks_s = batch_s['gt_masks'].to(device)  # [N, H, W]
                prompt_s = batch_s['prompt'] if isinstance(batch_s['prompt'], str) else batch_s['prompt'][0]
                intrinsics_s = batch_s.get('intrinsics')
                if intrinsics_s is not None:
                    intrinsics_s = intrinsics_s.to(device)
                extrinsics_s = batch_s.get('extrinsics')
                if extrinsics_s is not None:
                    extrinsics_s = extrinsics_s.to(device)

                spatial_prompt = f"{qualifier} {prompt_s}"
                sq_type_g, _ = parse_spatial_qualifier(spatial_prompt)
                sq_idx_g = get_spatial_qualifier_idx(sq_type_g)
                sq_tensor_g = torch.tensor([sq_idx_g], device=device, dtype=torch.long) if sq_idx_g > 0 else None

                N_s = images_s.shape[0]
                frame_ious = []

                with torch.no_grad():
                    with autocast('cuda', dtype=torch.float16):
                        for fi in range(N_s):
                            gt_fi = gt_masks_s[fi]
                            if gt_fi.sum() < 1:
                                continue
                            frame_img = images_s[fi:fi+1]
                            frame_intr = intrinsics_s[fi:fi+1] if intrinsics_s is not None else None
                            frame_ext = extrinsics_s[fi:fi+1] if extrinsics_s is not None else None
                            outputs_g = model.forward(
                                images=frame_img,
                                text_prompts=[spatial_prompt],
                                gt_masks=None,
                                gt_intrinsics=frame_intr,
                                gt_extrinsics=frame_ext,
                                spatial_qualifier_idx=sq_tensor_g,
                            )
                            pred_g = outputs_g.get('pred_masks')
                            if pred_g is None:
                                continue
                            if pred_g.dim() == 4:
                                pred_g = pred_g[:, 0]
                            pred_g = pred_g.squeeze(0)
                            if pred_g.shape != gt_fi.shape:
                                pred_g = F.interpolate(
                                    pred_g.unsqueeze(0).unsqueeze(0).float(),
                                    size=gt_fi.shape[-2:], mode='bilinear', align_corners=False
                                ).squeeze(0).squeeze(0)
                            gt_bin_g = (gt_fi > 0.5).float()
                            pred_bin_g = (torch.sigmoid(pred_g) > 0.5).float()
                            inter_g = (pred_bin_g * gt_bin_g).sum()
                            union_g = pred_bin_g.sum() + gt_bin_g.sum() - inter_g
                            frame_ious.append((inter_g / (union_g + 1e-6)).item())

                if frame_ious:
                    sample_iou = np.mean(frame_ious)
                    spatial_ious_generic.append(sample_iou)
                    spatial_details_generic[qualifier].append(sample_iou)

            except Exception:
                continue

        # DDP gather spatial results
        if ddp.is_distributed:
            spatial_local = {
                'spatial_ious': spatial_ious_generic,
                'spatial_details': {k: list(v) for k, v in spatial_details_generic.items()},
            }
            spatial_gathered = [None] * ddp.world_size if ddp.is_main else None
            dist.gather_object(spatial_local, spatial_gathered, dst=0)
            if ddp.is_main:
                spatial_ious_generic = []
                spatial_details_generic = defaultdict(list)
                for rd in spatial_gathered:
                    spatial_ious_generic.extend(rd['spatial_ious'])
                    for q, v in rd['spatial_details'].items():
                        spatial_details_generic[q].extend(v)

        if spatial_ious_generic:
            spatial_mean = np.mean(spatial_ious_generic)
            spatial_per_q = {q: np.mean(v) for q, v in spatial_details_generic.items()}
            results['spatial_eval'] = {
                'num_samples': len(spatial_ious_generic),
                'spatial_miou': float(spatial_mean),
                'baseline_miou': float(mean_iou),
                'delta': float(spatial_mean - mean_iou),
                'per_qualifier_iou': {q: float(v) for q, v in spatial_per_q.items()},
            }
            if ddp.is_main:
                ddp.print(f"\nSpatial Robustness")
                ddp.print(f"  Baseline mIoU:  {100*mean_iou:.2f}%")
                ddp.print(f"  Spatial mIoU:   {100*spatial_mean:.2f}%  (delta={100*(spatial_mean-mean_iou):+.2f}%)")
                for q in GENERIC_SPATIAL_QUALIFIERS:
                    if q in spatial_per_q:
                        ddp.print(f"    {q:<12} IoU={100*spatial_per_q[q]:5.1f}%  (n={len(spatial_details_generic[q])})")
                ddp.print("="*70)

    return results


def save_visualization(
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    prompt: str,
    iou: float,
    output_path: Path,
    max_views: int = 4,
):
    """Save a grid visualization of predictions vs GT."""
    N = min(images.shape[0], max_views)

    fig, axes = plt.subplots(2, N, figsize=(4*N, 8))
    if N == 1:
        axes = axes.reshape(2, 1)

    # Get image size for resizing masks
    img_h, img_w = images.shape[2], images.shape[3]

    for i in range(N):
        img = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt = gt_masks[i]
        pred = pred_masks[i]

        # Resize masks to match image size if needed
        if gt.shape[-2:] != (img_h, img_w):
            gt = F.interpolate(gt.unsqueeze(0).unsqueeze(0).float(),
                               size=(img_h, img_w), mode='nearest').squeeze()
        if pred.shape[-2:] != (img_h, img_w):
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(),
                                 size=(img_h, img_w), mode='nearest').squeeze()

        gt = gt.numpy()
        pred = pred.numpy()

        # GT overlay
        gt_overlay = overlay_mask_sam3_style(img, gt, MASK_COLORS[0])
        axes[0, i].imshow(gt_overlay)
        axes[0, i].set_title(f'GT: {prompt}')
        axes[0, i].axis('off')

        # Pred overlay
        pred_overlay = overlay_mask_sam3_style(img, pred, MASK_COLORS[1])
        axes[1, i].imshow(pred_overlay)
        axes[1, i].set_title(f'Pred IoU: {iou*100:.1f}%')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


# Paper-Quality Grid Visualization

def create_paper_grid(
    rows: List[Dict[str, np.ndarray]],
    dpi: int = 600,
    mask_color: str = 'white',
    overlay_alpha: float = 0.5,
    cell_size_inches: float = 2.0,
) -> plt.Figure:
    """Create a paper-quality 5-column grid with row labels and column headers.

    Each row dict must contain:
        'image': np.ndarray [H, W, 3] uint8 RGB image
        'gt_mask': np.ndarray [H, W] float32 binary mask (0/1)
        'pred_mask': np.ndarray [H, W] float32 binary mask (0/1)
        'label': str  (GT category name, shown as row label)

    Optional:
        'depth': np.ndarray [H, W] float32 depth map (if missing, column is blank)

    Grid columns: Image | Depth | GT Mask | Pred Mask | Overlay
    Row labels on the left show the GT category.
    Column headers at the top label each column.
    """
    n_rows = len(rows)
    n_cols = 5
    col_headers = ['Image', 'Depth', 'GT Mask', 'Pred Mask', 'Overlay']

    # Extra vertical space for column header row
    header_height_inches = 0.4
    fig_w = n_cols * cell_size_inches + 0.8  # extra left margin for row labels
    fig_h = n_rows * cell_size_inches + header_height_inches

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    for row_idx, row_data in enumerate(rows):
        img = row_data['image']
        gt = row_data['gt_mask']
        pred = row_data['pred_mask']
        label = row_data.get('label', '')
        depth = row_data.get('depth', None)

        # Resize masks to image resolution if needed
        img_h, img_w = img.shape[:2]
        if gt.shape[:2] != (img_h, img_w):
            gt = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize(
                (img_w, img_h), Image.NEAREST)).astype(np.float32) / 255.0
        if pred.shape[:2] != (img_h, img_w):
            pred = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize(
                (img_w, img_h), Image.NEAREST)).astype(np.float32) / 255.0

        # Column 0: Photo
        axes[row_idx, 0].imshow(img)

        # Column 1: Depth map
        if depth is not None:
            # Squeeze extra dimensions (depth may be [1, 1, H, W] or [1, H, W])
            depth = np.squeeze(depth)
            if depth.ndim != 2:
                depth = None  # Invalid shape, skip
            elif depth.shape[:2] != (img_h, img_w):
                # Resize float depth via uint8 conversion
                d_min, d_max = depth.min(), depth.max()
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                depth_resized = np.array(Image.fromarray(depth_uint8).resize(
                    (img_w, img_h), Image.BILINEAR)).astype(np.float32) / 255.0
                depth = depth_resized * (d_max - d_min) + d_min
            # Normalize for display: clip outliers, use viridis colormap
            valid = depth[depth > 0]
            if len(valid) > 0:
                vmin = np.percentile(valid, 2)
                vmax = np.percentile(valid, 98)
            else:
                vmin, vmax = 0.0, 1.0
            axes[row_idx, 1].imshow(depth, cmap='Spectral', vmin=vmin, vmax=vmax)
        else:
            # Blank placeholder
            axes[row_idx, 1].imshow(np.zeros((img_h, img_w, 3), dtype=np.uint8))

        # Column 2: GT Mask (standalone)
        gt_color = 'blue' if mask_color == 'colored' else mask_color
        axes[row_idx, 2].imshow(render_mask_standalone(gt, color=gt_color))

        # Column 3: Pred Mask (standalone)
        pred_color = 'green' if mask_color == 'colored' else mask_color
        axes[row_idx, 3].imshow(render_mask_standalone(pred, color=pred_color))

        # Column 4: Overlay (pred mask on photo)
        overlay = overlay_mask_sam3_style(img, pred, MASK_COLORS[1], alpha=overlay_alpha)
        axes[row_idx, 4].imshow(overlay)

        # Row label on the left (GT category) - uncomment to show labels
        # axes[row_idx, 0].set_ylabel(label, fontsize=10, fontweight='bold',
        #                              rotation=90, labelpad=8, va='center')

    # Remove ticks and spines, keep y-label visible
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_aspect('equal')

    # Column headers - uncomment to show headers
    # for col_idx, header in enumerate(col_headers):
    #     axes[0, col_idx].set_title(header, fontsize=10, fontweight='bold', pad=6)

    fig.subplots_adjust(wspace=0.03, hspace=0.06,
                        left=0.08, right=0.99, top=0.94, bottom=0.01)
    return fig


def collect_paper_viz_from_results(
    viz_pool: List[Dict],
    mode: str = 'multi_scene',
    target_objects: Optional[List[str]] = None,
    num_rows: int = 4,
    num_sets: int = 1,
    seed: int = 42,
) -> List[List[Dict]]:
    """Select and organize collected viz data into grid sets.

    Args:
        viz_pool: List of dicts with keys: image, gt_mask, pred_mask, label, scene_id, iou.
        mode: 'single_object', 'single_scene', or 'multi_scene'.
        target_objects: Specific object labels to include (None = auto).
        num_rows: Number of rows per grid set.
        num_sets: Number of grid sets to produce.
        seed: Random seed for reproducible selection.

    Returns:
        List of grid sets. Each set is a list of row dicts.
    """
    import random
    rng = random.Random(seed)

    if not viz_pool:
        return []

    # Filter by target objects if specified
    pool = viz_pool
    if target_objects:
        target_lower = [t.lower() for t in target_objects]
        pool = [v for v in pool if v['label'].lower() in target_lower]
        if not pool:
            print(f"  [paper-viz] Warning: no samples match target objects {target_objects}, using all")
            pool = viz_pool

    grid_sets = []

    if mode == 'single_object':
        # Group by (scene_id, label) to find objects with multiple views
        from collections import defaultdict
        obj_groups = defaultdict(list)
        for v in pool:
            key = (v.get('scene_id', 'unknown'), v['label'])
            obj_groups[key].append(v)

        # Find objects with enough views
        candidates = [(k, vs) for k, vs in obj_groups.items() if len(vs) >= num_rows]
        if not candidates:
            # Fallback: use objects with most views
            candidates = sorted(obj_groups.items(), key=lambda x: len(x[1]), reverse=True)

        for set_idx in range(num_sets):
            if set_idx < len(candidates):
                key, views = candidates[set_idx]
            else:
                key, views = rng.choice(candidates)
            selected = rng.sample(views, min(num_rows, len(views)))
            if len(selected) < num_rows:
                # Pad by repeating if needed
                while len(selected) < num_rows:
                    selected.append(rng.choice(views))
            grid_sets.append(selected[:num_rows])

    elif mode == 'single_scene':
        # Group by scene_id
        from collections import defaultdict
        scene_groups = defaultdict(list)
        for v in pool:
            scene_groups[v.get('scene_id', 'unknown')].append(v)

        # Pick scenes with enough distinct objects
        scene_candidates = []
        for scene_id, views in scene_groups.items():
            labels_seen = set()
            unique = []
            for v in views:
                if v['label'] not in labels_seen:
                    labels_seen.add(v['label'])
                    unique.append(v)
            if len(unique) >= num_rows:
                scene_candidates.append((scene_id, unique))

        if not scene_candidates:
            # Fallback: scenes with most unique objects
            for scene_id, views in scene_groups.items():
                labels_seen = set()
                unique = []
                for v in views:
                    if v['label'] not in labels_seen:
                        labels_seen.add(v['label'])
                        unique.append(v)
                scene_candidates.append((scene_id, unique))
            scene_candidates.sort(key=lambda x: len(x[1]), reverse=True)

        for set_idx in range(num_sets):
            if set_idx < len(scene_candidates):
                scene_id, unique = scene_candidates[set_idx]
            else:
                scene_id, unique = rng.choice(scene_candidates)
            selected = rng.sample(unique, min(num_rows, len(unique)))
            while len(selected) < num_rows:
                selected.append(rng.choice(unique))
            grid_sets.append(selected[:num_rows])

    else:  # multi_scene
        # Pick diverse objects across different scenes and categories
        # Group all samples by label, keep all (not just best IoU)
        from collections import defaultdict
        by_label = defaultdict(list)
        for v in pool:
            by_label[v['label']].append(v)

        # Sort labels by number of samples (most first) for variety
        sorted_labels = sorted(by_label.keys(), key=lambda l: len(by_label[l]), reverse=True)
        rng.shuffle(sorted_labels)

        # For each set, pick different labels and different samples
        used_samples = set()  # Track (scene_id, label) to avoid exact duplicates

        for set_idx in range(num_sets):
            selected = []
            # Rotate through labels for each set to get variety
            label_offset = (set_idx * num_rows) % len(sorted_labels)

            for i in range(num_rows):
                label_idx = (label_offset + i) % len(sorted_labels)
                label = sorted_labels[label_idx]
                candidates = by_label[label]

                # Try to find an unused sample
                chosen = None
                for c in candidates:
                    key = (c.get('scene_id', ''), c['label'], id(c))
                    if key not in used_samples:
                        chosen = c
                        used_samples.add(key)
                        break

                if chosen is None:
                    # All used, pick randomly
                    chosen = rng.choice(candidates)

                selected.append(chosen)

            grid_sets.append(selected)

    return grid_sets


def generate_paper_visualizations(
    viz_pool: List[Dict],
    args,
    output_dir: Path,
    ddp_rank: int = None,
) -> None:
    """Generate paper-quality grid PNGs from collected visualization data.

    Args:
        viz_pool: List of dicts with image/mask data collected during eval.
        args: Parsed command-line arguments with paper_viz_* fields.
        output_dir: Base output directory.
        ddp_rank: If set, include rank in filenames to avoid conflicts.
    """
    if not viz_pool:
        print("  [paper-viz] No visualization data collected, skipping.")
        return

    paper_dir = output_dir / 'paper_viz'
    paper_dir.mkdir(parents=True, exist_ok=True)

    mode = getattr(args, 'paper_viz_mode', 'multi_scene')
    target_objects = getattr(args, 'paper_viz_objects', None)
    num_rows = getattr(args, 'paper_viz_rows', 4)
    num_sets = getattr(args, 'paper_viz_sets', 1)
    dpi = getattr(args, 'paper_viz_dpi', 600)
    mask_color = getattr(args, 'paper_viz_mask_color', 'white')
    overlay_alpha = getattr(args, 'paper_viz_overlay_alpha', 0.5)
    seed = getattr(args, 'seed', 42)
    min_iou = getattr(args, 'viz_min_iou', 0.0)

    # Filter by minimum IoU if specified
    if min_iou > 0:
        original_count = len(viz_pool)
        viz_pool = [v for v in viz_pool if v.get('iou', 0) >= min_iou]
        print(f"  [paper-viz] IoU filter (>= {min_iou:.0%}): {original_count} → {len(viz_pool)} samples")
        if not viz_pool:
            print(f"  [paper-viz] No samples above {min_iou:.0%} IoU threshold, skipping.")
            return

    grid_sets = collect_paper_viz_from_results(
        viz_pool, mode=mode, target_objects=target_objects,
        num_rows=num_rows, num_sets=num_sets, seed=seed,
    )

    if not grid_sets:
        print("  [paper-viz] Could not assemble any grid sets from collected data.")
        return

    metadata = {
        'config': {
            'mode': mode,
            'num_rows': num_rows,
            'num_sets': num_sets,
            'dpi': dpi,
            'mask_color': mask_color,
            'overlay_alpha': overlay_alpha,
            'min_iou_threshold': min_iou,
        },
        'sets': [],
    }

    for set_idx, grid_rows in enumerate(grid_sets):
        # Print depth source info for debugging
        print(f"  [paper-viz] Set {set_idx} rows:")
        for row_idx, r in enumerate(grid_rows):
            depth_src = r.get('depth_source', 'unknown')
            frame = r.get('frame_name', 'unknown')
            label = r.get('label', '')
            scene = r.get('scene_id', '')
            print(f"    Row {row_idx}: {scene}/{frame} '{label}' depth={depth_src}")

        fig = create_paper_grid(
            grid_rows, dpi=dpi, mask_color=mask_color, overlay_alpha=overlay_alpha,
        )
        # Include rank in filename if running distributed
        if ddp_rank is not None:
            filename = f'paper_grid_rank{ddp_rank}_set{set_idx:02d}.png'
        else:
            filename = f'paper_grid_set{set_idx:02d}.png'
        fig_path = paper_dir / filename
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f"  [paper-viz] Saved {fig_path} ({dpi} DPI)")

        set_meta = {
            'set_id': set_idx,
            'filename': filename,
            'rows': [
                {
                    'label': r.get('label', ''),
                    'scene_id': r.get('scene_id', ''),
                    'frame_name': r.get('frame_name', ''),
                    'depth_source': r.get('depth_source', 'unknown'),
                    'iou': float(r.get('iou', 0)),
                }
                for r in grid_rows
            ],
        }
        metadata['sets'].append(set_meta)

    # Include rank in filename if running distributed
    if ddp_rank is not None:
        meta_path = paper_dir / f'paper_grid_metadata_rank{ddp_rank}.json'
    else:
        meta_path = paper_dir / 'paper_grid_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [paper-viz] Metadata saved to {meta_path}")


def generate_single_object_viz(
    viz_pool: List[Dict],
    args,
    output_dir: Path,
    ddp_rank: int = None,
) -> None:
    """Generate focused visualization for specific objects showing top-k IoU frames.

    This mode is designed for detailed analysis of model performance on specific objects.
    For each object, shows the top-k frames by IoU in a grid.

    Supports multiple scenes - when samples come from multiple scenes, organizes output
    by scene with per-scene subdirectories.

    Args:
        viz_pool: List of dicts with image/mask data collected during eval.
        args: Parsed command-line arguments with viz_* fields.
        output_dir: Base output directory.
        ddp_rank: If set, include rank in metadata/summary filenames to avoid conflicts.
    """
    if not viz_pool:
        print("  [single-object-viz] No visualization data collected, skipping.")
        return

    viz_dir = output_dir / 'single_object_viz'
    viz_dir.mkdir(parents=True, exist_ok=True)

    topk = getattr(args, 'viz_topk', 4)
    num_objects = getattr(args, 'viz_num_objects', 4)
    target_objects = getattr(args, 'viz_objects', None)
    min_iou = getattr(args, 'viz_min_iou', 0.0)

    # Filter by minimum IoU if specified
    if min_iou > 0:
        original_count = len(viz_pool)
        viz_pool = [v for v in viz_pool if v.get('iou', 0) >= min_iou]
        print(f"  [single-object-viz] IoU filter (>= {min_iou:.0%}): {original_count} → {len(viz_pool)} samples")
        if not viz_pool:
            print(f"  [single-object-viz] No samples above {min_iou:.0%} IoU threshold, skipping.")
            return
    dpi = getattr(args, 'paper_viz_dpi', 600)
    mask_color = getattr(args, 'paper_viz_mask_color', 'white')
    overlay_alpha = getattr(args, 'paper_viz_overlay_alpha', 0.5)

    # Check how many scenes we have
    from collections import defaultdict
    scenes_in_pool = set(v.get('scene_id', 'unknown') for v in viz_pool)
    multi_scene = len(scenes_in_pool) > 1
    print(f"  [single-object-viz] {len(viz_pool)} samples from {len(scenes_in_pool)} scene(s)")

    # Group samples by scene, then by label
    by_scene = defaultdict(lambda: defaultdict(list))
    for v in viz_pool:
        scene = v.get('scene_id', 'unknown')
        label = v['label']
        by_scene[scene][label].append(v)

    # Process each scene
    all_metadata = {
        'num_scenes': len(scenes_in_pool),
        'topk_per_object': topk,
        'num_objects_per_scene': num_objects,
        'min_iou_threshold': min_iou,
        'scenes': [],
    }

    for scene_id in sorted(by_scene.keys()):
        by_label = by_scene[scene_id]
        print(f"\n  [single-object-viz] Scene: {scene_id}")

        # Determine which objects to visualize for this scene
        if target_objects:
            objects_to_viz = [obj for obj in target_objects if obj in by_label]
            if not objects_to_viz:
                print(f"    No target objects found, skipping scene")
                continue
        else:
            # Pick objects with BEST IoU for this scene
            label_best_iou = {}
            for label, samples in by_label.items():
                best_iou = max(s.get('iou', 0) for s in samples)
                label_best_iou[label] = best_iou
            sorted_labels = sorted(label_best_iou.keys(), key=lambda l: label_best_iou[l], reverse=True)
            objects_to_viz = sorted_labels[:num_objects]

        print(f"    Visualizing {len(objects_to_viz)} objects, top-{topk} IoU frames each:")

        # Collect rows for this scene
        scene_rows = []
        scene_meta = {
            'scene_id': scene_id,
            'objects': [],
        }

        for obj_label in objects_to_viz:
            samples = by_label[obj_label]
            samples_sorted = sorted(samples, key=lambda x: x.get('iou', 0), reverse=True)
            top_samples = samples_sorted[:topk]

            print(f"      {obj_label}: {len(samples)} total, top {len(top_samples)}")

            obj_meta = {
                'label': obj_label,
                'total_samples': len(samples),
                'frames': [],
            }

            for rank, sample in enumerate(top_samples):
                scene_rows.append(sample)
                obj_meta['frames'].append({
                    'rank': rank + 1,
                    'frame_name': sample.get('frame_name', 'unknown'),
                    'iou': float(sample.get('iou', 0)),
                    'depth_source': sample.get('depth_source', 'unknown'),
                })

            scene_meta['objects'].append(obj_meta)

        if not scene_rows:
            continue

        all_metadata['scenes'].append(scene_meta)

        # Check if separate image output is requested
        save_separate = getattr(args, 'viz_separate', False)

        if save_separate:
            # Save to per-scene subdirectory if multiple scenes
            if multi_scene:
                separate_dir = viz_dir / scene_id
            else:
                separate_dir = viz_dir / 'separate'
            separate_dir.mkdir(parents=True, exist_ok=True)

            print(f"    Saving {len(scene_rows)} samples to: {separate_dir}")

            for idx, sample in enumerate(scene_rows):
                label = sample.get('label', 'unknown').replace(' ', '_').replace('/', '_')
                frame = sample.get('frame_name', f'frame{idx}')
                iou = sample.get('iou', 0)

                # Simpler prefix without scene (it's in the folder name)
                prefix = f"{idx:02d}_{label}_{frame}_iou{iou:.2f}"

                img = sample['image']
                gt = sample['gt_mask']
                pred = sample['pred_mask']
                depth = sample.get('depth', None)

                img_h, img_w = img.shape[:2]
                if gt.shape[:2] != (img_h, img_w):
                    gt = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize(
                        (img_w, img_h), Image.NEAREST)).astype(np.float32) / 255.0
                if pred.shape[:2] != (img_h, img_w):
                    pred = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize(
                        (img_w, img_h), Image.NEAREST)).astype(np.float32) / 255.0

                Image.fromarray(img).save(separate_dir / f"{prefix}_rgb.png")
                Image.fromarray((gt * 255).astype(np.uint8)).save(separate_dir / f"{prefix}_gt_mask.png")
                Image.fromarray((pred * 255).astype(np.uint8)).save(separate_dir / f"{prefix}_pred_mask.png")

                if depth is not None:
                    depth = np.squeeze(depth)
                    if depth.ndim == 2:
                        if depth.shape[:2] != (img_h, img_w):
                            d_min, d_max = depth.min(), depth.max()
                            depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
                            depth_uint8 = (depth_norm * 255).astype(np.uint8)
                            depth = np.array(Image.fromarray(depth_uint8).resize(
                                (img_w, img_h), Image.BILINEAR)).astype(np.float32) / 255.0
                            depth = depth * (d_max - d_min) + d_min
                        valid = depth[depth > 0]
                        vmin = np.percentile(valid, 2) if len(valid) > 0 else 0.0
                        vmax = np.percentile(valid, 98) if len(valid) > 0 else 1.0
                        depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0, 1)
                        depth_colored = (plt.cm.Spectral(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                        Image.fromarray(depth_colored).save(separate_dir / f"{prefix}_depth.png")

                overlay = overlay_mask_sam3_style(img, pred, MASK_COLORS[1], alpha=overlay_alpha)
                Image.fromarray(overlay).save(separate_dir / f"{prefix}_overlay.png")

        # Generate per-scene grid
        if multi_scene:
            grid_path = viz_dir / f'grid_{scene_id}.png'
        else:
            grid_path = viz_dir / 'single_object_grid.png'

        fig = create_paper_grid(scene_rows, dpi=dpi, mask_color=mask_color, overlay_alpha=overlay_alpha)
        fig.savefig(grid_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f"    Saved grid: {grid_path}")

    # Save detailed metadata JSON
    # Include rank in filename if running distributed to avoid conflicts
    if ddp_rank is not None:
        meta_path = viz_dir / f'metadata_rank{ddp_rank}.json'
    else:
        meta_path = viz_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\n  [single-object-viz] Saved metadata: {meta_path}")

    # Also save a summary table
    summary_lines = [
        f"Single Object Visualization Summary",
        f"=" * 50,
        f"Scenes: {all_metadata['num_scenes']}",
        f"Top-K per object: {topk}",
        f"Objects per scene: {num_objects}",
        f"",
    ]
    for scene_meta in all_metadata['scenes']:
        summary_lines.append(f"\n{'='*50}")
        summary_lines.append(f"Scene: {scene_meta['scene_id']}")
        summary_lines.append("=" * 50)
        for obj_meta in scene_meta['objects']:
            summary_lines.append(f"\n  {obj_meta['label']} ({obj_meta['total_samples']} total samples):")
            summary_lines.append("  " + "-" * 38)
            for frame_info in obj_meta['frames']:
                summary_lines.append(
                    f"    #{frame_info['rank']}: {frame_info['frame_name']} "
                    f"IoU={frame_info['iou']:.3f}"
                )

    if ddp_rank is not None:
        summary_path = viz_dir / f'summary_rank{ddp_rank}.txt'
    else:
        summary_path = viz_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"  [single-object-viz] Saved summary: {summary_path}")


def main():
    config = tyro.cli(BenchmarkConfig)
    args = config.to_namespace()

    # Validate: --checkpoint required unless --baseline-sam3
    if not args.baseline_sam3 and not args.checkpoint:
        raise SystemExit("Error: --checkpoint is required unless --baseline-sam3 is used")

    # Default mask_size to SAM3 native output: (image_size // 14) * 4
    # Deferred until after model loading if image_size is None (will use model.resolution)
    if args.mask_size is None and args.image_size is not None:
        args.mask_size = (args.image_size // 14) * 4


    # Initialize DDP for distributed evaluation
    ddp = DDPManager()
    ddp.init(timeout_minutes=120)  # 2h timeout for large evals

    random.seed(args.seed + ddp.rank)  # Different seed per rank for diversity
    np.random.seed(args.seed + ddp.rank)
    torch.manual_seed(args.seed + ddp.rank)

    device = ddp.device if ddp.is_distributed else ('cuda' if torch.cuda.is_available() else 'cpu')

    # CUDA performance settings
    torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms (~5-15% speedup)

    ddp.print(f"Device: {device}" + (f" (DDP: {ddp.world_size} GPUs)" if ddp.is_distributed else ""))

    # Setup output directory (like evaluate_mvimgnet2.py)
    if args.run_name:
        run_name = args.run_name
    elif args.baseline_sam3:
        run_name = f"baseline_sam3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        # Extract run name from checkpoint path
        ckpt_name = Path(args.checkpoint).parent.name
        run_name = f"eval_{ckpt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.dataset in ('lerf_ovs', 'lerf_loc'):
        output_dir = PROJECT_ROOT / 'runs' / 'lerf' / run_name
    else:
        output_dir = PROJECT_ROOT / 'runs' / 'final' / run_name
    if ddp.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / 'visualizations'

    if args.visualize and ddp.is_main:
        viz_dir.mkdir(parents=True, exist_ok=True)

    ddp.print(f"Output directory: {output_dir}")

    # Load model
    if args.baseline_sam3:
        # Round image_size to nearest multiple of 14 (SAM3 ViT patch size)
        import math as _math
        if args.image_size is None:
            args.image_size = 1008  # Default for baseline
        if args.mask_size is None:
            args.mask_size = (args.image_size // 14) * 4
        sam3_res = _math.ceil(args.image_size / 14) * 14
        ddp.print(f"\nLoading baseline SAM3 (native decoder, no GASA/depth/cross-view)...")
        ddp.print(f"  SAM3 img_size={sam3_res} (from --image-size {args.image_size})")
        sam3_model = build_sam3_image_model(bpe_path=_BPE_PATH, img_size=sam3_res).to(device)
        model = BaselineSAM3Wrapper(sam3_model, resolution=sam3_res)
        model.eval()

        total_params, trainable_params = count_parameters(model)
        gasa_params = 0  # No GASA decoder in baseline
        ddp.print(f"\nBaseline SAM3 Parameters: {total_params/1e6:.2f}M")
    else:
        ddp.print(f"\nLoading model from {args.checkpoint}...")
        model = load_model(args.checkpoint, device, da3_resolution=args.da3_resolution, num_queries=args.num_queries, skip_trained_seghead=args.skip_trained_seghead, train_config_path=args.train_config, resolution=args.image_size)

        # Sync args.image_size with model resolution (handles None default)
        if args.image_size is None:
            args.image_size = model.resolution
        # Compute mask_size now that image_size is resolved
        if args.mask_size is None:
            args.mask_size = (args.image_size // 14) * 4
        ddp.print(f"  Eval image_size={args.image_size}, model.resolution={model.resolution}, mask_size={args.mask_size}")

        # Auto-enable spatial eval if model was trained with spatial tokens
        ckpt_dir = Path(args.checkpoint).parent
        train_config_path = args.train_config or str(ckpt_dir / 'config.json')
        if Path(train_config_path).exists():
            with open(train_config_path) as f:
                train_config = json.load(f)
            if train_config.get('use_spatial_tokens', False) and not args.spatial_eval and not args.no_spatial_eval:
                args.spatial_eval = True
                ddp.print(f"  Auto-enabled --spatial-eval (model trained with use_spatial_tokens=True)")
            if args.no_spatial_eval:
                args.spatial_eval = False

        # Auto-enable text scoring selection at eval for text_scoring models
        if model.pred_logits_source == 'text_scoring' and not args.mask_selection:
            ddp.print(f"  Auto-setting mask_selection='confidence' for text_scoring model (bypasses oracle iou_match)")
            model.mask_selection = 'confidence'

        # Override mask selection if specified
        if args.mask_selection:
            ddp.print(f"  Overriding mask_selection: {model.mask_selection} -> {args.mask_selection}")
            model.mask_selection = args.mask_selection
            if args.mask_selection == 'predicted_iou' and not model.use_iou_head:
                ddp.print("  WARNING: predicted_iou requires use_iou_head=True but model was trained without it!")

        # Count parameters
        total_params, trainable_params = count_parameters(model)
        gasa_params = sum(p.numel() for p in model.gasa_decoder.parameters())
        ddp.print(f"\nModel Parameters:")
        ddp.print(f"  Total: {total_params/1e6:.2f}M")
        ddp.print(f"  Trainable: {trainable_params/1e6:.2f}M")
        ddp.print(f"  GASA Decoder: {gasa_params/1e6:.2f}M")

    # Get scene list - use dataset-specific default if not provided
    if args.data_root is None:
        if args.dataset == 'scannetpp':
            args.data_root = str(PROJECT_ROOT / 'data' / 'scannetpp')
        elif args.dataset == 'uco3d':
            args.data_root = str(PROJECT_ROOT / 'data' / 'uco3d')
        elif args.dataset == 'partimagenet':
            args.data_root = str(PROJECT_ROOT / 'data' / 'partimagenet' / 'PartImageNet')
        elif args.dataset in ('lerf_ovs', 'lerf_loc'):
            args.data_root = str(PROJECT_ROOT / 'data' / 'lerf_ovs')
        elif args.dataset == 'nvos':
            args.data_root = str(PROJECT_ROOT / 'data' / 'nvos')
        else:
            raise ValueError(f"--data-root required for dataset {args.dataset}")
    data_root = Path(args.data_root)

    # uCO3D uses different evaluation path (dataset-based instead of scene-based)
    if args.dataset == 'uco3d':
        from triangulang.data.uco3d_dataset import UCO3DMultiViewDataset, create_uco3d_eval_dataset

        # Default to per-frame mode for uCO3D (faster, lower memory, no cross-view attention)
        # Use --view-chunk-size N for chunked processing with cross-view attention
        per_frame_default = False
        if not args.per_frame and args.view_chunk_size == 8:  # Default values = use per_frame
            args.per_frame = True
            per_frame_default = True

        ddp.print(f"\nuCO3D Evaluation")
        ddp.print(f"Using UCO3DMultiViewDataset with:")
        ddp.print(f"  - 50 representative sequences")
        ddp.print(f"  - 50 frames per sequence (uniformly sampled)")
        ddp.print(f"  - Category name as text prompt")
        ddp.print(f"  - Prompt normalization: {args.normalize_prompts}")
        num_seq = args.num_sequences if args.num_sequences else 50
        frames_per_seq = args.frames_per_sequence if args.frames_per_sequence else 50
        ddp.print(f"  - Sequences: {num_seq}" + (" (paper default)" if args.num_sequences is None else ""))
        ddp.print(f"  - Frames/sequence: {frames_per_seq}" + (" (paper default)" if args.frames_per_sequence is None else ""))
        if per_frame_default:
            ddp.print(f"  - Per-frame mode (default for uCO3D, use --view-chunk-size N for cross-view)")
        elif args.per_frame:
            ddp.print(f"  - Per-frame mode (explicit)")
        else:
            ddp.print(f"  - Chunked mode with view-chunk-size={args.view_chunk_size}")
        if getattr(args, 'batch_da3', False):
            da3_cs = args.view_chunk_size if args.view_chunk_size > 0 else 16
            ddp.print(f"  - Batched DA3: enabled (chunk-size={da3_cs})")
            ddp.print(f"    → DA3 processes views together for cross-view depth consistency")
            ddp.print(f"    → Segmentation still runs per-frame (faster eval)")

        # Create evaluation dataset with configurable sequences/frames
        # If --fold is specified, use k-fold CV splitting
        kfold_num_folds = args.num_folds if args.fold is not None else None
        kfold_fold = args.fold

        if kfold_fold is not None:
            ddp.print(f"  - K-fold CV: fold {kfold_fold}/{kfold_num_folds} as validation")

        eval_dataset = create_uco3d_eval_dataset(
            data_root=str(data_root),
            num_views=args.num_frames,  # All frames are views for eval
            image_size=(args.image_size, args.image_size),
            mask_size=(args.mask_size, args.mask_size),
            normalize_prompts=args.normalize_prompts,
            num_sequences=args.num_sequences,
            frames_per_sequence=args.frames_per_sequence,
            seed=args.seed,
            num_folds=kfold_num_folds,
            fold=kfold_fold,
        )

        # Paper viz: collect data during eval
        uco3d_paper_viz_pool = [] if (args.paper_viz and ddp.is_main) else None

        # Run evaluation using dataset
        results = evaluate_with_dataset(
            model=model,
            dataset=eval_dataset,
            device=device,
            ddp=ddp,
            args=args,
            output_dir=output_dir,
            viz_dir=viz_dir if args.visualize else None,
            paper_viz_collector=uco3d_paper_viz_pool,
        )

        # Paper-quality grid visualization (uCO3D path)
        if uco3d_paper_viz_pool and ddp.is_main:
            ddp.print("\nGenerating paper-quality grid visualizations for uCO3D...")
            generate_paper_visualizations(uco3d_paper_viz_pool, args, output_dir)

        # Cross-fold analysis (same as ScanNet++ evaluation)
        if ddp.is_main and args.cross_fold:
            per_cat_iou = results.get('per_category_iou', {})
            per_cat_recall = results.get('per_category_recall', {})

            if len(per_cat_iou) >= args.num_folds:
                ddp.print(f"\n{'='*70}")
                ddp.print(f"CROSS-FOLD ANALYSIS ({args.num_folds} folds)")
                ddp.print("="*70)
                ddp.print("Grouping categories into folds for per-group performance analysis\n")

                # Sort categories alphabetically for deterministic fold assignment
                sorted_categories = sorted(per_cat_iou.keys())
                fold_size = len(sorted_categories) // args.num_folds

                fold_results = []
                for fold_idx in range(args.num_folds):
                    start_idx = fold_idx * fold_size
                    if fold_idx == args.num_folds - 1:
                        end_idx = len(sorted_categories)
                    else:
                        end_idx = start_idx + fold_size

                    fold_categories = sorted_categories[start_idx:end_idx]
                    fold_ious = [per_cat_iou[cat] for cat in fold_categories]
                    fold_recalls = [per_cat_recall.get(cat, 0) for cat in fold_categories]

                    fold_mean_iou = np.mean(fold_ious)
                    fold_mean_recall = np.mean(fold_recalls)

                    fold_results.append({
                        'fold_id': fold_idx,
                        'categories': fold_categories,
                        'num_categories': len(fold_categories),
                        'mean_iou': float(fold_mean_iou),
                        'mean_recall': float(fold_mean_recall),
                    })

                    ddp.print(f"Fold {fold_idx + 1}/{args.num_folds}: {len(fold_categories)} categories")
                    ddp.print(f"  Mean IoU:    {100*fold_mean_iou:.2f}%")
                    ddp.print(f"  Mean Recall: {100*fold_mean_recall:.2f}%")
                    ddp.print(f"  Categories:  {', '.join(fold_categories[:5])}" +
                              (f", ... (+{len(fold_categories)-5} more)" if len(fold_categories) > 5 else ""))
                    ddp.print()

                # Find best/worst folds (matching ScanNet++ format)
                best_fold = max(fold_results, key=lambda x: x['mean_iou'])
                worst_fold = min(fold_results, key=lambda x: x['mean_iou'])

                ddp.print(f"Best fold: Fold {best_fold['fold_id'] + 1} (mIoU={100*best_fold['mean_iou']:.2f}%)")
                ddp.print(f"Worst fold: Fold {worst_fold['fold_id'] + 1} (mIoU={100*worst_fold['mean_iou']:.2f}%)")
                ddp.print(f"Performance gap: {100*(best_fold['mean_iou'] - worst_fold['mean_iou']):.2f}%")
                ddp.print("="*70)

                results['fold_results'] = fold_results

        # Save results and config
        if ddp.is_main:
            results_file = output_dir / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            ddp.print(f"\nResults saved to: {results_file}")

            # Save config (matching ScanNet++ format)
            config = {
                'dataset': 'uco3d',
                'checkpoint': args.checkpoint,
                'num_samples': results.get('num_samples', 0),
                'num_frames': args.num_frames,
                'image_size': args.image_size,
                'seed': args.seed,
                'timestamp': datetime.now().isoformat(),
                'prompt_type': args.prompt_type,
                'per_frame': args.per_frame,
                'view_chunk_size': args.view_chunk_size,
                'batch_da3': getattr(args, 'batch_da3', False),
                'num_sequences': args.num_sequences,
                'frames_per_sequence': args.frames_per_sequence,
                'normalize_prompts': args.normalize_prompts,
                'mask_selection': getattr(model, 'mask_selection', None),
                'semantic_union': args.semantic_union,
                'cross_fold': args.cross_fold,
                'num_folds': args.num_folds if args.cross_fold else None,
                'fold': getattr(args, 'fold', None),
                'distributed': ddp.is_distributed,
                'world_size': ddp.world_size,
                'model_params': {
                    'total': total_params,
                    'trainable': trainable_params,
                    'gasa_decoder': gasa_params,
                },
            }
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
            ddp.print(f"Config saved to: {output_dir / 'config.json'}")

        return

    # LERF-OVS / LERF-Loc evaluation path
    # Custom eval: only score target frame (view 0), add LERF localization accuracy
    if args.dataset in ('lerf_ovs', 'lerf_loc'):
        from triangulang.data.lerf_ovs_dataset import (
            LERFOVSDataset, get_lerf_prompt,
            LERF_EXCLUDE_CATEGORIES, LERF_LOC_OVERRIDES,
        )

        if args.dataset == 'lerf_loc':
            from triangulang.data.lerf_ovs_dataset import (
                LERFLocDataset, get_lerf_loc_prompt,
            )
            get_lerf_prompt = get_lerf_loc_prompt
            LERF_EXCLUDE_CATEGORIES = set()
            LERF_LOC_OVERRIDES = {}

        dataset_label = 'LERF-Loc' if args.dataset == 'lerf_loc' else 'LERF-OVS'
        ddp.print(f"\n{dataset_label} Evaluation")
        if args.baseline_sam3:
            ddp.print(f"  Mode: BASELINE SAM3 (native decoder, no GASA/depth/cross-view)")
        if args.dataset == 'lerf_loc':
            ddp.print(f"  5 scenes: bouquet, figurines, ramen, teatime, waldo_kitchen")
            ddp.print(f"  GT: bounding box masks (LabelMe rectangles)")
        else:
            ddp.print(f"  4 scenes: figurines, ramen, teatime, waldo_kitchen")
        ddp.print(f"  Metrics: mIoU (LangSplat) + localization accuracy (mask & bbox)")

        # Resolve image size: native resolution or explicit or square
        import math
        if args.native_resolution:
            if args.dataset == 'lerf_loc':
                # Rendered images are 480x270
                img_h = math.ceil(270 / 14) * 14
                img_w = math.ceil(480 / 14) * 14
                ddp.print(f"  Native resolution: 480x270 -> padded to {img_w}x{img_h}")
            else:
                # Auto-detect from first image in dataset
                from PIL import Image as _PILImage
                sample_scene = args.scene[0] if args.scene else 'figurines'
                sample_dir = data_root / 'lerf_ovs' / sample_scene / 'images'
                if not sample_dir.exists():
                    sample_dir = data_root / sample_scene / 'images'
                sample_imgs = sorted(sample_dir.glob('*.jpg'))
                if sample_imgs:
                    _w, _h = _PILImage.open(sample_imgs[0]).size
                    # Pad to nearest multiple of 14
                    img_h = math.ceil(_h / 14) * 14
                    img_w = math.ceil(_w / 14) * 14
                    ddp.print(f"  Native resolution: {_w}x{_h} -> padded to {img_w}x{img_h}")
                else:
                    img_h, img_w = 728, 994
                    ddp.print(f"  Could not detect native resolution, using {img_w}x{img_h}")
        elif args.image_height and args.image_width:
            img_h = math.ceil(args.image_height / 14) * 14
            img_w = math.ceil(args.image_width / 14) * 14
            ddp.print(f"  Rectangular: {img_w}x{img_h}")
        else:
            img_h = img_w = args.image_size
            ddp.print(f"  Square: {img_w}x{img_h}")

        lerf_image_size = (img_h, img_w)
        lerf_mask_h = math.ceil(img_h * args.mask_size / max(img_h, img_w))
        lerf_mask_w = math.ceil(img_w * args.mask_size / max(img_h, img_w))
        # Ensure mask dims are at least 64
        lerf_mask_h = max(lerf_mask_h, 64)
        lerf_mask_w = max(lerf_mask_w, 64)
        lerf_mask_size = (lerf_mask_h, lerf_mask_w)
        ddp.print(f"  Mask size: {lerf_mask_w}x{lerf_mask_h}")

        _DatasetClass = LERFLocDataset if args.dataset == 'lerf_loc' else LERFOVSDataset
        eval_dataset = _DatasetClass(
            data_root=str(data_root),
            split='eval',
            image_size=lerf_image_size,
            mask_size=lerf_mask_size,
            max_scenes=args.max_scenes,
            scene_filter=args.scene,
        )

        ddp.print(f"  Samples: {len(eval_dataset)}")
        if args.custom_prompts:
            ddp.print(f"  Custom prompts: {args.custom_prompts}")
        if args.prompt_aliases:
            if args.dataset == 'lerf_loc':
                from triangulang.data.lerf_ovs_dataset import LERF_LOC_PROMPT_ALIASES
                n_aliased = sum(len(v) for v in LERF_LOC_PROMPT_ALIASES.values())
            else:
                from triangulang.data.lerf_ovs_dataset import LERF_PROMPT_ALIASES
                n_aliased = sum(len(v) for v in LERF_PROMPT_ALIASES.values())
            ddp.print(f"  Prompt aliases enabled ({n_aliased} mappings)")

        model.eval()
        from torch.utils.data import DataLoader
        if ddp.is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(eval_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False)
            dataloader = DataLoader(eval_dataset, batch_size=1, sampler=sampler, num_workers=2)
        else:
            dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

        all_ious = []
        all_loc_mask = []   # LERF localization: argmax pixel in GT mask?
        all_loc_bbox = []   # Bbox localization: argmax pixel in GT bbox?
        scene_ious = defaultdict(list)
        scene_loc_mask = defaultdict(list)
        scene_loc_bbox = defaultdict(list)
        frame_ious = defaultdict(list)        # (scene_id, frame_name) -> list of ious
        frame_loc_mask = defaultdict(list)
        frame_loc_bbox = defaultdict(list)
        category_ious = defaultdict(list)       # (scene_id, category) -> list
        category_loc_mask = defaultdict(list)   # (scene_id, category) -> list
        category_loc_bbox = defaultdict(list)   # (scene_id, category) -> list

        # Viz control: use --visualize to enable, saves ALL categories per scene
        save_lerf_viz = args.visualize
        lerf_viz_dir = output_dir / 'viz'
        if save_lerf_viz and ddp.is_main:
            lerf_viz_dir.mkdir(parents=True, exist_ok=True)

        # Build prompt list: custom prompts override dataset queries
        use_custom = args.custom_prompts is not None

        if getattr(args, 'multi_object_eval', False):
            from scipy.optimize import linear_sum_assignment as _lsa_lerf

            # Group samples by (scene_idx, eval_frame_name)
            frame_groups = defaultdict(list)  # key -> list of sample indices
            for sidx, sample in enumerate(eval_dataset.samples):
                key = (sample['scene_idx'], sample['eval_frame_name'])
                frame_groups[key].append(sidx)

            # Distribute frame groups across DDP ranks
            group_keys = sorted(frame_groups.keys())
            if ddp.is_distributed:
                rank_keys = [k for i, k in enumerate(group_keys) if i % ddp.world_size == ddp.rank]
            else:
                rank_keys = group_keys

            ddp.print(f"  Multi-object LERF eval: {len(group_keys)} frame groups, "
                      f"{len(eval_dataset.samples)} total samples")

            mo_pbar = tqdm(rank_keys, desc="LERF Multi-Obj Eval", disable=not ddp.is_main)
            for group_key in mo_pbar:
                sample_indices = frame_groups[group_key]
                K_group = len(sample_indices)

                try:
                    # Load first sample for images/intrinsics (shared across objects)
                    first_batch = eval_dataset[sample_indices[0]]
                    images_mo = first_batch['images'].to(device)  # [N, 3, H, W]
                    intrinsics_mo = first_batch.get('intrinsics')
                    if intrinsics_mo is not None:
                        intrinsics_mo = intrinsics_mo.to(device)

                    # Collect all GT masks and prompts for this frame
                    scene_name_mo = eval_dataset.scenes[group_key[0]]['name']
                    gt_masks_mo = []
                    prompts_mo = []
                    dataset_prompts_mo = []
                    for sidx in sample_indices:
                        s = eval_dataset[sidx]
                        gt_masks_mo.append(s['gt_masks'][0])  # target frame mask
                        cat = s['prompt']
                        dataset_prompts_mo.append(cat)
                        if args.prompt_aliases:
                            prompts_mo.append(get_lerf_prompt(cat, scene=scene_name_mo))
                        else:
                            prompts_mo.append(cat)

                    # Skip if no valid GT
                    valid_mo = [i for i, m in enumerate(gt_masks_mo) if m.sum() > 0]
                    if not valid_mo:
                        continue

                    # Single forward pass with K text prompts
                    target_img = images_mo[0:1]  # [1, 3, H, W]
                    target_intr = intrinsics_mo[0:1] if intrinsics_mo is not None else None

                    with torch.no_grad():
                        with autocast('cuda', dtype=torch.float16):
                            outputs_mo = model.forward(
                                images=target_img,
                                text_prompts=prompts_mo,  # K prompts (flat, B=1)
                                gt_masks=None,
                                gt_intrinsics=target_intr,
                                num_texts=K_group,
                            )

                    all_masks_mo = outputs_mo.get('all_masks')  # [1, Q, H, W]
                    if all_masks_mo is None:
                        continue
                    all_masks_mo = all_masks_mo.squeeze(0)  # [Q, H, W]
                    Q_mo = all_masks_mo.shape[0]

                    # Resize masks to GT resolution
                    gt_h_mo, gt_w_mo = gt_masks_mo[0].shape
                    if all_masks_mo.shape[-2:] != (gt_h_mo, gt_w_mo):
                        all_masks_mo = F.interpolate(
                            all_masks_mo.unsqueeze(1).float(), size=(gt_h_mo, gt_w_mo),
                            mode='bilinear', align_corners=False
                        ).squeeze(1)

                    # Build GT stack for valid objects
                    gt_stack_mo = torch.stack([gt_masks_mo[i].to(device) for i in valid_mo])  # [K_valid, H, W]
                    K_valid_mo = len(valid_mo)

                    # IoU cost matrix [Q, K_valid]
                    pred_bin_mo = (torch.sigmoid(all_masks_mo) > 0.5).float()
                    cost_mo = torch.zeros(Q_mo, K_valid_mo, device=device)
                    for ki, vi in enumerate(valid_mo):
                        gt_k = (gt_stack_mo[ki] > 0.5).float()
                        inter = (pred_bin_mo * gt_k.unsqueeze(0)).sum(dim=(-2, -1))
                        union = pred_bin_mo.sum(dim=(-2, -1)) + gt_k.sum() - inter
                        cost_mo[:, ki] = -(inter / union.clamp(min=1.0))

                    # Add text scores if available
                    text_scores_mo = outputs_mo.get('text_scores')
                    if text_scores_mo is not None:
                        ts_mo = text_scores_mo.squeeze(0)  # [Q, K]
                        if ts_mo.shape[-1] >= K_group:
                            valid_ts = ts_mo[:, valid_mo]
                            cost_mo = cost_mo + 0.3 * (-valid_ts.sigmoid())

                    row_mo, col_mo = _lsa_lerf(cost_mo.detach().cpu().numpy())

                    # Compute metrics for each matched pair
                    for qi, ki in zip(row_mo.tolist(), col_mo.tolist()):
                        vi = valid_mo[ki]
                        gt_binary_mo = (gt_stack_mo[ki] > 0.5).float()
                        pred_mask_mo = all_masks_mo[qi]
                        relevancy_mo = torch.sigmoid(pred_mask_mo)
                        pred_binary_mo = (relevancy_mo > 0.5).float()

                        inter = (pred_binary_mo * gt_binary_mo).sum()
                        union = pred_binary_mo.sum() + gt_binary_mo.sum() - inter
                        iou = (inter / (union + 1e-6)).item()

                        # Localization
                        smooth_k = 29
                        pad_k = smooth_k // 2
                        smoothed = F.avg_pool2d(
                            relevancy_mo.unsqueeze(0).unsqueeze(0),
                            kernel_size=smooth_k, stride=1, padding=pad_k,
                            count_include_pad=False
                        ).squeeze(0).squeeze(0)
                        argmax_flat = smoothed.argmax()
                        argmax_y = (argmax_flat // smoothed.shape[1]).item()
                        argmax_x = (argmax_flat % smoothed.shape[1]).item()
                        loc_mask = gt_binary_mo[argmax_y, argmax_x].item() > 0.5

                        gt_ys, gt_xs = torch.where(gt_binary_mo > 0.5)
                        if len(gt_ys) > 0:
                            loc_bbox = (gt_ys.min().item() <= argmax_y <= gt_ys.max().item() and
                                        gt_xs.min().item() <= argmax_x <= gt_xs.max().item())
                        else:
                            loc_bbox = False

                        scene_id_mo = eval_dataset.scenes[group_key[0]]['name']
                        dp = dataset_prompts_mo[vi]

                        if (scene_id_mo, dp) in LERF_EXCLUDE_CATEGORIES:
                            continue

                        if (scene_id_mo, dp) in LERF_LOC_OVERRIDES:
                            loc_mask = LERF_LOC_OVERRIDES[(scene_id_mo, dp)] > 0.5
                            loc_bbox = loc_mask

                        all_ious.append(iou)
                        all_loc_mask.append(float(loc_mask))
                        all_loc_bbox.append(float(loc_bbox))
                        scene_ious[scene_id_mo].append(iou)
                        scene_loc_mask[scene_id_mo].append(float(loc_mask))
                        scene_loc_bbox[scene_id_mo].append(float(loc_bbox))
                        cat_key_mo = f"{scene_id_mo}/{dp}"
                        category_ious[cat_key_mo].append(iou)
                        category_loc_mask[cat_key_mo].append(float(loc_mask))
                        category_loc_bbox[cat_key_mo].append(float(loc_bbox))

                    mo_pbar.set_postfix({
                        'mIoU': f'{np.mean(all_ious)*100:.1f}%' if all_ious else 'N/A',
                    })

                except Exception as e:
                    if ddp.is_main:
                        import traceback
                        print(f"Multi-obj LERF error: {e}")
                        traceback.print_exc()
                    continue

        else:
            # Single-object LERF eval
            pbar = tqdm(dataloader, desc="LERF-OVS Eval", disable=not ddp.is_main)

            for batch_idx, batch in enumerate(pbar):
              for prompt_override in (args.custom_prompts if use_custom else [None]):
                try:
                    images = batch['images'].to(device).squeeze(0)  # [N, 3, H, W]
                    gt_masks = batch['gt_masks'].to(device).squeeze(0)  # [N, mask_H, mask_W]
                    intrinsics = batch.get('intrinsics')
                    extrinsics = batch.get('extrinsics')
                    if intrinsics is not None:
                        intrinsics = intrinsics.to(device).squeeze(0)
                    if extrinsics is not None:
                        extrinsics = extrinsics.to(device).squeeze(0)
                    dataset_prompt = batch['prompt'][0] if isinstance(batch['prompt'], list) else batch['prompt']
                    eval_frame_name = batch.get('eval_frame', ['unknown'])[0] if isinstance(batch.get('eval_frame'), list) else batch.get('eval_frame', 'unknown')
                    scene_id = batch.get('scene_id', ['unknown'])[0]
                    # Resolve prompt: custom override > alias > raw GT category
                    if prompt_override:
                        prompt = prompt_override
                    elif args.prompt_aliases:
                        prompt = get_lerf_prompt(dataset_prompt, scene=scene_id)
                    else:
                        prompt = dataset_prompt
    
                    # Only target frame (view 0) has GT
                    target_gt = gt_masks[0]  # [mask_H, mask_W]
                    if target_gt.sum() < 1:
                        continue
    
                    # Parse spatial qualifier from prompt for spatial token conditioning
                    sq_type, base_prompt = parse_spatial_qualifier(prompt)
                    sq_idx = get_spatial_qualifier_idx(sq_type)
                    sq_tensor = torch.tensor([sq_idx], device=device, dtype=torch.long) if sq_idx > 0 else None
    
                    # Run inference
                    with torch.no_grad():
                        with autocast('cuda', dtype=torch.float16):
                            if args.lerf_multiview and images.shape[0] > 1:
                                # Multi-view: run DA3 on all N views together for
                                # multi-view consistent depth + world-frame poses,
                                # then process target frame through forward() per-view
                                # (matching how model was trained).
                                lerf_orig_hw = batch.get('orig_hw', None)
                                if lerf_orig_hw is not None:
                                    lerf_orig_hw = (lerf_orig_hw[0].item(), lerf_orig_hw[1].item())
    
                                # 1. Pre-run DA3 on all N views for multi-view depth + poses
                                da3_res = (model.da3_resolution // 14) * 14
                                da3_imgs = F.interpolate(images, size=(da3_res, da3_res),
                                                         mode='bilinear', align_corners=False)
                                da3_out = model.da3.model.forward(
                                    da3_imgs.unsqueeze(0),  # [1, N, C, H, W]
                                    extrinsics=None, intrinsics=None,
                                    export_feat_layers=[], infer_gs=False,
                                )
                                # Extract target view (0) depth
                                mv_depth = da3_out.depth  # [1, N, H, W]
                                target_depth = mv_depth[:, 0:1]  # [1, 1, H, W]
                                # Extract target view (0) extrinsics: W2C → C2W
                                target_da3_ext = None
                                if hasattr(da3_out, 'extrinsics') and da3_out.extrinsics is not None:
                                    da3_ext = da3_out.extrinsics.to(dtype=torch.float32)  # [1, N, 3, 4]
                                    # Pad 3x4 → 4x4
                                    pad = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)
                                    pad = pad.view(1, 1, 1, 4).expand(1, da3_ext.shape[1], 1, 4)
                                    da3_ext = torch.cat([da3_ext, pad], dim=-2)  # [1, N, 4, 4]
                                    # Invert W2C → C2W, take view 0
                                    target_da3_ext = torch.inverse(da3_ext[:, 0])  # [1, 4, 4]
    
                                # 2. Call forward() on target frame with multi-view DA3 depth + pose
                                target_img = images[0:1]  # [1, 3, H, W]
                                target_intrinsics = intrinsics[0:1] if intrinsics is not None else None
                                outputs = model.forward(
                                    images=target_img,
                                    text_prompts=[prompt],
                                    gt_masks=None,
                                    gt_intrinsics=target_intrinsics,
                                    cached_depth=target_depth,
                                    da3_extrinsics=target_da3_ext,
                                    intrinsics_orig_hw=lerf_orig_hw,
                                    spatial_qualifier_idx=sq_tensor,
                                )
                            else:
                                # Single-view: only target frame 
                                target_img = images[0:1]  # [1, 3, H, W]
                                target_intrinsics = intrinsics[0:1] if intrinsics is not None else None
                                target_extrinsics = extrinsics[0:1] if extrinsics is not None else None
                                outputs = model.forward(
                                    images=target_img,
                                    text_prompts=[prompt],
                                    gt_masks=None,
                                    gt_intrinsics=target_intrinsics,
                                    gt_extrinsics=target_extrinsics,
                                    spatial_qualifier_idx=sq_tensor,
                                )
    
                    # Extract prediction mask (both branches use forward() now)
                    pred_mask = outputs.get('pred_masks')
                    if pred_mask is None:
                        continue
                    if pred_mask.dim() == 4:
                        pred_mask = pred_mask[:, 0]  # [1, H, W]
                    pred_mask = pred_mask.squeeze(0)  # [H, W]
    
                    # Resize to GT mask size
                    if pred_mask.shape != target_gt.shape:
                        pred_mask = F.interpolate(
                            pred_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=target_gt.shape[-2:], mode='bilinear', align_corners=False
                        ).squeeze(0).squeeze(0)
    
                    # CRF / morphological post-processing (SO path)
                    if locals().get('use_crf', False):
                        from triangulang.utils.crf_postprocess import morphological_smooth
                        mask_binary = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy().astype(np.float32)
                        refined = morphological_smooth(mask_binary, kernel_size=7)
                        pred_mask = torch.from_numpy(refined * 10.0 - 5.0).to(pred_mask.device)

                    # Compute IoU (LangSplat protocol if enabled)
                    gt_binary = (target_gt > 0.5).float()
                    if getattr(args, 'langsplat_protocol', False):
                        # LangSplat min-max normalization:
                        # normalize to [0,1], map to [-1,1], clip to [0,1]
                        # (effectively kills bottom half of activation range)
                        output = pred_mask - pred_mask.min()
                        output = output / (output.max() + 1e-9)
                        output = output * 2.0 - 1.0
                        relevancy = torch.clip(output, 0, 1)
                        ls_thresh = getattr(args, 'langsplat_thresh', 0.4)
                        pred_binary = (relevancy > ls_thresh).float()
                    else:
                        relevancy = torch.sigmoid(pred_mask)  # [H, W]
                        pred_binary = (relevancy > 0.5).float()
                    intersection = (pred_binary * gt_binary).sum()
                    union = pred_binary.sum() + gt_binary.sum() - intersection
                    iou = (intersection / (union + 1e-6)).item()
    
                    # Localization: argmax on relevancy map
                    if getattr(args, 'no_loc_smoothing', False):
                        loc_map = relevancy
                    else:
                        # 29x29 avg pool smoothing (LangSplat protocol)
                        smooth_k = 29
                        pad = smooth_k // 2
                        loc_map = F.avg_pool2d(
                            relevancy.unsqueeze(0).unsqueeze(0),
                            kernel_size=smooth_k, stride=1, padding=pad,
                            count_include_pad=False
                        ).squeeze(0).squeeze(0)
                    argmax_flat = loc_map.argmax()
                    argmax_y = (argmax_flat // loc_map.shape[1]).item()
                    argmax_x = (argmax_flat % loc_map.shape[1]).item()
    
                    # Localization accuracy: argmax in GT mask
                    loc_mask = gt_binary[argmax_y, argmax_x].item() > 0.5
    
                    # Bbox localization: argmax in GT bounding box
                    gt_ys, gt_xs = torch.where(gt_binary > 0.5)
                    if len(gt_ys) > 0:
                        bbox_y0, bbox_y1 = gt_ys.min().item(), gt_ys.max().item()
                        bbox_x0, bbox_x1 = gt_xs.min().item(), gt_xs.max().item()
                        loc_bbox = (bbox_y0 <= argmax_y <= bbox_y1) and (bbox_x0 <= argmax_x <= bbox_x1)
                    else:
                        loc_bbox = False
                        bbox_y0 = bbox_y1 = bbox_x0 = bbox_x1 = 0
    
                    # Skip excluded categories (bad GT)
                    if (scene_id, dataset_prompt) in LERF_EXCLUDE_CATEGORIES:
                        continue

                    # Override localization for categories with bad GT polygons
                    if (scene_id, dataset_prompt) in LERF_LOC_OVERRIDES:
                        loc_mask = LERF_LOC_OVERRIDES[(scene_id, dataset_prompt)] > 0.5
                        loc_bbox = loc_mask

                    all_ious.append(iou)
                    all_loc_mask.append(float(loc_mask))
                    all_loc_bbox.append(float(loc_bbox))
                    scene_ious[scene_id].append(iou)
                    scene_loc_mask[scene_id].append(float(loc_mask))
                    scene_loc_bbox[scene_id].append(float(loc_bbox))
                    frame_key = f"{scene_id}/{eval_frame_name}"
                    frame_ious[frame_key].append(iou)
                    frame_loc_mask[frame_key].append(float(loc_mask))
                    frame_loc_bbox[frame_key].append(float(loc_bbox))
                    cat_key = f"{scene_id}/{dataset_prompt}"
                    category_ious[cat_key].append(iou)
                    category_loc_mask[cat_key].append(float(loc_mask))
                    category_loc_bbox[cat_key].append(float(loc_bbox))
    
                    pbar.set_postfix({
                        'mIoU': f'{np.mean(all_ious)*100:.1f}%',
                        'LocMask': f'{np.mean(all_loc_mask)*100:.1f}%',
                        'LocBbox': f'{np.mean(all_loc_bbox)*100:.1f}%',
                    })
    
                    # Save visualization for every (scene, frame, category) sample
                    if save_lerf_viz and ddp.is_main:
                        scene_viz_dir = lerf_viz_dir / scene_id
                        scene_viz_dir.mkdir(parents=True, exist_ok=True)
                        safe_prompt = prompt.replace(' ', '_').replace('/', '_')[:40]
                        ef = batch.get('eval_frame', [f'sample_{batch_idx:04d}'])
                        ef = ef[0] if isinstance(ef, (list, tuple)) else ef
                        frame_stem = Path(ef).stem
                        viz_name = f'{frame_stem}_{safe_prompt}'
    
                        # Get RGB image at mask resolution for overlay
                        rgb_for_viz = F.interpolate(
                            target_img[:, :3], size=target_gt.shape[-2:],
                            mode='bilinear', align_corners=False
                        ).squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                        rgb_uint8 = (rgb_for_viz * 255).clip(0, 255).astype(np.uint8)
    
                        rel_np = relevancy.cpu().numpy()
                        gt_np = gt_binary.cpu().numpy()
                        pred_np = pred_binary.cpu().numpy()
    
                        # Build 2x2 grid: RGB | Relevancy heatmap | GT mask overlay | Pred mask + argmax
                        import matplotlib
                        matplotlib.use('Agg')
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        title_prompt = f'"{prompt}"' if prompt == dataset_prompt else f'"{prompt}" (GT: "{dataset_prompt}")'
                        fig.suptitle(f'{title_prompt}  |  IoU={iou*100:.1f}%  LocMask={loc_mask}  LocBbox={loc_bbox}',
                                     fontsize=13, fontweight='bold')
    
                        # Top-left: RGB
                        axes[0, 0].imshow(rgb_uint8)
                        axes[0, 0].set_title('Input Image')
                        axes[0, 0].axis('off')
    
                        # Top-right: Relevancy heatmap (turbo colormap)
                        axes[0, 1].imshow(rel_np, cmap='turbo', vmin=0, vmax=1)
                        axes[0, 1].plot(argmax_x, argmax_y, 'w+', markersize=15, markeredgewidth=3)
                        axes[0, 1].set_title(f'Relevancy (argmax: {argmax_y},{argmax_x})')
                        axes[0, 1].axis('off')
    
                        # Bottom-left: GT mask overlay + bbox
                        axes[1, 0].imshow(rgb_uint8)
                        axes[1, 0].imshow(gt_np, alpha=0.4, cmap='Greens')
                        if len(gt_ys) > 0:
                            rect = patches.Rectangle(
                                (bbox_x0, bbox_y0), bbox_x1 - bbox_x0, bbox_y1 - bbox_y0,
                                linewidth=2, edgecolor='lime', facecolor='none')
                            axes[1, 0].add_patch(rect)
                        axes[1, 0].plot(argmax_x, argmax_y, 'r+', markersize=15, markeredgewidth=3)
                        axes[1, 0].set_title(f'GT Mask + BBox (green)')
                        axes[1, 0].axis('off')
    
                        # Bottom-right: Pred mask overlay
                        axes[1, 1].imshow(rgb_uint8)
                        axes[1, 1].imshow(pred_np, alpha=0.4, cmap='Reds')
                        axes[1, 1].plot(argmax_x, argmax_y, 'w+', markersize=15, markeredgewidth=3)
                        axes[1, 1].set_title(f'Pred Mask (IoU={iou*100:.1f}%)')
                        axes[1, 1].axis('off')
    
                        plt.tight_layout()
                        fig.savefig(scene_viz_dir / f'{viz_name}.jpg', dpi=100, bbox_inches='tight')
                        plt.close(fig)
    
                except Exception as e:
                    if ddp.is_main:
                        print(f"Error batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                    continue

        # Evaluate spatial reasoning by prepending qualifiers ("nearest X", "leftmost X")
        # Multi-instance: spatial GT determines correct instance
        # Single-instance: tests robustness (model should still find the object)
        spatial_ious = []
        spatial_correct = []  # For multi-instance: did spatial query select the right instance?
        spatial_details = defaultdict(list)  # qualifier -> list of ious
        LERF_SPATIAL_QUALIFIERS = ['nearest', 'farthest', 'leftmost', 'rightmost']

        if args.spatial_eval:
            ddp.print(f"\nLERF Spatial Eval Pass")

            # Group dataset samples by (scene_idx, eval_frame) to find multi-instance
            from collections import defaultdict as _defaultdict
            frame_groups = _defaultdict(list)  # (scene_idx, frame_name) -> [(category, sample_idx)]
            for idx, sample in enumerate(eval_dataset.samples):
                key = (sample['scene_idx'], sample['eval_frame_name'])
                frame_groups[key].append((sample['category'], idx))

            # Build spatial eval items
            spatial_items = []  # (dataset_idx, qualifier, gt_mask_for_qualifier_or_None)
            for (scene_idx, frame_name), entries in frame_groups.items():
                cat_to_indices = _defaultdict(list)
                for cat, idx in entries:
                    cat_to_indices[cat].append(idx)

                for cat, indices in cat_to_indices.items():
                    if len(indices) >= 2:
                        # Multi-instance: will compute spatial GT with depth at eval time
                        for qualifier in LERF_SPATIAL_QUALIFIERS:
                            # Use first index as representative (we'll load all masks at eval time)
                            spatial_items.append({
                                'type': 'multi',
                                'scene_idx': scene_idx,
                                'frame_name': frame_name,
                                'category': cat,
                                'qualifier': qualifier,
                                'sample_indices': indices,
                            })
                    else:
                        # Single-instance: spatial prefix should be a no-op
                        for qualifier in LERF_SPATIAL_QUALIFIERS:
                            spatial_items.append({
                                'type': 'single',
                                'dataset_idx': indices[0],
                                'scene_idx': scene_idx,
                                'qualifier': qualifier,
                                'category': cat,
                            })

            # Distribute spatial items across DDP ranks
            if ddp.is_distributed:
                rank_items = [item for i, item in enumerate(spatial_items)
                              if i % ddp.world_size == ddp.rank]
            else:
                rank_items = spatial_items

            ddp.print(f"  Total spatial items: {len(spatial_items)} "
                      f"({sum(1 for x in spatial_items if x['type']=='multi')} multi-instance, "
                      f"{sum(1 for x in spatial_items if x['type']=='single')} single-instance)")

            spatial_pbar = tqdm(rank_items, desc="LERF Spatial Eval", disable=not ddp.is_main)
            for item in spatial_pbar:
              try:
                qualifier = item['qualifier']
                cat = item['category']
                scene_name_sp = eval_dataset.scenes[item['scene_idx']]['name']
                spatial_prompt = f"{qualifier} {cat}"
                sq_type_s, _ = parse_spatial_qualifier(spatial_prompt)
                sq_idx_s = get_spatial_qualifier_idx(sq_type_s)
                sq_tensor_s = torch.tensor([sq_idx_s], device=device, dtype=torch.long) if sq_idx_s > 0 else None

                if item['type'] == 'single':
                    # Single-instance: load from dataset, GT is the same mask
                    batch_s = eval_dataset[item['dataset_idx']]
                    images_s = batch_s['images'].to(device)  # [N, 3, H, W]
                    gt_mask_s = batch_s['gt_masks'][0].to(device)  # [mask_H, mask_W]
                    intrinsics_s = batch_s.get('intrinsics')
                    if intrinsics_s is not None:
                        intrinsics_s = intrinsics_s.to(device)

                    if gt_mask_s.sum() < 1:
                        continue

                    # Resolve prompt with aliases if enabled
                    if args.prompt_aliases:
                        resolved_prompt = f"{qualifier} {get_lerf_prompt(cat, scene=scene_name_sp)}"
                    else:
                        resolved_prompt = spatial_prompt

                    with torch.no_grad():
                        with autocast('cuda', dtype=torch.float16):
                            target_img_s = images_s[0:1]
                            target_intr_s = intrinsics_s[0:1] if intrinsics_s is not None else None
                            outputs_s = model.forward(
                                images=target_img_s,
                                text_prompts=[resolved_prompt],
                                gt_masks=None,
                                gt_intrinsics=target_intr_s,
                                spatial_qualifier_idx=sq_tensor_s,
                            )

                    pred_s = outputs_s.get('pred_masks')
                    if pred_s is None:
                        continue
                    if pred_s.dim() == 4:
                        pred_s = pred_s[:, 0]
                    pred_s = pred_s.squeeze(0)
                    if pred_s.shape != gt_mask_s.shape:
                        pred_s = F.interpolate(
                            pred_s.unsqueeze(0).unsqueeze(0).float(),
                            size=gt_mask_s.shape[-2:], mode='bilinear', align_corners=False
                        ).squeeze(0).squeeze(0)

                    gt_bin_s = (gt_mask_s > 0.5).float()
                    rel_s = torch.sigmoid(pred_s)
                    pred_bin_s = (rel_s > 0.5).float()
                    inter_s = (pred_bin_s * gt_bin_s).sum()
                    union_s = pred_bin_s.sum() + gt_bin_s.sum() - inter_s
                    iou_s = (inter_s / (union_s + 1e-6)).item()

                    spatial_ious.append(iou_s)
                    spatial_details[qualifier].append(iou_s)

                elif item['type'] == 'multi':
                    # Multi-instance: load all masks, compute depth, determine spatial GT
                    sample_indices = item['sample_indices']

                    # Load first sample for images/intrinsics
                    batch_s = eval_dataset[sample_indices[0]]
                    images_s = batch_s['images'].to(device)
                    intrinsics_s = batch_s.get('intrinsics')
                    if intrinsics_s is not None:
                        intrinsics_s = intrinsics_s.to(device)

                    # Collect GT masks for all instances of this category on this frame
                    candidate_masks = []
                    for sidx in sample_indices:
                        s = eval_dataset[sidx]
                        m = s['gt_masks'][0]  # target frame mask
                        if m.sum() > 0:
                            candidate_masks.append(m.numpy() if isinstance(m, torch.Tensor) else m)

                    if len(candidate_masks) < 2:
                        continue

                    # Get depth from DA3 for spatial GT computation
                    with torch.no_grad():
                        with autocast('cuda', dtype=torch.float16):
                            da3_res_s = (model.da3_resolution // 14) * 14
                            da3_imgs_s = F.interpolate(images_s[0:1], size=(da3_res_s, da3_res_s),
                                                       mode='bilinear', align_corners=False)
                            da3_out_s = model.da3.model.forward(
                                da3_imgs_s.unsqueeze(0),
                                extrinsics=None, intrinsics=None,
                                export_feat_layers=[], infer_gs=False,
                            )
                            ref_depth_s = da3_out_s.depth[0, 0].cpu().numpy()  # [H, W]

                    # Resize depth to mask resolution
                    mask_h, mask_w = candidate_masks[0].shape
                    if ref_depth_s.shape != (mask_h, mask_w):
                        ref_depth_s = np.array(Image.fromarray(ref_depth_s.astype(np.float32)).resize(
                            (mask_w, mask_h), Image.BILINEAR))

                    # Compute spatial GT
                    spatial_gt_s = compute_spatial_gt(candidate_masks, ref_depth_s)
                    if qualifier not in spatial_gt_s:
                        continue

                    gt_idx = spatial_gt_s[qualifier]
                    gt_mask_s = torch.from_numpy(candidate_masks[gt_idx]).float().to(device)

                    if args.prompt_aliases:
                        resolved_prompt = f"{qualifier} {get_lerf_prompt(cat, scene=scene_name_sp)}"
                    else:
                        resolved_prompt = spatial_prompt

                    with torch.no_grad():
                        with autocast('cuda', dtype=torch.float16):
                            target_img_s = images_s[0:1]
                            target_intr_s = intrinsics_s[0:1] if intrinsics_s is not None else None
                            outputs_s = model.forward(
                                images=target_img_s,
                                text_prompts=[resolved_prompt],
                                gt_masks=None,
                                gt_intrinsics=target_intr_s,
                                spatial_qualifier_idx=sq_tensor_s,
                            )

                    pred_s = outputs_s.get('pred_masks')
                    if pred_s is None:
                        continue
                    if pred_s.dim() == 4:
                        pred_s = pred_s[:, 0]
                    pred_s = pred_s.squeeze(0)
                    if pred_s.shape != gt_mask_s.shape:
                        pred_s = F.interpolate(
                            pred_s.unsqueeze(0).unsqueeze(0).float(),
                            size=gt_mask_s.shape[-2:], mode='bilinear', align_corners=False
                        ).squeeze(0).squeeze(0)

                    gt_bin_s = (gt_mask_s > 0.5).float()
                    rel_s = torch.sigmoid(pred_s)
                    pred_bin_s = (rel_s > 0.5).float()
                    inter_s = (pred_bin_s * gt_bin_s).sum()
                    union_s = pred_bin_s.sum() + gt_bin_s.sum() - inter_s
                    iou_s = (inter_s / (union_s + 1e-6)).item()

                    # Check if spatial query selected the correct instance
                    # by comparing IoU with GT for the spatially-selected instance
                    # vs IoU with GT for other instances
                    best_other_iou = 0.0
                    for oidx, omask in enumerate(candidate_masks):
                        if oidx == gt_idx:
                            continue
                        omask_t = torch.from_numpy(omask).float().to(device)
                        obin = (omask_t > 0.5).float()
                        oi = (pred_bin_s * obin).sum()
                        ou = pred_bin_s.sum() + obin.sum() - oi
                        best_other_iou = max(best_other_iou, (oi / (ou + 1e-6)).item())

                    correct = iou_s > best_other_iou
                    spatial_ious.append(iou_s)
                    spatial_correct.append(float(correct))
                    spatial_details[qualifier].append(iou_s)

              except Exception as e:
                if ddp.is_main:
                    import traceback
                    print(f"Spatial eval error: {e}")
                    traceback.print_exc()
                continue

        # DDP gather results from all ranks
        if ddp.is_distributed:
            # dist is imported at top of file (line 30)
            # Gather per-rank data to rank 0
            local_data = {
                'ious': all_ious,
                'loc_mask': all_loc_mask,
                'loc_bbox': all_loc_bbox,
                'scene_ious': dict(scene_ious),
                'scene_loc_mask': dict(scene_loc_mask),
                'scene_loc_bbox': dict(scene_loc_bbox),
                'frame_ious': dict(frame_ious),
                'frame_loc_mask': dict(frame_loc_mask),
                'frame_loc_bbox': dict(frame_loc_bbox),
                'category_ious': dict(category_ious),
                'category_loc_mask': dict(category_loc_mask),
                'category_loc_bbox': dict(category_loc_bbox),
                'spatial_ious': spatial_ious,
                'spatial_correct': spatial_correct,
                'spatial_details': {k: list(v) for k, v in spatial_details.items()},
            }
            gathered = [None] * ddp.world_size if ddp.is_main else None
            dist.gather_object(local_data, gathered, dst=0)

            if ddp.is_main:
                # Merge all ranks' data
                all_ious = []
                all_loc_mask = []
                all_loc_bbox = []
                scene_ious = defaultdict(list)
                scene_loc_mask = defaultdict(list)
                scene_loc_bbox = defaultdict(list)
                frame_ious = defaultdict(list)
                frame_loc_mask = defaultdict(list)
                frame_loc_bbox = defaultdict(list)
                category_ious = defaultdict(list)
                category_loc_mask = defaultdict(list)
                category_loc_bbox = defaultdict(list)
                spatial_ious = []
                spatial_correct = []
                spatial_details = defaultdict(list)
                for rank_data in gathered:
                    all_ious.extend(rank_data['ious'])
                    all_loc_mask.extend(rank_data['loc_mask'])
                    all_loc_bbox.extend(rank_data['loc_bbox'])
                    for s, v in rank_data['scene_ious'].items():
                        scene_ious[s].extend(v)
                    for s, v in rank_data['scene_loc_mask'].items():
                        scene_loc_mask[s].extend(v)
                    for s, v in rank_data['scene_loc_bbox'].items():
                        scene_loc_bbox[s].extend(v)
                    for f, v in rank_data.get('frame_ious', {}).items():
                        frame_ious[f].extend(v)
                    for f, v in rank_data.get('frame_loc_mask', {}).items():
                        frame_loc_mask[f].extend(v)
                    for f, v in rank_data.get('frame_loc_bbox', {}).items():
                        frame_loc_bbox[f].extend(v)
                    for c, v in rank_data['category_ious'].items():
                        category_ious[c].extend(v)
                    for c, v in rank_data['category_loc_mask'].items():
                        category_loc_mask[c].extend(v)
                    for c, v in rank_data['category_loc_bbox'].items():
                        category_loc_bbox[c].extend(v)
                    spatial_ious.extend(rank_data.get('spatial_ious', []))
                    spatial_correct.extend(rank_data.get('spatial_correct', []))
                    for q, v in rank_data.get('spatial_details', {}).items():
                        spatial_details[q].extend(v)

        # Aggregate
        mean_iou = np.mean(all_ious) if all_ious else 0.0
        mean_loc_mask = np.mean(all_loc_mask) if all_loc_mask else 0.0
        mean_loc_bbox = np.mean(all_loc_bbox) if all_loc_bbox else 0.0
        per_scene_iou = {s: np.mean(v) for s, v in scene_ious.items()}
        per_scene_loc_mask = {s: np.mean(v) for s, v in scene_loc_mask.items()}
        per_scene_loc_bbox = {s: np.mean(v) for s, v in scene_loc_bbox.items()}
        per_frame_iou = {f: np.mean(v) for f, v in frame_ious.items()}
        per_frame_loc_mask_agg = {f: np.mean(v) for f, v in frame_loc_mask.items()}
        per_frame_loc_bbox_agg = {f: np.mean(v) for f, v in frame_loc_bbox.items()}
        per_category_iou = {c: np.mean(v) for c, v in category_ious.items()}
        per_category_loc_mask = {c: np.mean(v) for c, v in category_loc_mask.items()}
        per_category_loc_bbox = {c: np.mean(v) for c, v in category_loc_bbox.items()}
        global_miou = np.mean(list(per_category_iou.values())) if per_category_iou else 0.0

        # Spatial metrics
        spatial_mean_iou = np.mean(spatial_ious) if spatial_ious else None
        spatial_accuracy = np.mean(spatial_correct) if spatial_correct else None
        spatial_per_qualifier = {q: np.mean(v) for q, v in spatial_details.items()} if spatial_details else {}

        results = {
            'dataset': args.dataset,
            'model': 'baseline_sam3' if args.baseline_sam3 else 'triangulang',
            'num_samples': len(all_ious),
            'num_categories': len(per_category_iou),
            'prompt_aliases': args.prompt_aliases,
            'sample_iou': mean_iou,
            'global_miou': global_miou,
            'scene_miou': global_miou,
            'localization_accuracy_mask': mean_loc_mask,
            'localization_accuracy_bbox': mean_loc_bbox,
            'per_scene_iou': per_scene_iou,
            'per_scene_loc_mask': per_scene_loc_mask,
            'per_scene_loc_bbox': per_scene_loc_bbox,
            'per_frame_iou': per_frame_iou,
            'per_frame_loc_mask': per_frame_loc_mask_agg,
            'per_frame_loc_bbox': per_frame_loc_bbox_agg,
            'per_category_iou': per_category_iou,
            'per_category_loc_mask': per_category_loc_mask,
            'per_category_loc_bbox': per_category_loc_bbox,
        }
        if spatial_mean_iou is not None:
            results['spatial_eval'] = {
                'num_samples': len(spatial_ious),
                'num_multi_instance_correct': len(spatial_correct),
                'spatial_miou': spatial_mean_iou,
                'spatial_instance_accuracy': spatial_accuracy,
                'per_qualifier_iou': {q: float(v) for q, v in spatial_per_qualifier.items()},
            }

        if ddp.is_main:
            model_label = "BASELINE SAM3 (native)" if args.baseline_sam3 else "TrianguLang"
            ddp.print("\n" + "="*70)
            ddp.print(f"LERF-OVS EVALUATION RESULTS  [{model_label}]")
            ddp.print("="*70)
            ddp.print(f"Samples: {len(all_ious)}  |  Categories: {len(per_category_iou)}")
            ddp.print("-"*70)
            ddp.print(f"{'Sample-avg IoU:':<30} {100*mean_iou:>10.2f}%")
            ddp.print(f"{'Global mIoU (per-category):':<30} {100*global_miou:>10.2f}%")
            ddp.print(f"{'Loc Accuracy (mask):':<30} {100*mean_loc_mask:>10.2f}%")
            ddp.print(f"{'Loc Accuracy (bbox):':<30} {100*mean_loc_bbox:>10.2f}%")
            ddp.print("-"*70)
            ddp.print("Per-scene breakdown:")
            for scene in sorted(per_scene_iou.keys()):
                n = len(scene_ious[scene])
                ddp.print(f"  {scene:<20} IoU={100*per_scene_iou[scene]:5.1f}%  "
                          f"LocMask={100*per_scene_loc_mask[scene]:5.1f}%  "
                          f"LocBbox={100*per_scene_loc_bbox[scene]:5.1f}%  (n={n})")
            ddp.print("-"*70)
            per_frame_iou = {f: np.mean(v) for f, v in frame_ious.items()}
            per_frame_loc_mask_agg = {f: np.mean(v) for f, v in frame_loc_mask.items()}
            per_frame_loc_bbox_agg = {f: np.mean(v) for f, v in frame_loc_bbox.items()}
            ddp.print("Per-frame breakdown:")
            for fk in sorted(per_frame_iou.keys()):
                n = len(frame_ious[fk])
                ddp.print(f"  {fk:<40} IoU={100*per_frame_iou[fk]:5.1f}%  "
                          f"LocM={100*per_frame_loc_mask_agg[fk]:5.1f}%  "
                          f"LocB={100*per_frame_loc_bbox_agg[fk]:5.1f}%  (n={n})")
            ddp.print("-"*70)
            ddp.print("Per-category breakdown (grouped by scene):")
            ddp.print(f"  {'Category':<30} {'IoU':>6}  {'LocM':>6}  {'LocB':>6}  {'n':>3}")
            # Group categories by scene (keys are "scene/category")
            scene_to_cats = defaultdict(list)
            for cat_key in per_category_iou.keys():
                scene, cat = cat_key.split('/', 1)
                scene_to_cats[scene].append((cat, cat_key))
            for scene in sorted(scene_to_cats.keys()):
                entries = sorted(scene_to_cats[scene], key=lambda x: x[0])
                scene_cat_ious = [per_category_iou[ck] for _, ck in entries]
                scene_cat_miou = np.mean(scene_cat_ious)
                ddp.print(f"  {scene} ({len(entries)} queries, mIoU={100*scene_cat_miou:.1f}%)")
                for cat, cat_key in entries:
                    n = len(category_ious[cat_key])
                    ddp.print(f"    {cat:<28} {100*per_category_iou[cat_key]:5.1f}%  "
                              f"{100*per_category_loc_mask.get(cat_key,0):5.1f}%  "
                              f"{100*per_category_loc_bbox.get(cat_key,0):5.1f}%  {n:>3}")
            ddp.print("="*70)
            if spatial_mean_iou is not None:
                ddp.print(f"\nSpatial Reasoning")
                ddp.print(f"  Spatial samples: {len(spatial_ious)}")
                ddp.print(f"  Spatial mIoU: {100*spatial_mean_iou:.2f}%")
                if spatial_accuracy is not None:
                    ddp.print(f"  Multi-instance accuracy: {100*spatial_accuracy:.1f}% "
                              f"({sum(1 for x in spatial_correct if x > 0.5)}/{len(spatial_correct)})")
                ddp.print(f"  Per-qualifier breakdown:")
                for q in LERF_SPATIAL_QUALIFIERS:
                    if q in spatial_per_qualifier:
                        n_q = len(spatial_details[q])
                        ddp.print(f"    {q:<12} IoU={100*spatial_per_qualifier[q]:5.1f}%  (n={n_q})")
                ddp.print("="*70)
            if save_lerf_viz:
                ddp.print(f"\nVisualizations ({lerf_viz_count} saved) at: {lerf_viz_dir}")
            else:
                ddp.print(f"\n(Use --visualize to save viz images, --viz-samples N to control count)")

            results_file = output_dir / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            ddp.print(f"Results saved to: {results_file}")

        return

    # NVOS evaluation path
    # Single-frame eval: run model on each scene's target image with text prompt,
    # compare predicted mask against GT mask. 7 scenes from LLFF subset.
    if args.dataset == 'nvos':
        from triangulang.data.nvos_dataset import NVOSDataset, NVOS_PROMPTS

        ddp.print(f"\nNVOS Evaluation")
        if args.baseline_sam3:
            ddp.print(f"  Mode: BASELINE SAM3 (native decoder, no GASA/depth/cross-view)")
        ddp.print(f"  7 scenes from LLFF (orchid excluded)")
        ddp.print(f"  Metric: IoU on target frame (single-frame, text-only)")

        # Resolve image size
        resolution = args.image_size or model.resolution
        mask_res = args.mask_size or resolution

        eval_dataset = NVOSDataset(
            data_root=str(data_root),
            split='all',
            views_per_sample=2,
            image_size=(resolution, resolution),
            mask_size=(mask_res, mask_res),
            use_language=True,
        )

        if len(eval_dataset) == 0:
            ddp.print("ERROR: No NVOS scenes found. Check --data-root.")
            return

        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, num_workers=0,
        )

        all_ious = []
        scene_ious = defaultdict(list)

        pbar = tqdm(dataloader, desc="NVOS Eval", disable=not ddp.is_main)

        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['images'].to(device).squeeze(0)      # [2, 3, H, W]
                gt_masks = batch['gt_masks'].to(device).squeeze(0)  # [2, H, W]
                prompt = batch['prompt'][0] if isinstance(batch['prompt'], list) else batch['prompt']
                scene_id = batch['scene_id'][0] if isinstance(batch['scene_id'], list) else batch['scene_id']

                # Target frame (view 1) has GT mask; reference (view 0) has scribbles only
                target_gt = gt_masks[1]  # [H, W]
                if target_gt.sum() < 1:
                    continue

                target_img = images[1:2]  # [1, 3, H, W]

                # Parse spatial qualifier from prompt (e.g. "nearest fern plant")
                sq_type, base_prompt = parse_spatial_qualifier(prompt)
                sq_idx = get_spatial_qualifier_idx(sq_type)
                sq_tensor = torch.tensor([sq_idx], device=device, dtype=torch.long) if sq_idx > 0 else None

                with torch.no_grad():
                    with autocast('cuda', dtype=torch.float16):
                        outputs = model.forward(
                            images=target_img,
                            text_prompts=[base_prompt],
                            gt_masks=None,
                            gt_intrinsics=None,
                            spatial_qualifier_idx=sq_tensor,
                        )

                # Extract prediction
                pred_mask = outputs.get('pred_masks')
                if pred_mask is None:
                    continue
                if pred_mask.dim() == 4:
                    pred_mask = pred_mask[:, 0]
                pred_mask = pred_mask.squeeze(0)  # [H, W]

                # Resize to GT mask size if needed
                if pred_mask.shape != target_gt.shape:
                    pred_mask = F.interpolate(
                        pred_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=target_gt.shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)

                # CRF / morphological post-processing (SO spatial path)
                if use_crf:
                    from triangulang.utils.crf_postprocess import morphological_smooth
                    mask_binary = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy().astype(np.float32)
                    refined = morphological_smooth(mask_binary, kernel_size=7)
                    pred_mask = torch.from_numpy(refined * 10.0 - 5.0).to(pred_mask.device)

                # Compute IoU
                gt_binary = (target_gt > 0.5).float()
                relevancy = torch.sigmoid(pred_mask)
                pred_binary = (relevancy > 0.5).float()
                intersection = (pred_binary * gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum() - intersection
                iou = (intersection / (union + 1e-6)).item()

                all_ious.append(iou)
                scene_ious[scene_id].append(iou)

                pbar.set_postfix({'mIoU': f'{np.mean(all_ious)*100:.1f}%'})

                # Visualization
                if args.visualize and ddp.is_main and batch_idx < args.viz_samples:
                    nvos_viz_dir = viz_dir / 'nvos'
                    nvos_viz_dir.mkdir(parents=True, exist_ok=True)

                    rgb_for_viz = F.interpolate(
                        target_img[:, :3], size=target_gt.shape[-2:],
                        mode='bilinear', align_corners=False
                    ).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    rgb_for_viz = (rgb_for_viz * 255).clip(0, 255).astype(np.uint8)

                    gt_np = gt_binary.cpu().numpy()
                    pred_np = pred_binary.cpu().numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(rgb_for_viz)
                    axes[0].set_title(f'{scene_id}: "{prompt}"')
                    axes[1].imshow(gt_np, cmap='gray')
                    axes[1].set_title('GT Mask')
                    axes[2].imshow(pred_np, cmap='gray')
                    axes[2].set_title(f'Pred (IoU={iou*100:.1f}%)')
                    for ax in axes:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(nvos_viz_dir / f'{scene_id}.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)

            except Exception as e:
                if ddp.is_main:
                    import traceback
                    print(f"NVOS error on {batch.get('scene_id', '?')}: {e}")
                    traceback.print_exc()
                continue

        # Aggregate results
        if ddp.is_main:
            mean_iou = np.mean(all_ious) * 100 if all_ious else 0.0
            per_scene = {k: np.mean(v) * 100 for k, v in scene_ious.items()}
            scene_mean_iou = np.mean(list(per_scene.values())) if per_scene else 0.0

            ddp.print(f"\n{'='*50}")
            ddp.print(f"NVOS Results ({len(all_ious)} samples, {len(scene_ious)} scenes)")
            ddp.print(f"{'='*50}")
            ddp.print(f"  Sample mIoU: {mean_iou:.2f}%")
            ddp.print(f"  Scene  mIoU: {scene_mean_iou:.2f}%")
            ddp.print(f"\n  Per-scene IoU:")
            for scene, siou in sorted(per_scene.items()):
                ddp.print(f"    {scene:20s}: {siou:.2f}%")

            results = {
                'dataset': 'nvos',
                'sample_miou': mean_iou,
                'scene_miou': scene_mean_iou,
                'num_samples': len(all_ious),
                'num_scenes': len(scene_ious),
                'per_scene_iou': per_scene,
            }

            results_file = output_dir / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            ddp.print(f"\nResults saved to: {results_file}")

            if args.visualize:
                ddp.print(f"Visualizations saved to: {viz_dir / 'nvos'}")

        return

    # PartImageNet evaluation path
    if args.dataset == 'partimagenet':
        from triangulang.data.partimagenet_dataset import PartImageNetDataset, create_partimagenet_eval_dataset

        ddp.print(f"\nPartImageNet Evaluation")
        ddp.print(f"Using PartImageNetDataset with:")
        ddp.print(f"  - Split: {args.split}")
        ddp.print(f"  - Part query mode: all (evaluates all parts)")
        ddp.print(f"  - 11 super-categories with part-level annotations")

        # Create eval dataset
        eval_dataset = create_partimagenet_eval_dataset(
            data_root=str(data_root),
            split=args.split,
            image_size=(args.image_size, args.image_size),
            mask_size=(args.mask_size, args.mask_size),
            max_samples=args.max_scenes,  # Use max_scenes to limit samples
        )

        ddp.print(f"  - Samples: {len(eval_dataset)}")

        # Run evaluation using dataset (same as uCO3D)
        results = evaluate_with_dataset(
            model=model,
            dataset=eval_dataset,
            device=device,
            ddp=ddp,
            args=args,
            output_dir=output_dir,
            viz_dir=viz_dir if args.visualize else None,
        )

        # Save results
        if ddp.is_main:
            results_file = output_dir / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            ddp.print(f"\nResults saved to: {results_file}")

        return

    if 'train' in args.split:
        semantics_root = data_root / 'semantics_2d_train'
    else:
        # Prefer v2 (all frames) over v1 (GT-subset only)
        semantics_root = data_root / 'semantics_2d_val_v2'
        if not semantics_root.exists():
            semantics_root = data_root / 'semantics_2d_val'

    split_file = data_root / 'splits' / f'{args.split}.txt'

    if split_file.exists():
        with open(split_file) as f:
            split_scene_ids = [line.strip() for line in f if line.strip()]
    else:
        split_scene_ids = None

    available_semantics = {d.name for d in semantics_root.iterdir() if d.is_dir()}

    if split_scene_ids:
        scene_ids = [s for s in split_scene_ids if s in available_semantics]
    else:
        scene_ids = sorted(available_semantics)

    ddp.print(f"Found {len(scene_ids)} scenes with semantics_2d data")

    # Single-object-viz mode: focus on specific scene(s)
    # Track if we already distributed scenes (to avoid double-distribution)
    scenes_already_distributed = False

    # Filter to specific scene(s) if --scene is provided (works for ScanNet++ too)
    if args.scene and not args.single_object_viz:
        requested_scenes = args.scene if isinstance(args.scene, list) else [args.scene]
        scene_ids = [s for s in scene_ids if s in requested_scenes]
        if not scene_ids:
            ddp.print(f"ERROR: --scene {args.scene} not found. Available: {sorted(available_semantics)[:20]}")
            return
        ddp.print(f"Filtering to scene(s): {scene_ids}")

    if args.single_object_viz:
        num_viz_scenes = getattr(args, 'viz_num_scenes', 1)

        if args.viz_scene:
            # Specific scene requested
            if args.viz_scene in scene_ids:
                scene_ids = [args.viz_scene]
                ddp.print(f"\n[single-object-viz] Focusing on scene: {args.viz_scene}")
            else:
                ddp.print(f"ERROR: --viz-scene '{args.viz_scene}' not found in available scenes")
                ddp.print(f"Available scenes (first 20): {scene_ids[:20]}")
                return
        elif args.viz_random_scene:
            # Pick random scenes - with DDP, each rank gets different scenes
            random.seed(args.seed)  # Consistent base seed
            shuffled = scene_ids.copy()
            random.shuffle(shuffled)

            if ddp.is_distributed:
                # With DDP: distribute scenes across ranks
                # e.g., 8 scenes on 8 GPUs = 1 scene per GPU
                total_scenes = min(num_viz_scenes, len(shuffled))
                scenes_per_rank = max(1, total_scenes // ddp.world_size)
                start_idx = ddp.rank * scenes_per_rank
                end_idx = start_idx + scenes_per_rank
                scene_ids = shuffled[start_idx:end_idx]
                ddp.print(f"\n[single-object-viz] Rank {ddp.rank}: {len(scene_ids)} random scene(s): {scene_ids}")
                scenes_already_distributed = True  # Don't distribute again below
            else:
                # Single GPU: take first N random scenes
                scene_ids = shuffled[:num_viz_scenes]
                ddp.print(f"\n[single-object-viz] {len(scene_ids)} random scene(s): {scene_ids}")
        else:
            # Default: use first N scenes alphabetically
            scene_ids = scene_ids[:num_viz_scenes]
            ddp.print(f"\n[single-object-viz] Using first {len(scene_ids)} scene(s): {scene_ids}")
            ddp.print(f"    Tip: Use --viz-random-scene for random selection")
            ddp.print(f"    Tip: Use --viz-num-scenes N for multiple scenes")
    elif args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]

    # Split scenes across ranks for distributed evaluation
    # (Skip if single-object-viz already distributed scenes)
    total_scenes = len(scene_ids)
    if ddp.is_distributed and not scenes_already_distributed:
        # Each rank gets a subset of scenes
        scenes_per_rank = (total_scenes + ddp.world_size - 1) // ddp.world_size
        start_idx = ddp.rank * scenes_per_rank
        end_idx = min(start_idx + scenes_per_rank, total_scenes)
        scene_ids = scene_ids[start_idx:end_idx]
        ddp.print(f"Rank {ddp.rank}: evaluating scenes {start_idx}-{end_idx} ({len(scene_ids)} scenes)")

    ddp.print(f"\nEvaluating on {total_scenes} scenes from {args.split}...")
    if args.single_prompt:
        ddp.print(f"Mode: SINGLE-PROMPT (prompt view {args.prompt_view}, measure other views)")
        ddp.print(f"  This is our key differentiator from MV-SAM!")
        ddp.print(f"  Uses cross-view attention to propagate understanding.")
    else:
        ddp.print(f"Mode: MULTI-PROMPT (prompt all views)")
    ddp.print(f"Protocol: {args.num_frames} frames, {args.objects_per_scene} objects/scene, ≥{args.min_mask_coverage*100:.2g}% coverage")
    ddp.print(f"Frame sampling: {args.eval_sampling}")

    # Category filtering: collect categories across all scenes and filter rare ones
    allowed_categories = None

    # Spatial query parsing: maps original prompt -> (qualifier, base_prompt)
    # e.g., "leftmost towel" -> ("leftmost", "towel")
    spatial_query_map = {}

    # If --custom-prompts is specified for ScanNet++, use those as allowed categories
    # Parse spatial qualifiers and use base prompts for filtering
    if args.custom_prompts:
        base_prompts = set()
        for prompt in args.custom_prompts:
            qualifier, base = parse_spatial_query(prompt)
            spatial_query_map[prompt.lower()] = (qualifier, base)
            base_prompts.add(base.lower())
            if qualifier:
                ddp.print(f"  Spatial query: '{prompt}' -> qualifier='{qualifier}', base='{base}'")
        allowed_categories = base_prompts
        ddp.print(f"Custom prompts filter: {args.custom_prompts}")
        ddp.print(f"  Base categories for matching: {sorted(base_prompts)}")

    if args.min_category_samples > 1:
        ddp.print(f"Collecting category statistics for filtering (min {args.min_category_samples} samples)...")
        from collections import Counter
        category_counts = Counter()
        for scene_id in scene_ids:
            anno_path = data_root / 'data' / scene_id / 'scans' / 'segments_anno.json'
            if anno_path.exists():
                with open(anno_path) as f:
                    anno = json.load(f)
                for group in anno.get('segGroups', []):
                    label = normalize_label(group.get('label', '')).lower()
                    if label:
                        category_counts[label] += 1

        allowed_categories = {cat for cat, count in category_counts.items()
                              if count >= args.min_category_samples}
        rare_categories = {cat for cat, count in category_counts.items()
                          if count < args.min_category_samples}
        ddp.print(f"  Found {len(category_counts)} categories, keeping {len(allowed_categories)} "
                  f"(filtered {len(rare_categories)} with < {args.min_category_samples} samples)")
        if rare_categories:
            examples = sorted(rare_categories)[:5]
            ddp.print(f"  Filtered examples: {examples}")

    ddp.print(f"Prompt type: {args.prompt_type}")
    if args.prompt_type in ['text_point', 'text_box_point', 'all']:
        if args.sparse_prompts:
            ddp.print(f"  MV-SAM SPARSE PROMPTING: {args.num_pos_points} pos + {args.num_neg_points} neg = {args.num_pos_points + args.num_neg_points} points TOTAL")
            ddp.print(f"  Distributed across {args.num_prompted_frames} frames (out of {args.num_frames})")
            ddp.print(f"  Other frames receive text-only prompts (global semantic context)")
        else:
            ddp.print(f"  DENSE PROMPTING: {args.num_pos_points} pos + {args.num_neg_points} neg points PER FRAME")
            ddp.print(f"  WARNING: This is NOT the MV-SAM protocol. Use --sparse-prompts for fair comparison.")
    if args.consistency_metric:
        ddp.print(f"Computing cross-view consistency (3D centroid variance)")
    if args.visualize:
        ddp.print(f"Saving visualizations to: {viz_dir}")

    # Setup DA3 cache for depth and poses (one cache provides both)
    # Auto-select val_allframes cache when evaluating on val split
    da3_cache_dir = None
    if args.da3_nested_cache:
        da3_cache_dir = data_root / args.da3_nested_cache
        # For val splits, prefer the allframes cache (more frames for Procrustes alignment)
        if 'val' in args.split and not da3_cache_dir.name.endswith('_val_allframes'):
            val_allframes_dir = data_root / f"{args.da3_nested_cache}_val_allframes"
            if val_allframes_dir.exists():
                ddp.print(f"\n📦 Auto-selecting val_allframes cache for val split")
                da3_cache_dir = val_allframes_dir
        if not da3_cache_dir.exists():
            ddp.print(f"WARNING: DA3 cache not found at {da3_cache_dir}")
            ddp.print("         Run 'python scripts/preprocess_da3_nested.py' first.")
            ddp.print("         DA3 will run live (slower) and estimated poses unavailable.")
            da3_cache_dir = None
            if args.use_estimated_poses:
                ddp.print("         Falling back to camera-frame evaluation.")
                args.use_estimated_poses = False
        else:
            ddp.print(f"\n📦 DA3 CACHE: {da3_cache_dir}")
            ddp.print(f"   Provides: cached depth + estimated poses")
    else:
        if args.baseline_sam3:
            ddp.print(f"\n⚡ DA3 SKIPPED: Baseline SAM3 mode (no depth needed)")
        else:
            ddp.print(f"\n⚡ DA3 LIVE: No --da3-nested-cache specified")
            ddp.print(f"   DA3 will run live (slower), no estimated poses available")
        if args.use_estimated_poses:
            ddp.print("   Falling back to camera-frame evaluation.")
            args.use_estimated_poses = False

    # Load GT data for Procrustes evaluation
    gt_centroids_cache = {}
    gt_poses_cache = {}
    if args.procrustes:
        ddp.print(f"\n📐 PROCRUSTES EVALUATION: Enabled")
        ddp.print(f"   Scale estimation: {'7-DoF (with scale)' if args.procrustes_with_scale else '6-DoF (no scale)'}")
        gt_centroids_cache = load_gt_centroids(data_root)
        if gt_centroids_cache:
            # Filter to evaluated scenes if scene_ids available
            if scene_ids:
                gt_centroids_cache = {k: v for k, v in gt_centroids_cache.items() if k in set(scene_ids)}
            ddp.print(f"   Loaded GT centroids for {len(gt_centroids_cache)} scenes")
        else:
            ddp.print(f"   WARNING: centroid_cache.json not found - Procrustes disabled")
            args.procrustes = False

    if args.use_estimated_poses:
        ddp.print(f"\n📍 POSE-FREE EVALUATION: Using DA3-NESTED estimated poses")
    elif args.use_world_poses:
        ddp.print(f"\n📍 WORLD-FRAME EVALUATION: Using GT poses from transforms.json")
    else:
        ddp.print(f"\n📍 CAMERA-FRAME EVALUATION: Using identity poses (default)")

    if args.compare_pose_sources:
        ddp.print(f"\n🔬 POSE COMPARISON: Will run both GT and estimated poses")

    # Save config (only on main rank)
    config = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_scenes': total_scenes,
        'num_frames': args.num_frames,
        'objects_per_scene': args.objects_per_scene,
        'image_size': args.image_size,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'single_prompt': args.single_prompt,
        'prompt_view': args.prompt_view if args.single_prompt else None,
        'consistency_metric': args.consistency_metric,
        'prompt_type': args.prompt_type,
        'num_pos_points': args.num_pos_points,
        'num_neg_points': args.num_neg_points,
        'sparse_prompts': args.sparse_prompts,
        'num_prompted_frames': args.num_prompted_frames,
        'semantic_union': args.semantic_union,
        'use_synonyms': args.use_synonyms,
        'synonym_prob': args.synonym_prob if args.use_synonyms else 0.0,
        'cross_fold': args.cross_fold,
        'num_folds': args.num_folds if args.cross_fold else None,
        'distributed': ddp.is_distributed,
        'world_size': ddp.world_size,
        'model_params': {
            'total': total_params,
            'trainable': trainable_params,
            'gasa_decoder': gasa_params,
        },
        # Pose-free evaluation options
        'use_estimated_poses': args.use_estimated_poses,
        'use_world_poses': args.use_world_poses,
        'compare_pose_sources': args.compare_pose_sources,
        'da3_cache_dir': str(da3_cache_dir) if da3_cache_dir else None,
        # Filtering options
        'min_mask_coverage': args.min_mask_coverage,
        'min_category_samples': args.min_category_samples,
        'num_allowed_categories': len(allowed_categories) if allowed_categories else None,
        # Frame sampling
        'eval_sampling': args.eval_sampling,
        # Procrustes evaluation
        'procrustes': args.procrustes,
        'procrustes_with_scale': args.procrustes_with_scale,
    }

    if ddp.is_main:
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # Setup prompt augmentation for synonym robustness testing
    prompt_augmentor = None
    if args.use_synonyms:
        prompt_augmentor = PromptAugmentor(
            use_synonyms=True,
            synonym_prob=args.synonym_prob,
            use_templates=False,  # Keep prompts simple for eval
        )
        ddp.print(f"\n🔀 Synonym augmentation ENABLED (prob={args.synonym_prob})")
        ddp.print("   Testing robustness to prompt variations (e.g., 'tap' → 'faucet')")

    # Evaluate - use different mode based on single_prompt flag
    all_results = []
    all_category_metrics = defaultdict(lambda: {'iou': [], 'oracle_iou': [], 'pixel_acc': [], 'recall': []})

    if args.single_prompt:
        # SINGLE-PROMPT MODE: Multi-view batch evaluation
        print(f"\nSingle-Prompt Propagation Evaluation")
        for scene_id in tqdm(scene_ids, desc="Scenes (single-prompt)"):
            scene_path = data_root / 'data' / scene_id
            semantics_dir = semantics_root / scene_id

            if not scene_path.exists() or not semantics_dir.exists():
                continue

            result = evaluate_scene_single_prompt(
                model, scene_path, semantics_dir, device,
                num_views=4,  # Use 4 views for single-prompt eval
                objects_per_scene=args.objects_per_scene,
                min_pixel_fraction=args.min_mask_coverage,
                image_size=(args.image_size, args.image_size),
                prompt_view=args.prompt_view,
                use_world_poses=args.use_world_poses,
                use_estimated_poses=args.use_estimated_poses,
                da3_nested_cache_dir=da3_cache_dir,
                allowed_categories=allowed_categories,
            )

            if 'error' in result:
                print(f"  {scene_id}: {result['error']}")
            else:
                all_results.append(result)
                print(f"  {scene_id}: Prompted={100*result['mean_prompted_iou']:.1f}%, "
                      f"Unprompted={100*result['mean_unprompted_iou']:.1f}%, "
                      f"Propagation={result['mean_propagation_ratio']:.2f}x")

        if not all_results:
            print("No valid results!")
            return

        # Compute single-prompt specific metrics
        mean_prompted = np.mean([r['mean_prompted_iou'] for r in all_results])
        mean_unprompted = np.mean([r['mean_unprompted_iou'] for r in all_results])
        mean_propagation = np.mean([r['mean_propagation_ratio'] for r in all_results])

        # 3D localization metrics
        mean_acc_5cm = np.mean([r.get('acc_5cm', 0) for r in all_results])
        mean_acc_10cm = np.mean([r.get('acc_10cm', 0) for r in all_results])
        centroid_errors = [r.get('mean_centroid_error_m', float('inf')) for r in all_results if r.get('mean_centroid_error_m', float('inf')) != float('inf')]
        mean_centroid_error = np.mean(centroid_errors) if centroid_errors else float('inf')

        # World-frame metrics if available
        has_world_metrics = any('acc_5cm_world' in r for r in all_results)
        if has_world_metrics:
            mean_acc_5cm_world = np.mean([r.get('acc_5cm_world', 0) for r in all_results if 'acc_5cm_world' in r])
            mean_acc_10cm_world = np.mean([r.get('acc_10cm_world', 0) for r in all_results if 'acc_10cm_world' in r])
            centroid_errors_world = [r.get('mean_centroid_error_world_m', float('inf')) for r in all_results if r.get('mean_centroid_error_world_m', float('inf')) != float('inf')]
            mean_centroid_error_world = np.mean(centroid_errors_world) if centroid_errors_world else float('inf')

        print("\n" + "="*60)
        print("SINGLE-PROMPT EVALUATION RESULTS")
        print("="*60)
        print(f"Scenes evaluated: {len(all_results)}")
        print("-"*60)
        print(f"Mean Prompted IoU:    {100*mean_prompted:.2f}%")
        print(f"Mean Unprompted IoU:  {100*mean_unprompted:.2f}%")
        print(f"Propagation Ratio:    {mean_propagation:.2f}x")
        print("-"*60)
        print(f"3D Localization (prediction frame):")
        print(f"  Acc@5cm:            {100*mean_acc_5cm:.1f}%")
        print(f"  Acc@10cm:           {100*mean_acc_10cm:.1f}%")
        print(f"  Mean Error:         {mean_centroid_error*100:.1f} cm")
        if has_world_metrics:
            print(f"3D Localization (world frame, GT poses reference):")
            print(f"  Acc@5cm (world):    {100*mean_acc_5cm_world:.1f}%")
            print(f"  Acc@10cm (world):   {100*mean_acc_10cm_world:.1f}%")
            print(f"  Mean Error (world): {mean_centroid_error_world*100:.1f} cm")
        print("-"*60)
        print(f"  (Propagation > 0.8 means good cross-view transfer)")
        print("="*60)

        results_dict = {
            'mode': 'single_prompt',
            'checkpoint': args.checkpoint,
            'split': args.split,
            'num_scenes': len(all_results),
            'mean_prompted_iou': float(mean_prompted),
            'mean_unprompted_iou': float(mean_unprompted),
            'propagation_ratio': float(mean_propagation),
            # 3D localization metrics
            'acc_5cm': float(mean_acc_5cm),
            'acc_10cm': float(mean_acc_10cm),
            'mean_centroid_error_m': float(mean_centroid_error) if mean_centroid_error != float('inf') else None,
            # Pose options
            'use_world_poses': args.use_world_poses,
            'use_estimated_poses': args.use_estimated_poses,
            'per_scene_results': all_results,
        }

        # Add world-frame metrics if available
        if has_world_metrics:
            results_dict['acc_5cm_world'] = float(mean_acc_5cm_world)
            results_dict['acc_10cm_world'] = float(mean_acc_10cm_world)
            results_dict['mean_centroid_error_world_m'] = float(mean_centroid_error_world) if mean_centroid_error_world != float('inf') else None

        output_path = output_dir / 'results_single_prompt.json'
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_path}")
        return

    # MULTI-PROMPT MODE: Standard per-frame evaluation (MV-SAM protocol)
    ddp.print(f"\nStandard Evaluation (prompt_type={args.prompt_type})")
    if args.prompt_type != 'text_only':
        ddp.print(f"    Using {args.num_pos_points} positive + {args.num_neg_points} negative points per frame")

    # Paper viz: ALL ranks collect viz data (will be gathered to rank 0 later)
    collect_viz = (args.paper_viz or args.single_object_viz)
    paper_viz_pool = [] if collect_viz else None

    # Use tqdm only on main rank to avoid duplicate progress bars
    scene_iterator = tqdm(scene_ids, desc=f"Scenes (rank {ddp.rank})", disable=not ddp.is_main)
    for scene_id in scene_iterator:
        scene_path = data_root / 'data' / scene_id
        semantics_dir = semantics_root / scene_id

        if not scene_path.exists():
            continue

        if not semantics_dir.exists():
            continue

        result = evaluate_scene(
            model, scene_path, semantics_dir, device,
            num_frames=args.num_frames,
            objects_per_scene=args.objects_per_scene,
            min_pixel_fraction=args.min_mask_coverage,
            image_size=(args.image_size, args.image_size),
            save_viz=args.visualize and ddp.is_main,  # Only save viz on main rank
            viz_dir=viz_dir,
            viz_samples=args.viz_samples,
            prompt_type=args.prompt_type,
            num_pos_points=args.num_pos_points,
            num_neg_points=args.num_neg_points,
            sparse_prompts=args.sparse_prompts,
            num_prompted_frames=args.num_prompted_frames,
            output_localization=args.output_localization,
            output_depth=args.output_depth,
            prompt_augmentor=prompt_augmentor,
            semantic_union=args.semantic_union,
            da3_cache_dir=da3_cache_dir,
            # Procrustes evaluation
            procrustes=args.procrustes,
            procrustes_with_scale=args.procrustes_with_scale,
            gt_centroids_cache=gt_centroids_cache,
            data_root=data_root,
            # Category filtering
            allowed_categories=allowed_categories,
            # Spatial query filtering
            spatial_query_map=spatial_query_map,
            spatial_eval=args.spatial_eval,
            # Paper visualization collector
            paper_viz_collector=paper_viz_pool,
            # Frame selection
            frame_names=args.frame_names,
            eval_sampling=args.eval_sampling,
            multi_object_eval=getattr(args, 'multi_object_eval', False),
            temporal_smooth_alpha=getattr(args, 'temporal_smooth_alpha', 0.0),
            use_crf=getattr(args, 'use_crf', False),
        )

        if 'error' in result:
            print(f"  [rank {ddp.rank}] {scene_id}: {result['error']}")
        else:
            all_results.append(result)

            for cat, iou in result.get('per_category_iou', {}).items():
                all_category_metrics[cat]['iou'].append(iou)
            for cat, oracle_iou in result.get('per_category_oracle_iou', {}).items():
                all_category_metrics[cat]['oracle_iou'].append(oracle_iou)
            for cat, pixel_acc in result.get('per_category_pixel_acc', {}).items():
                all_category_metrics[cat]['pixel_acc'].append(pixel_acc)
            for cat, recall in result.get('per_category_recall', {}).items():
                all_category_metrics[cat]['recall'].append(recall)

            # Save per-rank partial results after each scene (never lose work)
            partial_dir = output_dir / 'partial'
            partial_dir.mkdir(parents=True, exist_ok=True)
            partial_path = partial_dir / f'rank{ddp.rank}_results.json'
            try:
                partial_ious = [r['mean_iou'] for r in all_results if 'mean_iou' in r]
                partial_summary = {
                    'rank': ddp.rank,
                    'scenes_done': len(all_results),
                    'scene_ids': [r.get('scene_id', '?') for r in all_results],
                    'running_miou': float(np.mean(partial_ious)) if partial_ious else 0.0,
                    'running_iou': float(np.mean([r['mean_iou'] for r in all_results])) if all_results else 0.0,
                    'last_updated': datetime.now().isoformat(),
                    'results': all_results,
                    'category_metrics': {cat: {k: v for k, v in metrics.items()} for cat, metrics in all_category_metrics.items()},
                }
                with open(partial_path, 'w') as f:
                    json.dump(partial_summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
            except Exception as e:
                print(f"  [rank {ddp.rank}] Warning: failed to save partial results: {e}")

    # Gather results from all ranks
    ddp.barrier()  # Sync before gathering

    if ddp.is_distributed:
        import pickle
        # Serialize results for gathering
        local_results_bytes = pickle.dumps(all_results)
        local_metrics_bytes = pickle.dumps(dict(all_category_metrics))

        # Gather all results to rank 0
        if ddp.is_main:
            gathered_results = [None] * ddp.world_size
            gathered_metrics = [None] * ddp.world_size
        else:
            gathered_results = None
            gathered_metrics = None

        # Use dist.gather_object for Python objects
        dist.gather_object(all_results, gathered_results, dst=0)
        dist.gather_object(dict(all_category_metrics), gathered_metrics, dst=0)

        # Gather viz data from all ranks to rank 0 for unified visualization
        if paper_viz_pool is not None:
            gathered_viz = [None] * ddp.world_size if ddp.is_main else None
            dist.gather_object(paper_viz_pool, gathered_viz, dst=0)
            if ddp.is_main:
                paper_viz_pool = []
                for rank_pool in gathered_viz:
                    if rank_pool:
                        paper_viz_pool.extend(rank_pool)
                print(f"[Viz] Gathered {len(paper_viz_pool)} viz samples from {ddp.world_size} ranks")

        if ddp.is_main:
            # Merge results from all ranks
            all_results = []
            for rank_results in gathered_results:
                all_results.extend(rank_results)

            # Merge category metrics
            merged_category_metrics = defaultdict(lambda: {'iou': [], 'oracle_iou': [], 'pixel_acc': [], 'recall': []})
            for rank_metrics in gathered_metrics:
                for cat, metrics in rank_metrics.items():
                    for key, values in metrics.items():
                        merged_category_metrics[cat][key].extend(values)
            all_category_metrics = merged_category_metrics

    if not all_results:
        ddp.print("No valid results!")
        ddp.cleanup()
        return

    # Only compute and print final metrics on main rank
    if not ddp.is_main:
        ddp.cleanup()
        return

    # Compute final metrics
    sample_iou = np.mean([r['sample_iou'] for r in all_results])
    sample_oracle_iou = np.mean([r.get('oracle_iou', r['sample_iou']) for r in all_results])
    scene_miou = np.mean([r['miou'] for r in all_results])
    scene_oracle_miou = np.mean([r.get('oracle_miou', r['miou']) for r in all_results])
    sample_pixel_acc = np.mean([r['pixel_acc'] for r in all_results])
    sample_recall = np.mean([r['recall'] for r in all_results])
    sample_precision = np.mean([r['precision'] for r in all_results])
    sample_f1 = np.mean([r['f1'] for r in all_results])

    # Global pixel accuracy from raw counts (more accurate)
    total_tp = sum(r['total_tp'] for r in all_results)
    total_fp = sum(r['total_fp'] for r in all_results)
    total_fn = sum(r['total_fn'] for r in all_results)
    total_tn = sum(r['total_tn'] for r in all_results)
    all_pixels = total_tp + total_fp + total_fn + total_tn
    global_pixel_acc = (total_tp + total_tn) / all_pixels if all_pixels > 0 else 0.0

    global_per_cat_iou = {cat: np.mean(m['iou']) for cat, m in all_category_metrics.items() if m['iou']}
    global_per_cat_oracle_iou = {cat: np.mean(m['oracle_iou']) for cat, m in all_category_metrics.items() if m.get('oracle_iou')}
    global_per_cat_pixel_acc = {cat: np.mean(m['pixel_acc']) for cat, m in all_category_metrics.items() if m['pixel_acc']}
    global_per_cat_recall = {cat: np.mean(m['recall']) for cat, m in all_category_metrics.items() if m['recall']}
    global_miou = np.mean(list(global_per_cat_iou.values())) if global_per_cat_iou else 0.0
    global_oracle_miou = np.mean(list(global_per_cat_oracle_iou.values())) if global_per_cat_oracle_iou else 0.0
    global_mean_class_recall = np.mean(list(global_per_cat_recall.values())) if global_per_cat_recall else 0.0

    # For spatial eval reporting
    spatial_qualifiers_set = {'nearest', 'farthest', 'leftmost', 'rightmost', 'topmost', 'bottommost',
                              'closest', 'left', 'right', 'top', 'bottom'}

    avg_preprocess_ms = np.mean([r['avg_preprocess_ms'] for r in all_results if 'avg_preprocess_ms' in r])
    avg_inference_ms = np.mean([r['avg_inference_ms'] for r in all_results if 'avg_inference_ms' in r])
    total_samples = sum(r['num_samples'] for r in all_results)

    # Aggregate Acc@m metrics for 3D localization (weighted by num_centroid_samples)
    total_centroid_samples = sum(r.get('num_centroid_samples', 0) for r in all_results)
    if total_centroid_samples > 0:
        global_acc_5cm = sum(r.get('acc_5cm', 0) * r.get('num_centroid_samples', 0) for r in all_results) / total_centroid_samples
        global_acc_10cm = sum(r.get('acc_10cm', 0) * r.get('num_centroid_samples', 0) for r in all_results) / total_centroid_samples
        global_acc_50cm = sum(r.get('acc_50cm', 0) * r.get('num_centroid_samples', 0) for r in all_results) / total_centroid_samples
        # Mean centroid error - filter out inf values
        valid_errors = [r.get('mean_centroid_error_m', float('inf')) for r in all_results if r.get('mean_centroid_error_m', float('inf')) != float('inf')]
        global_mean_centroid_error = np.mean(valid_errors) if valid_errors else float('inf')
    else:
        global_acc_5cm, global_acc_10cm, global_acc_50cm, global_mean_centroid_error = 0, 0, 0, float('inf')

    # Aggregate Procrustes-aligned localization metrics
    total_procrustes_samples = sum(r.get('procrustes_num_samples', 0) for r in all_results)
    if total_procrustes_samples > 0:
        global_procrustes_acc_5cm = sum(
            (r.get('procrustes_acc_5cm') or 0) * r.get('procrustes_num_samples', 0)
            for r in all_results
        ) / total_procrustes_samples
        global_procrustes_acc_10cm = sum(
            (r.get('procrustes_acc_10cm') or 0) * r.get('procrustes_num_samples', 0)
            for r in all_results
        ) / total_procrustes_samples
        valid_procrustes_errors = [
            r.get('procrustes_mean_error_m') for r in all_results
            if r.get('procrustes_mean_error_m') is not None
        ]
        global_procrustes_mean_error = np.mean(valid_procrustes_errors) if valid_procrustes_errors else None
        procrustes_scales = [r.get('procrustes_scale') for r in all_results if r.get('procrustes_scale') is not None]
        avg_procrustes_scale = np.mean(procrustes_scales) if procrustes_scales else None
    else:
        global_procrustes_acc_5cm = global_procrustes_acc_10cm = global_procrustes_mean_error = avg_procrustes_scale = None

    # Aggregate cross-view consistency
    valid_consistency = [r.get('consistency_iou') for r in all_results if r.get('consistency_iou') is not None]
    total_consistency_objects = sum(r.get('num_consistency_objects', 0) for r in all_results)
    global_consistency_iou = np.mean(valid_consistency) if valid_consistency else None

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Scenes evaluated: {len(all_results)}")
    print(f"Total samples: {total_samples}")
    print(f"Categories: {len(global_per_cat_iou)}")
    print("-"*70)
    print(f"{'Metric':<25} {'Selected':<15} {'Oracle':<15} {'Gap':<10}")
    print("-"*70)
    print(f"{'Sample-avg IoU:':<25} {100*sample_iou:>13.2f}%  {100*sample_oracle_iou:>13.2f}%  {100*(sample_oracle_iou-sample_iou):>+8.2f}%")
    print(f"{'Scene-avg mIoU:':<25} {100*scene_miou:>13.2f}%  {100*scene_oracle_miou:>13.2f}%  {100*(scene_oracle_miou-scene_miou):>+8.2f}%")
    print(f"{'Global mIoU:':<25} {100*global_miou:>13.2f}%  {100*global_oracle_miou:>13.2f}%  {100*(global_oracle_miou-global_miou):>+8.2f}%")
    print("-"*70)
    print(f"mAcc (Pixel Acc): {100*global_pixel_acc:.2f}%  (sample-avg: {100*sample_pixel_acc:.2f}%)")
    print(f"Mean Class Recall:{100*global_mean_class_recall:.2f}%  (sample-avg: {100*sample_recall:.2f}%)")
    print(f"Precision:        {100*sample_precision:.2f}%")
    print(f"F1 Score:         {100*sample_f1:.2f}%")
    print("-"*70)
    if total_centroid_samples > 0:
        print(f"3D Localization (IoU-based, same pointmap):")
        print(f"  Acc@5cm:        {100*global_acc_5cm:.2f}%")
        print(f"  Acc@10cm:       {100*global_acc_10cm:.2f}%")
        print(f"  Acc@50cm:       {100*global_acc_50cm:.2f}%")
        if global_mean_centroid_error != float('inf'):
            print(f"  Mean Error:     {global_mean_centroid_error*100:.1f} cm")
        print(f"  Samples:        {total_centroid_samples}")
        print("-"*60)

    # Procrustes-aligned localization 
    if total_procrustes_samples > 0:
        print(f"📐 Procrustes-aligned Localization (vs GT mesh centroids):")
        print(f"  Acc@5cm:        {100*global_procrustes_acc_5cm:.2f}%")
        print(f"  Acc@10cm:       {100*global_procrustes_acc_10cm:.2f}%")
        if global_procrustes_mean_error is not None:
            print(f"  Mean Error:     {global_procrustes_mean_error*100:.1f} cm")
        if avg_procrustes_scale is not None:
            print(f"  Avg Scale:      {avg_procrustes_scale:.3f}")
        print(f"  Samples:        {total_procrustes_samples}")
        print("-"*60)
    # Spatial eval metrics
    if args.spatial_eval:
        spatial_results = [r for r in all_results if r.get('spatial_miou') is not None]
        spatial_per_cat = {cat: np.mean(m['iou']) for cat, m in all_category_metrics.items()
                          if m['iou'] and cat.split()[0].lower() in spatial_qualifiers_set}
        spatial_global_miou = np.mean(list(spatial_per_cat.values())) if spatial_per_cat else 0.0
        total_spatial_queries = sum(r.get('spatial_num_queries', 0) for r in all_results)
        print(f"Spatial Language Evaluation:")
        print(f"  Spatial mIoU:    {100*spatial_global_miou:.2f}%")
        print(f"  Spatial queries: {total_spatial_queries} ({len(spatial_per_cat)} unique)")
        if spatial_per_cat:
            # Break down by qualifier type
            for qual in ['nearest', 'farthest', 'leftmost', 'rightmost']:
                qual_cats = {c: v for c, v in spatial_per_cat.items() if c.startswith(qual)}
                if qual_cats:
                    print(f"    {qual}: {100*np.mean(list(qual_cats.values())):.1f}% ({len(qual_cats)} cats)")
        print("-"*60)
    # Cross-view consistency metric
    if global_consistency_iou is not None:
        print(f"Cross-View Consistency:")
        print(f"  Consistency IoU: {100*global_consistency_iou:.2f}%")
        print(f"  Objects:         {total_consistency_objects}")
        print("-"*60)
    print(f"Avg preprocess:   {avg_preprocess_ms:.1f} ms")
    print(f"Avg inference:    {avg_inference_ms:.1f} ms")
    print(f"Total per sample: {avg_preprocess_ms + avg_inference_ms:.1f} ms")
    print("-"*60)

    sorted_cats = sorted(global_per_cat_iou.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 categories:")
    for cat, iou in sorted_cats[:5]:
        recall = global_per_cat_recall.get(cat, 0)
        print(f"  {cat}: IoU={100*iou:.1f}%, Recall={100*recall:.1f}%")
    print("\nBottom 5 categories:")
    for cat, iou in sorted_cats[-5:]:
        recall = global_per_cat_recall.get(cat, 0)
        print(f"  {cat}: IoU={100*iou:.1f}%, Recall={100*recall:.1f}%")

    # Cross-fold analysis (stratified category grouping)
    fold_results = None
    if args.cross_fold and len(global_per_cat_iou) >= args.num_folds:
        print(f"\n{'='*70}")
        print(f"CROSS-FOLD ANALYSIS ({args.num_folds} folds)")
        print("="*70)
        print("Grouping categories into folds for per-group performance analysis\n")

        # Sort categories alphabetically for deterministic fold assignment
        sorted_categories = sorted(global_per_cat_iou.keys())
        fold_size = len(sorted_categories) // args.num_folds

        fold_results = []
        for fold_idx in range(args.num_folds):
            start_idx = fold_idx * fold_size
            if fold_idx == args.num_folds - 1:
                # Last fold gets remaining categories
                end_idx = len(sorted_categories)
            else:
                end_idx = (fold_idx + 1) * fold_size

            fold_categories = sorted_categories[start_idx:end_idx]
            fold_ious = [global_per_cat_iou[cat] for cat in fold_categories]
            fold_recalls = [global_per_cat_recall.get(cat, 0) for cat in fold_categories]

            fold_mean_iou = np.mean(fold_ious)
            fold_mean_recall = np.mean(fold_recalls)

            fold_results.append({
                'fold_id': fold_idx,
                'categories': fold_categories,
                'num_categories': len(fold_categories),
                'mean_iou': float(fold_mean_iou),
                'mean_recall': float(fold_mean_recall),
            })

            print(f"Fold {fold_idx + 1}/{args.num_folds}: {len(fold_categories)} categories")
            print(f"  Mean IoU:    {100*fold_mean_iou:.2f}%")
            print(f"  Mean Recall: {100*fold_mean_recall:.2f}%")
            print(f"  Categories:  {', '.join(fold_categories[:5])}" +
                  (f", ... (+{len(fold_categories)-5} more)" if len(fold_categories) > 5 else ""))
            print()

        # Find best/worst folds
        best_fold = max(fold_results, key=lambda x: x['mean_iou'])
        worst_fold = min(fold_results, key=lambda x: x['mean_iou'])

        print(f"Best fold: Fold {best_fold['fold_id'] + 1} (mIoU={100*best_fold['mean_iou']:.2f}%)")
        print(f"Worst fold: Fold {worst_fold['fold_id'] + 1} (mIoU={100*worst_fold['mean_iou']:.2f}%)")
        print(f"Performance gap: {100*(best_fold['mean_iou'] - worst_fold['mean_iou']):.2f}%")
        print("="*70)

    # Build results dict
    results_dict = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_scenes': len(all_results),
        'total_samples': total_samples,
        # Selected mask metrics
        'sample_iou': float(sample_iou),
        'scene_miou': float(scene_miou),
        'global_miou': float(global_miou),
        # Oracle metrics (best possible mask selection)
        'oracle_sample_iou': float(sample_oracle_iou),
        'oracle_scene_miou': float(scene_oracle_miou),
        'oracle_global_miou': float(global_oracle_miou),
        # Gap between oracle and selected (room for improvement)
        'iou_gap': float(sample_oracle_iou - sample_iou),
        'miou_gap': float(global_oracle_miou - global_miou),
        # mAcc = global pixel accuracy (TP+TN)/total - standard in most papers
        'mAcc': float(global_pixel_acc),
        'sample_pixel_acc': float(sample_pixel_acc),
        # Mean class recall = TP/(TP+FN) per category, averaged
        'mean_class_recall': float(global_mean_class_recall),
        'sample_recall': float(sample_recall),
        'precision': float(sample_precision),
        'f1': float(sample_f1),
        'avg_preprocess_ms': float(avg_preprocess_ms),
        'avg_inference_ms': float(avg_inference_ms),
        # 3D Localization Accuracy (Acc@m) - IoU-based (same pointmap)
        'acc_5cm': float(global_acc_5cm),
        'acc_10cm': float(global_acc_10cm),
        'acc_50cm': float(global_acc_50cm),
        'mean_centroid_error_m': float(global_mean_centroid_error) if global_mean_centroid_error != float('inf') else None,
        'num_centroid_samples': total_centroid_samples,
        # Procrustes-aligned localization 
        'procrustes_acc_5cm': float(global_procrustes_acc_5cm) if global_procrustes_acc_5cm is not None else None,
        'procrustes_acc_10cm': float(global_procrustes_acc_10cm) if global_procrustes_acc_10cm is not None else None,
        'procrustes_mean_error_m': float(global_procrustes_mean_error) if global_procrustes_mean_error is not None else None,
        'procrustes_avg_scale': float(avg_procrustes_scale) if avg_procrustes_scale is not None else None,
        'procrustes_num_samples': total_procrustes_samples,
        # Cross-view consistency (do corresponding 3D points get same prediction?)
        'consistency_iou': float(global_consistency_iou) if global_consistency_iou is not None else None,
        'num_consistency_objects': total_consistency_objects,
        'per_category_iou': {k: float(v) for k, v in global_per_cat_iou.items()},
        'per_category_oracle_iou': {k: float(v) for k, v in global_per_cat_oracle_iou.items()},
        'per_category_pixel_acc': {k: float(v) for k, v in global_per_cat_pixel_acc.items()},
        'per_category_recall': {k: float(v) for k, v in global_per_cat_recall.items()},
        'fold_results': fold_results if fold_results else None,
        'per_scene_results': [
            {
                'scene_id': r['scene_id'],
                'miou': float(r['miou']),
                'oracle_miou': float(r.get('oracle_miou', r['miou'])),
                'pixel_acc': float(r['pixel_acc']),
                'global_pixel_acc': float(r['global_pixel_acc']),
                'recall': float(r['recall']),
                'precision': float(r['precision']),
                'f1': float(r['f1']),
                'num_samples': r['num_samples'],
                'consistency_iou': float(r['consistency_iou']) if r.get('consistency_iou') is not None else None,
            }
            for r in all_results
        ],
        'model_params': config['model_params'],
    }

    # Save results JSON
    output_path = output_dir / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate plots
    print("\nGenerating plots...")

    if global_per_cat_iou:
        plot_category_iou(global_per_cat_iou, output_dir / 'category_iou.png')

    if len(all_results) > 1:
        plot_scene_metrics(all_results, output_dir / 'scene_metrics.png')

    plot_summary(results_dict, output_dir / 'summary.png')

    # Paper-quality grid visualization (ScanNet++ path)
    # Viz data already gathered from all ranks above (before non-main ranks exit)
    if paper_viz_pool and args.paper_viz:
        print("\nGenerating paper-quality grid visualizations...")
        generate_paper_visualizations(paper_viz_pool, args, output_dir)

    # Single-object focused visualization
    if paper_viz_pool and args.single_object_viz:
        print("\nGenerating single-object focused visualization...")
        generate_single_object_viz(paper_viz_pool, args, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")

    # Cleanup DDP
    ddp.cleanup()


if __name__ == '__main__':
    main()
