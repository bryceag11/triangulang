"""
Spatial Reasoning Utilities for TrianguLang

Provides:
1. Spatial qualifier parsing (nearest, leftmost, etc.)
2. Spatial augmentation for training labels
3. Spatial-to-pseudo-point conversion
4. Relational query parsing (to the right of, above, etc.)
"""

import re
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch


# Spatial Qualifier Parsing

SPATIAL_QUALIFIERS = {
    # Depth-based (extremes)
    'nearest': 'depth_min', 'closest': 'depth_min', 'close': 'depth_min',
    'farthest': 'depth_max', 'far': 'depth_max', 'distant': 'depth_max',
    # Depth-based (ordinal)
    'second nearest': 'depth_2nd_min', 'second closest': 'depth_2nd_min',
    'second farthest': 'depth_2nd_max',
    # Horizontal position (extremes)
    'leftmost': 'x_min', 'left': 'x_min',
    'rightmost': 'x_max', 'right': 'x_max',
    # Horizontal position (ordinal)
    'second leftmost': 'x_2nd_min', 'second from left': 'x_2nd_min',
    'second rightmost': 'x_2nd_max', 'second from right': 'x_2nd_max',
    # Vertical position (extremes)
    'topmost': 'y_min', 'top': 'y_min', 'upper': 'y_min',
    'bottommost': 'y_max', 'bottom': 'y_max', 'lower': 'y_max',
    # Vertical position (ordinal)
    'second topmost': 'y_2nd_min', 'second from top': 'y_2nd_min',
    'second bottommost': 'y_2nd_max', 'second from bottom': 'y_2nd_max',
    # Middle/center position (for objects not at extremes)
    'middle': 'middle', 'central': 'middle', 'center': 'middle',
    'mid-depth': 'depth_mid', 'middle depth': 'depth_mid',
    # Size-based (using mask coverage)
    'largest': 'size_max', 'biggest': 'size_max', 'big': 'size_max',
    'smallest': 'size_min', 'small': 'size_min', 'tiny': 'size_min',
}

SPATIAL_QUALIFIER_TO_IDX = {
    None: 0,
    'depth_min': 1,      # nearest
    'depth_max': 2,      # farthest
    'x_min': 3,          # leftmost
    'x_max': 4,          # rightmost
    'y_min': 5,          # top
    'y_max': 6,          # bottom
    'middle': 7,         # middle/center
    # Extended qualifiers (share indices with related extremes for embedding)
    'depth_2nd_min': 1,  # second nearest -> nearest embedding
    'depth_2nd_max': 2,  # second farthest -> farthest embedding
    'x_2nd_min': 3,      # second leftmost -> leftmost embedding
    'x_2nd_max': 4,      # second rightmost -> rightmost embedding
    'y_2nd_min': 5,      # second topmost -> top embedding
    'y_2nd_max': 6,      # second bottommost -> bottom embedding
    'depth_mid': 7,      # middle depth -> middle embedding
    'size_max': 1,       # largest -> reuse nearest (both are "more")
    'size_min': 2,       # smallest -> reuse farthest (both are "less")
}

# Relational patterns for parsing
RELATION_PATTERNS = [
    # "X to the right of Y", "X on the right of Y"
    (r'(.+?)\s+(?:to\s+the\s+|on\s+the\s+)?right\s+of\s+(?:the\s+)?(.+)', 'right_of'),
    (r'(.+?)\s+(?:to\s+the\s+|on\s+the\s+)?left\s+of\s+(?:the\s+)?(.+)', 'left_of'),
    # "X above Y", "X over Y"
    (r'(.+?)\s+(?:above|over)\s+(?:the\s+)?(.+)', 'above'),
    (r'(.+?)\s+(?:below|under|beneath)\s+(?:the\s+)?(.+)', 'below'),
    # "X near Y", "X next to Y", "X beside Y"
    (r'(.+?)\s+(?:near|next\s+to|beside|by)\s+(?:the\s+)?(.+)', 'near'),
    # "X on Y", "X on top of Y"
    (r'(.+?)\s+on\s+(?:top\s+of\s+)?(?:the\s+)?(.+)', 'on_top_of'),
    # "X in front of Y"
    (r'(.+?)\s+in\s+front\s+of\s+(?:the\s+)?(.+)', 'in_front_of'),
    # "X behind Y"
    (r'(.+?)\s+behind\s+(?:the\s+)?(.+)', 'behind'),
]


def parse_spatial_qualifier(prompt: str) -> Tuple[Optional[str], str]:
    """Extract spatial qualifier and base object from prompt.

    Only matches qualifiers as leading words (possibly after an article).
    This avoids false matches on relational queries like "chair to the right of table"
    where "right" is NOT a spatial qualifier prefix.

    Args:
        prompt: e.g., "nearest chair", "the rightmost monitor"

    Returns:
        (qualifier_type, base_object): e.g., ('depth_min', 'chair')
        qualifier_type is one of: 'depth_min', 'depth_max', 'x_min', 'x_max', 'y_min', 'y_max', None
    """
    prompt_lower = prompt.lower().strip()

    # Strip leading article
    stripped = prompt_lower
    for prefix in ['the ', 'a ', 'an ']:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break

    # Check if prompt starts with a spatial qualifier word
    # Sort by length descending so "second nearest" matches before "nearest", etc.
    for word, qualifier_type in sorted(SPATIAL_QUALIFIERS.items(), key=lambda x: len(x[0]), reverse=True):
        if stripped.startswith(word + ' ') or stripped == word:
            base = stripped[len(word):].strip()
            return qualifier_type, base

    return None, prompt


def parse_relational_query(prompt: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse relational query into target, reference, and relation.

    Args:
        prompt: e.g., "chair to the right of the table"

    Returns:
        (target, reference, relation): e.g., ('chair', 'table', 'right_of')
        Returns (None, None, None) if not a relational query.
    """
    prompt_lower = prompt.lower().strip()

    for pattern, relation in RELATION_PATTERNS:
        match = re.match(pattern, prompt_lower)
        if match:
            target = match.group(1).strip()
            reference = match.group(2).strip()
            # Clean up articles
            for prefix in ['the ', 'a ', 'an ']:
                if target.startswith(prefix):
                    target = target[len(prefix):]
                if reference.startswith(prefix):
                    reference = reference[len(prefix):]
            return target, reference, relation

    return None, None, None


def get_spatial_qualifier_idx(qualifier_type: Optional[str]) -> int:
    """Convert qualifier type to index for embedding lookup."""
    return SPATIAL_QUALIFIER_TO_IDX.get(qualifier_type, 0)


# Spatial Augmentation for Training

def get_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Get centroid of binary mask as (x, y)."""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return (mask.shape[1] / 2, mask.shape[0] / 2)
    return (float(xs.mean()), float(ys.mean()))


def get_depth_at_centroid(mask: np.ndarray, depth: np.ndarray) -> float:
    """Get depth value at mask centroid."""
    cx, cy = get_mask_centroid(mask)
    cy, cx = int(cy), int(cx)
    cy = min(max(cy, 0), depth.shape[0] - 1)
    cx = min(max(cx, 0), depth.shape[1] - 1)
    return float(depth[cy, cx])


def augment_label_with_spatial(
    label: str,
    mask: np.ndarray,
    depth: np.ndarray,
    all_masks_same_label: List[np.ndarray],
    all_depths: Optional[List[np.ndarray]] = None,
    augment_prob: float = 0.5
) -> str:
    """Augment label with spatial qualifier based on object position.

    Args:
        label: Base label (e.g., "chair")
        mask: This object's mask [H, W]
        depth: Depth map [H, W]
        all_masks_same_label: List of all masks with same label in scene
        all_depths: Optional list of depth maps per mask (if different views)
        augment_prob: Probability of adding spatial qualifier

    Returns:
        Augmented label (e.g., "nearest chair") or original label
    """
    if len(all_masks_same_label) <= 1:
        return label  # Only one instance, no spatial qualifier needed

    if random.random() > augment_prob:
        return label  # Skip augmentation randomly

    # Use same depth for all if not provided per-mask
    if all_depths is None:
        all_depths = [depth] * len(all_masks_same_label)

    # Compute centroids and depths for all instances
    centroids = []
    depths_at_centroid = []
    for m, d in zip(all_masks_same_label, all_depths):
        cx, cy = get_mask_centroid(m)
        centroids.append((cx, cy))
        depths_at_centroid.append(get_depth_at_centroid(m, d))

    # Find index of current mask
    my_idx = None
    for i, m in enumerate(all_masks_same_label):
        if np.array_equal(m, mask):
            my_idx = i
            break

    if my_idx is None:
        return label

    my_cx, my_cy = centroids[my_idx]
    my_depth = depths_at_centroid[my_idx]

    # Check if this is the extreme in any dimension
    qualifiers = []

    # Depth-based
    if my_depth <= min(depths_at_centroid):
        qualifiers.extend(['nearest', 'closest'])
    if my_depth >= max(depths_at_centroid):
        qualifiers.extend(['farthest', 'far'])

    # X-coordinate (horizontal)
    all_x = [c[0] for c in centroids]
    if my_cx <= min(all_x):
        qualifiers.extend(['leftmost', 'left'])
    if my_cx >= max(all_x):
        qualifiers.extend(['rightmost', 'right'])

    # Y-coordinate (vertical)
    all_y = [c[1] for c in centroids]
    if my_cy <= min(all_y):
        qualifiers.extend(['topmost', 'top', 'upper'])
    if my_cy >= max(all_y):
        qualifiers.extend(['bottommost', 'bottom', 'lower'])

    if qualifiers:
        qualifier = random.choice(qualifiers)
        # Randomly choose format
        formats = [
            f"{qualifier} {label}",
            f"the {qualifier} {label}",
        ]
        return random.choice(formats)

    return label


# Spatial to Pseudo-Point Conversion

def spatial_to_pseudo_point(
    qualifier_type: Optional[str],
    depth_map: Optional[np.ndarray] = None,
    H: int = 518,
    W: int = 518
) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[int]]]:
    """Convert spatial qualifier to pseudo-point prompt.

    Args:
        qualifier_type: One of 'depth_min', 'depth_max', 'x_min', 'x_max', 'y_min', 'y_max'
        depth_map: [H, W] depth map (needed for depth-based qualifiers)
        H, W: Image dimensions

    Returns:
        (points, labels): Normalized point coordinates and labels (1=positive)
        Returns (None, None) if qualifier_type is None or conversion fails.
    """
    if qualifier_type is None:
        return None, None

    if qualifier_type == 'depth_min' and depth_map is not None:
        # Point at minimum depth location
        y, x = np.unravel_index(depth_map.argmin(), depth_map.shape)
        return [(x / W, y / H)], [1]

    elif qualifier_type == 'depth_max' and depth_map is not None:
        # Point at maximum depth location
        y, x = np.unravel_index(depth_map.argmax(), depth_map.shape)
        return [(x / W, y / H)], [1]

    elif qualifier_type == 'x_min':
        # Point on left edge, center height
        return [(0.05, 0.5)], [1]

    elif qualifier_type == 'x_max':
        # Point on right edge, center height
        return [(0.95, 0.5)], [1]

    elif qualifier_type == 'y_min':
        # Point on top edge, center width
        return [(0.5, 0.05)], [1]

    elif qualifier_type == 'y_max':
        # Point on bottom edge, center width
        return [(0.5, 0.95)], [1]

    elif qualifier_type == 'center':
        # Point at center
        return [(0.5, 0.5)], [1]

    return None, None


def spatial_to_pseudo_point_tensor(
    qualifier_type: Optional[str],
    depth_map: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Tensor version of spatial_to_pseudo_point.

    Returns:
        (points, labels): [1, 2] and [1] tensors, or (None, None)
    """
    if depth_map is not None:
        H, W = depth_map.shape[-2:]
        depth_np = depth_map.squeeze().cpu().numpy() if isinstance(depth_map, torch.Tensor) else depth_map
    else:
        H, W = 518, 518
        depth_np = None

    points, labels = spatial_to_pseudo_point(qualifier_type, depth_np, H, W)

    if points is None:
        return None, None

    points_t = torch.tensor(points, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    return points_t, labels_t


# Relational Filtering

def filter_by_relation(
    target_masks: List[np.ndarray],
    target_depths: List[np.ndarray],
    reference_mask: np.ndarray,
    reference_depth: np.ndarray,
    relation: str,
    threshold: float = 0.3
) -> Tuple[Optional[np.ndarray], int]:
    """Filter target masks by spatial relation to reference object.

    Args:
        target_masks: List of candidate masks for target object
        target_depths: Depth maps for each target mask
        reference_mask: Mask of reference object
        reference_depth: Depth map for reference object
        relation: One of 'right_of', 'left_of', 'above', 'below', 'near', 'on_top_of', etc.
        threshold: Distance threshold for 'near' relation (normalized)

    Returns:
        (best_mask, best_idx): Selected mask and its index, or (None, -1) if none match
    """
    if len(target_masks) == 0:
        return None, -1

    # Get reference centroid and depth
    ref_cx, ref_cy = get_mask_centroid(reference_mask)
    ref_depth = get_depth_at_centroid(reference_mask, reference_depth)

    valid_indices = []
    scores = []

    for i, (mask, depth) in enumerate(zip(target_masks, target_depths)):
        cx, cy = get_mask_centroid(mask)
        d = get_depth_at_centroid(mask, depth)

        # Normalize coordinates
        H, W = mask.shape
        cx_norm, cy_norm = cx / W, cy / H
        ref_cx_norm, ref_cy_norm = ref_cx / W, ref_cy / H

        # Check relation
        is_valid = False
        score = 0.0

        if relation == 'right_of':
            if cx_norm > ref_cx_norm:
                is_valid = True
                score = cx_norm - ref_cx_norm  # More to the right = better

        elif relation == 'left_of':
            if cx_norm < ref_cx_norm:
                is_valid = True
                score = ref_cx_norm - cx_norm

        elif relation == 'above':
            if cy_norm < ref_cy_norm:
                is_valid = True
                score = ref_cy_norm - cy_norm

        elif relation == 'below':
            if cy_norm > ref_cy_norm:
                is_valid = True
                score = cy_norm - ref_cy_norm

        elif relation == 'near':
            dist = np.sqrt((cx_norm - ref_cx_norm)**2 + (cy_norm - ref_cy_norm)**2)
            if dist < threshold:
                is_valid = True
                score = 1.0 - dist / threshold  # Closer = better

        elif relation == 'on_top_of':
            # Above AND close in depth
            if cy_norm < ref_cy_norm and abs(d - ref_depth) < 0.5:
                is_valid = True
                score = (ref_cy_norm - cy_norm) - abs(d - ref_depth)

        elif relation == 'in_front_of':
            if d < ref_depth:
                is_valid = True
                score = ref_depth - d

        elif relation == 'behind':
            if d > ref_depth:
                is_valid = True
                score = d - ref_depth

        if is_valid:
            valid_indices.append(i)
            scores.append(score)

    if len(valid_indices) == 0:
        # Fallback: return closest to reference
        distances = []
        for i, (mask, depth) in enumerate(zip(target_masks, target_depths)):
            cx, cy = get_mask_centroid(mask)
            dist = np.sqrt((cx - ref_cx)**2 + (cy - ref_cy)**2)
            distances.append(dist)
        best_idx = int(np.argmin(distances))
        return target_masks[best_idx], best_idx

    # Return best matching mask
    best_valid_idx = int(np.argmax(scores))
    best_idx = valid_indices[best_valid_idx]
    return target_masks[best_idx], best_idx


# Batch-level Spatial Augmentation for Training

class SpatialAugmentor:
    """Augments training labels with spatial qualifiers.

    Used in training to teach the model to understand spatial language.
    """

    def __init__(self, augment_prob: float = 0.3, use_relational: bool = False):
        """
        Args:
            augment_prob: Probability of adding spatial qualifier to a label
            use_relational: If True, also generate relational queries (experimental)
        """
        self.augment_prob = augment_prob
        self.use_relational = use_relational

    def augment_batch(
        self,
        labels: List[str],
        masks: List[np.ndarray],
        depth: np.ndarray,
        scene_labels: Optional[Dict[str, List[np.ndarray]]] = None
    ) -> List[str]:
        """Augment a batch of labels with spatial qualifiers.

        Args:
            labels: List of labels for batch items
            masks: List of masks for batch items
            depth: Depth map [H, W]
            scene_labels: Optional dict mapping label -> list of all masks with that label

        Returns:
            List of augmented labels
        """
        if scene_labels is None:
            # Build scene_labels from input
            scene_labels = {}
            for label, mask in zip(labels, masks):
                if label not in scene_labels:
                    scene_labels[label] = []
                scene_labels[label].append(mask)

        augmented = []
        for label, mask in zip(labels, masks):
            aug_label = augment_label_with_spatial(
                label=label,
                mask=mask,
                depth=depth,
                all_masks_same_label=scene_labels.get(label, [mask]),
                augment_prob=self.augment_prob
            )
            augmented.append(aug_label)

        return augmented

    def __call__(self, label: str, mask: np.ndarray, depth: np.ndarray,
                 all_masks: List[np.ndarray]) -> str:
        """Augment a single label."""
        return augment_label_with_spatial(
            label, mask, depth, all_masks, augment_prob=self.augment_prob
        )


from triangulang.utils.spatial_context import (
    InstanceSpatialInfo, SpatialContext,
    compute_instance_spatial_info, get_true_spatial_qualifiers,
    generate_relational_query, GTAwareSpatialAugmentor, build_spatial_context,
)
