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


# GT-Aware Spatial Context for Training

@dataclass
class InstanceSpatialInfo:
    """Spatial information for a single object instance."""
    obj_id: int
    label: str
    centroid_2d: Tuple[float, float]  # (x, y) normalized [0, 1]
    depth_at_centroid: float  # meters
    mask_coverage: float  # fraction of image covered by mask

    def __post_init__(self):
        # Ensure centroid is tuple of floats
        self.centroid_2d = (float(self.centroid_2d[0]), float(self.centroid_2d[1]))


@dataclass
class SpatialContext:
    """Spatial context for GT-aware augmentation."""
    target_instance: InstanceSpatialInfo  # The instance being trained on
    same_label_instances: List[InstanceSpatialInfo]  # Other instances with same label
    nearby_objects: Dict[str, List[InstanceSpatialInfo]]  # label -> instances for relational queries

    def has_multi_instance(self) -> bool:
        """Check if there are multiple instances of the same label."""
        return len(self.same_label_instances) > 0

    def get_reference_objects(self) -> List[InstanceSpatialInfo]:
        """Get all objects that can serve as references for relational queries."""
        refs = []
        for label, instances in self.nearby_objects.items():
            refs.extend(instances)
        return refs


def compute_instance_spatial_info(
    mask: np.ndarray,
    depth: np.ndarray,
    obj_id: int,
    label: str
) -> Optional[InstanceSpatialInfo]:
    """Compute spatial info for a single instance.

    Args:
        mask: Binary mask [H, W]
        depth: Depth map [H, W] in meters
        obj_id: Object ID
        label: Object label

    Returns:
        InstanceSpatialInfo or None if mask is empty
    """
    H, W = mask.shape

    # Get mask pixels
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None

    # Compute centroid (normalized)
    cx = float(xs.mean()) / W
    cy = float(ys.mean()) / H

    # Get depth at centroid (using actual centroid pixel)
    cx_px, cy_px = int(xs.mean()), int(ys.mean())
    cx_px = min(max(cx_px, 0), W - 1)
    cy_px = min(max(cy_px, 0), H - 1)
    depth_val = float(depth[cy_px, cx_px])

    # Handle invalid depth
    if depth_val <= 0 or np.isnan(depth_val) or np.isinf(depth_val):
        # Use median depth within mask
        mask_depths = depth[mask > 0.5]
        valid_depths = mask_depths[(mask_depths > 0) & np.isfinite(mask_depths)]
        if len(valid_depths) > 0:
            depth_val = float(np.median(valid_depths))
        else:
            depth_val = 1.0  # fallback

    # Compute mask coverage
    coverage = float(np.sum(mask > 0.5)) / (H * W)

    return InstanceSpatialInfo(
        obj_id=obj_id,
        label=label,
        centroid_2d=(cx, cy),
        depth_at_centroid=depth_val,
        mask_coverage=coverage
    )


def get_true_spatial_qualifiers(
    target: InstanceSpatialInfo,
    same_label_instances: List[InstanceSpatialInfo],
    tolerance: float = 0.05,
    include_ordinal: bool = True,
    include_middle: bool = True,
    include_size: bool = True
) -> List[str]:
    """Determine which spatial qualifiers are apply to the target instance.

    Args:
        target: The instance we're generating a qualifier for
        same_label_instances: Other instances with the same label
        tolerance: Tolerance for "approximately equal" comparisons (normalized coords)
        include_ordinal: Include ordinal qualifiers (second nearest, etc.)
        include_middle: Include middle/central qualifiers for non-extreme instances
        include_size: Include size-based qualifiers (largest, smallest)

    Returns:
        List of valid qualifiers (e.g., ['nearest', 'leftmost', 'largest'])
    """
    if len(same_label_instances) == 0:
        return []  # No comparison possible with single instance

    all_instances = [target] + same_label_instances
    n_instances = len(all_instances)

    # Extract values for comparison
    depths = [inst.depth_at_centroid for inst in all_instances]
    xs = [inst.centroid_2d[0] for inst in all_instances]
    ys = [inst.centroid_2d[1] for inst in all_instances]
    sizes = [inst.mask_coverage for inst in all_instances]

    target_depth = target.depth_at_centroid
    target_x = target.centroid_2d[0]
    target_y = target.centroid_2d[1]
    target_size = target.mask_coverage

    qualifiers = []

    # Helper to check if target is at rank k (0-indexed) in sorted list
    def is_at_rank(values, target_val, rank, ascending=True):
        sorted_vals = sorted(values, reverse=not ascending)
        if rank >= len(sorted_vals):
            return False
        # Check if target is approximately at this rank
        rank_val = sorted_vals[rank]
        val_range = max(values) - min(values) + 1e-6
        return abs(target_val - rank_val) <= tolerance * val_range

    # Depth-based
    min_depth, max_depth = min(depths), max(depths)
    depth_range = max_depth - min_depth + 1e-6

    # Extreme qualifiers
    if target_depth <= min_depth + tolerance * depth_range:
        qualifiers.extend(['nearest', 'closest'])
    elif target_depth >= max_depth - tolerance * depth_range:
        qualifiers.extend(['farthest', 'far'])
    else:
        # Not at extreme - check for ordinal or middle
        if include_ordinal and n_instances >= 3:
            if is_at_rank(depths, target_depth, 1, ascending=True):
                qualifiers.extend(['second nearest', 'second closest'])
            if is_at_rank(depths, target_depth, 1, ascending=False):
                qualifiers.append('second farthest')

        if include_middle and n_instances >= 3:
            # Check if in middle third of depth range
            depth_normalized = (target_depth - min_depth) / depth_range
            if 0.33 <= depth_normalized <= 0.67:
                qualifiers.extend(['middle depth', 'mid-depth'])

    # X-coordinate (left/right)
    min_x, max_x = min(xs), max(xs)
    x_range = max_x - min_x + 1e-6

    if target_x <= min_x + tolerance * x_range:
        qualifiers.extend(['leftmost', 'left'])
    elif target_x >= max_x - tolerance * x_range:
        qualifiers.extend(['rightmost', 'right'])
    else:
        if include_ordinal and n_instances >= 3:
            if is_at_rank(xs, target_x, 1, ascending=True):
                qualifiers.extend(['second leftmost', 'second from left'])
            if is_at_rank(xs, target_x, 1, ascending=False):
                qualifiers.extend(['second rightmost', 'second from right'])

    # Y-coordinate (top/bottom)
    min_y, max_y = min(ys), max(ys)
    y_range = max_y - min_y + 1e-6

    if target_y <= min_y + tolerance * y_range:
        qualifiers.extend(['topmost', 'top', 'upper'])
    elif target_y >= max_y - tolerance * y_range:
        qualifiers.extend(['bottommost', 'bottom', 'lower'])
    else:
        if include_ordinal and n_instances >= 3:
            if is_at_rank(ys, target_y, 1, ascending=True):
                qualifiers.extend(['second topmost', 'second from top'])
            if is_at_rank(ys, target_y, 1, ascending=False):
                qualifiers.extend(['second bottommost', 'second from bottom'])

    # Middle/central
    if include_middle and len(qualifiers) == 0:
        # This instance isn't extreme in any dimension - call it "middle" or "central"
        qualifiers.extend(['middle', 'central'])

    # Size-based
    if include_size:
        min_size, max_size = min(sizes), max(sizes)
        size_range = max_size - min_size + 1e-6

        if target_size >= max_size - tolerance * size_range:
            qualifiers.extend(['largest', 'biggest'])
        if target_size <= min_size + tolerance * size_range:
            qualifiers.extend(['smallest', 'small'])

    return qualifiers


def generate_relational_query(
    target: InstanceSpatialInfo,
    reference: InstanceSpatialInfo,
    depth_threshold: float = 0.3,
    position_threshold: float = 0.15
) -> Optional[Tuple[str, str]]:
    """Generate a relational query describing target relative to reference.

    Args:
        target: The instance we're describing
        reference: The reference object
        depth_threshold: Min depth difference for in_front_of/behind (meters)
        position_threshold: Min position difference for left/right/above/below (normalized)

    Returns:
        (query, relation) tuple or None if no clear relation
        e.g., ("chair to the right of the table", "right_of")
    """
    if target.label == reference.label:
        return None  # Don't relate objects of same type

    tx, ty = target.centroid_2d
    rx, ry = reference.centroid_2d
    td, rd = target.depth_at_centroid, reference.depth_at_centroid

    relations = []

    # Horizontal relations
    dx = tx - rx
    if dx > position_threshold:
        relations.append(('right_of', f"{target.label} to the right of the {reference.label}"))
    elif dx < -position_threshold:
        relations.append(('left_of', f"{target.label} to the left of the {reference.label}"))

    # Vertical relations
    dy = ty - ry
    if dy > position_threshold:
        relations.append(('below', f"{target.label} below the {reference.label}"))
    elif dy < -position_threshold:
        relations.append(('above', f"{target.label} above the {reference.label}"))

    # Depth relations
    dd = td - rd
    if dd < -depth_threshold:
        relations.append(('in_front_of', f"{target.label} in front of the {reference.label}"))
    elif dd > depth_threshold:
        relations.append(('behind', f"{target.label} behind the {reference.label}"))

    # Proximity (if close in both position and depth)
    dist_2d = np.sqrt(dx**2 + dy**2)
    if dist_2d < position_threshold * 2 and abs(dd) < depth_threshold:
        relations.append(('near', f"{target.label} near the {reference.label}"))
        relations.append(('next_to', f"{target.label} next to the {reference.label}"))

    if not relations:
        return None

    # Pick one relation randomly for variety
    relation, query = random.choice(relations)
    return query, relation


class GTAwareSpatialAugmentor:
    """GT-aware spatial augmentation for training.

    Unlike the basic SpatialAugmentor, this class:
    1. Uses actual GT masks to determine spatial qualifiers
    2. Only augments when the qualifier is CORRECT
    3. Supports relational queries ("chair next to table")
    """

    def __init__(
        self,
        augment_prob: float = 0.3,
        relational_prob: float = 0.1,
        multi_instance_only: bool = True,
        qualifier_diversity: bool = True
    ):
        """
        Args:
            augment_prob: Probability of adding spatial qualifier
            relational_prob: Probability of generating relational query (vs simple qualifier)
            multi_instance_only: Only augment when multiple instances exist
            qualifier_diversity: If True, vary qualifier words (nearest vs closest)
        """
        self.augment_prob = augment_prob
        self.relational_prob = relational_prob
        self.multi_instance_only = multi_instance_only
        self.qualifier_diversity = qualifier_diversity

        # Track augmentation stats
        self.stats = {
            'total_samples': 0,
            'augmented': 0,
            'skipped_single_instance': 0,
            'skipped_no_qualifier': 0,
            'relational_queries': 0,
            'single_instance_relational': 0,  # Single instance rescued by relational
        }

    def augment(
        self,
        label: str,
        spatial_context: Optional[SpatialContext],
        force_augment: bool = False
    ) -> Tuple[str, Optional[str], Optional[int]]:
        """Augment a label with spatial qualifier based on GT context.

        Args:
            label: Base label (e.g., "chair")
            spatial_context: SpatialContext from dataloader (None = no augmentation)
            force_augment: If True, always augment (ignore augment_prob)

        Returns:
            (augmented_label, qualifier_type, spatial_idx) tuple
            - augmented_label: e.g., "nearest chair" or "chair to the right of table"
            - qualifier_type: e.g., 'depth_min', 'right_of', or None
            - spatial_idx: Index for spatial token embedding (0=none, 1=nearest, etc.)
        """
        self.stats['total_samples'] += 1

        # No context = no augmentation
        if spatial_context is None:
            return label, None, 0

        # Check probability
        if not force_augment and random.random() > self.augment_prob:
            return label, None, 0

        has_multi_instance = spatial_context.has_multi_instance()
        has_nearby_objects = len(spatial_context.get_reference_objects()) > 0

        # For single-instance, we can still try relational queries if nearby objects exist
        is_single_instance = not has_multi_instance

        # Decide: relational query or simple qualifier?
        # For single instances, ONLY try relational (simple qualifiers need multi-instance)
        # For multi-instance, use relational_prob to decide
        if is_single_instance:
            use_relational = (
                self.relational_prob > 0 and
                random.random() < self.relational_prob and
                has_nearby_objects
            )
        else:
            use_relational = (
                random.random() < self.relational_prob and
                has_nearby_objects
            )

        if use_relational:
            # Try to generate relational query
            ref_objects = spatial_context.get_reference_objects()
            random.shuffle(ref_objects)

            for ref in ref_objects:
                result = generate_relational_query(
                    spatial_context.target_instance,
                    ref
                )
                if result:
                    query, relation = result
                    self.stats['augmented'] += 1
                    self.stats['relational_queries'] += 1
                    if is_single_instance:
                        self.stats['single_instance_relational'] += 1
                    # Map relation to spatial idx (relational uses idx 0 for now)
                    return query, relation, 0

        # If single instance and relational failed, skip
        if is_single_instance:
            self.stats['skipped_single_instance'] += 1
            return label, None, 0

        # Simple qualifier based on same-label instances (multi-instance only)
        true_qualifiers = get_true_spatial_qualifiers(
            spatial_context.target_instance,
            spatial_context.same_label_instances
        )

        if not true_qualifiers:
            self.stats['skipped_no_qualifier'] += 1
            return label, None, 0

        # Pick a qualifier
        if self.qualifier_diversity:
            qualifier = random.choice(true_qualifiers)
        else:
            # Prefer standard forms
            preferred = ['nearest', 'farthest', 'leftmost', 'rightmost', 'topmost', 'bottommost']
            available = [q for q in preferred if q in true_qualifiers]
            qualifier = random.choice(available) if available else random.choice(true_qualifiers)

        # Map qualifier to type and index
        qualifier_type = SPATIAL_QUALIFIERS.get(qualifier)
        spatial_idx = SPATIAL_QUALIFIER_TO_IDX.get(qualifier_type, 0)

        # Format augmented label
        formats = [
            f"{qualifier} {label}",
            f"the {qualifier} {label}",
        ]
        augmented_label = random.choice(formats)

        self.stats['augmented'] += 1
        return augmented_label, qualifier_type, spatial_idx

    def get_stats_summary(self) -> str:
        """Get summary of augmentation statistics."""
        total = self.stats['total_samples']
        if total == 0:
            return "No samples processed"

        aug_rate = self.stats['augmented'] / total * 100
        rel_rate = self.stats['relational_queries'] / max(1, self.stats['augmented']) * 100
        single_rel = self.stats['single_instance_relational']

        return (
            f"Spatial augmentation: {self.stats['augmented']}/{total} ({aug_rate:.1f}%), "
            f"relational: {self.stats['relational_queries']} ({rel_rate:.1f}% of augmented, {single_rel} from single-instance), "
            f"skipped: single={self.stats['skipped_single_instance']}, "
            f"no_qualifier={self.stats['skipped_no_qualifier']}"
        )

    def reset_stats(self):
        """Reset augmentation statistics."""
        for key in self.stats:
            self.stats[key] = 0


def build_spatial_context(
    target_mask: np.ndarray,
    target_obj_id: int,
    target_label: str,
    depth: np.ndarray,
    scene_obj_masks: Dict[int, np.ndarray],
    obj_to_label: Dict[int, str],
    max_nearby_objects: int = 10
) -> Optional[SpatialContext]:
    """Build spatial context for GT-aware augmentation.

    Args:
        target_mask: Binary mask for target object [H, W]
        target_obj_id: Object ID of target
        target_label: Label of target
        depth: Depth map [H, W]
        scene_obj_masks: Dict mapping obj_id -> mask for all visible objects
        obj_to_label: Dict mapping obj_id -> label
        max_nearby_objects: Max number of nearby objects to include

    Returns:
        SpatialContext or None if target mask is invalid
    """
    # Compute target spatial info
    target_info = compute_instance_spatial_info(
        target_mask, depth, target_obj_id, target_label
    )
    if target_info is None:
        return None

    # Find same-label instances
    same_label_instances = []
    for obj_id, mask in scene_obj_masks.items():
        if obj_id == target_obj_id:
            continue
        label = obj_to_label.get(obj_id, '')
        if label == target_label:
            info = compute_instance_spatial_info(mask, depth, obj_id, label)
            if info is not None:
                same_label_instances.append(info)

    # Find nearby objects (different labels) for relational queries
    nearby_objects: Dict[str, List[InstanceSpatialInfo]] = {}
    other_instances = []

    for obj_id, mask in scene_obj_masks.items():
        if obj_id == target_obj_id:
            continue
        label = obj_to_label.get(obj_id, '')
        if label == target_label or not label:
            continue
        info = compute_instance_spatial_info(mask, depth, obj_id, label)
        if info is not None:
            other_instances.append(info)

    # Sort by distance to target and take top N
    def dist_to_target(inst: InstanceSpatialInfo) -> float:
        dx = inst.centroid_2d[0] - target_info.centroid_2d[0]
        dy = inst.centroid_2d[1] - target_info.centroid_2d[1]
        return dx * dx + dy * dy

    other_instances.sort(key=dist_to_target)
    for inst in other_instances[:max_nearby_objects]:
        if inst.label not in nearby_objects:
            nearby_objects[inst.label] = []
        nearby_objects[inst.label].append(inst)

    return SpatialContext(
        target_instance=target_info,
        same_label_instances=same_label_instances,
        nearby_objects=nearby_objects
    )


# Testing

if __name__ == '__main__':
    # Test spatial qualifier parsing
    test_prompts = [
        "nearest chair",
        "the leftmost monitor",
        "farthest table",
        "bottom shelf",
        "chair",  # No qualifier
    ]

    print("Spatial Qualifier Parsing")
    for prompt in test_prompts:
        qualifier, base = parse_spatial_qualifier(prompt)
        print(f"  '{prompt}' -> qualifier={qualifier}, base='{base}'")

    # Test relational query parsing
    test_relational = [
        "chair to the right of the table",
        "monitor above the keyboard",
        "lamp near the bed",
        "bottle on the shelf",
        "cat behind the couch",
    ]

    print("\nRelational Query Parsing")
    for prompt in test_relational:
        target, ref, relation = parse_relational_query(prompt)
        print(f"  '{prompt}' -> target='{target}', ref='{ref}', relation='{relation}'")

    # Test pseudo-point conversion
    print("\nSpatial to Pseudo-Point")
    for qualifier_type in ['depth_min', 'x_min', 'x_max', 'y_min', 'y_max']:
        points, labels = spatial_to_pseudo_point(qualifier_type, None, 518, 518)
        print(f"  {qualifier_type} -> points={points}, labels={labels}")

    print("\nSpatial Augmentation")
    # Create dummy data
    mask1 = np.zeros((128, 128))
    mask1[20:40, 20:40] = 1  # Top-left
    mask2 = np.zeros((128, 128))
    mask2[80:100, 80:100] = 1  # Bottom-right
    depth = np.random.rand(128, 128) + 0.5

    augmentor = SpatialAugmentor(augment_prob=1.0)
    for _ in range(5):
        aug1 = augmentor("chair", mask1, depth, [mask1, mask2])
        aug2 = augmentor("chair", mask2, depth, [mask1, mask2])
        print(f"  mask1 (top-left): '{aug1}'")
        print(f"  mask2 (bottom-right): '{aug2}'")
