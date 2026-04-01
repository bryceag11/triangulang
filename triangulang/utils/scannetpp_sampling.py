"""ScanNet++ view-sampling utilities.

Standalone functions for camera-aware view selection.  Extracted from
ScanNetPPMultiViewDataset to keep scannetpp_loader.py under 1000 lines.

Strategies
----------
random       : uniform random subset
stratified   : evenly-spaced temporal subset
sequential   : consecutive window
chunk_aware  : random within a single DA3 temporal chunk
overlap_30   : high-overlap views within a DA3 chunk (min 0.30 pair overlap)
overlap_50   : very-high-overlap views within a DA3 chunk (min 0.50 pair overlap)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Low-level helpers

def estimate_view_overlap(
    pos1: np.ndarray, dir1: np.ndarray,
    pos2: np.ndarray, dir2: np.ndarray,
    baseline_scale: float = 1.5,
) -> float:
    """Estimate visual overlap between two camera views (0–1 score)."""
    baseline = np.linalg.norm(pos1 - pos2)
    cos_angle = np.clip(np.dot(dir1, dir2), -1, 1)
    dist_score = np.exp(-baseline / baseline_scale)
    angle_score = (cos_angle + 1) / 2
    return float(np.sqrt(dist_score * angle_score))


def get_chunk_groups(
    scene_id: str,
    images: List[str],
    scene_chunk_map: Dict[str, Dict[str, int]],
) -> Dict[int, List[str]]:
    """Group image names by their DA3 processing chunk index."""
    stem_to_chunk = scene_chunk_map.get(scene_id)
    if not stem_to_chunk:
        return {}

    chunk_groups: Dict[int, List[str]] = {}
    for img_name in images:
        stem = Path(img_name).stem
        chunk_idx = stem_to_chunk.get(stem)
        if chunk_idx is not None:
            chunk_groups.setdefault(chunk_idx, []).append(img_name)
    return chunk_groups


def select_chunk_group(
    chunk_groups: Dict[int, List[str]],
    n_views: int,
    prefer_overlap: bool = False,
    transforms: Optional[dict] = None,
) -> List[str]:
    """Select a DA3 chunk group that contains enough images for sampling."""
    if not chunk_groups:
        return []

    valid_chunks = [(k, v) for k, v in chunk_groups.items() if len(v) >= n_views]
    if not valid_chunks:
        # Return the largest chunk even if undersized
        return max(chunk_groups.items(), key=lambda x: len(x[1]))[1]

    if prefer_overlap and transforms:
        frame_lookup = {
            Path(fr.get('file_path', '')).name: fr
            for fr in transforms.get('frames', [])
        }
        best_chunk: Optional[List[str]] = None
        best_score = -1.0

        for _key, imgs in valid_chunks:
            positions = []
            for img in imgs:
                if img in frame_lookup:
                    c2w = np.array(frame_lookup[img]['transform_matrix'], dtype=np.float64)
                    pos = c2w[:3, 3]
                    view_dir = -c2w[:3, 2]
                    norm = np.linalg.norm(view_dir)
                    if norm > 1e-6:
                        view_dir = view_dir / norm
                    positions.append((pos, view_dir))

            if len(positions) >= 2:
                total_overlap = count = 0
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        total_overlap += estimate_view_overlap(
                            positions[i][0], positions[i][1],
                            positions[j][0], positions[j][1],
                        )
                        count += 1
                avg_overlap = total_overlap / count if count > 0 else 0.0
                if avg_overlap > best_score:
                    best_score = avg_overlap
                    best_chunk = imgs

        if best_chunk is not None:
            return best_chunk

    return random.choice(valid_chunks)[1]


def sample_views_overlap(
    images: List[str],
    n_views: int,
    transforms: Optional[dict],
    min_overlap: float = 0.3,
) -> List[str]:
    """Greedy high-overlap view selection within *images*."""
    if not transforms or len(images) <= n_views:
        return random.sample(images, min(n_views, len(images)))

    frame_lookup = {
        Path(fr.get('file_path', '')).name: fr
        for fr in transforms.get('frames', [])
    }
    camera_info: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for img in images:
        if img in frame_lookup:
            c2w = np.array(frame_lookup[img]['transform_matrix'], dtype=np.float64)
            pos = c2w[:3, 3]
            view_dir = -c2w[:3, 2]
            norm = np.linalg.norm(view_dir)
            if norm > 1e-6:
                view_dir = view_dir / norm
            camera_info[img] = (pos, view_dir)

    available = [img for img in images if img in camera_info]
    if len(available) < n_views:
        return random.sample(images, min(n_views, len(images)))

    selected = [random.choice(available)]
    remaining = set(available) - set(selected)

    while len(selected) < n_views and remaining:
        best_img: Optional[str] = None
        best_score = -1.0

        # First pass: require min_overlap
        for img in remaining:
            pos, vdir = camera_info[img]
            max_ovlp = max(
                estimate_view_overlap(pos, vdir, *camera_info[s])
                for s in selected
            )
            if max_ovlp >= min_overlap and max_ovlp > best_score:
                best_score = max_ovlp
                best_img = img

        # Second pass: relax constraint if nothing found
        if best_img is None:
            for img in remaining:
                pos, vdir = camera_info[img]
                max_ovlp = max(
                    estimate_view_overlap(pos, vdir, *camera_info[s])
                    for s in selected
                )
                if max_ovlp > best_score:
                    best_score = max_ovlp
                    best_img = img

        if best_img:
            selected.append(best_img)
            remaining.remove(best_img)
        else:
            break

    if len(selected) < n_views:
        remaining_pool = [img for img in images if img not in selected]
        needed = n_views - len(selected)
        selected.extend(random.sample(remaining_pool, min(needed, len(remaining_pool))))

    return selected


# Top-level dispatcher

def sample_views(
    images: List[str],
    n_views: int,
    strategy: str,
    scene_id: Optional[str] = None,
    transforms: Optional[dict] = None,
    use_cached_depth: bool = False,
    scene_chunk_map: Optional[Dict[str, Dict[str, int]]] = None,
    chunk_warning_ref: Optional[list] = None,
) -> List[str]:
    """Select *n_views* images from *images* using the given strategy.

    Parameters
    ----------
    chunk_warning_ref
        Pass a single-element list ``[False]``; set to ``[True]`` once a
        chunk-size warning has been printed, preventing duplicates across calls.
    """
    if len(images) <= n_views:
        return images

    needs_chunk = strategy in ('chunk_aware', 'overlap_30', 'overlap_50')

    if needs_chunk:
        chunk_groups_map: Dict[int, List[str]] = {}
        if scene_id and use_cached_depth and scene_chunk_map:
            chunk_groups_map = get_chunk_groups(scene_id, images, scene_chunk_map)

        needs_overlap = strategy in ('overlap_30', 'overlap_50')

        if chunk_groups_map:
            chunk_images = select_chunk_group(
                chunk_groups_map, n_views,
                prefer_overlap=needs_overlap,
                transforms=transforms,
            )
            if len(chunk_images) < n_views and chunk_warning_ref is not None and not chunk_warning_ref[0]:
                print(
                    f"[Sampling] Warning: No DA3 chunk has {n_views} views. "
                    f"Using largest chunk ({len(chunk_images)} views). "
                    f"Max views per chunk is 16."
                )
                chunk_warning_ref[0] = True
            images = chunk_images

        if len(images) <= n_views:
            return images

        if strategy == 'overlap_30':
            return sample_views_overlap(images, n_views, transforms, min_overlap=0.3)
        elif strategy == 'overlap_50':
            return sample_views_overlap(images, n_views, transforms, min_overlap=0.5)
        else:  # chunk_aware
            return random.sample(images, n_views)

    if strategy == 'random':
        return random.sample(images, n_views)
    elif strategy == 'stratified':
        step = len(images) / n_views
        indices = [int(i * step) for i in range(n_views)]
        return [images[i] for i in indices]
    else:  # sequential
        if len(images) == n_views:
            return images
        start = random.randint(0, len(images) - n_views)
        return images[start:start + n_views]
