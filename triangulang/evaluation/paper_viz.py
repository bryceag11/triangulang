"""Paper-quality visualization functions for TrianguLang results."""
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import triangulang

logger = triangulang.get_logger(__name__)

from triangulang.evaluation.visualization import MASK_COLORS, overlay_mask_sam3_style

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
            logger.warning(f"  [paper-viz] No samples match target objects {target_objects}, using all")
            pool = viz_pool

    grid_sets = []

    if mode == 'single_object':
        # Group by (scene_id, label) to find objects with multiple views

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
        logger.info("  [paper-viz] No visualization data collected, skipping.")
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
        logger.debug(f"  [paper-viz] IoU filter (>= {min_iou:.0%}): {original_count} -> {len(viz_pool)} samples")
        if not viz_pool:
            logger.info(f"  [paper-viz] No samples above {min_iou:.0%} IoU threshold, skipping.")
            return

    grid_sets = collect_paper_viz_from_results(
        viz_pool, mode=mode, target_objects=target_objects,
        num_rows=num_rows, num_sets=num_sets, seed=seed,
    )

    if not grid_sets:
        logger.info("  [paper-viz] Could not assemble any grid sets from collected data.")
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
        logger.debug(f"  [paper-viz] Set {set_idx} rows:")
        for row_idx, r in enumerate(grid_rows):
            depth_src = r.get('depth_source', 'unknown')
            frame = r.get('frame_name', 'unknown')
            label = r.get('label', '')
            scene = r.get('scene_id', '')
            logger.debug(f"    Row {row_idx}: {scene}/{frame} '{label}' depth={depth_src}")

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
        logger.info(f"  [paper-viz] Saved {fig_path} ({dpi} DPI)")

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
    logger.info(f"  [paper-viz] Metadata saved to {meta_path}")


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
        logger.info("  [single-object-viz] No visualization data collected, skipping.")
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
        logger.debug(f"  [single-object-viz] IoU filter (>= {min_iou:.0%}): {original_count} -> {len(viz_pool)} samples")
        if not viz_pool:
            logger.info(f"  [single-object-viz] No samples above {min_iou:.0%} IoU threshold, skipping.")
            return
    dpi = getattr(args, 'paper_viz_dpi', 600)
    mask_color = getattr(args, 'paper_viz_mask_color', 'white')
    overlay_alpha = getattr(args, 'paper_viz_overlay_alpha', 0.5)

    # Check how many scenes we have
    from collections import defaultdict
    scenes_in_pool = set(v.get('scene_id', 'unknown') for v in viz_pool)
    multi_scene = len(scenes_in_pool) > 1
    logger.info(f"  [single-object-viz] {len(viz_pool)} samples from {len(scenes_in_pool)} scene(s)")

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
        logger.info(f"  [single-object-viz] Scene: {scene_id}")

        # Determine which objects to visualize for this scene
        if target_objects:
            objects_to_viz = [obj for obj in target_objects if obj in by_label]
            if not objects_to_viz:
                logger.info(f"    No target objects found, skipping scene")
                continue
        else:
            # Pick objects with BEST IoU for this scene
            label_best_iou = {}
            for label, samples in by_label.items():
                best_iou = max(s.get('iou', 0) for s in samples)
                label_best_iou[label] = best_iou
            sorted_labels = sorted(label_best_iou.keys(), key=lambda l: label_best_iou[l], reverse=True)
            objects_to_viz = sorted_labels[:num_objects]

        logger.info(f"    Visualizing {len(objects_to_viz)} objects, top-{topk} IoU frames each:")

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

            logger.debug(f"      {obj_label}: {len(samples)} total, top {len(top_samples)}")

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

            logger.info(f"    Saving {len(scene_rows)} samples to: {separate_dir}")

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
        logger.info(f"    Saved grid: {grid_path}")

    # Save detailed metadata JSON
    # Include rank in filename if running distributed to avoid conflicts
    if ddp_rank is not None:
        meta_path = viz_dir / f'metadata_rank{ddp_rank}.json'
    else:
        meta_path = viz_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"  [single-object-viz] Saved metadata: {meta_path}")

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
    logger.info(f"  [single-object-viz] Saved summary: {summary_path}")
