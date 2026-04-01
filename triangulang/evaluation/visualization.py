"""Visualization utilities for benchmark results and paper figures."""
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


from triangulang.evaluation.paper_viz import (
    create_paper_grid, collect_paper_viz_from_results,
    generate_paper_visualizations, generate_single_object_viz,
)
