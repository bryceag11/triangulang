"""Grid visualization of decoder outputs (mask, IoU, presence, centroid).

Usage:
    python scripts/demo/grid_output.py \
        --image path/to/image.jpg \
        --prompt "chair" \
        --checkpoint checkpoints/<run>/best.pt \
        --output output/grid.png
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'sam3'))
sys.path.insert(0, str(PROJECT_ROOT / 'depth_anything_v3' / 'src'))

def create_paper_figure(
    image: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    centroid_3d: np.ndarray,
    iou_score: float,
    presence_score: float,
    prompt: str,
    output_path: str,
    dpi: int = 600,
    mask_alpha: float = 0.4,
):
    """
    Create publication-quality figure showing decoder outputs.

    Layout: Single image with mask overlay, plus annotation panel on right side
    showing IoU, Presence, Mask label, and Centroid.

    Args:
        image: [H, W, 3] RGB image (uint8)
        mask: [H, W] binary mask
        depth: [H, W] depth map in meters
        centroid_3d: [3] XYZ centroid in meters (camera frame)
        iou_score: predicted IoU confidence (0-1)
        presence_score: object presence probability (0-1)
        prompt: text query used
        output_path: where to save (600 DPI PNG)
        dpi: output resolution (default 600)
        mask_alpha: mask overlay opacity (default 0.4)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import AnchoredText
    from scipy import ndimage

    H, W = image.shape[:2]

    # Figure with two panels: main image + annotation strip
    fig_width = 8  # inches
    annotation_width_ratio = 0.3
    fig_height = fig_width * H / W / (1 + annotation_width_ratio)

    fig, (ax_main, ax_info) = plt.subplots(
        1, 2,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [1, annotation_width_ratio]},
        facecolor='white'
    )

    # Main panel
    ax_main.imshow(image)

    # Mask overlay at specified opacity
    mask_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    # Use a teal/cyan color for the mask
    mask_color = np.array([0.0, 0.75, 0.85])  # teal
    mask_overlay[mask > 0, :3] = mask_color
    mask_overlay[mask > 0, 3] = mask_alpha

    # Add edge glow
    if mask.sum() > 0:
        dilated = ndimage.binary_dilation(mask, iterations=2)
        edge = dilated & ~mask
        mask_overlay[edge, :3] = [1.0, 1.0, 1.0]  # white edge
        mask_overlay[edge, 3] = 0.6

    ax_main.imshow(mask_overlay)

    # Mark centroid on image (project 3D centroid to 2D)
    if mask.sum() > 0:
        ys, xs = np.where(mask > 0)
        cx_2d, cy_2d = xs.mean(), ys.mean()
        ax_main.plot(cx_2d, cy_2d, 'o', color='red', markersize=6,
                     markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        # Centroid label near the point
        ax_main.annotate(
            f'({centroid_3d[0]:+.2f}, {centroid_3d[1]:+.2f}, {centroid_3d[2]:.2f})m',
            xy=(cx_2d, cy_2d), xytext=(cx_2d + W * 0.05, cy_2d - H * 0.05),
            fontsize=5, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='white', lw=0.8),
            zorder=11
        )

    # "Mask" label with arrow
    if mask.sum() > 0:
        # Find a point on the mask boundary for the arrow
        ys, xs = np.where(mask > 0)
        # Pick a point near the top of the mask
        top_idx = ys.argmin()
        mx, my = xs[top_idx], ys[top_idx]
        ax_main.annotate(
            'Mask',
            xy=(mx, my), xytext=(mx + W * 0.08, max(my - H * 0.12, H * 0.05)),
            fontsize=6, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=tuple(mask_color), alpha=0.85, edgecolor='white', linewidth=0.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.0),
            zorder=12
        )

    # Query text at top
    ax_main.set_title(f'Query: "{prompt}"', fontsize=7, fontweight='bold', pad=4)
    ax_main.axis('off')

    # Annotation panel
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')
    ax_info.set_facecolor('#f5f5f5')

    # Draw info blocks
    blocks = [
        {
            'label': 'IoU',
            'sublabel': 'How confident am I?',
            'value': f'{iou_score:.1%}',
            'color': _score_color(iou_score),
            'y': 0.82,
        },
        {
            'label': 'Presence',
            'sublabel': 'Is there an object here?',
            'value': f'{presence_score:.1%}',
            'color': _score_color(presence_score),
            'y': 0.58,
        },
        {
            'label': 'Mask',
            'sublabel': 'Where is the object?',
            'value': f'{mask.sum():,} px',
            'color': tuple(mask_color),
            'y': 0.34,
        },
        {
            'label': 'Centroid',
            'sublabel': 'Where is it located?',
            'value': f'({centroid_3d[0]:+.1f}, {centroid_3d[1]:+.1f}, {centroid_3d[2]:.1f})m',
            'color': (0.8, 0.2, 0.2),
            'y': 0.10,
        },
    ]

    for block in blocks:
        y = block['y']
        # Block background
        rect = mpatches.FancyBboxPatch(
            (0.05, y), 0.9, 0.18,
            boxstyle='round,pad=0.02',
            facecolor='white', edgecolor='#cccccc', linewidth=0.5,
            transform=ax_info.transAxes, zorder=1
        )
        ax_info.add_patch(rect)

        # Color indicator bar on left
        bar = mpatches.FancyBboxPatch(
            (0.05, y), 0.04, 0.18,
            boxstyle='round,pad=0.01',
            facecolor=block['color'], edgecolor='none',
            transform=ax_info.transAxes, zorder=2
        )
        ax_info.add_patch(bar)

        # Label
        ax_info.text(0.15, y + 0.135, block['label'],
                     transform=ax_info.transAxes, fontsize=6,
                     fontweight='bold', color='#333333', va='center')
        # Sublabel
        ax_info.text(0.15, y + 0.09, block['sublabel'],
                     transform=ax_info.transAxes, fontsize=4,
                     color='#888888', va='center', style='italic')
        # Value
        ax_info.text(0.15, y + 0.04, block['value'],
                     transform=ax_info.transAxes, fontsize=5.5,
                     fontweight='bold', color=block['color'], va='center',
                     fontfamily='monospace')

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"Saved publication figure to {output_path} ({dpi} DPI)")

def _score_color(score: float) -> tuple:
    """Map 0-1 score to red-yellow-green color."""
    if score > 0.7:
        return (0.2, 0.7, 0.2)  # green
    elif score > 0.4:
        return (0.9, 0.7, 0.1)  # yellow
    else:
        return (0.8, 0.2, 0.2)  # red

def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.amp import autocast

    parser = argparse.ArgumentParser(description='Grid visualization output')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True, help='Text query')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='demo/grid_output.png', help='Output path')
    parser.add_argument('--dpi', type=int, default=600, help='Output DPI (default 600)')
    parser.add_argument('--mask-alpha', type=float, default=0.4, help='Mask overlay opacity (default 0.4)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--size', type=int, default=518, help='Input size for SAM3')

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Import model loader from demo_locate_3d
    from scripts.demo_locate_3d import (
        load_model, load_image, parse_spatial_qualifier,
        get_multi_instance_masks, filter_by_spatial_qualifier,
        unproject_to_3d
    )

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load image
    img_tensor, img_np, orig_size = load_image(args.image, size=args.size)

    # Parse spatial qualifier
    qualifier, base_prompt = parse_spatial_qualifier(args.prompt)
    effective_prompt = base_prompt if qualifier else args.prompt

    # Run inference
    with torch.no_grad():
        with autocast('cuda', dtype=torch.float16):
            outputs = model(img_tensor.to(device), [effective_prompt], gt_masks=None)

    all_masks = outputs['all_masks'][0]  # [Q, H, W]
    depth = outputs['depth'][0, 0].cpu().numpy()  # [H, W]

    # Scale depth for DA3METRIC
    da3_model_name = config.get('da3_model', 'depth-anything/DA3METRIC-LARGE')
    if 'DA3METRIC' in da3_model_name.upper():
        focal_length = args.size * 0.96
        depth = depth * (focal_length / 300.0)
    else:
        focal_length = args.size * 0.96

    # Get IoU predictions
    iou_preds = None
    if 'iou_pred' in outputs and outputs['iou_pred'] is not None:
        iou_preds = outputs['iou_pred'][0]

    # Get presence predictions
    presence_score = 0.0
    if 'presence_pred' in outputs and outputs['presence_pred'] is not None:
        presence_logit = outputs['presence_pred'][0]
        if hasattr(presence_logit, 'item'):
            presence_score = torch.sigmoid(presence_logit).item()
        else:
            presence_score = float(torch.sigmoid(torch.tensor(presence_logit)))

    # Select mask
    if qualifier:
        masks, scores = get_multi_instance_masks(model, all_masks, iou_preds)
        if len(masks) > 1:
            mask, _ = filter_by_spatial_qualifier(masks, depth, qualifier)
        elif len(masks) > 0:
            mask = masks[0]
        else:
            mask = np.zeros(all_masks.shape[-2:], dtype=bool)
        iou_score = scores[0] if scores else 0.0
    else:
        if iou_preds is not None:
            best_idx = iou_preds.argmax()
            iou_score = iou_preds[best_idx].item()
        else:
            pred_logits = all_masks.mean(dim=(1, 2))
            best_idx = pred_logits.argmax()
            iou_score = torch.sigmoid(pred_logits[best_idx]).item()

        best_mask = all_masks[best_idx]
        if best_mask.shape != img_tensor.shape[-2:]:
            best_mask = F.interpolate(
                best_mask.unsqueeze(0).unsqueeze(0),
                size=img_tensor.shape[-2:],
                mode='bilinear', align_corners=False
            ).squeeze()
        mask = (torch.sigmoid(best_mask) > 0.5).cpu().numpy()

    # Resize depth to match mask
    if depth.shape != mask.shape:
        depth = np.array(
            Image.fromarray(depth.astype(np.float32)).resize(
                mask.shape[::-1], Image.BILINEAR
            )
        )

    # Compute 3D centroid
    points_3d, centroid_3d = unproject_to_3d(mask, depth, fx=focal_length, fy=focal_length)

    print(f"Query: '{args.prompt}'")
    print(f"Mask pixels: {mask.sum():,}")
    print(f"IoU score: {iou_score:.3f}")
    print(f"Presence: {presence_score:.3f}")
    print(f"Centroid: ({centroid_3d[0]:+.2f}, {centroid_3d[1]:+.2f}, {centroid_3d[2]:.2f})m")

    # Create figure
    create_paper_figure(
        image=img_np,
        mask=mask,
        depth=depth,
        centroid_3d=centroid_3d,
        iou_score=iou_score,
        presence_score=presence_score,
        prompt=args.prompt,
        output_path=args.output,
        dpi=args.dpi,
        mask_alpha=args.mask_alpha,
    )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
