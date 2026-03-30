"""Demo: spatial and relational query disambiguation.

Supports queries like "nearest chair", "leftmost monitor", or
"chair to the right of the table" using depth-based spatial reasoning.

Usage:
    python scripts/demo/spatial_query.py \
        --checkpoint checkpoints/<run>/best.pt \
        --image path/to/image.jpg \
        --prompt "nearest chair"
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'sam3'))
sys.path.insert(0, str(PROJECT_ROOT / 'depth_anything_v3' / 'src'))

from triangulang.utils.spatial_reasoning import (
    parse_spatial_qualifier,
    parse_relational_query,
    get_spatial_qualifier_idx,
    filter_by_relation,
    get_mask_centroid,
    get_depth_at_centroid,
    SPATIAL_QUALIFIERS,
)
from depth_anything_3.utils.visualize import visualize_depth

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    from triangulang.training.train_2d_improved import TrianguLangModel
    from sam3 import build_sam3_image_model
    from depth_anything_3.api import DepthAnything3

    _BPE_PATH = str(PROJECT_ROOT / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    sam3_model = build_sam3_image_model(bpe_path=_BPE_PATH).to(device)
    da3_model = DepthAnything3.from_pretrained(
        config.get('da3_model', 'depth-anything/DA3METRIC-LARGE')
    ).to(device)

    model = TrianguLangModel(
        sam3_model=sam3_model,
        da3_model=da3_model,
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        num_queries=config.get('num_queries', 100),
        use_presence_token=config.get('use_presence_token', True),
        use_box_prompts=config.get('use_box_prompts', True),
        use_point_prompts=config.get('use_point_prompts', True),
        use_world_pe=config.get('use_world_pe', True),
        use_gasa=config.get('use_gasa', True),
        mask_selection=config.get('mask_selection', 'iou_match'),
        use_iou_head=config.get('use_iou_head', False),
        use_spatial_tokens=config.get('use_spatial_tokens', False),
        use_spatial_points=config.get('use_spatial_points', False),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    return model, config

def get_all_masks_nms(model, image_tensor, prompt, device, iou_threshold=0.5, max_masks=10):
    """Get all non-overlapping masks for a prompt using NMS."""
    with torch.no_grad():
        outputs = model(image_tensor, [prompt], gt_masks=None)

    all_masks = outputs['all_masks'][0]  # [Q, H, W]
    mask_logits = all_masks.mean(dim=(-2, -1))  # [Q]

    # Sort by confidence
    sorted_idx = torch.argsort(mask_logits, descending=True)

    # NMS
    selected_masks = []
    selected_depths = []
    depth = outputs['depth'][0, 0].cpu().numpy()

    for idx in sorted_idx[:max_masks * 3]:  # Check more masks than needed
        mask = (all_masks[idx] > 0).cpu().numpy()
        if mask.sum() < 100:  # Skip tiny masks
            continue

        # Check IoU with selected masks
        is_duplicate = False
        for sel_mask in selected_masks:
            intersection = (mask & sel_mask).sum()
            union = (mask | sel_mask).sum()
            iou = intersection / (union + 1e-6)
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected_masks.append(mask)
            selected_depths.append(depth)
            if len(selected_masks) >= max_masks:
                break

    return selected_masks, selected_depths

def filter_by_spatial_qualifier(masks, depths, qualifier_type):
    """Filter masks by spatial qualifier."""
    if len(masks) == 0:
        return None, -1

    if len(masks) == 1:
        return masks[0], 0

    # Compute centroids and depths
    centroids = []
    depths_at_centroid = []
    for mask, depth in zip(masks, depths):
        cx, cy = get_mask_centroid(mask)
        d = get_depth_at_centroid(mask, depth)
        centroids.append((cx, cy))
        depths_at_centroid.append(d)

    # Filter based on qualifier
    if qualifier_type == 'depth_min':  # nearest
        best_idx = int(np.argmin(depths_at_centroid))
    elif qualifier_type == 'depth_max':  # farthest
        best_idx = int(np.argmax(depths_at_centroid))
    elif qualifier_type == 'x_min':  # leftmost
        best_idx = int(np.argmin([c[0] for c in centroids]))
    elif qualifier_type == 'x_max':  # rightmost
        best_idx = int(np.argmax([c[0] for c in centroids]))
    elif qualifier_type == 'y_min':  # top
        best_idx = int(np.argmin([c[1] for c in centroids]))
    elif qualifier_type == 'y_max':  # bottom
        best_idx = int(np.argmax([c[1] for c in centroids]))
    else:
        best_idx = 0  # Default to first

    return masks[best_idx], best_idx

def process_spatial_query(model, image_tensor, prompt, device):
    """Process a query with spatial qualifier."""
    qualifier_type, base_prompt = parse_spatial_qualifier(prompt)

    print(f"  Parsed: qualifier={qualifier_type}, base='{base_prompt}'")

    # Get all candidate masks
    masks, depths = get_all_masks_nms(model, image_tensor, base_prompt, device)
    print(f"  Found {len(masks)} candidate masks for '{base_prompt}'")

    if len(masks) == 0:
        print("  No masks found!")
        return None, None

    # Filter by spatial qualifier
    if qualifier_type:
        best_mask, best_idx = filter_by_spatial_qualifier(masks, depths, qualifier_type)
        print(f"  Selected mask {best_idx} based on '{qualifier_type}'")
    else:
        best_mask = masks[0]
        best_idx = 0

    # Get depth for the selected mask
    depth = depths[best_idx]
    centroid = get_mask_centroid(best_mask)
    depth_val = get_depth_at_centroid(best_mask, depth)
    print(f"  Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}), Depth: {depth_val:.2f}m")

    return best_mask, depth

def process_relational_query_sequential(model, image_tensor, prompt, device):
    """Process relational query with two sequential queries."""
    target, reference, relation = parse_relational_query(prompt)

    if target is None:
        print(f"  Not a relational query, falling back to spatial query")
        return process_spatial_query(model, image_tensor, prompt, device)

    print(f"  Parsed: target='{target}', reference='{reference}', relation='{relation}'")

    # Query 1: Find reference object
    print(f"  Finding reference: '{reference}'")
    with torch.no_grad():
        ref_outputs = model(image_tensor, [reference], gt_masks=None)
    ref_mask = (ref_outputs['pred_masks'][0, 0] > 0).cpu().numpy()
    ref_depth = ref_outputs['depth'][0, 0].cpu().numpy()

    if ref_mask.sum() < 100:
        print(f"  Reference object '{reference}' not found!")
        return None, None

    ref_centroid = get_mask_centroid(ref_mask)
    print(f"  Reference centroid: ({ref_centroid[0]:.1f}, {ref_centroid[1]:.1f})")

    # Query 2: Find all target objects
    print(f"  Finding targets: '{target}'")
    target_masks, target_depths = get_all_masks_nms(model, image_tensor, target, device)
    print(f"  Found {len(target_masks)} candidate targets")

    if len(target_masks) == 0:
        print(f"  No target objects found!")
        return None, None

    # Filter by relation
    best_mask, best_idx = filter_by_relation(
        target_masks, target_depths, ref_mask, ref_depth, relation
    )

    if best_mask is not None:
        centroid = get_mask_centroid(best_mask)
        print(f"  Selected target {best_idx} at ({centroid[0]:.1f}, {centroid[1]:.1f})")

    return best_mask, ref_depth

def process_relational_query_parallel(model, image_tensor, prompt, device):
    """Process relational query with single multi-prompt query (faster)."""
    target, reference, relation = parse_relational_query(prompt)

    if target is None:
        print(f"  Not a relational query, falling back to spatial query")
        return process_spatial_query(model, image_tensor, prompt, device)

    print(f"  Parsed: target='{target}', reference='{reference}', relation='{relation}'")
    print(f"  Using parallel multi-prompt query")

    # Single forward pass with both prompts
    # Note: This requires the model to support multiple text prompts
    with torch.no_grad():
        # For now, run sequentially since multi-prompt batching needs special handling
        # In full implementation, would batch [target, reference] together
        target_outputs = model(image_tensor, [target], gt_masks=None)
        ref_outputs = model(image_tensor, [reference], gt_masks=None)

    # Get masks and depths
    ref_mask = (ref_outputs['pred_masks'][0, 0] > 0).cpu().numpy()
    ref_depth = ref_outputs['depth'][0, 0].cpu().numpy()

    if ref_mask.sum() < 100:
        print(f"  Reference object '{reference}' not found!")
        return None, None

    # Get all target masks
    target_masks, target_depths = get_all_masks_nms(model, image_tensor, target, device)

    if len(target_masks) == 0:
        print(f"  No target objects found!")
        return None, None

    # Filter by relation
    best_mask, best_idx = filter_by_relation(
        target_masks, target_depths, ref_mask, ref_depth, relation
    )

    return best_mask, ref_depth

def visualize_result(image, mask, depth, output_path, prompt):
    """Save visualization of result."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Query: '{prompt}'")
    axes[0].axis('off')

    # Mask overlay
    axes[1].imshow(image)
    if mask is not None:
        mask_overlay = np.zeros((*mask.shape, 4))
        mask_overlay[mask > 0] = [1, 0, 0, 0.5]  # Red with alpha
        axes[1].imshow(mask_overlay)
        axes[1].set_title("Selected Mask")
    else:
        axes[1].set_title("No mask found")
    axes[1].axis('off')

    # Depth (use DA3's Spectral colormap: warm=close, cool=far)
    if depth is not None:
        depth_vis = visualize_depth(depth, cmap="Spectral")
        axes[2].imshow(depth_vis)
        axes[2].set_title(f"Depth Map ({depth.min():.1f}-{depth.max():.1f}m)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Spatial and Relational Query Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Query prompt (e.g., "nearest chair", "chair to the right of the table")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output visualization path')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel multi-prompt for relational queries (faster)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        args.output = f"spatial_query_{Path(args.image).stem}.png"

    print(f"Query: '{args.prompt}'")
    print(f"Image: {args.image}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)

    # Load and preprocess image
    image = Image.open(args.image).convert('RGB')
    image_np = np.array(image)

    # Resize to model resolution
    resolution = 1008
    image_resized = image.resize((resolution, resolution), Image.BILINEAR)
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(args.device)

    # Check if relational or spatial query
    target, reference, relation = parse_relational_query(args.prompt)

    print(f"\nProcessing query...")
    if target is not None:
        # Relational query
        if args.parallel:
            mask, depth = process_relational_query_parallel(model, image_tensor, args.prompt, args.device)
        else:
            mask, depth = process_relational_query_sequential(model, image_tensor, args.prompt, args.device)
    else:
        # Spatial qualifier query
        mask, depth = process_spatial_query(model, image_tensor, args.prompt, args.device)

    # Resize mask back to original image size if needed
    if mask is not None and mask.shape != image_np.shape[:2]:
        from skimage.transform import resize
        mask = resize(mask.astype(float), image_np.shape[:2], order=0) > 0.5

    # Visualize
    print(f"\nSaving visualization...")
    visualize_result(image_np, mask, depth, args.output, args.prompt)

    print(f"\nDone!")

if __name__ == '__main__':
    main()
