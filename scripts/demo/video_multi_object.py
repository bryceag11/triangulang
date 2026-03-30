"""Multi-object video demo with colored mask overlays and text labels.

Supports live inference or replay from saved eval results.

Usage:
    python scripts/demo/video_multi_object.py \
        --checkpoint checkpoints/<run>/best.pt \
        --scene data/scannetpp/data/<scene_id> \
        --output viz/mo_demo.mp4
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'sam3'))
sys.path.insert(0, str(PROJECT_ROOT / 'depth_anything_v3' / 'src'))

# Distinct, high-contrast colors for up to 20 objects (RGB)
OBJECT_COLORS = [
    (0, 191, 217),    # cyan
    (255, 107, 107),   # coral
    (78, 205, 96),     # green
    (255, 193, 7),     # amber
    (156, 39, 176),    # purple
    (255, 152, 0),     # orange
    (33, 150, 243),    # blue
    (244, 67, 54),     # red
    (0, 188, 212),     # teal
    (205, 220, 57),    # lime
    (233, 30, 99),     # pink
    (121, 85, 72),     # brown
    (63, 81, 181),     # indigo
    (0, 150, 136),     # dark teal
    (255, 87, 34),     # deep orange
    (103, 58, 183),    # deep purple
    (139, 195, 74),    # light green
    (3, 169, 244),     # light blue
    (255, 235, 59),    # yellow
    (96, 125, 139),    # blue grey
]

def draw_text_with_bg(img, text, pos, font_scale=0.5, color=(255, 255, 255),
                      bg_color=None, thickness=1, padding=4):
    """Draw text with a background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if bg_color is not None:
        cv2.rectangle(img,
                      (x - padding, y - th - padding),
                      (x + tw + padding, y + baseline + padding),
                      bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return th + baseline + 2 * padding

def create_multi_object_overlay(image, masks_and_labels, alpha=0.45, show_legend=True):
    """Overlay multiple colored masks on an image with text labels.

    Args:
        image: RGB numpy array [H, W, 3]
        masks_and_labels: list of (mask, label, iou, color) tuples
        alpha: mask overlay opacity
        show_legend: whether to draw legend box

    Returns:
        RGB numpy array with overlays
    """
    from scipy import ndimage

    overlay = image.copy().astype(np.float32)
    H, W = image.shape[:2]

    # Sort by mask area (largest first) so small objects render on top
    masks_and_labels = sorted(masks_and_labels, key=lambda x: x[0].sum(), reverse=True)

    for mask, label, iou, color in masks_and_labels:
        if mask.sum() == 0:
            continue
        mask_bool = mask > 0.5
        color_f = np.array(color, dtype=np.float32)

        # Fill
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color_f * alpha

        # Edge highlight
        dilated = ndimage.binary_dilation(mask_bool, iterations=2)
        edge = dilated & ~mask_bool
        overlay[edge] = overlay[edge] * 0.3 + color_f * 0.7

    overlay = overlay.clip(0, 255).astype(np.uint8)

    # Draw per-object labels at mask centroids
    for mask, label, iou, color in masks_and_labels:
        if mask.sum() == 0:
            continue
        mask_bool = mask > 0.5
        ys, xs = np.where(mask_bool)
        cy, cx = int(ys.mean()), int(xs.mean())

        # Label text
        iou_str = f" {iou*100:.0f}%" if iou > 0 else ""
        text = f"{label}{iou_str}"
        bg = (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5))
        draw_text_with_bg(overlay, text, (cx - 20, cy), font_scale=0.45,
                         color=(255, 255, 255), bg_color=bg, thickness=1)

    # Legend box in top-right
    if show_legend and masks_and_labels:
        legend_h = 24 * len(masks_and_labels) + 10
        legend_w = 200
        lx = W - legend_w - 10
        ly = 10
        # Semi-transparent background
        roi = overlay[ly:ly+legend_h, lx:lx+legend_w].astype(np.float32)
        overlay[ly:ly+legend_h, lx:lx+legend_w] = (roi * 0.3).astype(np.uint8)

        for i, (mask, label, iou, color) in enumerate(masks_and_labels):
            yy = ly + 20 + i * 24
            # Color swatch
            cv2.rectangle(overlay, (lx + 8, yy - 12), (lx + 22, yy + 2),
                         color, -1)
            cv2.rectangle(overlay, (lx + 8, yy - 12), (lx + 22, yy + 2),
                         (255, 255, 255), 1)
            # Label
            iou_str = f" ({iou*100:.0f}%)" if iou > 0 else ""
            cv2.putText(overlay, f"{label}{iou_str}", (lx + 28, yy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay

def add_frame_header(frame, scene_id, frame_idx, total_frames, n_objects, header_h=40):
    """Add a header bar with scene info."""
    H, W = frame.shape[:2]
    # Darken top
    frame[:header_h] = (frame[:header_h].astype(np.float32) * 0.3).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Scene: {scene_id}  |  Frame {frame_idx+1}/{total_frames}  |  {n_objects} objects",
                (10, 26), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def load_scene_objects(scene_path, semantics_dir):
    """Load all object labels and their IDs from segments_anno.json + GT masks.

    Uses the same approach as benchmark.load_scene_data():
    1. Load labels from segments_anno.json
    2. Check which objects are actually visible in the GT mask files
    """
    from triangulang.utils.scannetpp_loader import LABEL_FIXES
    import json, torch

    scene_path = Path(scene_path)
    sem_path = Path(semantics_dir)

    # Load object labels from annotation
    anno_path = scene_path / "scans" / "segments_anno.json"
    if not anno_path.exists():
        print(f"  No segments_anno.json in {scene_path}")
        return {}

    with open(anno_path) as f:
        anno_data = json.load(f)

    all_objects = {}  # obj_id -> label
    for group in anno_data.get('segGroups', []):
        obj_id = group.get('objectId') or group.get('id')
        label = group.get('label', 'unknown').strip().lower()
        label = LABEL_FIXES.get(label, label)
        if obj_id is not None and label:
            all_objects[obj_id] = label

    # Filter to objects actually visible in GT masks (sample a few frames)
    skip_labels = {'remove', 'split', 'object', 'objects', 'stuff', 'unknown',
                   'wall', 'floor', 'ceiling', 'door', 'window', 'doorframe',
                   'windowframe', 'window frame', 'reflection', 'mirror', 'structure'}

    pth_files = sorted(sem_path.glob("*.pth"))
    if not pth_files:
        print(f"  No .pth files in {sem_path}")
        return {}

    # Sample a few frames to find visible objects
    import numpy as np
    sample_indices = np.linspace(0, len(pth_files) - 1, min(10, len(pth_files)), dtype=int)
    visible_ids = set()
    for idx in sample_indices:
        data = torch.load(pth_files[idx], map_location='cpu', weights_only=False)
        if isinstance(data, np.ndarray):
            visible_ids.update(np.unique(data).tolist())

    # Build label -> [obj_ids] for visible, non-skip objects
    object_labels = {}
    for obj_id, label in all_objects.items():
        if label in skip_labels or obj_id not in visible_ids:
            continue
        if label not in object_labels:
            object_labels[label] = []
        object_labels[label].append(obj_id)

    return object_labels

def load_gt_masks_for_frame(semantics_dir, frame_name):
    """Load GT masks from a .pth file for a frame.

    Returns dict: obj_id -> binary numpy mask.
    The pth files contain per-pixel object ID arrays (int32).
    """
    import torch
    import numpy as np

    sem_path = Path(semantics_dir)
    # frame_name may be 'DSC02798.JPG' — pth is 'DSC02798.JPG.pth'
    for candidate in [f"{frame_name}.pth", f"{Path(frame_name).stem}.pth"]:
        pth_path = sem_path / candidate
        if pth_path.exists():
            data = torch.load(pth_path, map_location='cpu', weights_only=False)
            if isinstance(data, np.ndarray):
                # Per-pixel object ID map → per-object binary masks
                masks = {}
                for obj_id in np.unique(data):
                    if obj_id <= 0:
                        continue
                    masks[int(obj_id)] = (data == obj_id).astype(np.uint8)
                return masks
            elif isinstance(data, dict):
                return data
    return {}

def main():
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    parser = argparse.ArgumentParser(description='Multi-object video demo')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--scene', type=str, required=True,
                       help='Scene dir (e.g., data/scannetpp/data/<scene_id>)')
    parser.add_argument('--objects', nargs='+', default=None,
                       help='Object labels to show (default: all objects in scene)')
    parser.add_argument('--max-objects', type=int, default=15,
                       help='Max objects to show per frame')
    parser.add_argument('--output', type=str, default='viz/mo_video.mp4')
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=0, help='0=all')
    parser.add_argument('--stride', type=int, default=3, help='Frame stride')
    parser.add_argument('--resolution', type=int, default=720, help='Output video height')
    parser.add_argument('--image-size', type=int, default=1008, help='Model input size')
    parser.add_argument('--mask-alpha', type=float, default=0.45)
    parser.add_argument('--show-gt', action='store_true', help='Side-by-side GT comparison')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--da3-cache', type=str, default='da3_nested_cache_1008',
                       help='DA3 cache name')
    parser.add_argument('--smooth-alpha', type=float, default=0.65,
                       help='Temporal EMA weight for current frame (0=full smoothing, 1=no smoothing)')

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    scene_path = Path(args.scene)
    scene_id = scene_path.name
    # Try both naming conventions
    image_dir = scene_path / 'dslr' / 'resized_undistorted_images'
    if not image_dir.exists():
        image_dir = scene_path / 'dslr' / 'resized_undistorted'
    if not image_dir.exists():
        image_dir = scene_path  # Allow passing image dir directly

    # Find semantics directory
    semantics_dir = None
    for sem_name in ['semantics_2d_val_v2', 'semantics_2d_val', 'semantics_2d_train']:
        sem_path = scene_path.parent.parent / sem_name / scene_id
        if sem_path.exists():
            semantics_dir = sem_path
            break
    if semantics_dir is None:
        # Try data root
        for sem_name in ['semantics_2d_val_v2', 'semantics_2d_val', 'semantics_2d_train']:
            sem_path = Path('data/scannetpp') / sem_name / scene_id
            if sem_path.exists():
                semantics_dir = sem_path
                break

    # Get frames
    frames = sorted(image_dir.glob('*.JPG')) + sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png'))
    frames = sorted(set(frames), key=lambda p: p.name)
    if args.stride > 1:
        frames = frames[::args.stride]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"Scene: {scene_id}, {len(frames)} frames (stride={args.stride})")

    # Initialize CUDA before model loading (SAM3 hardcodes device="cuda" in position encoding)
    if device != 'cpu':
        torch.cuda.init()
        torch.zeros(1, device=device)
        print(f"  CUDA initialized: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    from triangulang.evaluation.benchmark import load_model
    model = load_model(args.checkpoint, device, da3_resolution=args.image_size,
                       resolution=args.image_size)
    model.eval()

    image_size = (args.image_size, args.image_size)
    sam3_mo = getattr(model, 'sam3_multi_object', False)
    print(f"  SAM3 multi-object mode: {'ON' if sam3_mo else 'OFF'}")

    # Discover objects
    if args.objects:
        object_labels = {label: [] for label in args.objects}
        print(f"Using specified objects: {args.objects}")
    elif semantics_dir:
        object_labels = load_scene_objects(scene_path, semantics_dir)
        print(f"Found {len(object_labels)} object categories in GT")
    else:
        print("ERROR: No --objects specified and no GT semantics found. Pass --objects.")
        return

    # Limit objects
    label_list = list(object_labels.keys())[:args.max_objects]
    print(f"Tracking {len(label_list)} objects: {label_list}")

    # Assign colors
    label_colors = {label: OBJECT_COLORS[i % len(OBJECT_COLORS)]
                    for i, label in enumerate(label_list)}

    # DA3 cache
    da3_cache_dir = None
    for root in [Path('data/scannetpp'), Path('data/scannetpp')]:
        candidate = root / args.da3_cache
        if candidate.exists():
            da3_cache_dir = candidate
            break
    if da3_cache_dir:
        print(f"DA3 cache: {da3_cache_dir}")

    # Temporal smoothing state (per-label EMA of mask logits)
    prev_logits = {}  # label -> tensor [H, W] on CPU
    alpha = args.smooth_alpha
    if alpha < 1.0:
        print(f"Temporal smoothing: alpha={alpha:.2f} (0=full smooth, 1=no smooth)")

    # Process frames
    print(f"Processing {len(frames)} frames...")
    processed_frames = []

    for fidx, frame_path in enumerate(frames):
        t0 = time.perf_counter()
        frame_stem = frame_path.stem

        # Load image
        img = Image.open(frame_path).convert('RGB')
        orig_size = img.size  # (W, H)
        img_resized = img.resize(image_size, Image.BILINEAR)
        img_np = np.array(img_resized)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Load cached depth + poses
        cached_depth = None
        da3_intrinsics = None
        da3_extrinsics = None
        if da3_cache_dir:
            cache_file = da3_cache_dir / scene_id / f"{frame_stem}.pt"
            if cache_file.exists():
                cache_data = torch.load(cache_file, map_location='cpu', weights_only=True)
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

        # Run inference: one object at a time (avoids SAM3-MO batch expansion issues)
        masks_and_labels = []
        for label in label_list:
            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    outputs = model(
                        img_tensor, [label], None,
                        cached_depth=cached_depth,
                        da3_intrinsics=da3_intrinsics,
                        da3_extrinsics=da3_extrinsics,
                    )

            # Extract best mask from Q candidates
            all_masks = outputs['all_masks'][0]  # [Q, H, W]
            text_scores = outputs.get('text_scores')
            if text_scores is not None and text_scores.dim() == 3:
                scores = text_scores[0, :, 0]  # [B, Q, K] → [Q]
            elif text_scores is not None and text_scores.dim() == 2:
                scores = text_scores[0]  # [B, Q] → [Q]
            else:
                scores = all_masks.mean(dim=(1, 2))  # fallback: mean logit
            best_q = scores.argmax()
            mask_logit = all_masks[best_q]
            if mask_logit.shape != img_np.shape[:2]:
                mask_logit = F.interpolate(
                    mask_logit.unsqueeze(0).unsqueeze(0).float(),
                    size=img_np.shape[:2],
                    mode='bilinear', align_corners=False
                ).squeeze()

            # Temporal EMA smoothing on logits
            logit_cpu = mask_logit.cpu().float()
            if alpha < 1.0 and label in prev_logits:
                logit_cpu = alpha * logit_cpu + (1 - alpha) * prev_logits[label]
            prev_logits[label] = logit_cpu

            mask = (torch.sigmoid(logit_cpu) > 0.5).numpy()
            conf = torch.sigmoid(scores[best_q]).item()
            masks_and_labels.append((mask, label, conf, label_colors[label]))

        # Build overlay
        overlay = create_multi_object_overlay(img_np, masks_and_labels,
                                              alpha=args.mask_alpha)

        # GT comparison (side-by-side)
        if args.show_gt and semantics_dir:
            gt_data = load_gt_masks_for_frame(semantics_dir, frame_path.name)
            gt_masks_labels = []
            from triangulang.utils.scannetpp_loader import LABEL_FIXES
            for label in label_list:
                obj_ids = object_labels.get(label, [])
                gt_mask = None
                for oid in obj_ids:
                    if oid in gt_data and isinstance(gt_data[oid], dict) and 'mask' in gt_data[oid]:
                        m = gt_data[oid]['mask']
                        if isinstance(m, torch.Tensor):
                            m = m.numpy()
                        if gt_mask is None:
                            gt_mask = m.copy()
                        else:
                            gt_mask = np.maximum(gt_mask, m)
                    elif oid in gt_data and isinstance(gt_data[oid], np.ndarray):
                        m = gt_data[oid]
                        if gt_mask is None:
                            gt_mask = m.copy()
                        else:
                            gt_mask = np.maximum(gt_mask, m)
                if gt_mask is not None:
                    # Resize to model resolution
                    gt_resized = np.array(
                        Image.fromarray(gt_mask.astype(np.uint8) * 255).resize(
                            image_size, Image.NEAREST)
                    ) > 127
                    gt_masks_labels.append((gt_resized, label, 1.0, label_colors[label]))

            if gt_masks_labels:
                gt_overlay = create_multi_object_overlay(img_np, gt_masks_labels,
                                                        alpha=args.mask_alpha,
                                                        show_legend=False)
                # Add GT/Pred labels
                draw_text_with_bg(gt_overlay, "Ground Truth", (10, 25),
                                 font_scale=0.6, bg_color=(0, 0, 0))
                draw_text_with_bg(overlay, "Prediction", (10, 25),
                                 font_scale=0.6, bg_color=(0, 0, 0))
                overlay = np.concatenate([gt_overlay, overlay], axis=1)

        # Add header
        n_detected = sum(1 for m, _, _, _ in masks_and_labels if m.sum() > 0)
        overlay = add_frame_header(overlay, scene_id, fidx, len(frames), n_detected)

        processed_frames.append(overlay)

        elapsed = time.perf_counter() - t0
        if (fidx + 1) % 10 == 0 or fidx == 0:
            print(f"  [{fidx+1}/{len(frames)}] {n_detected}/{len(label_list)} objects detected, "
                  f"{elapsed:.2f}s/frame")

    # Write video
    if not processed_frames:
        print("ERROR: No frames processed.")
        return

    sample = processed_frames[0]
    h, w = sample.shape[:2]
    scale = args.resolution / h
    out_w, out_h = int(w * scale), int(h * scale)
    out_w += out_w % 2  # ensure even
    out_h += out_h % 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (out_w, out_h))

    for frame in processed_frames:
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

    writer.release()

    # Also encode with ffmpeg for better compatibility if available
    ffmpeg_out = output_path.with_suffix('.mp4')
    tmp_out = output_path.with_name(output_path.stem + '_tmp.mp4')
    import subprocess
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(output_path),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            str(tmp_out)
        ], capture_output=True, check=True)
        tmp_out.rename(ffmpeg_out)
        print(f"Re-encoded with H.264: {ffmpeg_out}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"(ffmpeg not available, using mp4v codec)")

    duration = len(processed_frames) / args.fps
    print(f"\nSaved: {output_path}")
    print(f"  {len(processed_frames)} frames, {args.fps} FPS, {out_w}x{out_h}, {duration:.1f}s")

if __name__ == '__main__':
    main()
