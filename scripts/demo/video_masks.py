"""Video demo: runs segmentation on a frame sequence and produces an MP4 with mask overlays.

Supports ScanNet++ scenes and uCO3D sequences, with optional depth overlay.

Usage:
    python scripts/demo/video_masks.py \
        --checkpoint checkpoints/<run>/best.pt \
        --scene-dir data/scannetpp/data/<scene>/dslr/resized_undistorted/ \
        --prompt "chair" \
        --output viz/video_demo.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'sam3'))
sys.path.insert(0, str(PROJECT_ROOT / 'depth_anything_v3' / 'src'))

def get_scannetpp_frames(scene_dir: str, max_frames: int = 0, stride: int = 1) -> list:
    """Get sorted image paths from a ScanNet++ scene directory."""
    scene_path = Path(scene_dir)
    extensions = ['.JPG', '.jpg', '.png', '.PNG']
    frames = []
    for ext in extensions:
        frames.extend(sorted(scene_path.glob(f'*{ext}')))
    frames = sorted(set(frames), key=lambda p: p.name)
    if stride > 1:
        frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames

def get_uco3d_frames(seq_dir: str, max_frames: int = 0, stride: int = 1) -> list:
    """Extract frames from a uCO3D sequence video."""
    import tempfile
    seq_path = Path(seq_dir)
    video_path = seq_path / 'rgb_video.mp4'
    if not video_path.exists():
        raise FileNotFoundError(f"No rgb_video.mp4 in {seq_dir}")

    # Extract frames to temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix='uco3d_frames_'))
    import subprocess
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-q:v', '2',
        str(tmp_dir / 'frame_%05d.jpg'),
        '-loglevel', 'error'
    ]
    subprocess.run(cmd, check=True)

    frames = sorted(tmp_dir.glob('frame_*.jpg'))
    if stride > 1:
        frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames

def create_overlay(image: np.ndarray, mask: np.ndarray, depth: np.ndarray = None,
                   centroid_3d: np.ndarray = None, prompt: str = '',
                   iou_score: float = 0.0, frame_idx: int = 0,
                   mask_alpha: float = 0.4, show_depth: bool = False,
                   mask_color: tuple = (0, 191, 217)) -> np.ndarray:
    """Create a single frame with mask overlay and info text."""
    from scipy import ndimage

    H, W = image.shape[:2]
    overlay = image.copy()

    # Apply mask overlay
    if mask.sum() > 0:
        mask_rgb = np.array(mask_color, dtype=np.uint8)
        mask_bool = mask > 0
        overlay[mask_bool] = (
            overlay[mask_bool].astype(float) * (1 - mask_alpha) +
            mask_rgb.astype(float) * mask_alpha
        ).astype(np.uint8)

        # Edge glow
        dilated = ndimage.binary_dilation(mask, iterations=2)
        edge = dilated & ~mask_bool
        overlay[edge] = (
            overlay[edge].astype(float) * 0.4 +
            np.array([255, 255, 255], dtype=float) * 0.6
        ).astype(np.uint8)

    # If showing depth side-by-side
    if show_depth and depth is not None:
        from depth_anything_3.utils.visualize import visualize_depth
        depth_vis = visualize_depth(depth, cmap="Spectral")
        if depth_vis.shape[:2] != (H, W):
            from PIL import Image as PILImage
            depth_vis = np.array(PILImage.fromarray(depth_vis).resize((W, H)))
        # Stack side by side
        overlay = np.concatenate([overlay, depth_vis], axis=1)

    return overlay

def add_text_to_frame(frame: np.ndarray, prompt: str, iou_score: float,
                      centroid_3d: np.ndarray, frame_idx: int, total_frames: int,
                      mask_pixels: int) -> np.ndarray:
    """Add text overlay to frame using OpenCV."""
    import cv2
    out = frame.copy()
    H, W = out.shape[:2]

    # Semi-transparent banner at top
    banner_h = 50
    banner = np.zeros((banner_h, W, 3), dtype=np.uint8)
    out[:banner_h] = (out[:banner_h].astype(float) * 0.4 + banner.astype(float) * 0.6).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Query text
    cv2.putText(out, f'Query: "{prompt}"', (10, 20),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # IoU and frame counter
    cv2.putText(out, f'IoU: {iou_score:.1%}  |  Frame {frame_idx+1}/{total_frames}',
                (10, 40), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom banner with centroid
    if centroid_3d is not None and mask_pixels > 0:
        bot_y = H - 30
        bot_banner = np.zeros((30, W, 3), dtype=np.uint8)
        out[bot_y:] = (out[bot_y:].astype(float) * 0.4).astype(np.uint8)
        cv2.putText(out, f'3D: ({centroid_3d[0]:+.2f}, {centroid_3d[1]:+.2f}, {centroid_3d[2]:.2f})m  |  {mask_pixels:,} px',
                    (10, H - 10), font, 0.4, (0, 220, 255), 1, cv2.LINE_AA)

    return out

def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.amp import autocast

    parser = argparse.ArgumentParser(description='Video demo: streaming multi-view segmentation masks')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scene-dir', type=str, help='ScanNet++ scene image directory')
    group.add_argument('--uco3d-seq', type=str, help='uCO3D sequence directory')

    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Text query')
    parser.add_argument('--output', type=str, default='demo_video.mp4', help='Output MP4 path')
    parser.add_argument('--fps', type=int, default=10, help='Output video FPS')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames (0=all)')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--size', type=int, default=518, help='Input size for SAM3')
    parser.add_argument('--mask-alpha', type=float, default=0.4, help='Mask overlay opacity')
    parser.add_argument('--show-depth', action='store_true', help='Show depth map side-by-side')
    parser.add_argument('--resolution', type=int, default=720, help='Output video height (pixels)')

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Import model utilities
    from scripts.demo_locate_3d import load_model, parse_spatial_qualifier, unproject_to_3d

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)

    # Get frames
    if args.scene_dir:
        frames = get_scannetpp_frames(args.scene_dir, args.max_frames, args.stride)
        print(f"Found {len(frames)} frames in {args.scene_dir}")
    else:
        frames = get_uco3d_frames(args.uco3d_seq, args.max_frames, args.stride)
        print(f"Extracted {len(frames)} frames from {args.uco3d_seq}")

    if len(frames) == 0:
        print("ERROR: No frames found.")
        return

    # Parse spatial qualifier
    qualifier, base_prompt = parse_spatial_qualifier(args.prompt)
    effective_prompt = base_prompt if qualifier else args.prompt

    # DA3 model info
    da3_model_name = config.get('da3_model', 'depth-anything/DA3METRIC-LARGE')
    focal_length = args.size * 0.96

    # Process frames
    print(f"Processing {len(frames)} frames with prompt: '{args.prompt}'...")
    processed_frames = []

    for idx, frame_path in enumerate(frames):
        # Load and preprocess
        img = Image.open(frame_path).convert('RGB')
        orig_w, orig_h = img.size
        img_resized = img.resize((args.size, args.size), Image.BILINEAR)
        img_np = np.array(img_resized)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            with autocast('cuda', dtype=torch.float16):
                outputs = model(img_tensor.to(device), [effective_prompt], gt_masks=None)

        all_masks = outputs['all_masks'][0]  # [Q, H, W]
        depth = outputs['depth'][0, 0].cpu().numpy()

        # Scale depth
        if 'DA3METRIC' in da3_model_name.upper():
            depth = depth * (focal_length / 300.0)

        # Get IoU predictions
        iou_preds = None
        if 'iou_pred' in outputs and outputs['iou_pred'] is not None:
            iou_preds = outputs['iou_pred'][0]

        # Select best mask
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

        # Resize depth to match
        if depth.shape != mask.shape:
            depth = np.array(
                Image.fromarray(depth.astype(np.float32)).resize(
                    mask.shape[::-1], Image.BILINEAR
                )
            )

        # Compute 3D centroid
        _, centroid_3d = unproject_to_3d(mask, depth, fx=focal_length, fy=focal_length)

        # Create overlay
        overlay = create_overlay(
            img_np, mask, depth, centroid_3d, args.prompt,
            iou_score, idx, args.mask_alpha, args.show_depth
        )

        # Add text
        overlay = add_text_to_frame(
            overlay, args.prompt, iou_score, centroid_3d,
            idx, len(frames), int(mask.sum())
        )

        processed_frames.append(overlay)

        if (idx + 1) % 10 == 0 or idx == len(frames) - 1:
            print(f"  [{idx+1}/{len(frames)}] IoU={iou_score:.2f}, mask={mask.sum():,}px")

    # Write video
    import cv2
    if len(processed_frames) == 0:
        print("ERROR: No frames processed.")
        return

    # Determine output size
    sample = processed_frames[0]
    h, w = sample.shape[:2]
    scale = args.resolution / h
    out_w, out_h = int(w * scale), int(h * scale)
    # Ensure even dimensions for video codec
    out_w = out_w + (out_w % 2)
    out_h = out_h + (out_h % 2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (out_w, out_h))

    for frame in processed_frames:
        # Resize to output resolution
        frame_resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        # RGB to BGR for OpenCV
        writer.write(cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"\nSaved video to {output_path} ({len(processed_frames)} frames, {args.fps} FPS, {out_w}x{out_h})")

if __name__ == '__main__':
    main()
