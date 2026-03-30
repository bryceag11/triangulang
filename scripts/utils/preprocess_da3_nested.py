"""Preprocess DA3-NESTED depth and poses for ScanNet++ dataset.

Computes metric depth and camera poses using DA3-NESTED-GIANT-LARGE with
overlapping-chunk Sim(3) alignment for globally consistent coordinates.
Outputs per-frame .pt files with depth, c2w extrinsics, and intrinsics.

Usage:
    torchrun --nproc_per_node=8 scripts/utils/preprocess_da3_nested.py \
        --data-root /path/to/scannetpp
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "sam3"))
sys.path.insert(0, str(project_root / "depth_anything_v3" / "src"))

def get_scenes_dir(data_root: Path) -> Path:
    """Get the directory containing scene folders (handles nested 'data' folder)."""
    nested = data_root / "data"
    if nested.exists() and nested.is_dir():
        return nested
    return data_root

def ensure_transforms_undistorted(scene_path: Path) -> bool:
    """Generate transforms_undistorted.json if missing, from fisheye transforms.json.

    Uses cv2.fisheye.estimateNewCameraMatrixForUndistortRectify to compute
    undistorted pinhole intrinsics - matches ScanNet++ toolkit exactly.

    Returns True if file exists (or was created), False if cannot generate.
    """
    nerfstudio_dir = scene_path / "dslr" / "nerfstudio"
    undistorted_path = nerfstudio_dir / "transforms_undistorted.json"

    if undistorted_path.exists():
        return True

    transforms_path = nerfstudio_dir / "transforms.json"
    if not transforms_path.exists():
        return False

    try:
        with open(transforms_path) as f:
            transforms = json.load(f)

        if transforms.get("camera_model") == "PINHOLE":
            # Already pinhole, just copy it
            with open(undistorted_path, "w") as f:
                json.dump(transforms, f, indent=4)
            return True

        height = int(transforms["h"])
        width = int(transforms["w"])
        distortion_params = np.array([
            float(transforms["k1"]), float(transforms["k2"]),
            float(transforms["k3"]), float(transforms["k4"]),
        ])
        K = np.array([
            [float(transforms["fl_x"]), 0, float(transforms["cx"])],
            [0, float(transforms["fl_y"]), float(transforms["cy"])],
            [0, 0, 1],
        ])

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (width, height), np.eye(3), balance=0.0,
        )
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        new_transforms = deepcopy(transforms)
        new_transforms["fl_x"] = float(new_K[0, 0])
        new_transforms["fl_y"] = float(new_K[1, 1])
        new_transforms["cx"] = float(new_K[0, 2])
        new_transforms["cy"] = float(new_K[1, 2])
        new_transforms["camera_model"] = "PINHOLE"
        for key in ("k1", "k2", "k3", "k4"):
            if key in new_transforms:
                new_transforms[key] = 0.0

        with open(undistorted_path, "w") as f:
            json.dump(new_transforms, f, indent=4)
        return True

    except Exception as e:
        print(f"  Warning: Could not generate transforms_undistorted.json: {e}")
        return False

def load_gt_poses(scene_path: Path) -> dict:
    """Load GT camera poses from transforms_undistorted.json.

    Returns dict mapping frame stem (e.g. 'DSC00001') to (4,4) c2w matrix,
    or empty dict if file not found.
    """
    tf_path = scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    if not tf_path.exists():
        return {}

    try:
        with open(tf_path) as f:
            tf = json.load(f)
        poses = {}
        for frame in tf.get("frames", []):
            key = Path(frame.get("file_path", "")).stem
            if key:
                poses[key] = np.array(frame["transform_matrix"])
        return poses
    except Exception:
        return {}

def procrustes_check(da3_c2w_dict: dict, gt_poses: dict, scene_id: str) -> dict:
    """Run Procrustes alignment between DA3 and GT camera positions.

    Args:
        da3_c2w_dict: dict mapping frame stem -> (4,4) c2w from DA3
        gt_poses: dict mapping frame stem -> (4,4) c2w from GT
        scene_id: for logging

    Returns dict with alignment stats, or None if insufficient data.
    """
    common = sorted(set(da3_c2w_dict.keys()) & set(gt_poses.keys()))
    if len(common) < 4:
        return None

    # Sample up to 100 evenly spaced frames for speed
    if len(common) > 100:
        indices = np.linspace(0, len(common) - 1, 100, dtype=int)
        common = [common[i] for i in indices]

    da3_pos = np.array([da3_c2w_dict[k][:3, 3] for k in common])
    gt_pos = np.array([gt_poses[k][:3, 3] for k in common])

    try:
        s, R, t = umeyama_sim3(da3_pos, gt_pos)
        aligned = s * (da3_pos @ R.T) + t
        errors = np.linalg.norm(aligned - gt_pos, axis=1)

        return {
            'scene_id': scene_id,
            'n_frames': len(common),
            'global_mean_cm': float(errors.mean() * 100),
            'global_median_cm': float(np.median(errors) * 100),
            'global_max_cm': float(errors.max() * 100),
            'scale': float(s),
        }
    except Exception:
        return None

def load_scene_list(data_root: Path, split: str) -> list:
    """Load scene IDs from split file.

    Handles formats like:
        scene_id
        scene_id  # comment
        scene_id  123.4cm
    Strips everything after first whitespace or '#'.
    """
    split_file = data_root / "splits" / f"{split}.txt"
    if not split_file.exists():
        for alt_path in [data_root / f"{split}.txt", data_root / "metadata" / f"{split}.txt"]:
            if alt_path.exists():
                split_file = alt_path
                break

    if not split_file.exists():
        print(f"Warning: Split file not found: {split_file}")
        return []

    scenes = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Take first token (scene ID), ignore comments/annotations
            scene_id = line.split()[0].split('#')[0].strip()
            if scene_id:
                scenes.append(scene_id)
    return scenes

def get_frames_with_gt_masks(data_root: Path, scene_id: str, split: str = 'train') -> set:
    """Get frame names that have GT masks in semantics_2d_train/val.

    Returns set of image stems (e.g., "DSC00001") without extension.
    GT files are named like "DSC00001.JPG.pth", so we strip both extensions.
    """
    if 'train' in split:
        obj_ids_dir = data_root / "semantics_2d_train" / scene_id
    else:
        obj_ids_dir = data_root / "semantics_2d_val" / scene_id

    if not obj_ids_dir.exists():
        return set()

    frames = set()
    for p in obj_ids_dir.glob("*.pth"):
        # p.stem = "DSC00001.JPG", then Path(p.stem).stem = "DSC00001"
        frames.add(Path(p.stem).stem)
    return frames

def build_overlapping_chunks(n_images: int, chunk_size: int, overlap: int) -> list:
    """Build overlapping chunk index ranges.

    Returns list of (start, end) tuples. Adjacent chunks share `overlap` frames.
    If n_images <= chunk_size, returns a single chunk covering all images.
    """
    if n_images <= chunk_size:
        return [(0, n_images)]

    stride = chunk_size - overlap
    chunks = []
    start = 0
    while start < n_images:
        end = min(start + chunk_size, n_images)
        chunks.append((start, end))
        if end == n_images:
            break
        start += stride
    return chunks

def umeyama_sim3(src_points, dst_points, weights=None):
    """Compute Sim(3) alignment: dst ≈ s * R @ src + t

    Uses the Umeyama algorithm to find scale s, rotation R, and translation t
    that minimizes the weighted sum of ||dst_i - (s * R @ src_i + t)||^2.

    Args:
        src_points: (N, 3) source points (e.g., camera centers in chunk B)
        dst_points: (N, 3) destination points (e.g., camera centers in chunk A)
        weights: (N,) optional per-point weights

    Returns:
        s: scalar scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert src_points.shape == dst_points.shape
    n = src_points.shape[0]
    if n < 3:
        # Not enough points for Sim(3), return identity
        return 1.0, np.eye(3), np.zeros(3)

    if weights is None:
        weights = np.ones(n)
    weights = weights / weights.sum()

    # Weighted centroids
    mu_src = (weights[:, None] * src_points).sum(axis=0)
    mu_dst = (weights[:, None] * dst_points).sum(axis=0)

    # Center points
    q_src = src_points - mu_src
    q_dst = dst_points - mu_dst

    # Weighted cross-covariance
    H = (weights[:, None, None] * (q_src[:, :, None] @ q_dst[:, None, :])).sum(axis=0)
    # H is (3, 3): H = sum_i w_i * q_src_i @ q_dst_i^T

    # SVD
    U, S_vals, Vt = np.linalg.svd(H)

    # Handle reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.diag([1, 1, np.sign(d)])

    # Rotation
    R = Vt.T @ sign_mat @ U.T

    # Weighted variance of source
    var_src = (weights * np.sum(q_src ** 2, axis=1)).sum()
    if var_src < 1e-10:
        return 1.0, R, mu_dst - mu_src

    # Scale
    s = np.trace(np.diag([1, 1, np.sign(d)]) @ np.diag(S_vals)) / var_src

    # Translation
    t = mu_dst - s * R @ mu_src

    return float(s), R, t

def apply_sim3_to_c2w(c2w, s, R, t):
    """Apply Sim(3) transform to camera-to-world poses.

    Given c2w in frame B and Sim(3) mapping B->A: P_A = s * R_align @ P_B + t_align
    New c2w rotation: R_align @ R_cam (keeps rotation orthogonal)
    New c2w translation: s * R_align @ t_cam + t_align (scales + rotates camera position)

    Args:
        c2w: (N, 3, 4) or (N, 4, 4) camera-to-world poses
        s: scalar scale
        R: (3, 3) rotation
        t: (3,) translation

    Returns:
        aligned c2w: same shape as input
    """
    N = c2w.shape[0]
    is_4x4 = (c2w.shape[-2] == 4)

    rot_cam = c2w[:, :3, :3]  # (N, 3, 3)
    t_cam = c2w[:, :3, 3]     # (N, 3)

    new_rot = R[None] @ rot_cam  # (N, 3, 3)
    new_t = s * (R[None] @ t_cam[:, :, None]).squeeze(-1) + t[None]  # (N, 3)

    if is_4x4:
        result = np.zeros_like(c2w)
        result[:, :3, :3] = new_rot
        result[:, :3, 3] = new_t
        result[:, 3, 3] = 1.0
    else:
        result = np.zeros_like(c2w)
        result[:, :3, :3] = new_rot
        result[:, :3, 3] = new_t

    return result

def extract_c2w_from_extrinsics(extrinsics):
    """Convert model output extrinsics (w2c) to c2w 4x4 matrices.

    IMPORTANT: The model's CameraDec actually outputs WORLD-TO-CAMERA (w2c),
    NOT camera-to-world! See da3.py line 225:
        output.extrinsics = affine_inverse(c2w)  # Returns w2c!

    This function pads to 4x4 and inverts to get actual c2w.

    Args:
        extrinsics: (N, 3, 4) or (N, 4, 4) - w2c from model

    Returns:
        c2w: (N, 4, 4) - actual camera-to-world transforms
    """
    N = extrinsics.shape[0]

    # Pad to 4x4 if needed
    if extrinsics.shape[-2] == 3:
        w2c = np.zeros((N, 4, 4), dtype=extrinsics.dtype)
        w2c[:, :3, :] = extrinsics
        w2c[:, 3, 3] = 1.0
    else:
        w2c = extrinsics.copy()

    # Invert w2c to get c2w
    c2w = np.linalg.inv(w2c)
    return c2w

def main():
    parser = argparse.ArgumentParser(description='Preprocess DA3-NESTED depth and poses for ScanNet++')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to ScanNet++ data root')
    parser.add_argument('--split', type=str, default='nvs_sem_train',
                        help='Dataset split to process')
    parser.add_argument('--da3-model', type=str, default='depth-anything/DA3NESTED-GIANT-LARGE',
                        choices=['depth-anything/DA3NESTED-GIANT-LARGE', 'depth-anything/DA3-GIANT',
                                 'depth-anything/DA3-LARGE'],
                        help='DA3 model (must support multi-view and pose estimation)')
    parser.add_argument('--resolution', type=int, default=504,
                        help='DA3 processing resolution (504 recommended for speed, 756 for quality)')
    parser.add_argument('--chunk-size', type=int, default=8,
                        help='Views processed jointly with cross-view attention. '
                             'Higher = better depth consistency but more VRAM. '
                             '80GB at 504: use 16. 80GB at 756: use 8.')
    parser.add_argument('--overlap', type=int, default=None,
                        help='Number of overlapping frames between consecutive chunks. '
                             'Default: chunk_size // 2. Must be < chunk_size. '
                             'More overlap = better alignment but more computation.')
    parser.add_argument('--use-undistorted', action='store_true', default=True,
                        help='Use undistorted images')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Max scenes to process (for testing)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing cache files')
    parser.add_argument('--cache-name', type=str, default='da3_nested_cache',
                        help='Name of cache directory under data_root')
    parser.add_argument('--only-with-gt', action='store_true', default=True,
                        help='Only SAVE frames that have GT masks (all frames still '
                             'participate in chunks for spatial coherence)')
    parser.add_argument('--use-ray-pose', action='store_true',
                        help='Use ray-based pose estimation instead of camera decoder')
    parser.add_argument('--no-only-with-gt', dest='only_with_gt', action='store_false',
                        help='Save all frames, not just those with GT masks')
    parser.add_argument('--procrustes', action='store_true', default=True,
                        help='Run Procrustes alignment check against GT poses after each scene')
    parser.add_argument('--no-procrustes', dest='procrustes', action='store_false',
                        help='Skip Procrustes alignment check')
    parser.add_argument('--procrustes-only', action='store_true', default=False,
                        help='Skip DA3 inference, only run Procrustes check on existing cache')
    parser.add_argument('--procrustes-threshold', type=float, default=20.0,
                        help='Global Procrustes error threshold (cm) for flagging bad scenes')
    args = parser.parse_args()

    # Default overlap to half chunk size (following DA3-Streaming convention)
    if args.overlap is None:
        args.overlap = args.chunk_size // 2
    if args.overlap >= args.chunk_size:
        raise ValueError(f"Overlap ({args.overlap}) must be < chunk_size ({args.chunk_size})")

    # Setup distributed mode if available
    local_rank = 0
    world_size = 1

    if 'LOCAL_RANK' in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (local_rank == 0)

    if is_main:
        print(f"Preprocessing DA3-NESTED depth + poses for ScanNet++")
        print(f"  Data root: {args.data_root}")
        print(f"  Cache name: {args.cache_name}")
        print(f"  Split: {args.split}")
        print(f"  DA3 model: {args.da3_model}")
        print(f"  Resolution: {args.resolution}")
        print(f"  Chunk size: {args.chunk_size} views (cross-view attention window)")
        print(f"  Overlap: {args.overlap} frames between chunks")
        print(f"  Only save GT frames: {args.only_with_gt}")
        print(f"  Procrustes check: {args.procrustes} (threshold: {args.procrustes_threshold}cm)")
        print(f"  World size: {world_size} GPUs")
        print()

    # Load DA3 model (skip for --procrustes-only)
    da3 = None
    input_processor = None
    if not args.procrustes_only:
        if is_main:
            print(f"Loading DA3 model (this may take a moment for 1.4B params)...")

        from depth_anything_3.api import DepthAnything3
        da3 = DepthAnything3.from_pretrained(args.da3_model).to(device)
        da3.eval()

        # Load InputProcessor for correct preprocessing (ImageNet norm + aspect-ratio resize)
        from depth_anything_3.utils.io.input_processor import InputProcessor
        input_processor = InputProcessor()

        if is_main:
            print(f"  DA3 loaded successfully")
            print()
    else:
        if is_main:
            print(f"[Procrustes-only mode] Skipping DA3 model loading")
            print()

    # Get scene list
    data_root = Path(args.data_root)
    scenes_dir = get_scenes_dir(data_root)
    scenes_from_split = load_scene_list(data_root, args.split)

    # Filter to scenes that exist
    scenes = []
    for scene_id in scenes_from_split:
        scene_path = scenes_dir / scene_id
        if args.use_undistorted:
            images_dir = scene_path / "dslr" / "resized_undistorted_images"
        else:
            images_dir = scene_path / "dslr" / "resized_images"
        if images_dir.exists():
            scenes.append(scene_id)

    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    if is_main:
        print(f"Found {len(scenes)} scenes with images")

    # Create cache directory
    cache_dir = data_root / args.cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect scene metadata
    scene_data = {}
    total_gt_frames = 0
    total_all_frames = 0
    transforms_generated = 0

    for scene_id in scenes:
        scene_path = scenes_dir / scene_id
        if args.use_undistorted:
            images_dir = scene_path / "dslr" / "resized_undistorted_images"
        else:
            images_dir = scene_path / "dslr" / "resized_images"

        if not images_dir.exists():
            continue

        # Auto-generate transforms_undistorted.json if missing
        if args.use_undistorted:
            if ensure_transforms_undistorted(scene_path):
                tf_path = scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
                if not (scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json.was_generated").exists():
                    # Check if we just created it (didn't exist before this run)
                    pass
            else:
                if is_main:
                    print(f"  Warning: {scene_id} - cannot generate transforms_undistorted.json, skipping Procrustes")

        gt_frames = get_frames_with_gt_masks(data_root, scene_id, args.split)
        if args.only_with_gt and not gt_frames:
            continue

        # Get ALL sorted image paths (for spatial coherence in chunks)
        all_img_paths = sorted(images_dir.glob("*.JPG"))
        if not all_img_paths:
            continue

        scene_cache_dir = cache_dir / scene_id
        scene_cache_dir.mkdir(exist_ok=True)

        # Check which GT frames need processing
        gt_to_process = set()
        if not args.procrustes_only:
            for img_path in all_img_paths:
                if args.only_with_gt and img_path.stem not in gt_frames:
                    continue
                cache_path = scene_cache_dir / f"{img_path.stem}.pt"
                if cache_path.exists() and not args.overwrite:
                    continue
                gt_to_process.add(img_path.stem)

            if not gt_to_process:
                continue
        else:
            # Procrustes-only: include scene if cache dir has files
            cached_files = list(scene_cache_dir.glob("*.pt"))
            if not cached_files:
                continue

        # Pre-load GT poses for Procrustes check
        gt_poses = {}
        if args.procrustes:
            gt_poses = load_gt_poses(scene_path)

        scene_data[scene_id] = {
            'all_img_paths': all_img_paths,
            'gt_frames': gt_frames,
            'gt_to_process': gt_to_process,
            'cache_dir': scene_cache_dir,
            'gt_poses': gt_poses,
        }
        total_gt_frames += len(gt_to_process)
        total_all_frames += len(all_img_paths)

    if is_main:
        print(f"  Scenes to process: {len(scene_data)}")
        print(f"  GT frames to save: {total_gt_frames}")
        print(f"  Total frames in chunks (for context): {total_all_frames}")
        if args.procrustes:
            n_with_gt_poses = sum(1 for sd in scene_data.values() if sd['gt_poses'])
            print(f"  Scenes with GT poses (for Procrustes): {n_with_gt_poses}/{len(scene_data)}")
        print()

    # Distribute scenes across GPUs
    scene_list = list(scene_data.keys())
    my_scenes = scene_list[local_rank::world_size]

    my_gt_count = sum(len(scene_data[s]['gt_to_process']) for s in my_scenes)
    print(f"[Rank {local_rank}] Processing {len(my_scenes)} scenes, {my_gt_count} GT frames")

    # Process each scene
    chunk_size = args.chunk_size
    overlap = args.overlap

    pbar = tqdm(my_scenes, disable=(not is_main), desc="Scenes")

    # Track Procrustes results across scenes
    procrustes_results = []
    bad_alignment_scenes = []

    for scene_id in pbar:
        scene_start_time = time.time()
        pbar.set_postfix({'scene': scene_id[:12]})
        sd = scene_data[scene_id]
        all_img_paths = sd['all_img_paths']
        gt_frames = sd['gt_frames']
        gt_to_process = sd['gt_to_process']
        scene_cache_dir = sd['cache_dir']

        if args.procrustes_only:
            # Procrustes-only mode
            da3_c2w_dict = {}
            cached_files = sorted(scene_cache_dir.glob("*.pt"))
            for cf in cached_files:
                data = torch.load(cf, map_location='cpu', weights_only=False)
                da3_c2w_dict[cf.stem] = data['extrinsics'].float().numpy()

            scene_elapsed = time.time() - scene_start_time
            n_cached = len(cached_files)

            if sd['gt_poses'] and da3_c2w_dict:
                pr = procrustes_check(da3_c2w_dict, sd['gt_poses'], scene_id)
                if pr:
                    procrustes_results.append(pr)
                    status = "OK" if pr['global_mean_cm'] < args.procrustes_threshold else "BAD"
                    if status == "BAD":
                        bad_alignment_scenes.append(pr)
                    tqdm.write(
                        f"  [{status}] {scene_id}: global={pr['global_mean_cm']:.1f}cm "
                        f"(max={pr['global_max_cm']:.1f}cm) scale={pr['scale']:.3f} "
                        f"cached={n_cached} time={scene_elapsed:.1f}s"
                    )
                else:
                    tqdm.write(
                        f"  [SKIP] {scene_id}: insufficient GT overlap "
                        f"cached={n_cached} time={scene_elapsed:.1f}s"
                    )
            else:
                tqdm.write(
                    f"  [SKIP] {scene_id}: no GT poses or cache "
                    f"cached={n_cached} time={scene_elapsed:.1f}s"
                )
            continue

        # DA3 inference + Sim3 alignment
        n_images = len(all_img_paths)
        chunks = build_overlapping_chunks(n_images, chunk_size, overlap)

        # Store per-chunk results for alignment
        chunk_results = []  # list of dicts with depth, c2w, intrinsics, conf, indices, etc.

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_paths = all_img_paths[chunk_start:chunk_end]
            chunk_path_strs = [str(p) for p in chunk_paths]

            try:
                # Use InputProcessor for correct preprocessing:
                # - Aspect-ratio-preserving resize (longest side to process_res)
                # - Pad to patch_size=14 aligned dimensions
                # - ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                batch_tensor, _, _ = input_processor(
                    chunk_path_strs,
                    extrinsics=None,
                    intrinsics=None,
                    process_res=args.resolution,
                    process_res_method="upper_bound_resize",
                    sequential=True,  # Avoid forking in CUDA processes
                )
                # batch_tensor: (1, N, 3, H, W) - already normalized

                batch_tensor = batch_tensor.to(device)
                # InputProcessor returns (N, 3, H, W); model expects (B, S, 3, H, W)
                if batch_tensor.dim() == 4:
                    batch_tensor = batch_tensor.unsqueeze(0)  # (1, N, 3, H, W)
                N = batch_tensor.shape[1]

                # Run through model with proper autocast
                output = da3.forward(
                    batch_tensor,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False,
                    use_ray_pose=args.use_ray_pose,
                )

                # Extract outputs - all (1, N, ...) with batch dim
                depths = output['depth'][0].cpu().numpy()          # (N, H, W)
                extrinsics_raw = output['extrinsics'][0].cpu().numpy()  # (N, 3, 4) w2c (NOT c2w!)
                intrinsics_out = output['intrinsics'][0].cpu().numpy()  # (N, 3, 3)

                conf = None
                if 'depth_conf' in output and output['depth_conf'] is not None:
                    conf = output['depth_conf'][0].cpu().numpy()   # (N, H, W)

                # Convert to 4x4 c2w
                c2w = extract_c2w_from_extrinsics(extrinsics_raw)  # (N, 4, 4)

                proc_h, proc_w = depths.shape[1], depths.shape[2]

                chunk_results.append({
                    'chunk_idx': chunk_idx,
                    'start': chunk_start,
                    'end': chunk_end,
                    'depths': depths,
                    'c2w': c2w,
                    'intrinsics': intrinsics_out,
                    'conf': conf,
                    'proc_hw': (proc_h, proc_w),
                    'frame_names': [p.stem for p in chunk_paths],
                    'orig_hws': [],  # filled below
                })

                # Get original image sizes
                from PIL import Image as PILImage
                for p in chunk_paths:
                    img = PILImage.open(p)
                    chunk_results[-1]['orig_hws'].append((img.height, img.width))
                    img.close()

            except Exception as e:
                print(f"\n[Rank {local_rank}] Error processing chunk {chunk_idx} "
                      f"in {scene_id}: {e}")
                import traceback
                traceback.print_exc()
                # Store empty result so alignment indexing stays correct
                chunk_results.append(None)
                continue

        # Cross-chunk alignment
        # Chain Sim(3) transforms to bring all chunks into chunk 0's coordinate frame.
        # For each pair of adjacent chunks, use overlapping frames' camera centers
        # to compute the alignment.

        # Accumulated transforms: sim3_chain[i] maps chunk i's frame to chunk 0's frame
        # sim3_chain[0] = identity
        sim3_chain = [(1.0, np.eye(3), np.zeros(3))]  # (s, R, t) for chunk 0

        for i in range(1, len(chunk_results)):
            prev_cr = chunk_results[i - 1]
            curr_cr = chunk_results[i]

            if prev_cr is None or curr_cr is None:
                # Can't align, use previous transform (best effort)
                sim3_chain.append(sim3_chain[-1])
                continue

            # Find overlapping global indices
            overlap_start = curr_cr['start']
            overlap_end = prev_cr['end']
            n_overlap = overlap_end - overlap_start

            if n_overlap < 3:
                # Not enough overlap for Sim(3), use previous transform
                sim3_chain.append(sim3_chain[-1])
                continue

            # Indices within each chunk's local arrays
            # In prev chunk: overlap frames are at local indices [overlap_start - prev_cr['start'] : ]
            prev_local_start = overlap_start - prev_cr['start']
            prev_local_end = overlap_end - prev_cr['start']
            # In curr chunk: overlap frames are at local indices [0 : n_overlap]
            curr_local_end = n_overlap

            # Camera centers in each chunk's frame
            prev_centers = prev_cr['c2w'][prev_local_start:prev_local_end, :3, 3]  # (O, 3)
            curr_centers = curr_cr['c2w'][:curr_local_end, :3, 3]                  # (O, 3)

            # Use depth confidence as weights if available
            weights = None
            if prev_cr['conf'] is not None and curr_cr['conf'] is not None:
                # Average confidence per frame as weight
                w_prev = prev_cr['conf'][prev_local_start:prev_local_end].mean(axis=(1, 2))
                w_curr = curr_cr['conf'][:curr_local_end].mean(axis=(1, 2))
                weights = np.sqrt(w_prev * w_curr)

            # Apply chain to prev_centers: bring them into chunk 0's frame
            s_prev, R_prev, t_prev = sim3_chain[-1]
            prev_centers_aligned = s_prev * (R_prev @ prev_centers.T).T + t_prev

            # Compute Sim(3) from curr chunk's frame to chunk 0's frame
            # curr_centers -> prev_centers_aligned
            s_local, R_local, t_local = umeyama_sim3(curr_centers, prev_centers_aligned, weights)

            sim3_chain.append((s_local, R_local, t_local))

        # Save GT frames with aligned poses
        # For each chunk, apply its Sim(3) transform and save GT frames

        # Track which global indices have been saved (avoid duplicates from overlaps)
        saved_frames = set()

        for cr_idx, cr in enumerate(chunk_results):
            if cr is None:
                continue

            s, R, t = sim3_chain[cr_idx]

            # Apply Sim(3) to this chunk's c2w poses
            aligned_c2w = apply_sim3_to_c2w(cr['c2w'], s, R, t)
            # Scale depth by s
            aligned_depths = cr['depths'] * s

            chunk_frame_names = cr['frame_names']

            for local_idx in range(len(chunk_frame_names)):
                frame_name = chunk_frame_names[local_idx]
                global_idx = cr['start'] + local_idx

                # Skip if already saved from a previous overlapping chunk
                if global_idx in saved_frames:
                    continue

                # Skip if not a GT frame (when only_with_gt)
                if args.only_with_gt and frame_name not in gt_frames:
                    continue

                # Skip if not in the set needing processing
                if frame_name not in gt_to_process:
                    continue

                cache_path = scene_cache_dir / f"{frame_name}.pt"

                cache_data = {
                    'depth': torch.from_numpy(aligned_depths[local_idx]).half(),
                    'extrinsics': torch.from_numpy(aligned_c2w[local_idx]).float(),
                    'intrinsics': torch.from_numpy(cr['intrinsics'][local_idx]).float(),
                    'proc_hw': cr['proc_hw'],
                    'orig_hw': cr['orig_hws'][local_idx],
                    'chunk_id': cr_idx,
                    'chunk_frames': chunk_frame_names,
                    'chunk_idx_in_chunk': local_idx,
                }
                if cr['conf'] is not None:
                    cache_data['depth_conf'] = torch.from_numpy(
                        cr['conf'][local_idx]).half()

                torch.save(cache_data, cache_path)
                saved_frames.add(global_idx)

        # Procrustes alignment check
        scene_elapsed = time.time() - scene_start_time
        n_chunks = len(chunk_results)
        n_saved = len(saved_frames)

        if args.procrustes and sd['gt_poses']:
            # Collect DA3 aligned c2w for all saved frames
            da3_c2w_dict = {}
            for cr_idx, cr in enumerate(chunk_results):
                if cr is None:
                    continue
                s, R, t_vec = sim3_chain[cr_idx]
                aligned_c2w = apply_sim3_to_c2w(cr['c2w'], s, R, t_vec)
                for local_idx, fname in enumerate(cr['frame_names']):
                    if fname not in da3_c2w_dict:  # first occurrence wins (avoid overlap dups)
                        da3_c2w_dict[fname] = aligned_c2w[local_idx]

            pr = procrustes_check(da3_c2w_dict, sd['gt_poses'], scene_id)
            if pr:
                procrustes_results.append(pr)
                status = "OK" if pr['global_mean_cm'] < args.procrustes_threshold else "BAD"
                if status == "BAD":
                    bad_alignment_scenes.append(pr)
                tqdm.write(
                    f"  [{status}] {scene_id}: global={pr['global_mean_cm']:.1f}cm "
                    f"(max={pr['global_max_cm']:.1f}cm) scale={pr['scale']:.3f} "
                    f"chunks={n_chunks} saved={n_saved} time={scene_elapsed:.0f}s"
                )
            else:
                tqdm.write(
                    f"  [SKIP] {scene_id}: insufficient GT overlap for Procrustes "
                    f"chunks={n_chunks} saved={n_saved} time={scene_elapsed:.0f}s"
                )
        else:
            tqdm.write(
                f"  [DONE] {scene_id}: chunks={n_chunks} saved={n_saved} "
                f"time={scene_elapsed:.0f}s"
            )

    # Write Procrustes summary
    if procrustes_results:
        # Write per-rank results to a temp file, main rank aggregates after barrier
        rank_results_path = cache_dir / f".procrustes_rank{local_rank}.json"
        with open(rank_results_path, 'w') as f:
            json.dump({'results': procrustes_results, 'bad': bad_alignment_scenes}, f)

    # Cleanup
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        # Aggregate Procrustes results from all ranks
        all_procrustes = []
        all_bad = []
        for rank in range(world_size):
            rank_path = cache_dir / f".procrustes_rank{rank}.json"
            if rank_path.exists():
                with open(rank_path) as f:
                    data = json.load(f)
                all_procrustes.extend(data['results'])
                all_bad.extend(data['bad'])
                rank_path.unlink()  # cleanup temp file

        print()
        print(f"Done! Cache saved to: {cache_dir}")
        print(f"  Processing resolution: {args.resolution} (aspect-ratio preserved)")
        print(f"  Chunk size: {chunk_size}, Overlap: {overlap}")
        print(f"  Each .pt file contains:")
        print(f"    - depth: [H, W] float16 (metric meters, globally aligned)")
        print(f"    - extrinsics: [4, 4] float32 (camera-to-world, globally aligned)")
        print(f"    - intrinsics: [3, 3] float32")
        print(f"    - depth_conf: [H, W] float16 (confidence >= 1.0)")
        print(f"    - chunk_id: int (for chunk_aware sampling)")

        # Procrustes summary
        if all_procrustes:
            errs = [r['global_mean_cm'] for r in all_procrustes]
            scales = [r['scale'] for r in all_procrustes]
            print()
            print(f"  Procrustes Alignment Summary ({len(all_procrustes)} scenes):")
            print(f"    Global error:  median={np.median(errs):.1f}cm  mean={np.mean(errs):.1f}cm  max={max(errs):.1f}cm")
            print(f"    Scale:         min={min(scales):.3f}  max={max(scales):.3f}  median={np.median(scales):.3f}")
            print(f"    Good (<{args.procrustes_threshold}cm): {sum(1 for e in errs if e < args.procrustes_threshold)}")
            print(f"    Bad  (>={args.procrustes_threshold}cm): {sum(1 for e in errs if e >= args.procrustes_threshold)}")

            if all_bad:
                bad_log_path = cache_dir / "bad_alignment.txt"
                with open(bad_log_path, 'w') as f:
                    for r in sorted(all_bad, key=lambda x: -x['global_mean_cm']):
                        f.write(f"{r['scene_id']} {r['global_mean_cm']:.1f}cm "
                                f"max={r['global_max_cm']:.1f}cm scale={r['scale']:.3f}\n")
                print(f"    Bad scenes written to: {bad_log_path}")

        print()
        print(f"To use in training:")
        print(f"  --use-cached-depth --da3-cache-name {args.cache_name}")
        print()

if __name__ == "__main__":
    main()
